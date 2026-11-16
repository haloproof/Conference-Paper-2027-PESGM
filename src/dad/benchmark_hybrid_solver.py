"""Frozen hybrid evaluation and bounded worker support for B4 and B5-v4.

No dispatch policy, teacher rollout or training runs on import.
"""
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch

from .common import digest, read_json
from .hybrid import HeatNetwork, HybridModel, InitialState, Physics


def load_model(path):
    path = Path(path)
    complete = read_json(path.parent / 'complete.json')
    if complete.get('best') != path.name or complete.get('sha256') != digest(path):
        raise ValueError('hybrid dispatch requires the unchanged completed stage-2 model')
    # Local training artifacts contain tensors and normalizer metadata.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('stage') != 'stage2':
        raise ValueError('hybrid dispatch requires stage2-best.pt, not a random/untrained network')
    stats = saved['normalizers']
    heat = HeatNetwork(**{k: stats[k] for k in
                         ('input_center', 'input_scale', 'heat_center_w', 'heat_scale_w')})
    heat.load_state_dict(saved['network'], strict=True)
    heat.eval().requires_grad_(False)
    physics = Physics(**saved['physics'])
    if (physics.eta_charge, physics.eta_discharge, physics.internal_dt_s) != (.91, .97, 10.):
        raise ValueError('model does not match confirmed SOC/thermal coefficients')
    if any(not torch.isfinite(v).all() for v in heat.state_dict().values()):
        raise ValueError('nonfinite trained model')
    return HybridModel(physics, heat).eval()


class Deadline(Exception):
    pass


class HybridSearch:
    """Shared full-model scoring and linear planning feasibility, without a search policy."""
    def __init__(self, model, env, cp, options, request, publish):
        self.model, self.env, self.cp, self.o = model, env, cp, options
        self.prices = np.asarray(request['prices'], dtype=float)
        self.initial = InitialState.from_checkpoint(request['checkpoint'])
        self.n = len(self.prices)
        self.charge_cap = -request['power_bounds_ac_kw'][0]
        self.discharge_cap = request['power_bounds_ac_kw'][1]
        self.energy = model.physics.energy_kwh * (1-self.initial.q_calendar-self.initial.q_cyclic)
        self.deadline = request['deadline']
        self.kappa, self.ambient = request['kappa'], request['ambient_c']
        self.low, self.high = request['soc_bounds']
        self.publish = publish
        self.best = None
        self.trace = []
        self.stats = dict(forwards=0, backwards=0, invalid_evaluations=0)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise Deadline()

    def soc(self, p):
        ph = self.model.physics
        return self.initial.soc + np.cumsum(
            .25/self.energy * (ph.eta_charge*np.maximum(-p, 0)-np.maximum(p, 0)/ph.eta_discharge))

    def feasible(self, p):
        tol = self.o['feasibility_tolerance']
        p = np.asarray(p, dtype=float)
        if np.shape(p) != (self.n,) or not np.isfinite(p).all():
            return False
        s = self.soc(p)
        return bool(np.min(p) >= -self.charge_cap-tol and np.max(p) <= self.discharge_cap+tol and
                    0 <= self.initial.soc <= 1 and
                    np.min(s) >= self.low-tol and np.max(s) <= self.high+tol)

    def evaluate(self, p, gradient=False):
        self.check()
        if not self.feasible(p):
            return None
        power = torch.tensor(p, dtype=torch.float64, requires_grad=gradient)
        self.stats['forwards'] += 1
        try:
            with torch.set_grad_enabled(gradient):
                result = self.model(power, self.initial, self.ambient, .25)
                # Reject floating-point loss of a tiny active SOC increment;
                # never silently keep its old half-cycle duration.
                movement = np.sign(np.diff(result.soc.detach().numpy()))
                if not np.array_equal(movement, -np.sign(p)):
                    self.stats['invalid_evaluations'] += 1
                    return None
                objective = -.25*torch.dot(torch.tensor(self.prices), power) + self.kappa*result.total_damage
                values = [objective, result.calendar_damage, result.cyclic.damage,
                          *result.soc, *result.temperature_c, *result.heat_w]
                if not all(bool(torch.isfinite(v)) for v in values):
                    self.stats['invalid_evaluations'] += 1
                    return None
                self.check()
                grad = None
                if gradient:
                    self.stats['backwards'] += 1
                    grad = torch.autograd.grad(objective, power)[0].detach().numpy()
                    if not np.isfinite(grad).all():
                        self.stats['invalid_evaluations'] += 1
                        return None
            self.check()
            calendar, cyclic = float(result.calendar_damage.detach()), float(result.cyclic.damage.detach())
            revenue = float(.25*np.dot(self.prices, p))
            plan = dict(success=True, power_ac_kw=p.tolist(), mode=np.sign(p).astype(int).tolist(),
                        predicted_soc=result.soc[1:].detach().tolist(),
                        predicted_temperature_c=result.temperature_c[1:].detach().tolist(),
                        predicted_damage_components_pu=[calendar, cyclic],
                        predicted_total_damage_pu=calendar+cyclic,
                        predicted_revenue_eur=revenue,
                        predicted_aging_cost_eur=self.kappa*(calendar+cyclic),
                        optimizer_aging_penalty_eur=self.kappa*(calendar+cyclic),
                        predicted_objective_eur=revenue-self.kappa*(calendar+cyclic),
                        minimized_objective_eur=self.kappa*(calendar+cyclic)-revenue,
                        aging_prediction_available=True)
            return plan, grad
        except (ValueError, FloatingPointError, OverflowError):
            self.stats['invalid_evaluations'] += 1
            return None


class HybridSolver:
    """Lifecycle and result checks; subclasses supply their own worker and method."""
    intervals = 96

    def __init__(self, options, model_path):
        self.options = dict(options)
        self.model_path = str(Path(model_path).resolve())
        self.process = self.connection = None
        self.metadata = None

    def _stop(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=.1)
            if not self.process.is_alive():
                self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        self._start(self.options['startup_seconds'])
        return self

    def __exit__(self, *args):
        self._stop()

    def solve(self, prices, *, checkpoint, kappa, power_bounds_ac_kw, ambient_c=25., soc_min=.1, soc_max=.9):
        began = time.monotonic()
        deadline = began+self.options['max_seconds']
        work_end = deadline-self.options['return_reserve_seconds']
        prices = np.asarray(prices, dtype=float)
        if prices.shape != (self.intervals,) or not np.isfinite(prices).all():
            raise ValueError(f'{self.method} requires exactly {self.intervals} finite day-ahead prices in EUR/kWh')
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError('finite nonnegative degradation valuation required')
        bounds = np.asarray(power_bounds_ac_kw, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not bounds[0] < 0 < bounds[1]:
            raise ValueError('finite directional AC power bounds required')
        request = dict(power_bounds_ac_kw=bounds.tolist(), prices=prices.tolist(), checkpoint=checkpoint, kappa=kappa,
                       ambient_c=ambient_c, soc_bounds=[soc_min, soc_max], deadline=work_end)
        best = None
        info = dict(reason='watchdog_deadline', counts={}, candidates=[])
        finished = False
        try:
            if self.process is None or not self.process.is_alive():
                self._stop()
                self._start(max(0., work_end-time.monotonic()))
            self.connection.send(request)
            inbox = queue.Queue()
            connection = self.connection
            def receive():
                # poll may be readable before a complete message has arrived.
                # The thread owning the deadline must not block in recv.
                try:
                    while True:
                        message = connection.recv()
                        inbox.put(message)
                        if message[0] in ('done', 'error'):
                            return
                except (EOFError, OSError) as exc:
                    inbox.put(('error', f'worker connection closed: {exc}'))
            threading.Thread(target=receive, daemon=True).start()
            while time.monotonic() < work_end:
                try:
                    kind, value = inbox.get(timeout=min(.05, max(0., work_end-time.monotonic())))
                except queue.Empty:
                    continue
                if kind == 'incumbent':
                    # Parent rechecks every published feasible result before storing it.
                    self.validate_plan(value, request)
                    best = value
                elif kind == 'done':
                    info, finished = value, True
                    break
                elif kind == 'error':
                    raise RuntimeError(value)
        except TimeoutError:
            info['reason'] = 'worker_restart_time_budget'
        finally:
            if not finished:
                self._stop()
        if best is not None:
            self.validate_plan(best, request)
            if not info['counts']:
                info['counts'] = best.get('search_counts', {})
                info['initialization_seconds'] = best.get('initialization_seconds', 0.)
                info['initial_milp'] = best.get('initial_milp', {})
                info['statistics_scope'] = 'last_published_incumbent'
            else:
                info['statistics_scope'] = 'completed_search'
        elapsed = time.monotonic()-began
        backend = 'copt MILP + frozen hybrid forward evaluation'
        info.update(backend=backend, method=self.method, elapsed_seconds=elapsed,
                    max_seconds=self.options['max_seconds'], deadline_exceeded=elapsed > self.options['max_seconds'],
                    model_sha256=self.metadata['model_sha256'] if self.metadata else None,
                    terminal_soc_required=False, hard_watchdog_used=not finished)
        if best is None:
            return dict(success=False, reason=f'no checked feasible {self.method} plan', solver=info)
        return dict(best, solver=info)

    def validate_plan(self, plan, request):
        p = np.asarray(plan['power_ac_kw'], dtype=float)
        ph = Physics(**self.metadata['physics'])
        initial = InitialState.from_checkpoint(request['checkpoint'])
        energy = ph.energy_kwh*(1-initial.q_calendar-initial.q_cyclic)
        s = initial.soc+np.cumsum(.25/energy*(ph.eta_charge*np.maximum(-p, 0)-np.maximum(p, 0)/ph.eta_discharge))
        tol = self.options['feasibility_tolerance']
        lower, upper = request['power_bounds_ac_kw']
        if (p.shape != (self.intervals,) or not np.isfinite(p).all() or
                np.min(p) < lower-tol or np.max(p) > upper+tol):
            raise ValueError('invalid hybrid dispatch power result')
        if not 0 <= initial.soc <= 1:
            raise ValueError('invalid initial planning SOC')
        if np.min(s) < request['soc_bounds'][0]-tol or np.max(s) > request['soc_bounds'][1]+tol:
            raise ValueError('hybrid dispatch result violates SOC bounds')
        if not np.allclose(s, plan['predicted_soc'], atol=tol, rtol=0) or not np.array_equal(np.sign(p), plan['mode']):
            raise ValueError('hybrid dispatch result mode/state mismatch')
        cal, cyc = plan['predicted_damage_components_pu']
        revenue = float(.25*np.dot(request['prices'], p))
        penalty = request['kappa']*(cal+cyc)
        for key, value in dict(predicted_revenue_eur=revenue, predicted_aging_cost_eur=penalty,
                               optimizer_aging_penalty_eur=penalty, predicted_objective_eur=revenue-penalty,
                               minimized_objective_eur=penalty-revenue).items():
            if not np.isfinite(value) or not np.isclose(value, plan[key], atol=1e-8, rtol=1e-9):
                raise ValueError(f'hybrid objective mismatch: {key}')

