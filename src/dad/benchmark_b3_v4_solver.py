"""b3-v4: fixed-mode search with B2 health-reference economic weights.

Only explicit solve calls optimize; no native teacher or training is launched here.
"""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata as md
import multiprocessing as mp
import queue
import threading
from pathlib import Path
import time

import numpy as np
import torch

from .common import digest, read_json
from .hybrid import HeatNetwork, HybridModel, InitialState, Physics
from .benchmark_b2_tables import AgingTables


def load_model(path):
    path = Path(path)
    complete = read_json(path.parent / 'complete.json')
    if complete.get('best') != path.name or complete.get('sha256') != digest(path):
        raise ValueError('B3 requires the unchanged completed stage-2 model')
    # Local training artifacts contain tensors and normalizer metadata.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('stage') != 'stage2':
        raise ValueError('B3 requires stage2-best.pt, not a random/untrained network')
    stats = saved['normalizers']
    heat = HeatNetwork(**{k: stats[k] for k in
                         ('input_center', 'input_scale', 'heat_center_w', 'heat_scale_w')})
    heat.load_state_dict(saved['network'], strict=True)
    heat.eval().requires_grad_(False)
    physics = Physics(**saved['physics'])
    if (physics.eta_charge, physics.eta_discharge, physics.internal_dt_s) != (.91, .97, 10.):
        raise ValueError('model does not match confirmed B3 SOC/thermal coefficients')
    if any(not torch.isfinite(v).all() for v in heat.state_dict().values()):
        raise ValueError('nonfinite trained model')
    return HybridModel(physics, heat).eval()


class WorkLimit(Exception):
    pass


class Deadline(Exception):
    pass


from copy import deepcopy

def build_aging_scaling(checkpoint, energy_kwh, tables=None):
    """Reuse B2's native-total-SOH endpoint ratios, without changing model health."""
    if tables is None:
        tables = AgingTables()
    native_soh = float(checkpoint['battery']['soh_Q'])
    return deepcopy(tables.select(native_soh, energy_kwh)['audit'])


class Search:
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
        self.aging_scaling = deepcopy(request['aging_scaling'])
        self.beta_cal = float(self.aging_scaling['cal']['scaling_factor'])
        self.beta_cyc = float(self.aging_scaling['cycl']['scaling_factor'])
        if (self.aging_scaling['native_soh'] != request['checkpoint']['battery']['soh_Q'] or
                not np.isfinite([self.beta_cal, self.beta_cyc]).all() or
                min(self.beta_cal, self.beta_cyc) <= 0):
            raise ValueError('invalid daily economic aging weights')
        self.low, self.high = request['soc_bounds']
        self.publish = publish
        self.emit_seed = lambda record: None
        self.best = None
        self.branches, self.seeds = [], []
        self.current_branch = None
        self.initialization_seconds = 0.
        self.stats = dict(milp_solves=0, projections=0, forwards=0, backwards=0,
                          main_steps=0, rounds=0, invalid_evaluations=0,
                          boundary_trials=0, power_shift_attempts=0, power_shift_accepted=0)
        self.reason = 'initializing'

    def check(self):
        if self.stats['forwards'] >= self.o['max_evaluations']:
            raise WorkLimit('evaluation_budget')
        if time.monotonic() >= self.deadline:
            raise Deadline()

    def bump(self, key):
        self.stats[key] += 1
        if self.current_branch is not None:
            counts = self.current_branch['record']['counts']
            counts[key] = counts.get(key, 0)+1

    def snapshot(self):
        return [dict(record, counts=dict(record['counts'])) for record in self.seeds]

    def same_mode(self, power):
        if self.current_branch is None:
            return False
        p = np.asarray(power, dtype=float)
        return (p.shape == (self.n,) and np.isfinite(p).all() and
                np.array_equal(np.sign(p), self.current_branch['initial_mode']) and
                not np.any((abs(p) < self.o['zero_tolerance_kw']) & (p != 0)))

    def feasible(self, p):
        """Check power and the initial state; SOC constraints remain in MILP/QP."""
        tol = self.o['feasibility_tolerance']
        p = np.asarray(p, dtype=float)
        if np.shape(p) != (self.n,) or not np.isfinite(p).all():
            return False
        return bool(np.min(p) >= -self.charge_cap-tol and np.max(p) <= self.discharge_cap+tol and
                    0 <= self.initial.soc <= 1)

    def canonicalize(self, power):
        """Normalize tiny powers before complete-model evaluation."""
        p = np.asarray(power, dtype=float).copy()
        zeroed = p.copy()
        zeroed[np.abs(zeroed) < self.o['zero_tolerance_kw']] = 0.
        return zeroed

    def recovery_mode(self):
        """Only future endpoints obey the scheduling band; keep the measured initial SOC."""
        if self.initial.soc < self.low:
            return -1
        if self.initial.soc > self.high:
            return 1
        return None

    def arbitrage_power(self, record):
        """Each cap uses the original MILP's AC throughput, never the previous seed."""
        self.check()
        self.stats['milp_solves'] += 1
        record['stop_reason'] = 'milp_running'
        self.emit_seed(deepcopy(record))
        cp, ph = self.cp, self.model.physics
        cap = record['q_cap_kwh']
        with self.env.createModel('B3_v4_arbitrage') as problem:
            for key, value in dict(Logging=0, Threads=self.o['threads'], FeasTol=1e-9, IntTol=1e-9).items():
                problem.setParam(key, value)
            charge = [problem.addVar(lb=0, ub=self.charge_cap) for _ in range(self.n)]
            discharge = [problem.addVar(lb=0, ub=self.discharge_cap) for _ in range(self.n)]
            bc = [problem.addVar(vtype=cp.COPT.BINARY) for _ in range(self.n)]
            bd = [problem.addVar(vtype=cp.COPT.BINARY) for _ in range(self.n)]
            soc = [problem.addVar(lb=self.low, ub=self.high) for _ in range(self.n)]
            for t in range(self.n):
                problem.addConstr(charge[t] <= self.charge_cap*bc[t])
                problem.addConstr(discharge[t] <= self.discharge_cap*bd[t])
                problem.addConstr(bc[t]+bd[t] <= 1)
                problem.addConstr(soc[t] == (self.initial.soc if t == 0 else soc[t-1])
                                  + .25/self.energy*(ph.eta_charge*charge[t]-discharge[t]/ph.eta_discharge))
            if cap is not None:
                problem.addConstr(.25*cp.quicksum(charge[t]+discharge[t]
                                                for t in range(self.n)) <= cap)
            required = self.recovery_mode()
            if required is not None:
                problem.addConstr(bc[0] == int(required < 0))
                problem.addConstr(bd[0] == int(required > 0))
            problem.setObjective(cp.quicksum(.25*float(self.prices[t])*(discharge[t]-charge[t])
                                             for t in range(self.n)), cp.COPT.MAXIMIZE)
            problem.setParam('TimeLimit', min(self.o['subproblem_seconds'],
                                             max(.001, self.deadline-time.monotonic())))
            problem.solve()
            self.check()
            status = int(problem.getAttr('Status'))
            has = bool(problem.getAttr('HasSol'))
            record.update(milp_status=status, milp_has_solution=has,
                          milp_optimal=status == cp.COPT.OPTIMAL)
            if not has:
                record['stop_reason'] = 'milp_infeasible' if status == cp.COPT.INFEASIBLE else 'milp_no_solution'
                return None
            c = np.array([v.x for v in charge])
            d = np.array([v.x for v in discharge])
            tol = self.o['feasibility_tolerance']
            if (not np.isfinite(c).all() or not np.isfinite(d).all() or
                    np.any(c < -tol) or np.any(d < -tol) or
                    np.any(c > self.charge_cap+tol) or np.any(d > self.discharge_cap+tol) or
                    np.any((c > tol) & (d > tol))):
                record['stop_reason'] = 'milp_invalid_power'
                return None
            power = self.canonicalize(np.clip(d-c, -self.charge_cap, self.discharge_cap))
            q = float(.25*np.sum(abs(power)))
            if not self.feasible(power) or (cap is not None and q > cap+tol):
                record['stop_reason'] = 'milp_invalid_power_or_throughput'
                return None
            record['initial_throughput_kwh'] = q
            record['initial_revenue_eur'] = float(.25*np.dot(self.prices, power))
            return power

    def add_seed(self, power, record):
        """Keep the MILP power itself; no power initialization or mode deduplication."""
        if not any(existing is record for existing in self.seeds):
            self.seeds.append(record)
        p = self.canonicalize(power)
        h = np.sign(p).astype(np.int8)
        record.update(initial_power_ac_kw=p.tolist(), initial_mode=h.tolist(),
                      initial_objective_eur=None, final_objective_eur=None,
                      initial_unscaled_objective_eur=None, final_unscaled_objective_eur=None,
                      objective_basis='optimizer_objective_eur', stop_reason='initial_evaluation')
        record.setdefault('counts', {})
        branch = dict(record=record, initial_mode=h, x=self.to_x(p), plan=None,
                      alpha=None, next_step='pgd', round_start=None, active=False)
        self.branches.append(branch)
        self.current_branch = branch
        self.emit_seed(deepcopy(record))
        evaluated = self.evaluate(p)
        if evaluated is None:
            record['stop_reason'] = 'initial_evaluation_failed'
            self.emit_seed(deepcopy(record))
            return branch
        plan, _ = evaluated
        branch.update(plan=plan, round_start=plan['minimized_objective_eur'], active=bool(h.any()))
        record.update(initial_objective_eur=plan['optimizer_objective_eur'],
                      final_objective_eur=plan['optimizer_objective_eur'],
                      initial_unscaled_objective_eur=plan['predicted_objective_eur'],
                      final_unscaled_objective_eur=plan['predicted_objective_eur'],
                      stop_reason='ready' if h.any() else 'all_idle')
        self.keep(plan)
        self.emit_seed(deepcopy(record))
        return branch

    def evaluate(self, p, gradient=False):
        self.check()
        if not self.same_mode(p) or not self.feasible(p):
            return None
        power = torch.tensor(p, dtype=torch.float64, requires_grad=gradient)
        self.bump('forwards')
        try:
            with torch.set_grad_enabled(gradient):
                result = self.model(power, self.initial, self.ambient, .25)
                # Reject floating-point loss of a tiny active SOC increment;
                # never silently keep its old half-cycle duration.
                movement = np.sign(np.diff(result.soc.detach().numpy()))
                if not np.array_equal(movement, -np.sign(p)):
                    self.bump('invalid_evaluations')
                    return None
                # These day-start coefficients are constant across powers and branches.
                objective = (-.25*torch.dot(torch.tensor(self.prices), power) + self.kappa*
                             (self.beta_cal*result.calendar_damage + self.beta_cyc*result.cyclic.damage))
                values = [objective, result.calendar_damage, result.cyclic.damage,
                          *result.soc, *result.temperature_c, *result.heat_w]
                if not all(bool(torch.isfinite(v)) for v in values):
                    self.bump('invalid_evaluations')
                    return None
                grad = None
                if gradient:
                    self.bump('backwards')
                    grad = torch.autograd.grad(objective, power)[0].detach().numpy()
                    if not np.isfinite(grad).all():
                        self.bump('invalid_evaluations')
                        return None
            calendar, cyclic = float(result.calendar_damage.detach()), float(result.cyclic.damage.detach())
            revenue = float(.25*np.dot(self.prices, p))
            penalty = self.kappa*(self.beta_cal*calendar+self.beta_cyc*cyclic)
            plan = dict(success=True, power_ac_kw=p.tolist(), mode=np.sign(p).astype(int).tolist(),
                        branch_id=self.current_branch['record']['branch_id'],
                        initial_mode=self.current_branch['initial_mode'].tolist(),
                        predicted_soc=result.soc[1:].detach().tolist(),
                        predicted_temperature_c=result.temperature_c[1:].detach().tolist(),
                        predicted_damage_components_pu=[calendar, cyclic],
                        predicted_total_damage_pu=calendar+cyclic,
                        predicted_revenue_eur=revenue,
                        predicted_aging_cost_eur=self.kappa*(calendar+cyclic),
                        optimizer_aging_penalty_eur=penalty,
                        optimizer_objective_eur=revenue-penalty,
                        predicted_objective_eur=revenue-self.kappa*(calendar+cyclic),
                        minimized_objective_eur=penalty-revenue,
                        aging_scaling=deepcopy(self.aging_scaling),
                        aging_prediction_available=True)
            return plan, grad
        except (ValueError, FloatingPointError, OverflowError):
            self.bump('invalid_evaluations')
            return None
    def keep(self, plan):
        # This gate is before both branch and global publication, including rejected line-search trials.
        if (not self.same_mode(plan['power_ac_kw']) or
                plan.get('branch_id') != self.current_branch['record']['branch_id'] or
                not np.array_equal(plan.get('initial_mode'), self.current_branch['initial_mode'])):
            raise ValueError('fixed branch mode changed before publication')
        if plan.get('aging_scaling') != self.aging_scaling:
            raise ValueError('day-start economic weights changed before publication')
        record = self.current_branch['record']
        objective = plan['optimizer_objective_eur']
        if record['final_objective_eur'] is None or objective > record['final_objective_eur']:
            record['final_objective_eur'] = objective
            record['final_unscaled_objective_eur'] = plan['predicted_objective_eur']
            self.emit_seed(deepcopy(record))
        if self.best is None or plan['minimized_objective_eur'] < self.best['minimized_objective_eur']:
            self.best = plan
            self.publish(dict(plan, search_counts=dict(self.stats),
                              initialization_seconds=self.initialization_seconds,
                              seeds=self.snapshot()))

    def to_x(self, p):
        ph = self.model.physics
        return np.where(p < 0, ph.eta_charge*p, p/ph.eta_discharge)

    def to_p(self, x):
        ph = self.model.physics
        return np.where(x < 0, x/ph.eta_charge, ph.eta_discharge*x)

    def x_eval(self, x, gradient=False):
        p = self.canonicalize(self.to_p(x))
        if not self.same_mode(p):
            self.bump('boundary_trials')
            return None
        value = self.evaluate(p, gradient)
        if value is None:
            return None
        plan, gp = value
        self.keep(plan)
        actual = self.to_x(p)
        if gp is not None:
            ph = self.model.physics
            gp = gp*np.where(actual < 0, 1/ph.eta_charge, ph.eta_discharge)
            gp[self.current_branch['initial_mode'] == 0] = 0.
        return actual, plan, gp

    def project_x(self, center):
        self.check()
        self.bump('projections')
        cp, ph = self.cp, self.model.physics
        lo, hi = -ph.eta_charge*self.charge_cap, self.discharge_cap/ph.eta_discharge
        mode = self.current_branch['initial_mode']
        with self.env.createModel('b3-v4-projection') as problem:
            for key, value in dict(Logging=0, Threads=self.o['threads'], FeasTol=1e-6, Presolve=-1).items():
                problem.setParam(key, value)
            xs = [problem.addVar(lb=lo if m < 0 else 0., ub=hi if m > 0 else 0.) for m in mode]
            ss = [problem.addVar(lb=self.low, ub=self.high) for _ in range(self.n)]
            for t in range(self.n):
                problem.addConstr(ss[t] == (self.initial.soc if t == 0 else ss[t-1])
                                  - .25/self.energy*xs[t])
            problem.setObjective(cp.quicksum(.5*xs[t]*xs[t]-float(center[t])*xs[t]
                                             for t in range(self.n)), cp.COPT.MINIMIZE)
            self.check()
            problem.setParam('TimeLimit', min(self.o['subproblem_seconds'],
                                             max(.001, self.deadline-time.monotonic())))
            problem.solve()
            self.check()
            status = int(problem.getAttr('Status'))
            if status == cp.COPT.INFEASIBLE:
                return None, True
            if status != cp.COPT.OPTIMAL or not problem.getAttr('HasLpSol'):
                return None, False
            out = np.array([v.x for v in xs])
            out[mode == 0] = 0.
            if not self.feasible(self.to_p(out)):
                return None, False
            return out, True

    def candidate(self, x, plan, g, y):
        # Canonicalization can turn an otherwise same-sign point into idle.
        p = self.canonicalize(self.to_p(y))
        if not self.same_mode(p):
            self.bump('boundary_trials')
            return None, 'mode_boundary'
        evaluated = self.x_eval(y)
        if evaluated is None:
            return None, 'numerical_failure'
        w, trial, _ = evaluated
        slope = float(g@(w-x))
        threshold = max(self.o['improvement_eur'], -self.o['armijo']*slope)
        if slope < 0 and plan['minimized_objective_eur']-trial['minimized_objective_eur'] >= threshold:
            return (w, trial), None
        return None, None

    def full_step(self, x, plan):
        self.check()
        evaluated = self.x_eval(x, True)
        if evaluated is None:
            return None, 'gradient_evaluation_failed'
        x, plan, g = evaluated
        ph = self.model.physics
        scale = np.max(abs(g*np.where(x < 0, 1/ph.eta_charge, ph.eta_discharge)))
        if scale == 0:
            return None, None
        branch = self.current_branch
        alpha = branch['alpha'] if branch['alpha'] is not None else self.o['initial_step_fraction']*9.6/scale
        boundary = False
        for _ in range(self.o['backtracks']):
            y, ok = self.project_x(x-alpha*g)
            if not ok:
                return None, 'projection_numerical_failure'
            if y is not None:
                if np.array_equal(y, x):
                    return None, None
                result, status = self.candidate(x, plan, g, y)
                boundary |= status == 'mode_boundary'
                if status == 'numerical_failure':
                    return None, status
                if result is not None:
                    branch['alpha'] = alpha
                    return result, None
            alpha *= .5
        return None, 'mode_boundary_backtracks' if boundary else None

    def shift_direction(self, x, g):
        ph = self.model.physics
        lo, hi = -ph.eta_charge*self.charge_cap, self.discharge_cap/ph.eta_discharge
        soc = self.initial.soc-.25/self.energy*np.cumsum(x)
        best, score = None, 0.
        mode = self.current_branch['initial_mode']
        for i in range(self.n):
            self.check()
            for j in range(i+1, self.n):
                if mode[i] == 0 or mode[i] != mode[j] or g[i] == g[j]:
                    continue
                sign = -np.sign(g[i]-g[j])
                v = np.zeros(self.n)
                v[i], v[j] = sign, -sign
                power = min(hi-x[i], x[j]-lo) if sign > 0 else min(x[i]-lo, hi-x[j])
                slack = min(soc[i:j]-self.low) if sign > 0 else min(self.high-soc[i:j])
                zero = min(abs(x[t]) for t in (i,j) if x[t]*v[t] < 0)
                delta = min(power, slack*self.energy/.25, zero)
                value = abs(g[i]-g[j])*delta
                if delta > 0 and value > score:
                    best, score = (v, delta), value
        return best

    def shift_step(self, x, plan):
        self.check()
        if np.count_nonzero(x) < 2:
            return None, None
        evaluated = self.x_eval(x, True)
        if evaluated is None:
            return None, 'gradient_evaluation_failed'
        x, plan, g = evaluated
        pair = self.shift_direction(x, g)
        if pair is None:
            return None, None
        self.bump('power_shift_attempts')
        v, delta = pair
        boundary = False
        for _ in range(self.o['backtracks']):
            result, status = self.candidate(x, plan, g, x+delta*v)
            boundary |= status == 'mode_boundary'
            if status == 'numerical_failure':
                return None, status
            if result is not None:
                self.current_branch['alpha'] = None
                self.bump('power_shift_accepted')
                return result, None
            delta *= .5
        return None, 'mode_boundary_backtracks' if boundary else None

    def stop_branch(self, branch, reason):
        branch['active'] = False
        branch['record']['stop_reason'] = reason
        self.emit_seed(deepcopy(branch['record']))

    def finish(self, reason):
        self.reason = reason
        for branch in self.branches:
            if branch['active']:
                self.stop_branch(branch, reason)
        for record in self.seeds:
            if record['stop_reason'] in ('not_started', 'milp_running', 'initial_evaluation'):
                record['stop_reason'] = reason
                self.emit_seed(deepcopy(record))

    def solve(self):
        self.seeds = [dict(branch_id=i, alpha=1. if i == 0 else i/10,
                           q_reference_kwh=None, q_cap_kwh=None, stop_reason='not_started', counts={})
                      for i in range(10)]
        began = time.monotonic()
        q_reference = None
        try:
            for record in self.seeds:
                self.current_branch = None
                record['q_reference_kwh'] = q_reference
                record['q_cap_kwh'] = None if record['branch_id'] == 0 else record['alpha']*q_reference
                p = self.arbitrage_power(record)
                if p is None:
                    self.emit_seed(deepcopy(record))
                    if record['branch_id'] == 0:
                        for remaining in self.seeds[1:]:
                            remaining['stop_reason'] = 'missing_reference_milp'
                            self.emit_seed(deepcopy(remaining))
                        self.reason = 'initial_milp_failed'
                        return
                    continue
                if record['branch_id'] == 0:
                    q_reference = float(.25*np.sum(abs(p)))
                    record['q_reference_kwh'] = q_reference
                self.initialization_seconds = time.monotonic()-began
                self.add_seed(p, record)
        finally:
            self.initialization_seconds = time.monotonic()-began
        # One main step per live branch per pass; each branch alternates PGD and shift.
        while any(branch['active'] for branch in self.branches):
            for branch in self.branches:
                if not branch['active']:
                    continue
                self.current_branch = branch
                self.check()
                if self.stats['main_steps'] >= self.o['max_main_steps']:
                    self.finish('main_step_budget')
                    return
                self.bump('main_steps')
                is_pgd = branch['next_step'] == 'pgd'
                if is_pgd:
                    branch['round_start'] = branch['plan']['minimized_objective_eur']
                step = self.full_step if is_pgd else self.shift_step
                result, stop = step(branch['x'], branch['plan'])
                if result is not None:
                    branch['x'], branch['plan'] = result
                if stop is not None:
                    self.stop_branch(branch, stop)
                    continue
                branch['next_step'] = 'shift' if is_pgd else 'pgd'
                if not is_pgd:
                    self.bump('rounds')
                    gain = branch['round_start']-branch['plan']['minimized_objective_eur']
                    if gain < self.o['progress_eur']:
                        self.stop_branch(branch, 'stalled_fixed_mode')
        self.reason = 'branches_finished'


def worker(connection, model_path, options, search_class=Search):
    env = None
    try:
        torch.set_num_threads(options['threads'])
        if md.version('coptpy') != '8.0.7':
            raise RuntimeError('B3 requires coptpy==8.0.7')
        import coptpy as cp
        settings = cp.EnvrConfig()
        settings.set('nobanner', '1')
        env = cp.Envr(settings)
        model = load_model(model_path)
        connection.send(('ready', dict(physics=asdict(model.physics), model_sha256=digest(model_path))))
        while True:
            request = connection.recv()
            if request is None:
                return
            search = search_class(model, env, cp, options, request,
                            lambda plan: connection.send(('incumbent', plan)))
            search.emit_seed = lambda record: connection.send(('seed', record))
            try:
                search.solve()
            except WorkLimit as exc:
                search.finish(str(exc))
            except Deadline:
                search.finish('decision_time_budget')
            connection.send(('done', dict(reason=search.reason, counts=search.stats, seeds=search.snapshot(),
                                          initialization_seconds=search.initialization_seconds)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(('error', f'{type(exc).__name__}: {exc}'))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            env.__exit__(None, None, None)
        connection.close()


class B3V4Solver:
    method = 'b3-v4'
    intervals = 96
    search_class = Search

    def __init__(self, options, model_path):
        self.options = dict(options)
        self.model_path = str(Path(model_path).resolve())
        self.process = self.connection = None
        self.metadata = None
        self.seed_records = {}
        self.aging_tables = None

    def _start(self, seconds):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker,
            args=(child, self.model_path, self.options, self.search_class), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(max(0., seconds)):
            self._stop()
            raise TimeoutError('B3 worker/model/license initialization timed out')
        kind, value = self.connection.recv()
        if kind != 'ready':
            self._stop()
            raise RuntimeError(f'B3 worker startup: {value}')
        if self.metadata is not None and value != self.metadata:
            self._stop()
            raise ValueError('B3 trained model changed during run')
        self.metadata = value

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

    def register_seed(self, record, request):
        """Bind a branch to its separately received original MILP power and signs."""
        branch_id = record['branch_id']
        if type(branch_id) is not int or not 0 <= branch_id <= 9:
            raise ValueError('invalid b3-v4 branch id')
        expected_alpha = 1. if branch_id == 0 else branch_id/10
        if record['alpha'] != expected_alpha:
            raise ValueError('invalid fixed throughput fraction')
        previous = self.seed_records.get(branch_id)
        if previous is not None and 'initial_power_ac_kw' in previous:
            for key in ('initial_power_ac_kw', 'initial_mode', 'q_reference_kwh', 'q_cap_kwh'):
                if record.get(key) != previous.get(key):
                    raise ValueError('registered MILP seed changed')
        if 'initial_power_ac_kw' in record:
            p = np.asarray(record['initial_power_ac_kw'], dtype=float)
            h = np.asarray(record['initial_mode'])
            lower, upper = request['power_bounds_ac_kw']
            tol = self.options['feasibility_tolerance']
            if (p.shape != (self.intervals,) or not np.isfinite(p).all() or
                    np.min(p) < lower-tol or np.max(p) > upper+tol or
                    not np.array_equal(np.sign(p), h) or
                    np.any((abs(p) < self.options['zero_tolerance_kw']) & (p != 0))):
                raise ValueError('invalid original MILP power/mode')
            q = float(.25*np.sum(abs(p)))
            if branch_id == 0:
                if record['q_cap_kwh'] is not None or not np.isclose(q, record['q_reference_kwh'], atol=1e-10, rtol=0):
                    raise ValueError('invalid original MILP throughput reference')
            else:
                original = self.seed_records.get(0, {})
                reference = original.get('q_reference_kwh')
                if (reference is None or record['q_reference_kwh'] != reference or
                        record['q_cap_kwh'] != expected_alpha*reference or q > record['q_cap_kwh']+tol):
                    raise ValueError('seed does not use the common throughput reference')
        self.seed_records[branch_id] = deepcopy(record)

    def solve(self, prices, *, checkpoint, kappa, power_bounds_ac_kw, ambient_c=25., soc_min=.1, soc_max=.9):
        self.seed_records = {i: dict(branch_id=i, alpha=1. if i == 0 else i/10,
                                     q_reference_kwh=None, q_cap_kwh=None,
                                     stop_reason='not_started', counts={}) for i in range(10)}
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
        info = dict(reason='watchdog_deadline', counts={}, seeds=[])
        aging_scaling_seconds = 0.
        finished = False
        try:
            if self.process is None or not self.process.is_alive():
                self._stop()
                self._start(max(0., work_end-time.monotonic()))
            # Table loading and the day's selection consume this same decision budget.
            if time.monotonic() >= work_end:
                raise Deadline()
            scaling_began = time.monotonic()
            try:
                if self.aging_tables is None:
                    self.aging_tables = AgingTables()
                energy_kwh = (self.metadata['physics']['energy_kwh']*
                              float(checkpoint['battery']['soh_Q']))
                request['aging_scaling'] = build_aging_scaling(checkpoint, energy_kwh, self.aging_tables)
            finally:
                aging_scaling_seconds = time.monotonic()-scaling_began
            if time.monotonic() >= work_end:
                raise Deadline()
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
                elif kind == 'seed':
                    self.register_seed(value, request)
                elif kind == 'error':
                    raise RuntimeError(value)
        except TimeoutError:
            info['reason'] = 'worker_restart_time_budget'
        except Deadline:
            info['reason'] = 'aging_scaling_time_budget'
        finally:
            if not finished:
                self._stop()
        if best is not None:
            self.validate_plan(best, request)
            if not info['counts']:
                info['counts'] = best.get('search_counts', {})
                info['initialization_seconds'] = best.get('initialization_seconds', 0.)
                info['seeds'] = best.get('seeds', [])
                info['statistics_scope'] = 'last_published_incumbent'
            else:
                info['statistics_scope'] = 'completed_search'
        if not finished:
            info['seeds'] = [deepcopy(self.seed_records[i]) for i in sorted(self.seed_records)]
            for record in info['seeds']:
                if record['stop_reason'] in ('ready', 'milp_running', 'initial_evaluation'):
                    record['stop_reason'] = info['reason']
        elapsed = time.monotonic()-began
        backend = 'copt MILP/QP + torch autograd'
        info.update(backend=backend, method=self.method, elapsed_seconds=elapsed,
                    max_seconds=self.options['max_seconds'], deadline_exceeded=elapsed > self.options['max_seconds'],
                    model_sha256=self.metadata['model_sha256'] if self.metadata else None,
                    terminal_soc_required=False, hard_watchdog_used=not finished,
                    aging_scaling=deepcopy(request.get('aging_scaling')),
                    aging_scaling_seconds=aging_scaling_seconds)
        if best is None:
            return dict(success=False, reason='no checked feasible B3 plan', solver=info)
        return dict(best, solver=info)

    def validate_plan(self, plan, request):
        p = np.asarray(plan['power_ac_kw'], dtype=float)
        branch_id = plan.get('branch_id')
        registered = self.seed_records.get(branch_id)
        if registered is None or 'initial_mode' not in registered:
            raise ValueError('b3-v4 plan lacks a registered original seed')
        fixed_mode = registered['initial_mode']
        if (not np.array_equal(plan.get('initial_mode'), fixed_mode) or
                not np.array_equal(np.sign(p), fixed_mode)):
            raise ValueError('b3-v4 result crossed its original fixed mode')
        initial = InitialState.from_checkpoint(request['checkpoint'])
        tol = self.options['feasibility_tolerance']
        lower, upper = request['power_bounds_ac_kw']
        if (p.shape != (self.intervals,) or not np.isfinite(p).all() or
                np.min(p) < lower-tol or np.max(p) > upper+tol):
            raise ValueError('invalid B3 power result')
        if not 0 <= initial.soc <= 1:
            raise ValueError('invalid initial planning SOC')
        # SOC comes from the hybrid model; do not recompute or reject its scheduling-band deviation here.
        predicted_soc = np.asarray(plan['predicted_soc'], dtype=float)
        if (predicted_soc.shape != (self.intervals,) or not np.isfinite(predicted_soc).all() or
                not np.array_equal(np.sign(p), plan['mode'])):
            raise ValueError('B3 result mode/state mismatch')
        scaling = request['aging_scaling']
        if plan.get('aging_scaling') != scaling:
            raise ValueError('b3-v4 plan changed the day-start economic weights')
        beta_cal = float(scaling['cal']['scaling_factor'])
        beta_cyc = float(scaling['cycl']['scaling_factor'])
        if (scaling['native_soh'] != request['checkpoint']['battery']['soh_Q'] or
                not np.isfinite([beta_cal, beta_cyc]).all() or min(beta_cal, beta_cyc) <= 0):
            raise ValueError('invalid day-start economic weights')
        cal, cyc = plan['predicted_damage_components_pu']
        revenue = float(.25*np.dot(request['prices'], p))
        raw_cost = request['kappa']*(cal+cyc)
        penalty = request['kappa']*(beta_cal*cal+beta_cyc*cyc)
        for key, value in dict(predicted_total_damage_pu=cal+cyc,
                               predicted_revenue_eur=revenue, predicted_aging_cost_eur=raw_cost,
                               optimizer_aging_penalty_eur=penalty, predicted_objective_eur=revenue-raw_cost,
                               optimizer_objective_eur=revenue-penalty,
                               minimized_objective_eur=penalty-revenue).items():
            if not np.isfinite(value) or not np.isclose(value, plan[key], atol=1e-8, rtol=1e-9):
                raise ValueError(f'B3 objective mismatch: {key}')

        if np.any((np.abs(p) < self.options['zero_tolerance_kw']) & (p != 0)):
            raise ValueError('b3-v4 result was not zero-normalized')
