"""b5-v4 (paper M4): ten fixed-throughput candidates with day-start health-scaled utility.

The MILP remains pure arbitrage: pure first, then 0.1--0.9 of its AC throughput.
Raw hybrid damage, including signed residual accounting, is never rescaled in place.
"""
from dataclasses import asdict
import copy
import importlib.metadata as md
import multiprocessing as mp
import queue
import threading
import time

import numpy as np
import torch

from .benchmark_b1_solver import validate_solution
from .benchmark_hybrid_solver import HybridSolver, HybridSearch, Deadline, load_model
from .benchmark_b2_tables import AgingTables
from .benchmark_b3_v4_solver import build_aging_scaling
from .common import digest


class Search(HybridSearch):
    def __init__(self, model, env, cp, options, request, publish):
        super().__init__(model, env, cp, options, request, publish)
        self.method = 'b5-v4'
        if options.get('method', 'b5-v4') != 'b5-v4':
            raise ValueError('b5-v4 options required')
        self.aging_scaling = copy.deepcopy(request['aging_scaling'])
        self.beta_cal = float(self.aging_scaling['cal']['scaling_factor'])
        self.beta_cyc = float(self.aging_scaling['cycl']['scaling_factor'])
        if (self.aging_scaling['native_soh'] != request['checkpoint']['battery']['soh_Q'] or
                not np.isfinite([self.beta_cal, self.beta_cyc]).all() or
                min(self.beta_cal, self.beta_cyc) <= 0):
            raise ValueError('invalid daily economic aging weights')
        self.stats = dict(milp_solves=0, forwards=0, invalid_evaluations=0,
                          candidate_evaluations=0)
        self.selected = None
        self.reason = 'iteration_limit'
        self.milp_attempts = []
        self.converged = False

    def evaluate(self, p, gradient=False):
        """Keep the frozen model's raw ledger; scale only the economic score."""
        if gradient:
            raise ValueError('b5-v4 uses forward-only candidate evaluation')
        evaluated = super().evaluate(p, gradient=False)
        if evaluated is None:
            return None
        plan, _ = evaluated
        calendar, cyclic = plan['predicted_damage_components_pu']
        penalty = self.kappa*(self.beta_cal*calendar+self.beta_cyc*cyclic)
        score = plan['predicted_revenue_eur']-penalty
        if not np.isfinite([penalty, score]).all():
            self.stats['invalid_evaluations'] += 1
            return None
        plan.update(optimizer_aging_penalty_eur=penalty,
                    optimizer_objective_eur=score, minimized_objective_eur=-score,
                    aging_scaling=copy.deepcopy(self.aging_scaling))
        return plan, None

    def milp(self, throughput_limit):
        self.check()
        if throughput_limit is not None and (not np.isfinite(throughput_limit) or throughput_limit < 0):
            raise ValueError('invalid throughput limit')
        cp = self.cp
        began = time.monotonic()
        self.stats['milp_solves'] += 1
        with self.env.createModel(self.method+'_throughput') as problem:
            for key, value in dict(Logging=0, Threads=self.o['threads'],
                    NodeLimit=self.o['node_limit'], RelGap=self.o['relative_gap'],
                    AbsGap=self.o['absolute_gap_eur'], FeasTol=1e-9, IntTol=1e-9).items():
                problem.setParam(key, value)
            c = [problem.addVar(lb=0, ub=self.charge_cap) for _ in self.prices]
            d = [problem.addVar(lb=0, ub=self.discharge_cap) for _ in self.prices]
            b = [problem.addVar(vtype=cp.COPT.BINARY) for _ in self.prices]
            soc = [problem.addVar(lb=self.low, ub=self.high) for _ in self.prices]
            ph = self.model.physics
            for t in range(self.n):
                problem.addConstr(c[t] <= self.charge_cap*b[t])
                problem.addConstr(d[t] <= self.discharge_cap*(1-b[t]))
                problem.addConstr(soc[t] == (self.initial.soc if t == 0 else soc[t-1])+
                    .25/self.energy*(ph.eta_charge*c[t]-d[t]/ph.eta_discharge))
            total = .25*cp.quicksum(c[t]+d[t] for t in range(self.n))
            if throughput_limit is not None:
                problem.addConstr(total <= throughput_limit)
            problem.setObjective(.25*cp.quicksum(float(self.prices[t])*(d[t]-c[t])
                                 for t in range(self.n)), cp.COPT.MAXIMIZE)
            self.check()
            problem.setParam('TimeLimit', min(self.o['subproblem_seconds'],
                                             self.deadline-time.monotonic()))
            problem.solve()
            status = int(problem.getAttr('Status'))
            has = bool(problem.getAttr('HasSol'))
            info = dict(status_code=status, has_incumbent=has,
                        optimal_to_tolerance=status == cp.COPT.OPTIMAL,
                        coefficient_eur_per_kwh=0.,
                        throughput_limit_kwh=throughput_limit,
                        elapsed_seconds=time.monotonic()-began)
            self.milp_attempts.append(info)
            self.check()
            if status == cp.COPT.INFEASIBLE:
                return None, info
            if status not in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT, cp.COPT.NODELIMIT):
                raise RuntimeError(f'{self.method} unexpected MILP status {status}')
            if not has:
                return None, info
            result = validate_solution([v.x for v in c], [v.x for v in d], [v.x for v in b],
                [v.x for v in soc], self.prices, .25, self.initial.soc, self.energy,
                self.charge_cap, self.discharge_cap, ph.eta_charge, ph.eta_discharge,
                0., self.low, self.high)
            power = np.asarray(result['power_ac_kw'], dtype=float)
            throughput = float(.25*np.abs(power).sum())
            if throughput_limit is not None and throughput > throughput_limit+self.o['feasibility_tolerance']:
                raise ValueError('MILP result violates BCL')
            objective = float(problem.getAttr('ObjVal'))
            if not np.isclose(objective, result['predicted_objective_eur'], atol=1e-7, rtol=1e-8):
                raise ValueError('MILP objective verification failed')
            for attr, key in [('ObjBound', 'bound_eur'), ('BestGap', 'relative_gap')]:
                value = float(problem.getAttr(attr))
                info[key] = value if np.isfinite(value) and abs(value) < 1e20 else None
            info['objective_eur'] = objective
            return power, info

    def round_power(self, power):
        """Treat sub-1e-7 kW solver residues as zero before hybrid evaluation."""
        rounded = np.asarray(power, dtype=float).copy()
        rounded[np.abs(rounded) < 1e-7] = 0.
        if not self.feasible(rounded):
            raise ValueError('B5 power rounding violates power/SOC feasibility')
        return rounded

    def publish_selected(self):
        if self.selected is not None:
            self.publish(dict(self.selected, search_counts=dict(self.stats),
                              iteration_trace=copy.deepcopy(self.trace),
                              milp_attempts=copy.deepcopy(self.milp_attempts)))

    def solve(self):
        # Match b3-v4's seed schedule: every cap refers to the PURE plan.
        # A low cap can be infeasible (e.g. SOC recovery); later, larger caps
        # must still be attempted. Equal/zero plans are not deduplicated.
        q_reference = None
        previous = None
        for k in range(min(10, self.o['max_iterations'])):
            self.check()
            alpha = 1. if k == 0 else k/10
            limit = None if k == 0 else alpha*q_reference
            power, info = self.milp(limit)
            info.update(iteration=k, alpha=alpha, q_reference_kwh=q_reference)
            if power is None:
                if k == 0:
                    self.reason = 'initial_milp_failed'
                    return None
                continue
            power = self.round_power(power)
            throughput = float(.25*np.abs(power).sum())
            if k == 0:
                q_reference = throughput
                info['q_reference_kwh'] = q_reference
            evaluated = self.evaluate(power, gradient=False)
            if evaluated is None:
                raise ValueError('hybrid model evaluation invalid; damage is not clipped')
            plan, _ = evaluated
            distance = None if previous is None else float(np.max(np.abs(power-previous)))
            plan.update(iteration=k, alpha=alpha, q_reference_kwh=q_reference,
                        throughput_kwh=throughput,
                        lambda_eur_per_kwh=0., throughput_limit_kwh=limit,
                        milp_aging_penalty_eur=0.,
                        milp_objective_eur=plan['predicted_revenue_eur'],
                        selected_milp=copy.deepcopy(info))
            self.stats['candidate_evaluations'] += 1
            self.trace.append(dict(iteration=k, alpha=alpha, q_reference_kwh=q_reference,
                throughput_kwh=throughput, lambda_eur_per_kwh=0., throughput_limit_kwh=limit,
                power_update_inf_kw=distance, predicted_damage_pu=plan['predicted_total_damage_pu'],
                revenue_eur=plan['predicted_revenue_eur'], utility_eur=plan['optimizer_objective_eur'],
                unscaled_utility_eur=plan['predicted_objective_eur'],
                predicted_objective_eur=plan['predicted_objective_eur'],
                optimizer_objective_eur=plan['optimizer_objective_eur'],
                predicted_damage_components_pu=list(plan['predicted_damage_components_pu']),
                optimizer_aging_penalty_eur=plan['optimizer_aging_penalty_eur'],
                milp=copy.deepcopy(info)))
            if self.best is None or plan['optimizer_objective_eur'] > self.best['optimizer_objective_eur']:
                self.best = plan
            self.selected = self.best
            self.publish_selected()
            previous = power.copy()
        self.reason = ('candidate_schedule_complete' if self.o['max_iterations'] >= 10
                       else 'iteration_limit')
        return self.selected


def worker(connection, model_path, options):
    env = None
    try:
        torch.set_num_threads(options['threads'])
        if md.version('coptpy') != '8.0.7':
            raise RuntimeError('b5-v4 requires tested coptpy==8.0.7')
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
            search = Search(model, env, cp, options, request,
                            lambda plan: connection.send(('incumbent', plan)))
            try:
                search.solve()
            except Deadline:
                search.reason = 'decision_time_budget'
            connection.send(('done', dict(reason=search.reason, counts=search.stats,
                candidates=search.trace, milp_attempts=search.milp_attempts,
                converged=search.converged,
                selected_iteration=None if search.selected is None else search.selected['iteration'])))
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


class B5V4Solver(HybridSolver):
    """Bounded b5-v4 worker with independent raw/scaled result validation."""
    def __init__(self, options, model_path):
        if options.get('method', 'b5-v4') != 'b5-v4':
            raise ValueError('b5-v4 options required')
        super().__init__(dict(options, method='b5-v4'), model_path)
        self.aging_tables = None
        self.method = 'b5-v4'

    def _start(self, seconds):
        if seconds <= 0:
            raise TimeoutError('no worker startup allowance')
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, self.model_path, self.options), daemon=True)
        self.process.start()
        child.close()
        inbox = queue.Queue()
        connection = self.connection
        def ready():
            try:
                inbox.put(connection.recv())
            except (EOFError, OSError) as exc:
                inbox.put(('error', str(exc)))
        threading.Thread(target=ready, daemon=True).start()
        try:
            kind, value = inbox.get(timeout=seconds)
        except queue.Empty:
            self._stop()
            raise TimeoutError(f'{self.method} worker/model/license initialization timed out')
        if kind != 'ready':
            self._stop()
            raise RuntimeError(f'{self.method} worker startup: {value}')
        if self.metadata is not None and value != self.metadata:
            self._stop()
            raise ValueError('trained model changed during run')
        self.metadata = value

    def validate_plan(self, plan, request):
        # Preserve the original B5 power/SOC and raw ledger arithmetic checks.
        view = dict(plan, optimizer_aging_penalty_eur=plan['predicted_aging_cost_eur'],
                    minimized_objective_eur=-plan['predicted_objective_eur'])
        super().validate_plan(view, request)
        scaling = request['aging_scaling']
        if plan.get('aging_scaling') != scaling:
            raise ValueError('b5-v4 plan changed the day-start economic weights')
        beta_cal = float(scaling['cal']['scaling_factor'])
        beta_cyc = float(scaling['cycl']['scaling_factor'])
        if (scaling['native_soh'] != request['checkpoint']['battery']['soh_Q'] or
                not np.isfinite([beta_cal, beta_cyc]).all() or min(beta_cal, beta_cyc) <= 0):
            raise ValueError('invalid day-start economic weights')
        cal, cyc = plan['predicted_damage_components_pu']
        penalty = request['kappa']*(beta_cal*cal+beta_cyc*cyc)
        throughput = float(.25*np.abs(plan['power_ac_kw']).sum())
        coefficient = plan['lambda_eur_per_kwh']
        limit = plan['throughput_limit_kwh']
        if coefficient != 0:
            raise ValueError('b5-v4 MILP must have no aging penalty')
        if limit is not None and throughput > limit+self.options['feasibility_tolerance']:
            raise ValueError('selected b5-v4 plan violates its own BCL')
        expected = dict(throughput_kwh=throughput,
                        milp_aging_penalty_eur=0.,
                        milp_objective_eur=plan['predicted_revenue_eur'],
                        predicted_total_damage_pu=cal+cyc,
                        optimizer_aging_penalty_eur=penalty,
                        optimizer_objective_eur=plan['predicted_revenue_eur']-penalty,
                        minimized_objective_eur=penalty-plan['predicted_revenue_eur'])
        for key, value in expected.items():
            if not np.isfinite(value) or not np.isclose(plan[key], value, atol=1e-8, rtol=1e-9):
                raise ValueError(f'{self.method} result mismatch: {key}')

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
        aging_scaling_seconds = 0.
        finished = False
        try:
            if self.process is None or not self.process.is_alive():
                self._stop()
                self._start(max(0., work_end-time.monotonic()))
            # Reuse b3-v4's exact B2 selection, once at day start, within this budget.
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
                info['initial_milp'] = best.get('initial_milp', {})
                info['statistics_scope'] = 'last_published_incumbent'
            else:
                info['statistics_scope'] = 'completed_search'
        elapsed = time.monotonic()-began
        backend = 'copt MILP + frozen hybrid forward evaluation'
        info.update(backend=backend, method=self.method, elapsed_seconds=elapsed,
                    max_seconds=self.options['max_seconds'], deadline_exceeded=elapsed > self.options['max_seconds'],
                    model_sha256=self.metadata['model_sha256'] if self.metadata else None,
                    terminal_soc_required=False, hard_watchdog_used=not finished,
                    aging_scaling=copy.deepcopy(request.get('aging_scaling')),
                    aging_scaling_seconds=aging_scaling_seconds, converged=False,
                    selection_rule='best_evaluated_health_scaled_hybrid_utility',
                    utility_early_stop_rule='none_fixed_candidate_schedule',
                    candidate_schedule='pure_then_0.1_to_0.9_of_pure_ac_throughput')
        if best is None:
            return dict(success=False, reason=f'no checked feasible {self.method} plan', solver=info)
        info.setdefault('selected_iteration', best['iteration'])
        if not info.get('candidates'):
            info['candidates'] = best.get('iteration_trace', [])
        info.setdefault('milp_attempts', best.get('milp_attempts', []))
        result = dict(best, solver=info)
        result.pop('iteration_trace', None)
        result.pop('milp_attempts', None)
        return result

