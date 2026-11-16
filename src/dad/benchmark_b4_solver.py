"""B4: dynamic throughput-price feedback with frozen hybrid evaluation.

The iterative algorithm and its worker are independent of B5.
No fitting, teacher rollout, gradient search or BCL tightening.
"""
from dataclasses import asdict
import copy
import importlib.metadata as md
import multiprocessing as mp
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np
import torch

from .benchmark_b1_solver import validate_solution
from .benchmark_hybrid_solver import HybridSolver, HybridSearch, Deadline, load_model
from .common import digest, read_json


def historical_price(path, completed_days):
    """B4 v2-1: committed actual cash / signed teacher damage, last 30 days.

    Rebuild from immutable reports on every new decision, including resume/stage
    continuation. Pending plans retain their already selected price. The 1e-12
    pu floor is the native ledger reconciliation precision, not a damage clip.
    """
    if type(completed_days) is not int or completed_days < 0:
        raise ValueError('invalid completed history count')
    audit = dict(rule='B4-history-ratio-v2-1', window_days=30,
                 minimum_resolved_damage_pu=1e-12, completed_days=completed_days,
                 days=[], report_sha256=[], revenue_eur=0., damage_pu=0.,
                 previous_rho_eur_per_pu=6480., rho_eur_per_pu=6480., reason='initial')
    if completed_days == 0:
        return audit
    cash, damage = [], []
    previous_report = None
    for i in range(max(0, completed_days-30), completed_days):
        file = Path(path)/'days'/f'{i+1:05d}.json'
        report = read_json(file)
        if (report['index'] != i or report['method'] != 'B4' or
                not report['complete_day'] or len(report['intervals']) != 96):
            raise ValueError('historical price requires complete consecutive B4 days')
        if previous_report is not None and report['health_start'] != previous_report['health_end']:
            raise ValueError('historical price ledger handoff mismatch')
        components = np.asarray(report['damage_components_pu'], dtype=float)
        start = np.asarray(report['health_start']['ledger_components_pu'], dtype=float)
        end = np.asarray(report['health_end']['ledger_components_pu'], dtype=float)
        revenue = float(report['revenue_eur'])
        if (components.shape != (3,) or start.shape != (3,) or end.shape != (3,) or
                not np.isfinite([*components, *start, *end, revenue]).all()):
            raise ValueError('nonfinite or invalid history ledger')
        if not np.allclose(end-start, components, atol=1e-12, rtol=0):
            raise ValueError('historical price damage does not telescope')
        cash.append(revenue)
        damage.extend(components.tolist())
        audit['days'].append(i+1)
        audit['report_sha256'].append(digest(file))
        previous_report = report
    previous = float(previous_report['mpc_windows'][0]['plan']['rho_eur_per_pu'])
    if not np.isfinite(previous) or previous < 0:
        raise ValueError('invalid previous B4 historical price')
    revenue, loss = math.fsum(cash), math.fsum(damage)
    rho = max(0., revenue/loss) if loss > 1e-12 else previous
    if not np.isfinite(rho):
        raise ValueError('nonfinite B4 historical price')
    audit.update(revenue_eur=revenue, damage_pu=loss,
                 previous_rho_eur_per_pu=previous, rho_eur_per_pu=rho,
                 reason='window_ratio' if loss > 1e-12 else 'unresolved_damage_keep_previous')
    return audit


class Search(HybridSearch):
    def __init__(self, model, env, cp, options, request, publish):
        super().__init__(model, env, cp, options, request, publish)
        self.method = 'B4'
        # In v2-1 the inherited request field 'kappa' carries today's rho,
        # fixed by committed history; native evaluation kappa stays 6480.
        if options.get('method', 'B4') != 'B4':
            raise ValueError('B4 options required')
        self.stats = dict(milp_solves=0, forwards=0, invalid_evaluations=0,
                          candidate_evaluations=0)
        self.utilities = []
        self.selected = None
        self.reason = 'iteration_limit'
        self.milp_attempts = []
        self.converged = False

    def milp(self, coefficient):
        self.check()
        if not np.isfinite(coefficient):
            raise ValueError('nonfinite throughput price')
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
            problem.setObjective(.25*cp.quicksum(float(self.prices[t])*(d[t]-c[t])
                                 for t in range(self.n))-coefficient*total, cp.COPT.MAXIMIZE)
            self.check()
            problem.setParam('TimeLimit', min(self.o['subproblem_seconds'],
                                             self.deadline-time.monotonic()))
            problem.solve()
            status = int(problem.getAttr('Status'))
            has = bool(problem.getAttr('HasSol'))
            info = dict(status_code=status, has_incumbent=has,
                        optimal_to_tolerance=status == cp.COPT.OPTIMAL,
                        coefficient_eur_per_kwh=float(coefficient),
                        throughput_limit_kwh=None,
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
                coefficient, self.low, self.high)
            power = np.asarray(result['power_ac_kw'], dtype=float)
            throughput = float(.25*np.abs(power).sum())
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
            raise ValueError('B4 power rounding violates power/SOC feasibility')
        return rounded

    def publish_selected(self):
        if self.selected is not None:
            self.publish(dict(self.selected, search_counts=dict(self.stats),
                              iteration_trace=copy.deepcopy(self.trace),
                              milp_attempts=copy.deepcopy(self.milp_attempts)))

    def solve(self):
        coefficient = self.o['initial_lambda_eur_per_kwh']
        limit = None
        previous = None
        for k in range(self.o['max_iterations']):
            self.check()
            power, info = self.milp(coefficient)
            if power is None:
                self.reason = ('throughput_infeasible' if info['status_code'] == self.cp.COPT.INFEASIBLE
                               else 'milp_no_incumbent')
                break
            power = self.round_power(power)
            evaluated = self.evaluate(power, gradient=False)
            if evaluated is None:
                raise ValueError('hybrid model evaluation invalid; damage is not clipped')
            plan, _ = evaluated
            throughput = float(.25*np.abs(power).sum())
            distance = None if previous is None else float(np.max(np.abs(power-previous)))
            penalty = float(coefficient*throughput)
            plan.update(iteration=k, throughput_kwh=throughput,
                        lambda_eur_per_kwh=float(coefficient), throughput_limit_kwh=limit,
                        optimizer_aging_penalty_eur=penalty,
                        milp_objective_eur=plan['predicted_revenue_eur']-penalty,
                        selected_milp=copy.deepcopy(info))
            self.stats['candidate_evaluations'] += 1
            self.utilities.append(plan['predicted_objective_eur'])
            self.trace.append(dict(iteration=k, throughput_kwh=throughput,
                lambda_eur_per_kwh=float(coefficient), throughput_limit_kwh=limit,
                power_update_inf_kw=distance, predicted_damage_pu=plan['predicted_total_damage_pu'],
                revenue_eur=plan['predicted_revenue_eur'], utility_eur=plan['predicted_objective_eur'],
                milp=copy.deepcopy(info)))
            if self.best is None or plan['predicted_objective_eur'] > self.best['predicted_objective_eur']:
                self.best = plan
            self.selected = self.best
            if distance is not None and distance <= self.o['power_tolerance_kw']:
                # The specified normal return is the converged iterate, not the best historical one.
                self.selected = plan
                self.reason, self.converged = 'power_converged', True
                plan['power_converged'] = True
            elif throughput == 0:
                self.reason = 'zero_throughput'
            self.publish_selected()
            if self.reason != 'iteration_limit':
                break
            previous = power.copy()
            # Signed residual corrections remain signed, including in the resulting price.
            coefficient = self.kappa*plan['predicted_total_damage_pu']/throughput
            if not np.isfinite(coefficient):
                raise ValueError('nonfinite B4 price update')
        return self.selected


def worker(connection, model_path, options):
    env = None
    try:
        torch.set_num_threads(options['threads'])
        if md.version('coptpy') != '8.0.7':
            raise RuntimeError('B4 requires tested coptpy==8.0.7')
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


class B4Solver(HybridSolver):
    """Bounded B4 worker with independent result validation."""
    def __init__(self, options, model_path):
        if options.get('method', 'B4') != 'B4':
            raise ValueError('B4 options required')
        super().__init__(dict(options, method='B4'), model_path)
        self.method = 'B4'

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
        # The inherited validator checks hybrid utility. Here the MILP penalty is separate.
        view = dict(plan, optimizer_aging_penalty_eur=plan['predicted_aging_cost_eur'])
        super().validate_plan(view, request)
        throughput = float(.25*np.abs(plan['power_ac_kw']).sum())
        coefficient = plan['lambda_eur_per_kwh']
        limit = plan['throughput_limit_kwh']
        if limit is not None:
            raise ValueError('B4 must not add a BCL')
        expected = dict(throughput_kwh=throughput,
                        optimizer_aging_penalty_eur=coefficient*throughput,
                        milp_objective_eur=plan['predicted_revenue_eur']-coefficient*throughput,
                        predicted_total_damage_pu=sum(plan['predicted_damage_components_pu']))
        for key, value in expected.items():
            if not np.isfinite(value) or not np.isclose(plan[key], value, atol=1e-8, rtol=1e-9):
                raise ValueError(f'{self.method} result mismatch: {key}')

    def solve(self, prices, **kwargs):
        result = super().solve(prices, **kwargs)
        info = result['solver']
        info['backend'] = 'copt MILP + frozen hybrid forward evaluation'
        info['converged'] = bool(info.get('converged', False) or result.get('power_converged', False))
        info['selection_rule'] = ('converged_iterate' if info['converged']
                                  else 'best_evaluated_hybrid_utility')
        if result['success']:
            info.setdefault('selected_iteration', result['iteration'])
            if not info.get('candidates'):
                info['candidates'] = result.get('iteration_trace', [])
            info.setdefault('milp_attempts', result.get('milp_attempts', []))
            result.pop('iteration_trace', None)
            result.pop('milp_attempts', None)
        else:
            result['reason'] = f'no checked feasible {self.method} plan'
        return result
