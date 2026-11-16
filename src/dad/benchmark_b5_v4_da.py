"""Independent b5-v4 (paper M4) day-ahead entry point.

B5-v3 execution/accounting and day-start B2 health scaling, with ten
pure / fixed-throughput candidates matching the b3-v4 seed schedule. No dispatch, optimization or training on import.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np

from .common import Budget, LimitReached, canonical, digest, read_json
from .teacher import Teacher
from .dispatch_health import health
from .hybrid import Physics
from .power_limits import ac_power_bounds
from .benchmark_checks import execution_checks
from .benchmark_prices import load_windows
from .benchmark_hybrid_solver import load_model
from .benchmark_b5_v4_solver import B5V4Solver
from .benchmark_b2_tables import AgingTables
from .benchmark_hybrid_common import (exclusive, save, plain, log, new_day, append_segment,
    day_result as base_day_result, summary as base_summary, signature as base_signature)


def load_spec(path, *, required_method=None):
    spec=read_json(path)
    if 'base_config' in spec:
        base=read_json(Path(path).parent/Path(spec['base_config']).name)
        for key,value in spec.items():
            if key!='base_config':base[key]={**base.get(key,{}),**value} if isinstance(value,dict) else value
        spec=base
    validate_spec(spec, required_method=required_method)
    return spec


def validate_spec(c, *, required_method=None):
    if required_method is not None and (required_method != 'b5-v4' or c.get('method') != required_method):
        raise ValueError(f'{required_method} entry requires method={required_method}')
    if c.get('kind') != 'production' or c.get('method') != 'b5-v4' or c.get('algorithm') != 'b5-v4':
        raise ValueError('explicit production b5-v4 method/algorithm config required')
    expected_teacher = dict(series=100, parallel=10, initial_soc=.5, initial_temperature_c=25,
        soc_min=0, soc_max=1, temperature_min_c=0, temperature_max_c=50, internal_dt_s=10,
        cooling_area_fraction=.5, eta_charge=.96, eta_discharge=.96, dc_power_kw=9.6,
        power_tolerance_kw=.05, terminal_soc_tolerance=None, report_only=True)
    expected_mpc = dict(soc_min=.1, soc_max=.9, horizon_intervals=96, execute_intervals=96,
        day_end_soc=None, dt_h=.25, eta_charge=.91, eta_discharge=.97, ambient_c=25)
    if c['teacher'] != expected_teacher or c['mpc'] != expected_mpc:
        raise ValueError('b5-v4 retain B1 physical constraints, 96x15min and SOC coefficients .91/.97')
    if c['economics'] != dict(cell_eur_per_kwh=135, installed_eur_per_kwh=300, fec_eol=6000, eol_soh=.8):
        raise ValueError('b5-v4 retain common 6000 FEC and economic assumptions')
    if c.get('aging') != dict(kappa_eur_per_pu=6480, scaled=True, reference_soh=.95):
        raise ValueError('b5-v4 requires kappa=6480 and B2 endpoint-ratio scaling at reference SOH .95')
    positive = {'max_seconds', 'startup_seconds', 'return_reserve_seconds', 'subproblem_seconds',
        'threads', 'node_limit', 'feasibility_tolerance', 'max_iterations',
        'initial_lambda_eur_per_kwh', 'power_tolerance_kw', 'shrink_fraction'}
    nonnegative = {'relative_gap', 'absolute_gap_eur'}
    o = c['solver']
    if set(o) != positive | nonnegative:
        raise ValueError('unknown/missing b5-v4 solver setting')
    for key in positive | nonnegative:
        if isinstance(o[key], bool) or not np.isfinite(o[key]) or o[key] < 0 or (key in positive and o[key] == 0):
            raise ValueError('invalid solver work limit or tolerance: '+key)
    for key in ('threads', 'node_limit', 'max_iterations'):
        if type(o[key]) is not int:
            raise ValueError('integer solver setting required: '+key)
    if o['max_iterations'] > 10:
        raise ValueError('b5-v4 allows at most the ten fixed-throughput candidates')
    if not o['return_reserve_seconds'] < o['max_seconds'] or o['relative_gap'] >= 1:
        raise ValueError('invalid decision reserve or relative gap')
    if o['initial_lambda_eur_per_kwh'] != 135/(2*6000) or o['power_tolerance_kw'] != .01 or o['shrink_fraction'] != .03:
        raise ValueError('b5-v4 retains legacy B5 settings; shrink_fraction is unused by the fixed schedule')
    if type(c['seed']) is not int or type(c['scenario']['max_days']) is not int or c['scenario']['max_days'] < 1:
        raise ValueError('integer seed and positive cumulative day count required')
    limits = c['limits']
    if set(limits) != {'max_seconds', 'max_steps', 'max_queries', 'segment_seconds'}:
        raise ValueError('explicit execution work limits required')
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError('positive finite execution work limits required')
    if any(type(limits[k]) is not int for k in ('max_steps', 'max_queries')):
        raise ValueError('integer integration step/query limits required')
    if limits['segment_seconds'] <= o['max_seconds']:
        raise ValueError('segment budget must include optimization and native replay')


def signature(c, price_audit, certificate, model_path):
    sig = base_signature(c, price_audit, certificate, model_path)
    root = Path(__file__).parent
    for name in ('benchmark_b5_v4_da.py', 'benchmark_b5_v4_solver.py',
                 'benchmark_b3_v4_solver.py', 'benchmark_b2_tables.py',
                 'benchmark_b1_solver.py'):
        sig['source'][name] = digest(root/name)
    sig['algorithm'] = 'b5-v4'
    sig['paper_method'] = 'M4'
    sig['candidate_schedule'] = dict(order=['pure']+[i/10 for i in range(1, 10)],
        reference='rounded pure-arbitrage AC throughput in kWh',
        cap='alpha * pure throughput; no recursive shrink or utility early stop')
    sig['aging_tables'] = AgingTables().hashes
    sig['optimizer_scaling'] = dict(scaled=True, reference_soh=.95,
        source='B2 author-table endpoint ratios; hybrid damage remains unscaled',
        health_input='day-start native total SOH for both calendar and cyclic factors',
        factor_formula='table endpoint at reference SOH / table endpoint at day-start native SOH',
        factor_scope='fixed for all candidate evaluations and best-plan selection of one day',
        optimizer_objective='predicted revenue - 6480 * (beta_cal * signed hybrid calendar damage + beta_cycl * signed hybrid cyclic damage)',
        prediction_objective='predicted revenue - 6480 * unscaled signed hybrid total damage',
        evaluation_objective='actual delivered revenue - 6480 * signed native ledger damage')
    return sig


def day_result(day, teacher, kappa):
    report = base_day_result(day, teacher, kappa)
    plans = [window['plan'] for window in day['mpc_windows']]
    revenue = sum(plan['predicted_revenue_eur'] for plan in plans)
    raw_cost = sum(plan['predicted_aging_cost_eur'] for plan in plans)
    scaled_cost = sum(plan['optimizer_aging_penalty_eur'] for plan in plans)
    report.update(method='b5-v4', algorithm='b5-v4', paper_method='M4',
        predicted_revenue_eur=revenue, predicted_hybrid_aging_cost_eur=raw_cost,
        predicted_utility_eur=revenue-raw_cost,
        optimizer_aging_penalty_eur=scaled_cost,
        optimizer_objective_eur=revenue-scaled_cost,
        milp_aging_penalty_eur=sum(plan['milp_aging_penalty_eur'] for plan in plans),
        evaluation_kappa_eur_per_pu=kappa)
    return report


def commit_day(path,state,teacher,kappa):
    """Write immutable report before checkpoint; deterministic orphan recovery."""
    report=day_result(state['day'],teacher,kappa)
    target=path/'days'/f'{report["index"]+1:05d}.json'
    if target.exists():
        if read_json(target)!=report:raise ValueError('conflicting previously written daily report')
    else:save(target,report)
    state['daily_chain']=canonical([state['daily_chain'],digest(target)])
    state['reported_days']+=1
    state['completed_days']+=int(report['complete_day'])
    state['failed_days']+=int(report['failed'])
    state['native_temperature_failure_days']+=int(report['native_temperature_violation'])
    state['scheduling_soc_violation_days']+=int(report['scheduling_soc_band_violated'])
    for key in report['findings']:state['failure_days_by_type'][key]=state['failure_days_by_type'].get(key,0)+1
    state['day']=None
    return report


def summary(state, c):
    result = base_summary(state, c)
    result.update(schema='b5-v4-summary-v1', method='b5-v4', algorithm='b5-v4',
        paper_method='M4', intraday_reoptimization=False,
        predicted_hybrid_aging_cost_eur=state['predicted_hybrid_aging_cost_eur'],
        predicted_utility_eur=state['predicted_revenue_eur']-state['predicted_hybrid_aging_cost_eur'],
        optimizer_objective_eur=state['predicted_revenue_eur']-state['optimizer_aging_penalty_eur'],
        milp_aging_penalty_eur=0., evaluation_kappa_eur_per_pu=6480.,
        failed_day_definition='physical/temperature/delivery finding or configured decision deadline exceeded; scheduling SOC deviation reported separately; no repair')
    timing = result['decision_timing']
    timing.pop('above_30_seconds')
    timing['budget_seconds'] = c['solver']['max_seconds']
    timing['above_budget_seconds'] = sum(t > c['solver']['max_seconds'] for t in state['decision_seconds'])
    return result


def run(c,windows,path,sig,*,model_path,resume=False,extensions=None,solver_factory=B5V4Solver,
        segment_seconds=None, required_method='b5-v4'):
    validate_spec(c, required_method=required_method)
    if solver_factory is None:
        raise ValueError('b5-v4 requires its independent solver')
    np.random.seed(c['seed'])  # Retain the explicit run seed.
    path=Path(path);extensions=extensions or {};target=c['scenario']['max_days']*96
    if set(extensions)-{'seconds','steps','queries'} or any(not np.isfinite(v) or v<0 for v in extensions.values()):
        raise ValueError('invalid work extensions')
    if segment_seconds is not None and (not np.isfinite(segment_seconds) or
                                       segment_seconds < c['limits']['segment_seconds']):
        raise ValueError('segment allowance must be finite and at least the configured allowance')
    da=True;execute=c['mpc']['execute_intervals']
    steps_per_segment=int(execute*.25*3600/c['teacher']['internal_dt_s'])
    with exclusive(Path(str(path)+'.lock')):
        if resume:
            manifest = read_json(path/'manifest.json')
            previous = manifest['signature']
            if previous != sig:
                raise ValueError('resume source/config/input/environment mismatch')
            state=read_json(path/'checkpoint.json')
            chain=None
            for i in range(state['reported_days']):
                chain=canonical([chain,digest(path/'days'/f'{i+1:05d}.json')])
            if chain!=state['daily_chain']:raise ValueError('committed report hash chain mismatch')
            if state['status']=='complete':return summary(state,c)
            for key,amount in extensions.items():state['work_limits'][key]+=amount
            if state.get('reservation'):
                for key,value in state['reservation'].items():state['usage'][key]+=value
                state['crashes_reserved']+=1;state['reservation']=None
        else:
            if any(extensions.values()):raise ValueError('work extensions require --resume')
            path.mkdir(parents=True,exist_ok=False)
            save(path/'manifest.json',dict(signature=sig,run_id=path.name,created=time.time(),executable=sys.executable))
            t=Teacher(c['teacher'],Budget(30,1,1));h=health(t,t.checkpoint())
            state=dict(status='ready',index=0,teacher=t.checkpoint(),ledger=h['ledger_components_pu'],initial_health=h,
                day=None,reported_days=0,completed_days=0,failed_days=0,failure_days_by_type={},daily_chain=None,
                native_temperature_failure_days=0,scheduling_soc_violation_days=0,delivered_charge_kwh=0.,delivered_discharge_kwh=0.,
                usage=dict(seconds=0.,steps=0,queries=0),work_limits=dict(seconds=c['limits']['max_seconds'],
                    steps=c['limits']['max_steps'],queries=c['limits']['max_queries']),
                crashes_reserved=0,reservation=None,revenue_eur=0.,
                decision_seconds=[],decision_budget_exhausted=0,native_replay_seconds=0.,worker_startup_seconds=0.,revenue_at_2920_days_eur=None,
                predicted_hybrid_aging_cost_eur=0.,optimizer_aging_penalty_eur=0.,predicted_revenue_eur=0.,payback_days=None,temperature_range_c=[c['teacher']['initial_temperature_c']]*2)
        old_segment = state['work_limits'].get('segment_seconds', c['limits']['segment_seconds'])
        effective_segment = segment_seconds if segment_seconds is not None else old_segment
        state['work_limits']['segment_seconds'] = effective_segment
        save(path/'checkpoint.json',state)
        log(path,'start',resume=resume,extensions=extensions,segment_seconds=effective_segment)
        nominal=c['teacher']['series']*c['teacher']['parallel']*.0096
        kappa=nominal*c['economics']['cell_eur_per_kwh']/(1-c['economics']['eol_soh'])
        capex=nominal*c['economics']['installed_eur_per_kwh']
        try:
            startup_began=time.monotonic()
            with solver_factory(c['solver'], model_path) as solver:
                startup_elapsed=time.monotonic()-startup_began
                state['worker_startup_seconds']+=startup_elapsed
                state['usage']['seconds']+=startup_elapsed
                save(path/'checkpoint.json',state)
                while True:
                    teacher=Teacher(c['teacher'],Budget(30,1,1));teacher.restore(state['teacher'])
                    if vars(Physics.from_teacher(teacher)) != solver.metadata['physics']:
                        raise ValueError('trained model physical constants differ from the native device')
                    if solver.metadata['model_sha256'] != sig['model_sha256']:
                        raise ValueError('model changed after the run signature was prepared')
                    if state['day'] and (len(state['day']['intervals'])==96 or state['index']==target):
                        report=commit_day(path,state,teacher,kappa)
                        save(path/'checkpoint.json',state)
                        save(path/'summary.json',summary(state,c))
                        print(f'{c["method"]} day {state["reported_days"]}: revenue={report["revenue_eur"]:.4f} EUR '
                              f'SOH={report["health_end"]["native_soh"]:.6f} failed={report["failed"]}',flush=True)
                    if state['index']>=target or (state['day'] is None and teacher.battery.state.soh_Q<=c['economics']['eol_soh']):
                        state.update(status='complete',reason='native_soh_retirement' if teacher.battery.state.soh_Q<=c['economics']['eol_soh'] else 'specified_intervals')
                        break
                    progress=state.get('execution_progress')
                    done=len(progress['states']) if progress else 0
                    if done and (not state.get('pending_plan') or not 0<done<=execute):
                        raise ValueError('native progress requires a matching fixed plan')
                    reserve=dict(seconds=state['work_limits']['segment_seconds'],
                        steps=steps_per_segment-done*int(900/c['teacher']['internal_dt_s']),
                        queries=0 if progress else 1)
                    if any(state['usage'][k]+v>state['work_limits'][k] for k,v in reserve.items()):
                        state.update(status='paused',reason='work_limit: extend allowance explicitly on resume');break
                    prices,sources=windows.window(state['index'],c['mpc']['horizon_intervals'])
                    if da and c['kind']=='production' and state['index']%96:
                        raise ValueError('b5-v4 must start at a 96-interval simulation-day boundary')
                    if state['day'] is None:
                        state['day']=new_day(teacher,state['index']//96)
                        state['day']['method']=c['method']
                    state.update(reservation=reserve,status='running',reason=None)
                    save(path/'checkpoint.json',state)
                    budget=Budget(reserve['seconds'],reserve['steps'],reserve['queries']);teacher.budget=budget
                    try:
                        pending=state.get('pending_plan') if da else None
                        if pending:
                            if pending['index']!=state['index']:
                                raise ValueError('fixed DA plan does not match execution checkpoint')
                            plan=pending['plan']
                            if pending['sha256']!=canonical(plan):raise ValueError('fixed plan hash mismatch')
                        else:
                            plan=solver.solve(prices,checkpoint=teacher.checkpoint(),kappa=kappa,
                                power_bounds_ac_kw=ac_power_bounds(c['teacher']),
                                ambient_c=c['mpc']['ambient_c'],soc_min=c['mpc']['soc_min'],soc_max=c['mpc']['soc_max'])
                            state['decision_seconds'].append(plan['solver']['elapsed_seconds'])
                            if plan['solver']['reason'] in ('decision_time_budget','watchdog_deadline','worker_restart_time_budget','aging_scaling_time_budget'):
                                state['decision_budget_exhausted']+=1
                        if not plan['success']:
                            log(path,'solver_failure',index=state['index'],plan=plan)
                            raise RuntimeError('b5-v4 has no checked feasible plan; no invented execution or repair')
                        if da and pending is None:
                            # Save the selected full-day commitment before teacher execution.
                            # Interrupted execution resumes the saved native interval checkpoint;
                            # it never reoptimizes against partially observed teacher states.
                            state['pending_plan']=dict(index=state['index'],plan=plan,sha256=canonical(plan))
                            save(path/'checkpoint.json',state)
                        budget.check()
                        # Execute the entire saved commitment. No candidate teacher queries.
                        u=plan['power_ac_kw'][:execute]
                        if len(u)!=execute:raise ValueError('incomplete execution plan')
                        progress=state.get('execution_progress')
                        if progress:
                            if state.get('execution_progress_sha256')!=canonical(progress):
                                raise ValueError('native execution progress hash mismatch')
                            progress=copy.deepcopy(progress)
                            progress['min_margin']=np.asarray(progress['min_margin'],float)
                        def persist_progress(value):
                            # Commit native progress only after complete 15min intervals.
                            # The day-start state remains the basis of damage/cash accounting.
                            if not value['states']:return
                            state['execution_progress']=plain(value)
                            state['execution_progress_sha256']=canonical(state['execution_progress'])
                            save(path/'checkpoint.json',state)
                        replay_started=time.monotonic()
                        try:
                            result=teacher.rollout(u,np.full(execute,c['mpc']['ambient_c']),np.full(execute,.25),
                                resume_state=progress,checkpoint_sink=persist_progress)
                        finally:
                            state['native_replay_seconds']+=time.monotonic()-replay_started
                        budget.check()
                        checks=execution_checks(result,teacher)
                        actual_soc=checks['fine_state_ranges']['soc']
                        checks['scheduling_soc']=dict(bounds=[c['mpc']['soc_min'],c['mpc']['soc_max']],
                            actual_range=actual_soc,
                            passed=bool(actual_soc[0]>=c['mpc']['soc_min']-1e-9 and
                                        actual_soc[1]<=c['mpc']['soc_max']+1e-9))
                        end=np.asarray(result['damage'])+np.asarray(state['ledger'])
                        teacher.restore(result['checkpoint'])
                        if not np.allclose(end,teacher.ledger(),atol=1e-12,rtol=0):
                            raise ArithmeticError('segment residual ledger does not telescope')
                        # All numerical checks passed: account atomically on an isolated state copy.
                        state=copy.deepcopy(state)
                        next_day=append_segment(state['day'],result,checks,plan,prices,sources,state['index'],windows)
                        for row in next_day['intervals'][-execute:]:
                            before=state['revenue_eur']-capex
                            if state['payback_days'] is None and before<0<=before+row['revenue_eur']:
                                j=next_day['intervals'].index(row)
                                state['payback_days']=(next_day['index']*96+j-before/row['revenue_eur'])/96
                            state['revenue_eur']+=row['revenue_eur']
                            state['delivered_charge_kwh']+=max(0.,-row['delivered_ac_kw'])*row['dt_h']
                            state['delivered_discharge_kwh']+=max(0.,row['delivered_ac_kw'])*row['dt_h']
                        temps=checks['fine_state_ranges']['temperature_c']
                        state['temperature_range_c']=[min(state['temperature_range_c'][0],temps[0]),max(state['temperature_range_c'][1],temps[1])]
                        state.update(day=next_day,index=state['index']+execute,teacher=teacher.checkpoint(),ledger=end.tolist())
                        if state['index']==2920*96:
                            state['revenue_at_2920_days_eur']=state['revenue_eur']
                        if da:state['pending_plan']=None
                        state['execution_progress']=None;state['execution_progress_sha256']=None
                        state['predicted_hybrid_aging_cost_eur']+=plan['predicted_aging_cost_eur']
                        state['optimizer_aging_penalty_eur']+=plan['optimizer_aging_penalty_eur']
                        state['predicted_revenue_eur']+=plan['predicted_revenue_eur']
                    finally:
                        for key,value in budget.report().items():state['usage'][key]+=value
                        state['reservation']=None
                    save(path/'checkpoint.json',state)
        except (LimitReached,TimeoutError,KeyboardInterrupt) as exc:
            state.update(status='paused',reason=str(exc) or 'interrupted')
            log(path,'paused',index=state['index'],reason=state['reason'])
        except Exception as exc:
            state.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
            log(path,'failed',index=state['index'],reason=state['reason'])
            save(path/'checkpoint.json',state);save(path/'summary.json',summary(state,c))
            raise
        save(path/'checkpoint.json',state);out=summary(state,c);save(path/'summary.json',out)
        log(path,'stop',status=state['status'],index=state['index'],reason=state.get('reason'))
        return out


def main(argv=None):
    method = 'b5-v4'
    parser=argparse.ArgumentParser(description=f'{method} day-ahead run and preflight')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','run'):
        p=sub.add_parser(name)
        p.add_argument('--config',required=True)
        p.add_argument('--model',required=True,help='Completed hybrid training stage2-best.pt')
        p.add_argument('--prices',default='data/dispatch-prices/prices.csv')
        p.add_argument('--certificate',default='reports/teacher_certificate.json')
        if name=='run':
            p.add_argument('--out',required=True)
            p.add_argument('--resume',action='store_true')
            p.add_argument('--segment-seconds',type=float,default=None,
                           help='Set per-execution-segment wall allowance; persists on resume, never changes the configured daily optimization budget')
            p.add_argument('--add-seconds',type=float,default=0)
            p.add_argument('--add-steps',type=int,default=0)
            p.add_argument('--add-queries',type=int,default=0)
    args=parser.parse_args(argv)
    c=load_spec(args.config, required_method=method)
    from .data import check_certificate
    check_certificate(args.certificate)
    # Loading checks format, completion hash and constants; no forward/training/rollout here.
    model=load_model(args.model)
    nominal=c['teacher']['series']*c['teacher']['parallel']*.0096
    if not np.isclose(model.physics.energy_kwh,nominal,atol=1e-12,rtol=0):
        raise ValueError('model nominal energy does not match b5-v4 device')
    native = Teacher(c['teacher'], Budget(30, 1, 1))
    if vars(Physics.from_teacher(native)) != vars(model.physics):
        raise ValueError('model physics differs from the native device')
    del model, native
    windows,audit=load_windows(args.prices,c['scenario'],96,96)
    sig=signature(c,audit,args.certificate,args.model)
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',simulation_executed=False,
            optimizer_executed=False,short_tests_executed=False,signature_sha256=canonical(sig),
            method=c['method'],days=c['scenario']['max_days'],model_sha256=sig['model_sha256'],
            power_bounds_ac_kw=list(ac_power_bounds(c['teacher'])),
            decision_seconds=c['solver']['max_seconds'],
            evaluation_kappa_eur_per_pu=6480, algorithm=sig['algorithm'],
            paper_method='M4', optimizer_scaling=sig['optimizer_scaling'],
            aging_tables=sig['aging_tables'],
            prices=audit,copt_license_tested=False),indent=2))
        return 0
    ext=dict(seconds=args.add_seconds,steps=args.add_steps,queries=args.add_queries)
    if not np.isfinite(list(ext.values())).all() or min(ext.values())<0:
        raise ValueError('nonnegative finite extensions required')
    out=run(c,windows,args.out,sig,model_path=args.model,resume=args.resume,extensions=ext,
            segment_seconds=args.segment_seconds, solver_factory=B5V4Solver, required_method=method)
    print(f'{c["method"]} {out["status"]}: {out["reason"]}; results: {args.out}',flush=True)
    return 0 if out['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
