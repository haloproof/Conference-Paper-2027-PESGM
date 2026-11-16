"""b5-v4 (paper M4) continuation until native SoH retirement.

This entry removes only the cumulative day endpoint. The original method,
physical execution, signed accounting and finite work budgets are unchanged.
Importing this module never runs dispatch, native integration or training.
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
from .benchmark_prices import load_windows as bounded_load_windows
from .benchmark_hybrid_solver import load_model
from .benchmark_b5_v4_solver import B5V4Solver
from .benchmark_b5_v4_da import (
    exclusive, save, plain, log, new_day, append_segment, commit_day,
    validate_spec as validate_day_spec, signature as day_signature,
    summary as day_summary,
)


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
    scenario = c.get('scenario', {})
    if 'max_days' not in scenario or scenario['max_days'] is not None:
        raise ValueError('retirement-only b5-v4 requires scenario.max_days=null')
    if scenario.get('price_scenario') != 'repeat_history_year':
        raise ValueError('retirement-only b5-v4 requires repeat_history_year prices')
    # Reuse the original physical/solver/economic/work-limit validation exactly.
    # The temporary finite day count is not a simulation endpoint.
    finite = copy.deepcopy(c)
    finite['scenario']['max_days'] = 1
    validate_day_spec(finite, required_method=required_method)


def load_windows(path, scenario, horizon, execute):
    if ('max_days' not in scenario or scenario['max_days'] is not None or
            scenario.get('price_scenario') != 'repeat_history_year'):
        raise ValueError('retirement-only price windows require a null day limit and a repeated year')
    # The existing reader validates the entire 365-day source year. There is no
    # last requested day to check; all later windows use its unchanged modulo map.
    finite = copy.deepcopy(scenario)
    finite['max_days'] = 1
    windows, audit = bounded_load_windows(path, finite, horizon, execute)
    windows.spec = copy.deepcopy(scenario)
    return windows, audit


def termination(c):
    return dict(rule='native_soh_at_completed_day_boundary',
                native_soh_threshold=c['economics']['eol_soh'],
                cumulative_day_limit=None, work_limit_action='pause_and_resume')


def signature(c, price_audit, certificate, model_path):
    sig = day_signature(c, price_audit, certificate, model_path)
    sig['source'][Path(__file__).name] = digest(Path(__file__))
    sig['termination'] = termination(c)
    return sig


def summary(state, c):
    result = day_summary(state, c)
    result.update(termination=termination(c), cumulative_day_limit=None)
    return result


def run(c,windows,path,sig,*,model_path,resume=False,extensions=None,solver_factory=B5V4Solver,
        segment_seconds=None, required_method='b5-v4'):
    validate_spec(c, required_method=required_method)
    if solver_factory is None:
        raise ValueError('b5-v4 requires its independent solver')
    np.random.seed(c['seed'])  # Retain the explicit run seed.
    path=Path(path);extensions=extensions or {}
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
                    if state['day'] and len(state['day']['intervals'])==96:
                        report=commit_day(path,state,teacher,kappa)
                        save(path/'checkpoint.json',state)
                        save(path/'summary.json',summary(state,c))
                        print(f'{c["method"]} day {state["reported_days"]}: revenue={report["revenue_eur"]:.4f} EUR '
                              f'SOH={report["health_end"]["native_soh"]:.6f} failed={report["failed"]}',flush=True)
                    if state['day'] is None and teacher.battery.state.soh_Q<=c['economics']['eol_soh']:
                        state.update(status='complete',reason='native_soh_retirement')
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
    parser=argparse.ArgumentParser(description=f'{method} until native SoH retirement; no cumulative day endpoint')
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
            termination=sig['termination'],
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
