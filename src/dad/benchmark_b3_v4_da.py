"""Independent b3-v4 30-day runner; no simulation or training on import.

Preserves the existing native interval checkpoint and residual-ledger protocol.
The B3 optimizer never calls the teacher to select or repair a candidate.
"""
import argparse
import copy
from contextlib import contextmanager
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time
import numpy as np

from .common import Budget, LimitReached, canonical, digest, read_json, write_json
from .teacher import Teacher, native_fingerprint
from .dispatch_health import health
from .benchmark_b3_v4_solver import B3V4Solver, load_model
from .hybrid import Physics
from .benchmark_b2_tables import AgingTables
from .power_limits import ac_power_bounds
from .benchmark_checks import execution_checks
from .benchmark_prices import load_windows, PriceWindows


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def save(path,value):
    write_json(path,plain(value))


@contextmanager
def exclusive(path):
    import os
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def load_spec(path):
    spec=read_json(path)
    if 'base_config' in spec:
        base=read_json(Path(path).parent/Path(spec['base_config']).name)
        for key,value in spec.items():
            if key!='base_config':base[key]={**base.get(key,{}),**value} if isinstance(value,dict) else value
        spec=base
    validate_spec(spec)
    return spec


def validate_spec(c):
    if c.get('kind') != 'production' or c.get('method') != 'b3-v4':
        raise ValueError('explicit production B3 config required')
    expected_teacher = dict(series=100, parallel=10, initial_soc=.5, initial_temperature_c=25,
        soc_min=0, soc_max=1, temperature_min_c=0, temperature_max_c=50, internal_dt_s=10,
        cooling_area_fraction=.5, eta_charge=.96, eta_discharge=.96, dc_power_kw=9.6,
        power_tolerance_kw=.05, terminal_soc_tolerance=None, report_only=True)
    if c['teacher'] != expected_teacher:
        raise ValueError('b3-v4 requires DC +/-9.6 kW (AC -10..9.216 kW), converter efficiencies .96/.96, and native SOC 0..1')
    expected_mpc = dict(soc_min=.1, soc_max=.9, horizon_intervals=96, execute_intervals=96,
                        day_end_soc=None, dt_h=.25, eta_charge=.91, eta_discharge=.97, ambient_c=25)
    if c['mpc'] != expected_mpc:
        raise ValueError('B3 requires 96x15min, SOC .1.. .9, eta .91/.97, no active floor or terminal target')
    if c['economics'] != dict(cell_eur_per_kwh=135, installed_eur_per_kwh=300, fec_eol=6000, eol_soh=.8):
        raise ValueError('B3 retains kappa=6480 and investment=2880 EUR')
    if c.get('aging') != dict(kappa_eur_per_pu=6480, scaled=True, reference_soh=.95):
        raise ValueError('b3-v4 requires kappa=6480 and B2 endpoint-ratio scaling at reference SOH .95')
    o = c['solver']
    if c.get('algorithm') != 'b3-v4':
        raise ValueError('b3-v4 algorithm identity required')
    required = {'max_seconds', 'startup_seconds', 'return_reserve_seconds', 'subproblem_seconds',
        'threads', 'max_main_steps', 'max_evaluations', 'backtracks',
        'initial_step_fraction', 'armijo', 'feasibility_tolerance', 'zero_tolerance_kw',
        'improvement_eur', 'progress_eur'}
    if set(o) != required or any(not np.isfinite(v) or v <= 0 for v in o.values()):
        raise ValueError('invalid b3-v4 solver options')
    for key in ('threads','max_main_steps','max_evaluations','backtracks'):
        if type(o[key]) is not int:
            raise ValueError('integer work limit required: '+key)
    if not 0 < o['return_reserve_seconds'] < o['max_seconds'] <= 30:
        raise ValueError('invalid decision deadline')
    if max(o['initial_step_fraction'], o['armijo']) >= 1:
        raise ValueError('invalid step/Armijo fraction')
    if o['progress_eur'] < o['improvement_eur'] or o['zero_tolerance_kw'] != 1e-6:
        raise ValueError('invalid progress or zero tolerance')
    if type(c['seed']) is not int or type(c['scenario']['max_days']) is not int or c['scenario']['max_days'] < 1:
        raise ValueError('integer seed and positive day count required')
    if any(not np.isfinite(v) or v <= 0 for v in c['limits'].values()):
        raise ValueError('finite positive execution work limits required')
    if any(type(c['limits'][k]) is not int for k in ('max_steps', 'max_queries')):
        raise ValueError('integer teacher step/query budgets required')
    if c['limits']['segment_seconds'] <= o['max_seconds']:
        raise ValueError('execution segment budget must include decision plus native replay')


def signature(c, price_audit, certificate, model_path):
    root = Path(__file__).parent
    names = ['benchmark_b3_v4_da.py', 'benchmark_b3_v4_solver.py', 'benchmark_b2_tables.py', 'hybrid.py',
             'benchmark_checks.py', 'benchmark_prices.py', 'teacher.py', 'common.py',
             'power_limits.py', 'dispatch_health.py', 'prices.py', 'dispatch_continuous_prices.py']
    return dict(config=c, prices=price_audit, native=native_fingerprint(), certificate=digest(certificate),
        model_sha256=digest(model_path), model_complete_sha256=digest(Path(model_path).parent/'complete.json'),
        training_provenance={name:digest(Path(model_path).parent/name) for name in ('manifest.json','data_usage.json')
                             if (Path(model_path).parent/name).exists()},
        market_splits_sha256=digest(root.parents[1]/'configs/market_splits.json'),
        aging_tables=AgingTables().hashes,
        optimizer_scaling=dict(scaled=True, reference_soh=.95,
            source='B2 author-table endpoint ratios; hybrid damage remains unscaled',
            health_input='day-start native total SOH for both calendar and cyclic factors',
            factor_formula='table endpoint at reference SOH / table endpoint at day-start native SOH',
            factor_scope='fixed for all candidate evaluations and gradients of one day',
            optimizer_objective='predicted revenue - 6480 * (beta_cal * signed hybrid calendar damage + beta_cycl * signed hybrid cyclic damage)',
            prediction_objective='predicted revenue - 6480 * unscaled signed hybrid total damage',
            evaluation_objective='actual delivered revenue - 6480 * signed native ledger damage'),
        evaluation_scope='fixed trained hybrid model; historical perfect-foresight evaluation; no fitting in dispatch',
        units=dict(power='AC kW, positive discharge', price='EUR/kWh', temperature='degC',
                   voltage='pack V', current='pack A, positive discharge', damage='capacity-loss pu', time='UTC'),
        source={n:digest(root/n) for n in names}, python=sys.version,
        versions={k:md.version(k) for k in ('simses', 'torch', 'coptpy', 'numpy', 'pandas')})


def log(path,event,**values):
    with (path/'events.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(plain(dict(time=time.time(),event=event,**values)),allow_nan=False)+'\n')


def new_day(teacher,index):
    return dict(index=index,health_start=health(teacher,teacher.checkpoint()),
                intervals=[],mpc_windows=[],findings=[],native_temperature_violation=False)


def append_segment(day,result,checks,plan,prices,sources,index,windows):
    day=copy.deepcopy(day)
    day['findings']=sorted(set(day['findings'])|set(checks['findings']))
    if plan['solver']['deadline_exceeded']:
        day['findings']=sorted(set(day['findings'])|{'decision_time'})
    day['native_temperature_violation']|=checks['native_temperature_violation']
    day['mpc_windows'].append(dict(start_utc=windows.timestamp(index),plan=plan,execution_checks=checks))
    for j,(p,actual,dt) in enumerate(zip(result['u'],result['delivered'],result['dt'])):
        day['intervals'].append(dict(start_utc=windows.timestamp(index+j),end_utc=windows.timestamp(index+j+1),
            source_start_utc=sources[j],dt_h=float(dt),price_eur_kwh=float(prices[j]),command_ac_kw=float(p),
            delivered_ac_kw=float(actual),revenue_eur=float(prices[j]*actual*dt),
            predicted_soc=float(plan['predicted_soc'][j]),teacher_state=result['x'][j].tolist(),
            fine_extrema=result['extrema'][j].tolist()))
    return day


def day_result(day,teacher,kappa):
    end=health(teacher,teacher.checkpoint())
    damage=np.asarray(end['ledger_components_pu'])-day['health_start']['ledger_components_pu']
    rows=day['intervals'];revenue=sum(r['revenue_eur'] for r in rows)
    ranges=np.asarray([r['fine_extrema'] for r in rows])
    predicted_revenue=sum(w['plan']['predicted_revenue_eur'] for w in day['mpc_windows'])
    predicted_cost=sum(w['plan']['predicted_aging_cost_eur'] for w in day['mpc_windows'])
    optimizer_penalty=sum(w['plan']['optimizer_aging_penalty_eur'] for w in day['mpc_windows'])
    return plain(dict(**day,complete_day=len(rows)==96,failed=bool(day['findings']),
        health_end=end,damage_components_pu=damage,revenue_eur=revenue,
        predicted_revenue_eur=predicted_revenue,predicted_aging_cost_eur=predicted_cost,
        predicted_utility_eur=predicted_revenue-predicted_cost,
        optimizer_aging_penalty_eur=optimizer_penalty,
        optimizer_objective_eur=predicted_revenue-optimizer_penalty,
        scheduling_soc_band_violated=any(not w['execution_checks']['scheduling_soc']['passed'] for w in day['mpc_windows']),
        teacher_aging_cost_eur=kappa*float(damage.sum()),teacher_utility_eur=revenue-kappa*float(damage.sum()),
        temperature_range_c=[float(ranges[:,0,1].min()),float(ranges[:,1,1].max())],
        delivered_charge_kwh=sum(max(0,-r['delivered_ac_kw'])*r['dt_h'] for r in rows),
        delivered_discharge_kwh=sum(max(0,r['delivered_ac_kw'])*r['dt_h'] for r in rows),
        planned_revenue_eur=sum(r['command_ac_kw']*r['price_eur_kwh']*r['dt_h'] for r in rows),
        undelivered_charge_kwh=sum(max(0,r['delivered_ac_kw']-r['command_ac_kw'])*r['dt_h']
                                  for r in rows if r['command_ac_kw']<0),
        undelivered_discharge_kwh=sum(max(0,r['command_ac_kw']-r['delivered_ac_kw'])*r['dt_h']
                                     for r in rows if r['command_ac_kw']>0),
        solver_seconds=sum(x['plan']['solver']['elapsed_seconds'] for x in day['mpc_windows']),
        teacher_used_for_selection=False,temperature_check_enforced=True,
        terminal_soc_required=False,day_end_soc_check=None,aging_prediction_available=True,
        predicted_damage_components_pu=np.sum([w['plan']['predicted_damage_components_pu'] for w in day['mpc_windows']],axis=0),
        total_damage_prediction_error_pu=sum(w['plan']['predicted_total_damage_pu'] for w in day['mpc_windows'])-float(damage.sum()),
        decision_deadline_exceeded=any(w['plan']['solver']['deadline_exceeded'] for w in day['mpc_windows'])))


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


def summary(state,c):
    teacher=state['teacher'];revenue=state['revenue_eur'];nominal=c['teacher']['series']*c['teacher']['parallel']*.0096
    cost=nominal*c['economics']['installed_eur_per_kwh']
    kappa=nominal*c['economics']['cell_eur_per_kwh']/(1-c['economics']['eol_soh'])
    damage=np.asarray(state['ledger'])-state['initial_health']['ledger_components_pu']
    active=state['day']
    da=c.get('method') == 'b3-v4'
    return dict(schema='b3-v4-summary-v1',
        method=c.get('method','b3-v4'),prediction_intervals=c['mpc']['horizon_intervals'],
        execution_intervals=c['mpc']['execute_intervals'],
        intraday_reoptimization=not da,terminal_soc_required=c['mpc'].get('day_end_soc') is not None,
        terminal_inventory_value=False,settlement_includes_imbalance_market=False,
        pending_fixed_plan=bool(state.get('pending_plan')),
        status=state['status'],reason=state.get('reason'),
        completed_days=state['completed_days'],reported_days=state['reported_days'],executed_intervals=state['index'],
        failed_days=state['failed_days'],failure_days_by_type=state['failure_days_by_type'],
        failed_day_fraction=state['failed_days']/state['reported_days'] if state['reported_days'] else None,
        native_temperature_failure_days=state['native_temperature_failure_days'],
        scheduling_soc_violation_days=state['scheduling_soc_violation_days'],
        pending_day_findings=active['findings'] if active else [],
        pending_day_intervals=len(active['intervals']) if active else 0,
        revenue_eur=revenue,aging_prediction_available=True,
        optimizer_aging_penalty_eur=state['optimizer_aging_penalty_eur'],
        optimizer_objective_eur=state['predicted_revenue_eur']-state['optimizer_aging_penalty_eur'],
        predicted_revenue_eur=state['predicted_revenue_eur'],
        predicted_aging_cost_eur=state['predicted_aging_cost_eur'],
        predicted_utility_eur=state['predicted_revenue_eur']-state['predicted_aging_cost_eur'],
        initial_investment_eur=cost,cash_balance_eur=revenue-cost,
        delivered_charge_kwh=state['delivered_charge_kwh'],delivered_discharge_kwh=state['delivered_discharge_kwh'],
        payback_days=state['payback_days'],damage_components_pu=damage.tolist(),
        teacher_aging_cost_eur=float(kappa*damage.sum()),teacher_utility_eur=float(revenue-kappa*damage.sum()),
        final_soc=teacher['battery']['soc'],final_soh=teacher['battery']['soh_Q'],
        temperature_range_c=state['temperature_range_c'],usage=state['usage'],work_limits=state['work_limits'],
        crashes_reserved=state['crashes_reserved'],daily_chain=state['daily_chain'],
        failed_day_definition='physical/temperature/delivery finding or decision over 30s; scheduling SOC deviation reported separately; no teacher repair',
        cash_basis='actual delivered AC cash flow minus CAPEX once; no second depreciation cash deduction',
        benchmark='perfect-foresight deterministic historical-price scenario; not executable trading',
        source_price_scenario=c['scenario']['price_scenario'],teacher_used_for_selection=False,
        decision_timing=dict(attempts=len(state['decision_seconds']),
            mean_seconds=float(np.mean(state['decision_seconds'])) if state['decision_seconds'] else None,
            p95_seconds=float(np.percentile(state['decision_seconds'],95)) if state['decision_seconds'] else None,
            max_seconds=max(state['decision_seconds'],default=None),
            above_30_seconds=sum(x>30 for x in state['decision_seconds']),
            budget_exhausted_attempts=state['decision_budget_exhausted']),
        native_replay_seconds=state['native_replay_seconds'],worker_startup_seconds=state['worker_startup_seconds'],
        payback_target_days=2920,payback_target_met=state['payback_days'] is not None and state['payback_days']<=2920,
        revenue_at_2920_days_eur=state['revenue_at_2920_days_eur'],
        trained_model_frozen=True,optimizer_short_tests_run=False)


def compatible_budget_resume(previous, current):
    """Never migrate another algorithm or source signature into b3-v4."""
    return False


def run(c,windows,path,sig,*,model_path,resume=False,extensions=None,solver_factory=None,
        segment_seconds=None):
    validate_spec(c)
    if solver_factory is None:
        solver_factory = B3V4Solver
    np.random.seed(c['seed'])  # Deterministic mode search; retain explicit run seed.
    path=Path(path);extensions=extensions or {};target=c['scenario']['max_days']*96
    if set(extensions)-{'seconds','steps','queries'} or any(not np.isfinite(v) or v<0 for v in extensions.values()):
        raise ValueError('invalid work extensions')
    if segment_seconds is not None and (not np.isfinite(segment_seconds) or
                                       segment_seconds < c['limits']['segment_seconds']):
        raise ValueError('segment allowance must be finite and at least the configured allowance')
    da=c.get('method') == 'b3-v4';execute=c['mpc']['execute_intervals']
    steps_per_segment=int(execute*.25*3600/c['teacher']['internal_dt_s'])
    with exclusive(Path(str(path)+'.lock')):
        if resume:
            manifest = read_json(path/'manifest.json')
            previous = manifest['signature']
            migrate = previous != sig
            if migrate and not (segment_seconds is not None and compatible_budget_resume(previous, sig)):
                raise ValueError('resume source/config/input/environment mismatch')
            state=read_json(path/'checkpoint.json')
            chain=None
            for i in range(state['reported_days']):
                chain=canonical([chain,digest(path/'days'/f'{i+1:05d}.json')])
            if chain!=state['daily_chain']:raise ValueError('committed report hash chain mismatch')
            if state['status']=='complete':return summary(state,c)
            if migrate:
                manifest.setdefault('resume_migrations', []).append(dict(
                    time=time.time(), reason='explicit segment wall-time allowance adjustment',
                    previous_signature=previous, segment_seconds=segment_seconds))
                manifest['signature'] = sig
                save(path/'manifest.json', manifest)
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
                optimizer_aging_penalty_eur=0.,predicted_revenue_eur=0.,predicted_aging_cost_eur=0.,
                payback_days=None,temperature_range_c=[c['teacher']['initial_temperature_c']]*2)
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
                        print(f'{c.get("method","B3")} day {state["reported_days"]}: revenue={report["revenue_eur"]:.4f} EUR '
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
                        raise ValueError('B3 must start at a 96-interval simulation-day boundary')
                    if state['day'] is None:
                        state['day']=new_day(teacher,state['index']//96)
                        state['day']['method']=c.get('method','b3-v4')
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
                            raise RuntimeError('B3 has no checked feasible plan; no invented execution or repair')
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
                        state['optimizer_aging_penalty_eur']+=plan['optimizer_aging_penalty_eur']
                        state['predicted_aging_cost_eur']+=plan['predicted_aging_cost_eur']
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
    parser=argparse.ArgumentParser(description=__doc__)
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
                           help='Set per-execution-segment wall allowance; persists on resume, never changes optimizer 30s')
            p.add_argument('--add-seconds',type=float,default=0)
            p.add_argument('--add-steps',type=int,default=0)
            p.add_argument('--add-queries',type=int,default=0)
    args=parser.parse_args(argv)
    c=load_spec(args.config)
    from .data import check_certificate
    check_certificate(args.certificate)
    # Loading checks format, completion hash and constants; no forward/training/rollout here.
    model=load_model(args.model)
    nominal=c['teacher']['series']*c['teacher']['parallel']*.0096
    if not np.isclose(model.physics.energy_kwh,nominal,atol=1e-12,rtol=0):
        raise ValueError('model nominal energy does not match B3 device')
    del model
    windows,audit=load_windows(args.prices,c['scenario'],96,96)
    sig=signature(c,audit,args.certificate,args.model)
    if args.command=='preflight':
        print(json.dumps(dict(status='preflight_passed',simulation_executed=False,
            optimizer_executed=False,short_tests_executed=False,signature_sha256=canonical(sig),
            method=c['method'],days=c['scenario']['max_days'],model_sha256=sig['model_sha256'],
            power_bounds_ac_kw=list(ac_power_bounds(c['teacher'])),
            decision_seconds=c['solver']['max_seconds'],
            evaluation_kappa_eur_per_pu=6480,aging_tables=sig['aging_tables'],
            optimizer_scaling=sig['optimizer_scaling'],prices=audit,copt_license_tested=False),indent=2))
        return 0
    ext=dict(seconds=args.add_seconds,steps=args.add_steps,queries=args.add_queries)
    if not np.isfinite(list(ext.values())).all() or min(ext.values())<0:
        raise ValueError('nonnegative finite extensions required')
    out=run(c,windows,args.out,sig,model_path=args.model,resume=args.resume,extensions=ext,
            segment_seconds=args.segment_seconds)
    print(f'{c["method"]} {out["status"]}: {out["reason"]}; results: {args.out}',flush=True)
    return 0 if out['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
