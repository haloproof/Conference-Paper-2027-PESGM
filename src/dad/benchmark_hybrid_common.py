"""Shared daily accounting, provenance and stage checks for hybrid dispatch.

These helpers do not select a schedule or run an experiment.
"""
import copy
from contextlib import contextmanager
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time

import numpy as np

from .common import digest, write_json
from .teacher import native_fingerprint
from .dispatch_health import health


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


def signature(c, price_audit, certificate, model_path):
    root = Path(__file__).parent
    names = ['benchmark_hybrid_common.py', 'benchmark_hybrid_solver.py', 'hybrid.py',
             'benchmark_checks.py', 'benchmark_prices.py', 'teacher.py', 'common.py',
             'power_limits.py', 'dispatch_health.py', 'prices.py', 'dispatch_continuous_prices.py']
    return dict(config=c, prices=price_audit, native=native_fingerprint(), certificate=digest(certificate),
        model_sha256=digest(model_path), model_complete_sha256=digest(Path(model_path).parent/'complete.json'),
        training_provenance={name:digest(Path(model_path).parent/name) for name in ('manifest.json','data_usage.json')
                             if (Path(model_path).parent/name).exists()},
        market_splits_sha256=digest(root.parents[1]/'configs/market_splits.json'),
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
    return plain(dict(**day,complete_day=len(rows)==96,failed=bool(day['findings']),
        health_end=end,damage_components_pu=damage,revenue_eur=revenue,
        optimizer_aging_penalty_eur=sum(w['plan']['optimizer_aging_penalty_eur'] for w in day['mpc_windows']),
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


def summary(state,c):
    teacher=state['teacher'];revenue=state['revenue_eur'];nominal=c['teacher']['series']*c['teacher']['parallel']*.0096
    cost=nominal*c['economics']['installed_eur_per_kwh']
    kappa=nominal*c['economics']['cell_eur_per_kwh']/(1-c['economics']['eol_soh'])
    damage=np.asarray(state['ledger'])-state['initial_health']['ledger_components_pu']
    active=state['day']
    da=c['mpc']['horizon_intervals'] == c['mpc']['execute_intervals']
    return dict(schema='hybrid-summary-v1',
        method=c['method'],prediction_intervals=c['mpc']['horizon_intervals'],
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
        optimizer_aging_penalty_eur=state['optimizer_aging_penalty_eur'],predicted_revenue_eur=state['predicted_revenue_eur'],
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


def validate_daily_extension(old, new, state):
    a,b=copy.deepcopy(old),copy.deepcopy(new)
    sa,sb=a.pop('scenario'),b.pop('scenario')
    if a!=b or a['kind']!='production':
        raise ValueError('continuation may change scenario only, never method/physics/solver/economics')
    if sa['start_utc']!=sb['start_utc']:
        raise ValueError('continuation must retain original start date')
    if sb['max_days']<=sa['max_days']:
        raise ValueError('new cumulative day limit must increase')
    if state['status']!='complete' or state['reason']!='specified_intervals':
        raise ValueError('source must be completed by day limit, not EOL or interruption')
    if (state['index']!=sa['max_days']*96 or state['completed_days']!=sa['max_days'] or
        state['reported_days']!=sa['max_days'] or state.get('day') is not None or
        state.get('pending_plan') or state.get('execution_progress') or state.get('reservation')):
        raise ValueError('source must end at a fully committed daily boundary')
    if state['teacher']['battery']['soh_Q']<=old['economics']['eol_soh']:
        raise ValueError('source already reached native EOL')
    if sa['price_scenario']=='historical' and sb['price_scenario']=='historical':
        if {k:v for k,v in sa.items() if k!='max_days'}!={k:v for k,v in sb.items() if k!='max_days'}:
            raise ValueError('unexpected historical scenario change')
    elif sa['price_scenario']=='historical' and sb['price_scenario']=='repeat_history_year':
        if sb.get('repeat_start_utc')!=sa['start_utc'] or sa['max_days']>365:
            raise ValueError('repeat year must start at original historical start')
        if set(sb)-{'start_utc','max_days','price_scenario','repeat_start_utc','repeat_end_utc'}:
            raise ValueError('unexpected repeat scenario field')
    elif {k:v for k,v in sa.items() if k!='max_days'}!={k:v for k,v in sb.items() if k!='max_days'}:
        raise ValueError('unsupported scenario transition')
