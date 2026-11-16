"""Prepare a longer B1-DA stage from a completed checkpoint; never simulate."""
import argparse
import copy
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from .common import Budget, canonical, digest, read_json
from .benchmark_b1 import exclusive, load_spec, save, signature, summary
from .benchmark_prices import load_windows
from .teacher import Teacher
from .dispatch_health import health


def validate_extension(old, new, state):
    a,b=copy.deepcopy(old),copy.deepcopy(new)
    sa,sb=a.pop('scenario'),b.pop('scenario')
    if a!=b or a.get('method')!='B1-DA' or a['kind']!='production':
        raise ValueError('continuation may change scenario only, never method/physics/solver/economics')
    if sa['start_utc']!=sb['start_utc']:
        raise ValueError('continuation must retain original start date')
    if sb['max_days']<=sa['max_days']:
        raise ValueError('new cumulative day limit must increase')
    if state['status']!='complete' or state['reason']!='specified_intervals':
        raise ValueError('source must be completed by day limit, not EOL or interruption')
    if (state['index']!=sa['max_days']*96 or state['completed_days']!=sa['max_days'] or
        state['reported_days']!=sa['max_days'] or state.get('day') is not None or
        state.get('pending_plan') or state.get('reservation')):
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


def prepare(source, out, config, prices, certificate, check_only=False):
    source=Path(source).resolve();out=Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('use separate, non-nested stage directories')
    with exclusive(Path(str(source)+'.lock')), exclusive(Path(str(out)+'.lock')):
        if out.exists():raise FileExistsError(str(out)+' already exists; resume it instead of preparing again')
        manifest=read_json(source/'manifest.json');old=manifest['signature']['config']
        new=load_spec(config);state=read_json(source/'checkpoint.json')
        validate_extension(old,new,state)
        from .data import check_certificate
        check_certificate(certificate)
        previous,audit=load_windows(prices,old['scenario'],96,96)
        if signature(old,audit,certificate)!=manifest['signature']:
            raise ValueError('source code/config/prices/native/environment changed since source run')
        future,new_audit=load_windows(prices,new['scenario'],96,96)
        sig=signature(new,new_audit,certificate)
        chain=None;last=None;cash=0.
        for i in range(state['reported_days']):
            p=source/'days'/f'{i+1:05d}.json';d=read_json(p)
            chain=canonical([chain,digest(p)])
            x,xt=previous.window(i*96,96);y,yt=future.window(i*96,96)
            if not np.array_equal(x,y) or xt!=yt:
                raise ValueError('new price scenario changes already executed history')
            if not d['complete_day'] or d['index']!=i:
                raise ValueError('invalid committed day')
            origin=state['initial_health'] if last is None else last['health_end']
            if d['health_start']!=origin:raise ValueError('broken native day handoff')
            cash+=d['revenue_eur'];last=d
        if chain!=state['daily_chain']:raise ValueError('committed report hash chain mismatch')
        t=Teacher(new['teacher'],Budget(30,1,1));t.restore(state['teacher'])
        if health(t,t.checkpoint())!=last['health_end'] or not np.allclose(t.ledger(),state['ledger'],rtol=0,atol=1e-12):
            raise ValueError('teacher checkpoint disagrees with committed ledger/state')
        if not np.isclose(cash,state['revenue_eur'],rtol=0,atol=1e-9):
            raise ValueError('cumulative cash mismatch')
        if not np.isclose(t.time_s,state['index']*900,rtol=0,atol=1e-6):
            raise ValueError('native clock disagrees with execution index')
        record=dict(schema='B1-DA-stage-continuation-v1',source=str(source),destination=str(out),
            source_manifest_sha256=digest(source/'manifest.json'),
            source_checkpoint_sha256=digest(source/'checkpoint.json'),
            preparation_source_sha256=digest(Path(__file__)),
            inherited_days=state['completed_days'],next_day=state['completed_days']+1,
            cumulative_target_days=new['scenario']['max_days'],
            native_soh=t.battery.state.soh_Q,soc=t.battery.state.soc,
            preserved='full teacher, half-cycle history, cash, damage, usage, daily hash chain',
            simulation_executed=False)
        if check_only:return record
        staged=Path(tempfile.mkdtemp(prefix=out.name+'.preparing-',dir=out.parent))
        try:
            shutil.copytree(source/'days',staged/'days')
            if (source/'events.jsonl').exists():shutil.copy2(source/'events.jsonl',staged/'events.jsonl')
            save(staged/'source-manifest.json',manifest)
            save(staged/'source-checkpoint.json',state)
            state=copy.deepcopy(state);state.update(status='ready',reason='stage_extended')
            save(staged/'checkpoint.json',state)
            save(staged/'manifest.json',dict(signature=sig,run_id=out.name,created=time.time(),
                executable=manifest.get('executable'),continuation=record))
            save(staged/'continuation.json',record)
            save(staged/'summary.json',summary(state,new))
            staged.rename(out)
        except BaseException:
            # Remove only our explicitly created staging directory; never the source.
            if staged.exists():shutil.rmtree(staged)
            raise
        return record


def main(argv=None):
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--prices',default='data/dispatch-prices/prices.csv')
    p.add_argument('--certificate',default='reports/teacher_certificate.json')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv)
    print(json.dumps(prepare(a.source,a.out,a.config,a.prices,a.certificate,a.check_only),indent=2))


if __name__=='__main__':main()
