"""Prepare a longer B4 stage without running dispatch or native integration."""
import argparse
import copy
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from .common import Budget, canonical, digest, read_json
from .benchmark_b4_da import exclusive, load_spec, save, signature, summary
from .benchmark_hybrid_common import validate_daily_extension
from .benchmark_prices import load_windows
from .teacher import Teacher
from .dispatch_health import health


def validate_extension(old, new, state, *, required_method='B4'):
    if required_method is not None and old.get('method') != required_method:
        raise ValueError(f'{required_method} continuation requires a {required_method} source')
    method = old.get('method')
    if method != 'B4' or new.get('method') != method:
        raise ValueError('continuation must keep the same B4 method')
    validate_daily_extension(old, new, state)


def prepare(source, out, config, prices, certificate, model_path, check_only=False, *, required_method='B4'):
    new=load_spec(config, required_method=required_method)
    source=Path(source).resolve();out=Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('use separate, non-nested stage directories')
    with exclusive(Path(str(source)+'.lock')), exclusive(Path(str(out)+'.lock')):
        if out.exists():raise FileExistsError(str(out)+' already exists; resume it instead of preparing again')
        manifest=read_json(source/'manifest.json');old=manifest['signature']['config']
        state=read_json(source/'checkpoint.json')
        validate_extension(old,new,state,required_method=required_method)
        from .data import check_certificate
        from .benchmark_hybrid_solver import load_model
        check_certificate(certificate)
        load_model(model_path)  # Validate frozen model completion/hash; no forward pass.
        previous,audit=load_windows(prices,old['scenario'],96,96)
        if signature(old,audit,certificate,model_path)!=manifest['signature']:
            raise ValueError('source code/config/prices/native/environment changed since source run')
        future,new_audit=load_windows(prices,new['scenario'],96,96)
        sig=signature(new,new_audit,certificate,model_path)
        chain=None;last=None;cash=0.;penalty=0.;predicted=0.;hybrid_cost=0.
        for i in range(state['reported_days']):
            p=source/'days'/f'{i+1:05d}.json';d=read_json(p)
            chain=canonical([chain,digest(p)])
            x,xt=previous.window(i*96,96);y,yt=future.window(i*96,96)
            if not np.array_equal(x,y) or xt!=yt:
                raise ValueError('new price scenario changes already executed history')
            if not d['complete_day'] or d['index']!=i or len(d['intervals'])!=96:
                raise ValueError('invalid committed day')
            origin=state['initial_health'] if last is None else last['health_end']
            if d['health_start']!=origin:raise ValueError('broken native day handoff')
            cash+=d['revenue_eur'];last=d
            penalty+=d['optimizer_aging_penalty_eur'];predicted+=sum(w['plan']['predicted_revenue_eur'] for w in d['mpc_windows'])
            hybrid_cost+=sum(w['plan']['predicted_aging_cost_eur'] for w in d['mpc_windows'])
        if chain!=state['daily_chain']:raise ValueError('committed report hash chain mismatch')
        t=Teacher(new['teacher'],Budget(30,1,1));t.restore(state['teacher'])
        if health(t,t.checkpoint())!=last['health_end'] or not np.allclose(t.ledger(),state['ledger'],rtol=0,atol=1e-12):
            raise ValueError('teacher checkpoint disagrees with committed ledger/state')
        if not np.isclose(cash,state['revenue_eur'],rtol=0,atol=1e-9):
            raise ValueError('cumulative cash mismatch')
        for key,value in [('optimizer_aging_penalty_eur',penalty),('predicted_revenue_eur',predicted),
                          ('predicted_hybrid_aging_cost_eur',hybrid_cost)]:
            if not np.isclose(value,state[key],rtol=1e-12,atol=1e-9):raise ValueError('cumulative B4 prediction accounting mismatch')
        if not np.isclose(t.time_s,state['index']*900,rtol=0,atol=1e-6):
            raise ValueError('native clock disagrees with execution index')
        record=dict(schema=old['method']+'-stage-continuation-v1',source=str(source),destination=str(out),
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
            if staged.exists():
                if staged.resolve().parent != out.parent or not staged.name.startswith(out.name+'.preparing-'):
                    raise RuntimeError('unexpected staging cleanup path')
                shutil.rmtree(staged)
            raise
        return record


def main(argv=None):
    method = 'B4'
    import json
    p=argparse.ArgumentParser(description=f'Prepare a longer {method} stage; never simulate')
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--prices',default='data/dispatch-prices/prices.csv')
    p.add_argument('--certificate',default='reports/teacher_certificate.json')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv)
    print(json.dumps(prepare(a.source,a.out,a.config,a.prices,a.certificate,a.model,a.check_only,
                             required_method=method),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
