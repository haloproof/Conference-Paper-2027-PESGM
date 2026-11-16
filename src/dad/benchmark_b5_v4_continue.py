"""Prepare native-SOH retirement continuation of b5-v4; never simulate.

The original bounded-day runner and results remain unchanged. A separate EOL
runner consumes the full copied checkpoint with no cumulative day stopping limit.
"""
import argparse
import copy
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np

from .common import Budget, canonical, digest, read_json
from .benchmark_b5_v4_eol import exclusive, load_spec, save, signature, summary, load_windows
from .benchmark_b5_v4_da import signature as source_signature
from .benchmark_b5_v4_solver import load_model
from .benchmark_prices import load_windows as source_load_windows
from .teacher import Teacher
from .dispatch_health import health


def validate_extension(old, new, state):
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    sa, sb = a.pop('scenario'), b.pop('scenario')
    if a != b or a.get('method') != 'b5-v4' or a.get('algorithm') != 'b5-v4' or a.get('kind') != 'production':
        raise ValueError('continuation must retain the b5-v4 method, physics, scaling, solver and economics')
    if {k: v for k, v in sa.items() if k != 'max_days'} != {k: v for k, v in sb.items() if k != 'max_days'}:
        raise ValueError('continuation may change only the cumulative day limit, not the price scenario')
    if (type(sa.get('max_days')) is not int or sa['max_days'] < 1 or
            'max_days' not in sb or sb['max_days'] is not None or
            sa.get('price_scenario') != 'repeat_history_year'):
        raise ValueError('retirement continuation requires a bounded repeated-price source and max_days=null target')
    if state['status'] != 'complete' or state['reason'] != 'specified_intervals':
        raise ValueError('source must be completed by day limit, not EOL or interruption')
    if (state['index'] != sa['max_days']*96 or state['completed_days'] != sa['max_days'] or
            state['reported_days'] != sa['max_days'] or state.get('day') is not None or
            state.get('pending_plan') or state.get('execution_progress') or state.get('reservation')):
        raise ValueError('source must end at a fully committed daily boundary')
    if state['teacher']['battery']['soh_Q'] <= old['economics']['eol_soh']:
        raise ValueError('source already reached native EOL')


def prepare(source, out, config, prices, certificate, model_path, check_only=False):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('use separate, non-nested stage directories')
    with exclusive(Path(str(source)+'.lock')), exclusive(Path(str(out)+'.lock')):
        if out.exists():
            raise FileExistsError(str(out)+' already exists; resume it instead of preparing again')
        manifest = read_json(source/'manifest.json')
        old = manifest['signature']['config']
        new = load_spec(config)
        state = read_json(source/'checkpoint.json')
        validate_extension(old, new, state)
        from .data import check_certificate
        check_certificate(certificate)
        load_model(model_path)  # Completion and hash validation only; no model forward.
        previous, audit = source_load_windows(prices, old['scenario'], 96, 96)
        if source_signature(old, audit, certificate, model_path) != manifest['signature']:
            raise ValueError('source code/config/prices/native/environment changed since source run')
        future, new_audit = load_windows(prices, new['scenario'], 96, 96)
        # Only max_days may differ. Equal mapping and full source axes imply equal
        # prices and source timestamps for every historical window, including repeats.
        # Avoid formatting the same 96 timestamps for thousands of repeated days.
        if (audit != new_audit or previous.start != future.start or
                previous.repeat != future.repeat or previous.offset != future.offset or
                not np.array_equal(previous.values, future.values) or
                not previous.frame[['start_utc', 'end_utc']].equals(future.frame[['start_utc', 'end_utc']])):
            raise ValueError('new price scenario changes already executed history')
        sig = signature(new, new_audit, certificate, model_path)
        chain, last = None, None
        cash = penalty = predicted = raw_cost = 0.
        for i in range(state['reported_days']):
            path = source/'days'/f'{i+1:05d}.json'
            day = read_json(path)
            chain = canonical([chain, digest(path)])
            if not day['complete_day'] or day['index'] != i or len(day['intervals']) != 96:
                raise ValueError('invalid committed day')
            origin = state['initial_health'] if last is None else last['health_end']
            if day['health_start'] != origin:
                raise ValueError('broken native day handoff')
            cash += day['revenue_eur']
            penalty += day['optimizer_aging_penalty_eur']
            predicted += sum(w['plan']['predicted_revenue_eur'] for w in day['mpc_windows'])
            raw_cost += sum(w['plan']['predicted_aging_cost_eur'] for w in day['mpc_windows'])
            last = day
        if chain != state['daily_chain']:
            raise ValueError('committed report hash chain mismatch')
        teacher = Teacher(new['teacher'], Budget(30, 1, 1))
        teacher.restore(state['teacher'])
        if (health(teacher, teacher.checkpoint()) != last['health_end'] or
                not np.allclose(teacher.ledger(), state['ledger'], rtol=0, atol=1e-12)):
            raise ValueError('teacher checkpoint disagrees with committed ledger/state')
        if not np.isclose(cash, state['revenue_eur'], rtol=0, atol=1e-9):
            raise ValueError('cumulative cash mismatch')
        for key, value in [('optimizer_aging_penalty_eur', penalty),
                           ('predicted_revenue_eur', predicted), ('predicted_hybrid_aging_cost_eur', raw_cost)]:
            if not np.isclose(value, state[key], rtol=1e-12, atol=1e-9):
                raise ValueError('cumulative b5-v4 prediction accounting mismatch: '+key)
        if not np.isclose(teacher.time_s, state['index']*900, rtol=0, atol=1e-6):
            raise ValueError('native clock disagrees with execution index')
        record = dict(schema='b5-v4-stage-continuation-v1', source=str(source), destination=str(out),
            source_manifest_sha256=digest(source/'manifest.json'),
            source_checkpoint_sha256=digest(source/'checkpoint.json'),
            preparation_source_sha256=digest(Path(__file__)),
            inherited_days=state['completed_days'], next_day=state['completed_days']+1,
            cumulative_target_days=None,
            termination=dict(native_soh_at_most=new['economics']['eol_soh'],
                             max_days=None, checked_at='complete daily boundary'),
            native_soh=teacher.battery.state.soh_Q, soc=teacher.battery.state.soc,
            price_prefix_verification='unchanged scenario, full source price/time axes and index mapping',
            preserved='full teacher, half-cycle history, cash, damage, scaled penalty, raw prediction cost, usage and daily hash chain',
            simulation_executed=False)
        if check_only:
            return record
        staged = Path(tempfile.mkdtemp(prefix=out.name+'.preparing-', dir=out.parent))
        try:
            shutil.copytree(source/'days', staged/'days')
            if (source/'events.jsonl').exists():
                shutil.copy2(source/'events.jsonl', staged/'events.jsonl')
            save(staged/'source-manifest.json', manifest)
            save(staged/'source-checkpoint.json', state)
            next_state = copy.deepcopy(state)
            next_state.update(status='ready', reason='stage_extended')
            save(staged/'checkpoint.json', next_state)
            save(staged/'manifest.json', dict(signature=sig, run_id=out.name, created=time.time(),
                executable=manifest.get('executable'), continuation=record))
            save(staged/'continuation.json', record)
            save(staged/'summary.json', summary(next_state, new))
            if (staged.resolve().parent != out.parent or
                    not staged.resolve().name.startswith(out.name+'.preparing-')):
                raise RuntimeError('unexpected staging move path')
            if out.exists():
                raise FileExistsError(str(out)+' appeared during preparation')
            staged.rename(out)
        except BaseException:
            # Restrict recursive cleanup to the exact temporary stage under out.parent.
            if staged.exists():
                target = staged.resolve()
                if target.parent != out.parent or not target.name.startswith(out.name+'.preparing-'):
                    raise RuntimeError('unexpected staging cleanup path')
                shutil.rmtree(staged)
            raise
        return record


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prices', default='data/dispatch-prices/prices.csv')
    parser.add_argument('--certificate', default='reports/teacher_certificate.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.out, args.config, args.prices,
                             args.certificate, args.model, args.check_only), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
