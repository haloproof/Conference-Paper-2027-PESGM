"""Explicit estimator comparison; separate from the deployed hybrid trainer.

Only this CLI with an explicit method/config starts production fitting. No test
labels, teacher integration, dispatch, or plotting is performed. Trust only local
project .pt artifacts. Resume always restores the last fully committed unit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .common import (LimitReached, canonical, digest, environment, read_json,
                     write_json)
from .hybrid_data import load_state, save_state
from .hybrid_train import local_arrays, local_loss, read_days, update_selection
from .estimator_compare_models import (METHODS, build_model, daily_batch_loss,
                                      evaluate, hnn_day_losses)

SCHEMA = 'estimator-comparison-v1'
SLUGS = {'HNN-compare': 'hnn-compare', 'SOC-only': 'soc-only',
         'SOC NN': 'soc-nn', 'End-to-End NN': 'end-to-end-nn'}
SOURCE_FILES = ('estimator_compare.py', 'estimator_compare_models.py',
                'hybrid.py', 'hybrid_train.py', 'hybrid_data.py', 'common.py',
                'teacher.py')


def load_config(path):
    cfg = read_json(path)
    if cfg.get('schema') != SCHEMA or cfg.get('kind') != 'production':
        raise ValueError('explicit estimator-comparison-v1 production config required')
    for key in ('seed', 'evaluate_every'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'invalid {key}')
    settings = cfg['training']
    for stage in ('stage1', 'stage2', 'mlp'):
        s = settings[stage]
        if not math.isfinite(s['lr']) or s['lr'] <= 0:
            raise ValueError('positive finite learning rate required')
        for key in ('batch_size', 'epochs', 'patience'):
            if type(s[key]) is not int or s[key] <= 0:
                raise ValueError(f'positive integer {stage}.{key} required')
    if type(settings['stage2']['local_batch_size']) is not int or settings['stage2']['local_batch_size'] <= 0:
        raise ValueError('positive local heat batch size required')
    if settings['weights'] != [0.1, 1, 1] or settings['epsilon_calendar'] != 1e-6:
        raise ValueError('HNN comparison preserves the existing joint objective')
    if (len(settings['betas']) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in settings['betas'])
            or not math.isfinite(settings['adam_eps']) or settings['adam_eps'] <= 0
            or not math.isfinite(settings['weight_decay']) or settings['weight_decay'] < 0):
        raise ValueError('invalid Adam parameters')
    if not 0 <= settings['min_relative_improvement'] < 1:
        raise ValueError('invalid early stopping threshold')
    if type(settings['threads']) is not int or settings['threads'] < 1:
        raise ValueError('positive thread count required')
    cap = cfg['limits']['max_stage_seconds']
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('positive finite stage time limit required')
    return cfg


def dataset_header(dataset):
    path = Path(dataset)
    complete = read_json(path / 'complete.json')
    if complete['days'] != 960 or len(complete['files']) != 480:
        raise ValueError('completed 960-day HNN dataset required')
    if digest(path / 'dataset.json') != complete['metadata']:
        raise ValueError('dataset metadata changed')
    return dict(path=str(path.resolve()), complete_sha256=digest(path / 'complete.json'),
                metadata_sha256=complete['metadata'])


def signature(config, method, dataset):
    src = Path(__file__).parent
    return dict(schema=SCHEMA, method=method,
                config={k: v for k, v in config.items() if k != 'limits'},
                dataset=dataset_header(dataset),
                sources={name: digest(src / name) for name in SOURCE_FILES})


@contextmanager
def exclusive(path):
    """OS lock released even on process termination; the lock file is harmless."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('another process is using this comparison output') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def checkpoint_payload(model, stats, physics, method, stage, epoch):
    return dict(schema=SCHEMA, method=method, stage=stage, epoch=epoch,
                model=copy.deepcopy(model.state_dict()), normalizers=stats,
                physics=vars(physics), dtype='float64', output_unit='pu',
                label='calendar + cyclic (signed residual already included)')


def restore_model(model, payload, method):
    if payload.get('schema') != SCHEMA or payload.get('method') != method:
        raise ValueError('checkpoint is not from this comparison method; old hybrid weights are not accepted')
    model.load_state_dict(payload['model'])


def export_history(out, state):
    """Derived, replayable reports; latest.pt is authoritative after interruption."""
    write_json(out / 'history.json', dict(schema=SCHEMA, method=state['method'],
        epochs=state['epochs'], frozen=state['frozen'], selected=state['selected'],
        scope='training/validation only; test not loaded; no figure generated'))
    fields = ['method', 'stage', 'epoch', 'split', 'target', 'unit', 'count',
              'raw_sse', 'raw_mse', 'raw_mae', 'objective', 'checkpoint', 'checkpoint_sha256', 'roles']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in state['frozen']:
        for split, metrics in record['metrics'].items():
            row = {key: metrics[key] for key in fields if key in metrics}
            row.update(method=state['method'], stage=record['stage'], epoch=record['epoch'],
                       split=split, checkpoint=record['checkpoint'],
                       checkpoint_sha256=record['checkpoint_sha256'], roles=';'.join(record['roles']))
            writer.writerow(row)
    # Only this run's derived report is replaced. Original runs are never touched.
    tmp = out / 'metrics.csv.tmp'
    tmp.write_text(stream.getvalue(), encoding='utf-8', newline='')
    os.replace(tmp, out / 'metrics.csv')


def new_progress(stage):
    return dict(stage=stage, epoch=0, offset=0, order=None, phase='initial',
                selection=dict(best=float('inf'), reference=float('inf'), bad_epochs=0),
                train_sum=0., train_count=0, best_epoch=None, stop_reason=None)


def optimizer_for(model, settings, stage):
    if stage == 'fixed':
        return None
    return torch.optim.Adam(model.parameters(), lr=settings[stage]['lr'],
                            betas=tuple(settings['betas']), eps=settings['adam_eps'],
                            weight_decay=settings['weight_decay'])


def frozen_due(epoch, every, stopping=False):
    return epoch == 0 or epoch % every == 0 or stopping


def train(config, dataset, out, method, resume=False, stage_seconds=None):
    if method not in METHODS:
        raise ValueError('unknown comparison method')
    out = Path(out).resolve()
    # Also catches accidental use of runs/hybrid-fit-1020 on both fresh/resume paths.
    sig = signature(config, method, dataset)
    with exclusive(Path(str(out) + '.lock')):
        if resume:
            manifest = read_json(out / 'manifest.json')
            if manifest.get('signature') != sig:
                raise ValueError('resume requires the same comparison method, config, sources and dataset')
            if (out / 'complete.json').exists():
                print('Comparison already complete; use a new output directory for a new run.', flush=True)
                return 0
        else:
            out.mkdir(parents=True, exist_ok=False)
            write_json(out / 'manifest.json', dict(schema=SCHEMA, signature=sig,
                environment=environment(), run_id=out.name, created=time.time(),
                initial_limits=config['limits']))
        return _fit(config, dataset, out, method, resume, stage_seconds, canonical(sig))


def _fit(config, dataset, out, method, resume, stage_seconds, sig_hash):
    settings = config['training']
    latest = out / (SLUGS[method] + '-latest.pt')
    splits, physics = read_days(dataset)  # verifies hashes; never loads test labels
    model, stats = build_model(method, physics, splits['train'], config['seed'])
    torch.set_num_threads(settings['threads'])
    rng = np.random.default_rng(config['seed'])
    stages = ['stage1', 'stage2'] if method == 'HNN-compare' else ['fixed' if method == 'SOC-only' else 'mlp']
    if resume and latest.exists():
        state = load_state(latest)
        if state.get('signature_sha256') != sig_hash:
            raise ValueError('latest checkpoint provenance mismatch')
        restore_model(model, state, method)
        stats = state['normalizers']
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'])
    else:
        state = checkpoint_payload(model, stats, physics, method, stages[0], 0)
        state.update(progress=new_progress(stages[0]), epochs=[], frozen=[], selected={},
                     signature_sha256=sig_hash, total_work_seconds=0.)
    write_json(out / 'data_usage.json', dict(training_days=len(splits['train']),
        validation_days=len(splits['validation']), test_used=False,
        split_policy='original independent history groups; no reshuffle between splits',
        input_policy='command power; retain all samples including curtailment',
        label='calendar + cyclic; cyclic already contains signed residual correction',
        normalization='training only; frozen across epochs and splits',
        parameter_count=sum(p.numel() for p in model.parameters()),
        thermal_pretraining=(method == 'HNN-compare'), fresh_initialization=True,
        original_hybrid_weights_loaded=False, figure_generated=False))
    xy = local_arrays(splits['train']) if method == 'HNN-compare' else None
    stage = state['progress']['stage']
    optimizer = optimizer_for(model, settings, stage)
    if state.get('optimizer') is not None:
        optimizer.load_state_dict(state['optimizer'])
    timer = time.monotonic()
    committed_at = timer
    cap = stage_seconds if stage_seconds is not None else config['limits']['max_stage_seconds']
    max_unit_seconds = 0.

    def check():
        remaining = cap - (time.monotonic() - timer)
        if remaining <= max(2., 1.2 * max_unit_seconds):
            raise LimitReached('stage invocation budget reached; resume the last committed checkpoint')

    def commit(report=False):
        nonlocal committed_at
        now = time.monotonic()
        state.update(checkpoint_payload(model, stats, physics, method,
                     state['progress']['stage'], state['progress']['epoch']))
        state.update(optimizer=None if optimizer is None else copy.deepcopy(optimizer.state_dict()),
                     rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                     total_work_seconds=state['total_work_seconds'] + now - committed_at)
        save_state(latest, state)
        committed_at = now
        if report:
            export_history(out, state)

    def metrics_for(stage_name):
        return {split: evaluate(model, method, splits[split], stats, settings, stage_name, check)
                for split in ('train', 'validation')}

    def freeze(stage_name, epoch, metrics, roles):
        name = f'{SLUGS[method]}-{stage_name}-epoch-{epoch:04d}.pt'
        save_state(out / name, checkpoint_payload(model, stats, physics, method, stage_name, epoch))
        return dict(stage=stage_name, epoch=epoch, checkpoint=name,
                    checkpoint_sha256=digest(out / name), metrics=metrics, roles=roles)

    if not latest.exists():
        commit(report=True)  # recoverable even if the very first evaluation is interrupted
    else:
        export_history(out, state)
    try:
        while True:
            progress = state['progress']
            stage = progress['stage']
            check()
            if progress['phase'] == 'initial':
                metrics = metrics_for(stage)
                state['frozen'].append(freeze(stage, 0, metrics, ['initial'] if stage != 'fixed' else ['fixed', 'best', 'final']))
                progress['phase'] = 'done' if stage == 'fixed' else 'train'
                if stage == 'fixed':
                    state['selected'][stage] = state['frozen'][-1]
                    progress['stop_reason'] = 'zero_trainable_parameters'
                commit(report=True)
                print(f'{method} {stage}: initial frozen train/validation evaluation saved', flush=True)
                continue
            if progress['phase'] == 'train':
                s = settings[stage]
                count = len(xy[0]) if stage == 'stage1' else len(splits['train'])
                if progress['order'] is None:
                    progress['order'] = rng.permutation(count)
                    commit()
                model.train()
                while progress['offset'] < count:
                    check()
                    started = time.monotonic()
                    ids = progress['order'][progress['offset']:progress['offset'] + s['batch_size']]
                    optimizer.zero_grad(set_to_none=True)
                    if stage == 'stage1':
                        loss = local_loss(model.heat, xy[0][ids], xy[1][ids])
                    elif stage == 'stage2':
                        local_ids = rng.integers(len(xy[0]), size=s['local_batch_size'])
                        q = local_loss(model.heat, xy[0][local_ids], xy[1][local_ids])
                        t, c, _ = hnn_day_losses(model, [splits['train'][int(i)] for i in ids],
                                                stats, settings['epsilon_calendar'], check)
                        w = settings['weights']
                        loss = w[0] * q + w[1] * t + w[2] * c
                    else:
                        loss = daily_batch_loss(model, [splits['train'][int(i)] for i in ids])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite comparison training loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite comparison gradient')
                    check()
                    optimizer.step()
                    progress['offset'] += len(ids)
                    progress['train_sum'] += float(loss.detach()) * len(ids)
                    progress['train_count'] += len(ids)
                    commit()
                    max_unit_seconds = max(max_unit_seconds, time.monotonic() - started)
                # Keep progress unchanged until all required evaluation/saving succeeds.
                # An interrupted evaluation restarts from the final batch, not a new epoch.
                validation = evaluate(model, method, splits['validation'], stats, settings, stage, check)
                selection = copy.deepcopy(progress['selection'])
                improved = update_selection(selection, validation['objective'], settings['min_relative_improvement'])
                epoch = progress['epoch'] + 1
                stopping = selection['bad_epochs'] >= s['patience'] or epoch >= s['epochs']
                record = None
                if frozen_due(epoch, config['evaluate_every'], stopping):
                    train_metrics = evaluate(model, method, splits['train'], stats, settings, stage, check)
                    record = freeze(stage, epoch, dict(train=train_metrics, validation=validation),
                                    ['scheduled'] + (['final'] if stopping else []))
                if improved:
                    best_name = f'{SLUGS[method]}-{stage}-best.pt'
                    payload = checkpoint_payload(model, stats, physics, method, stage, epoch)
                    payload.update(validation=validation)
                    save_state(out / best_name, payload)
                    state['selected'][stage] = dict(epoch=epoch, checkpoint=best_name,
                        checkpoint_sha256=digest(out / best_name), validation=validation)
                state['epochs'].append(dict(stage=stage, epoch=epoch,
                    online_training_objective=progress['train_sum'] / progress['train_count'],
                    validation=validation, learning_rate=s['lr']))
                if record is not None:
                    state['frozen'].append(record)
                progress.update(selection=selection, epoch=epoch, offset=0, order=None,
                                train_sum=0., train_count=0., phase='finalize' if stopping else 'train')
                if improved:
                    progress['best_epoch'] = epoch
                if stopping:
                    progress['stop_reason'] = 'early_stop' if selection['bad_epochs'] >= s['patience'] else 'epoch_limit'
                commit(report=True)
                print(f'{method} {stage} epoch {epoch}: validation objective={validation["objective"]:.8g}; '
                      f'raw MSE={validation["raw_mse"]:.8g} {validation["unit"]}', flush=True)
                continue
            if progress['phase'] == 'finalize':
                best = state['selected'][stage]
                matches = [r for r in state['frozen'] if r['stage'] == stage and r['epoch'] == best['epoch']]
                if matches:
                    if 'best' not in matches[0]['roles']:
                        matches[0]['roles'].append('best')
                else:
                    last_weights = copy.deepcopy(model.state_dict())
                    try:
                        restore_model(model, load_state(out / best['checkpoint']), method)
                        metrics = metrics_for(stage)
                        record = freeze(stage, best['epoch'], metrics, ['best'])
                    finally:
                        model.load_state_dict(last_weights)
                    state['frozen'].append(record)
                progress['phase'] = 'done'
                commit(report=True)
                continue
            if progress['phase'] != 'done':
                raise ValueError('invalid resume phase')
            if stage == stages[-1]:
                chosen = state['selected'][stage]
                write_json(out / 'complete.json', dict(schema=SCHEMA, method=method,
                    selected=chosen, stop_reason=progress['stop_reason'],
                    history='history.json', frozen_metrics='metrics.csv',
                    total_work_seconds=state['total_work_seconds'],
                    test_evaluated=False, figure_generated=False))
                print(f'{method}: complete. Frozen metrics: {out / "metrics.csv"}', flush=True)
                return 0
            # HNN starts Stage 2 from its own freshly trained best heat network.
            restore_model(model, load_state(out / state['selected'][stage]['checkpoint']), method)
            state['progress'] = new_progress('stage2')
            optimizer = optimizer_for(model, settings, 'stage2')
            timer = time.monotonic()
            max_unit_seconds = 0.
            commit(report=True)
    except (LimitReached, KeyboardInterrupt) as exc:
        # Do not save half-computed gradients, consumed RNG, or an incomplete unit.
        # On resume model, optimizer, RNG and cursor all come from latest.pt.
        committed = load_state(latest)
        export_history(out, committed)
        print(f'{method}: paused ({type(exc).__name__}); resume with --resume. '
              f'Last committed stage={committed["progress"]["stage"]}, '
              f'epoch={committed["progress"]["epoch"]}, offset={committed["progress"]["offset"]}', flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=list(METHODS), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-stage-seconds', type=float)
    parser.add_argument('--check', action='store_true', help='config/header checks only; no model fitting or inference')
    args = parser.parse_args()
    if args.max_stage_seconds is not None and (not math.isfinite(args.max_stage_seconds) or args.max_stage_seconds <= 0):
        parser.error('positive finite --max-stage-seconds required')
    cfg = load_config(args.config)
    if args.check:
        sig = signature(cfg, args.method, args.dataset)
        if args.resume:
            if read_json(Path(args.out) / 'manifest.json')['signature'] != sig:
                parser.error('resume signature mismatch')
        elif Path(args.out).exists():
            parser.error('output already exists; use --resume or a new path')
        print(f'Checked {args.method}; no training/inference performed. Dataset header: {sig["dataset"]}')
        return
    return train(cfg, args.dataset, args.out, args.method, args.resume, args.max_stage_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
