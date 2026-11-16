from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import Budget, LimitReached, Run, canonical, digest, read_json, write_json
from .teacher import SCHEMA, Teacher, native_fingerprint


def check_certificate(path):
    import inspect
    c = read_json(path)
    if (c.get('schema') != SCHEMA or c.get('native_fingerprint') != native_fingerprint()
            or c.get('adapter_sha256') != digest(inspect.getfile(Teacher)) or not c.get('passed')):
        raise ValueError('missing/stale native accounting audit; run: bash scripts/check_teacher_certificate.sh')


def split_groups(count, train_fraction, validation_fraction, seed):
    if count < 5:
        raise ValueError('production data requires >=5 independent history groups')
    order = np.random.default_rng(seed).permutation(count)
    ntrain = max(1, int(count*train_fraction))
    nval = max(1, int(count*validation_fraction))
    if ntrain+nval >= count:
        raise ValueError('empty test partition')
    return {int(g): ('train' if i<ntrain else 'validation' if i<ntrain+nval else 'test') for i,g in enumerate(order)}


def trajectory(rng, h, amp, family, eta_ch, eta_dis):
    u = np.zeros(h)
    half = h//2
    if family == 'rest':
        return u
    if family == 'block':
        u[:half] = -amp
        u[half:] = amp*eta_ch*eta_dis*half/(h-half)
    elif family == 'pulse':
        u[::2] = -amp
        u[1::2] = amp*eta_ch*eta_dis*len(u[::2])/len(u[1::2])
    else:
        u = rng.uniform(0.2, 1, h)*amp
        u[:half] *= -1
        u[half:] *= -u[:half].sum()*eta_ch*eta_dis/u[half:].sum()
        u = np.roll(u, int(rng.integers(h)))
    return u


def direction(rng, u, dt, eta_ch, eta_dis, family):
    h = len(u)
    v = rng.normal(size=h)
    if family == 'shift':
        v = np.roll(u, 1)-u
    elif family == 'spread':
        v = (np.roll(u, 1)+np.roll(u, -1))/2-u
    a = np.where(u>=0, 1/eta_dis, eta_ch)*dt
    v -= a*(v@a)/(a@a)
    norm = np.linalg.norm(v)
    return v/norm if norm > 1e-12 else None


def serial_result(r):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}


def generate(config, out, certificate, resume=False):
    if config['data'].get('strategy')=='coverage-v2':
        raise ValueError('coverage-v2 was retired; no generator is available for that strategy')
    if config['kind'] != 'production':
        raise ValueError('dataset generation requires explicit production config; smoke is tests only')
    check_certificate(certificate)
    run = Run(out, config, 'generate', resume, dict(teacher=native_fingerprint()))
    budget = Budget(**config['limits'])
    d, t = config['data'], config['teacher']
    split = split_groups(d['groups'], d['train_fraction'], d['validation_fraction'], config['seed'])
    write_json(run.path/'splits.json', split)
    write_json(run.path/'units.json',dict(power='AC kW positive discharge',ambient='degC',dt='h',
        state=['SOC pu','cell degC','pack V','pack A positive discharge'],
        damage=['calendar capacity pu','settled cycle capacity pu','signed residual capacity pu'],
        direction='unit L2 vector in normalized power coordinates; epsilon dimensionless',
        teacher=native_fingerprint(),accounting_schema=SCHEMA))
    (run.path/'groups').mkdir(exist_ok=True)
    done = 0
    try:
        for g in range(d['groups']):
            target = run.path/'groups'/f'{g:06d}.json'
            if target.exists():
                done += 1
                continue
            budget.check()
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'], g]))
            teacher = Teacher(t, budget)
            ambient = float(rng.choice(d['temperature_splits_c'][split[g]]))
            history = rng.uniform(-d['amplitude_kw'], d['amplitude_kw'], d['history_intervals'])
            teacher.rollout(history, ambient, d['dt_h'], commit=True)
            checkpoint = teacher.checkpoint()
            samples, pairs = [], []
            # Entire history, all windows and perturbation families remain in one split.
            for family in ['rest', 'block', 'pulse', 'distributed']:
                u = trajectory(rng, d['horizon'], d['amplitude_kw'], family, t['eta_charge'], t['eta_discharge'])
                base = teacher.rollout(u, ambient, d['dt_h'])
                base['family']=family
                if not base['feasible']:
                    continue
                samples.append(serial_result(base))
                for perturb in ['shift', 'spread', 'random']:
                    v = direction(rng,u,d['dt_h'],t['eta_charge'],t['eta_discharge'],perturb)
                    if v is None:
                        continue
                    for eps in d['epsilon_splits'][split[g]]:
                        minus = teacher.rollout(u-eps*t['power_kw']*v, ambient, d['dt_h'])
                        plus = teacher.rollout(u+eps*t['power_kw']*v, ambient, d['dt_h'])
                        # Reject infeasible endpoints; never independently repair a centered pair.
                        if not (minus['feasible'] and plus['feasible']):
                            continue
                        a = len(samples)
                        samples.extend([serial_result(minus),serial_result(plus)])
                        pairs.append(dict(minus=a, plus=a+1,
                            center=u.tolist(), direction=v.tolist(), epsilon=eps, family=perturb,
                            response=float((plus['damage'].sum()-minus['damage'].sum())/(2*eps))))
            if not samples or not pairs:
                raise ValueError(f'group {g} has no feasible paired data; revise explicit configuration')
            write_json(target, dict(group=g, split=split[g], initial_checkpoint=checkpoint,
                       samples=samples, pairs=pairs, ambient=ambient))
            done += 1
            run.log('group_committed', group=g, split=split[g], samples=len(samples), pairs=len(pairs), **budget.report())
    except LimitReached as exc:
        run.log('paused', reason=str(exc), committed_groups=done, **budget.report())
        return
    write_json(run.path/'complete.json', dict(groups=done, manifest_hash=canonical(split)))
    run.log('complete', **budget.report())


def load_groups(path, split):
    path = Path(path)
    if not (path/'complete.json').exists():
        raise ValueError('dataset incomplete; resume generation before training/evaluation')
    groups = [read_json(p) for p in sorted((path/'groups').glob('*.json'))]
    return [g for g in groups if g['split']==split]
