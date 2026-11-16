"""idea-1016 2.2/2.3: explicit native history and price-free data commands.

No work runs on import. All persisted .pt files are trusted local artifacts.
"""
from __future__ import annotations
import argparse
import copy
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .common import Budget, LimitReached, Run, canonical, digest, read_json, write_json
from .data import check_certificate
from .hybrid import InitialState, Physics
from .teacher import Teacher, native_fingerprint

FAMILIES = ('random', 'shallow', 'deep', 'idle_crossday')
HISTORIES = ('rest_heavy', 'cycle_heavy', 'mixed')
FINE_COLUMNS = ['time_s', 'soc_pu', 'temperature_c', 'voltage_v', 'current_a_native',
                'delivered_ac_kw', 'q_calendar_pu', 'q_cyclic_settled_pu', 'heat_w']


def save_state(path, value):
    """Commit a closed, unique temporary file; never remove the old checkpoint.

    Windows scanners/readers may temporarily deny replace. On persistent denial,
    retain the fully serialized candidate for recovery and report its exact path.
    """
    path = Path(path)
    tmp = None
    ready = False
    committed = False
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        ready = True
        delays = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 3.2)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(tmp, path)
                committed = True
                return
            except PermissionError as exc:
                if attempt == len(delays):
                    raise PermissionError(
                        f'Cannot replace checkpoint {path} after 8 attempts. '
                        f'The previous checkpoint is unchanged; the new candidate is retained at {tmp}. '
                        'Check file readers, other writers and directory access, then resume. '
                        'Do not delete the previous checkpoint.'
                    ) from exc
                time.sleep(delays[attempt])
    finally:
        if tmp is not None and (committed or not ready):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass  # cleanup must not mask the original save error


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_config(path):
    c = read_json(path)
    if c['kind'] != 'production':
        raise ValueError('explicit production config required; smoke uses synthetic tests only')
    d, t = c['data'], c['teacher']
    if d['groups'] != 30 or d['branch_intervals'] != 192 or d['dt_h'] != .25:
        raise ValueError('this design requires 30 groups and 192 quarter-hour intervals')
    if t['internal_dt_s'] != 10 or (t['soc_min'], t['soc_max']) != (.1, .9):
        raise ValueError('native grid/SOC protocol changed')
    if d['ambient_c'] != 25 or 'dc_power_kw' in t or t.get('max_c_rate') is not None:
        raise ValueError('use 25 C and common AC nominal-energy 1C protocol')
    if any(v <= 0 for v in c['limits'].values()) or d['history_max_intervals'] <= 0 or d['save_every_intervals'] <= 0:
        raise ValueError('positive work limits required')
    return c


def group_specs(seed=1016):
    result = []
    proportions = ((7, 2, 1), (7, 1, 2), (6, 2, 2))
    ranges = ((.05, .15, .1, .4), (.60, .85, .3, 1.), (.25, .50, .1, .8))
    for family, counts, bounds in zip(HISTORIES, proportions, ranges):
        fi = HISTORIES.index(family)
        rng = np.random.default_rng(np.random.SeedSequence([seed, fi, 100]))
        order = rng.permutation(10)
        splits = ['train']*counts[0] + ['validation']*counts[1] + ['test']*counts[2]
        for j in range(10):
            center = rng.uniform(.2+.06*j, .2+.06*(j+1))
            width = rng.uniform(.25, .95) * 2*min(center-.1, .9-center)
            result.append(dict(group=10*fi+j, family=family, split=splits[int(np.where(order==j)[0][0])],
                center=center, low=center-width/2, high=center+width/2,
                duty=rng.uniform(*bounds[:2]), rate=rng.uniform(*bounds[2:]),
                targets=[rng.uniform(lo, hi) for lo, hi in ((.95, 1.), (.9, .95), (.85, .9), (.8, .85))]))
    # One reachable fresh training anchor supplies a genuine empty detector.
    # All other stage targets are sampled; no aged detector is ever cleared.
    anchor=next(s for s in result if s["family"]=="rest_heavy" and s["split"]=="train")
    anchor["targets"][0]=1.
    anchor["fresh_anchor"]=True
    return result


def make_teacher(config, budget):
    teacher = Teacher(config['teacher'], budget)
    nominal = teacher.battery.nominal_energy_capacity / 1000
    if not math.isclose(config['teacher']['power_kw'], nominal, rel_tol=1e-10):
        raise ValueError(f'AC 1C must equal nominal energy / 1 h: {nominal} kW')
    return teacher


def safe_power(teacher, target, rate, length=1):
    """Preselect a full-interval command, never rewrite delivered power as input.

    Use a 5% headroom against the confirmed SOC approximation; native limits
    remain authoritative and every curtailed interval is retained and flagged.
    """
    b = teacher.battery.state
    energy = teacher.battery.nominal_energy_capacity/1000 * b.soh_Q
    delta = target-b.soc
    if abs(delta) < 1e-8:
        return 0.
    amplitude = min(rate*teacher.config['power_kw'],
                    .95*abs(delta)*energy/(.25*length*(.91 if delta>0 else 1/.97)))
    return float(-amplitude if delta>0 else amplitude)


def interval(teacher, power):
    """One atomic 15-minute interval, including all 90 native observations."""
    before = teacher.checkpoint()
    fine = []
    active_seconds = 0.
    max_delivery_error = 0.
    try:
        teacher.thermal.T_ambient = 25.
        for _ in range(90):
            teacher.budget.step()
            previous_soc = teacher.battery.state.soc
            teacher.battery.step(teacher.ac_to_native(power), 10.)
            teacher.thermal.step(10.)
            teacher.time_s += 10.
            b, d = teacher.battery.state, teacher.battery.degradation.state
            delivered = teacher.delivered_ac()
            active_seconds += 10. if b.soc != previous_soc else 0.
            max_delivery_error = max(max_delivery_error, abs(delivered-power))
            fine.append([teacher.time_s, b.soc, b.T, b.v, b.i, delivered, d.qloss_cal, d.qloss_cyc, b.heat])
        fine = np.asarray(fine, dtype=np.float64)
        if not np.isfinite(fine).all():
            raise FloatingPointError('nonfinite native record')
    except BaseException:
        teacher.restore(before)  # retry starts at the last fully committed interval
        raise
    return dict(power=float(power), before=before, after=teacher.checkpoint(), fine=fine,
                delivered=float(fine[:,5].mean()), active_hours=active_seconds/3600,
                curtailed=max_delivery_error > teacher.config['power_tolerance_kw'],
                max_delivery_error=max_delivery_error)


def history_command(teacher, spec, control, rng):
    if control.get('idle', 0):
        control['idle'] -= 1
        return 0.
    if not control.get('left', 0):
        length = int(rng.choice([1,2,4,8]))
        control['left'] = length
        control['next_idle'] = int(round(length*(1-spec['duty'])/spec['duty']))
        control['rate'] = spec['rate']*rng.uniform(.7, 1.)
    target = control.setdefault('target', spec['high'])
    if abs(teacher.battery.state.soc-target) < .005:
        target = spec['low'] if target == spec['high'] else spec['high']
        control['target'] = target
    p = safe_power(teacher, target, control['rate'])
    control['left'] -= 1
    if control['left'] == 0:
        control['idle'] = control['next_idle']
    return p


def branch_command(teacher, family, index, control, rng, template=0):
    soc = teacher.battery.state.soc
    if family == 'idle_crossday':
        # Dedicated placements around midnight (96); repeated signed events
        # permit inherited continuation, rest across midnight and next-day reversal.
        schedules = [((0,16,-1),(40,56,-1),(104,120,1)),
                     ((0,16,-1),(40,56,1),(104,120,-1)),
                     ((88,104,-1),(136,152,1)),
                     ((72,88,-1),(104,120,-1),(160,176,1)),
                     ((72,88,1),(104,120,-1),(160,176,1))]
        for start, stop, sign in schedules[template%len(schedules)]:
            if start <= index < stop:
                return safe_power(teacher, .85 if sign<0 else .15, .15, stop-index)
        return 0.
    if family in ('shallow','deep') or (family == 'random' and index < 32):
        if 'low' not in control:
            depth = float(rng.uniform(*({'shallow':(.02,.15),'deep':(.4,.8)}.get(family,(.15,.4)))))
            low = float(rng.uniform(.1, .9-depth))
            control.update(low=low, high=low+depth, target=low+depth)
        if abs(soc-control['target']) < .002:
            control['target'] = control['low'] if control['target']==control['high'] else control['high']
            control['idle'] = int(rng.integers(0,3)) if family=='shallow' else 0
        if control.get('idle',0):
            control['idle'] -= 1
            return 0.
        return safe_power(teacher, control['target'], float(rng.uniform(.05,1.)))
    if not control.get('left',0):
        control['left'] = int(rng.choice([1,2,4,8]))
        sign = int(rng.choice([-1,1,0],p=[.4,.4,.2]))
        band = int(rng.choice(3,p=[.2,.6,.2]))
        rate = float(rng.uniform(*[(0,.05),(.05,.8),(.8,1.)][band]))
        if sign and rate == 0:
            rate = np.nextafter(0.,1.)
        control['power'] = 0. if not sign else safe_power(teacher,.9 if sign<0 else .1,rate,control['left'])
    control['left'] -= 1
    return control['power']


def day_record(rows, physics, group, stage, family, day):
    """Native labels; retain signed residual correction and full checkpoints."""
    start, end = rows[0]['before'], rows[-1]['after']
    # Ledger on restored complete states, never insert a reversal or mutate history.
    budget = Budget(30,1,1)
    teacher = Teacher(start['config'], budget)
    teacher.restore(start)
    l0 = teacher.ledger()
    teacher.restore(end)
    l1 = teacher.ledger()
    damage = l1-l0
    if (damage[:2] < 0).any():
        raise ArithmeticError('inconsistent settled loss; labels are never clipped')
    initial = InitialState.from_checkpoint(start)
    a = physics.alpha**90
    b = physics.resistance_k_per_w*(1-a)
    x, heat = [], []
    for row in rows:
        s, e = row['before']['battery'], row['after']['battery']
        x.append([row['power'],s['soc'],s['T'],initial.q_calendar,initial.q_cyclic])
        heat.append((e['T']-a*s['T']-(1-a)*25.)/b)
    return dict(group=group,stage=stage,family=family,day=day,initial=start,final=end,
                x=np.asarray(x),heat=np.asarray(heat),power=np.array([r['power'] for r in rows]),
                temperature=np.array([r['after']['battery']['T'] for r in rows]),
                soc=np.array([r['after']['battery']['soc'] for r in rows]),
                calendar=float(damage[0]),cyclic=float(damage[1]+damage[2]),
                ledger_start=l0,ledger_end=l1,
                delivered=np.array([r['delivered'] for r in rows]),
                active_hours=np.array([r['active_hours'] for r in rows]),
                curtailed=np.array([r['curtailed'] for r in rows]))


def histories(config, out, certificate, resume=False):
    check_certificate(certificate)
    run = Run(out,config,'hybrid-histories',resume,dict(native=native_fingerprint(),certificate=digest(certificate)))
    budget = Budget(**config['limits'])
    specs = group_specs(config['seed'])
    write_json(run.path/'groups.json',specs)
    try:
        for spec in specs:
            path = run.path/f"history-{spec['group']:02d}.pt"
            state = load_state(path) if path.exists() else dict(index=0,stages=[],control={},checkpoint=None,rng=None)
            if len(state['stages']) == 4:
                continue
            teacher = make_teacher(config,budget)
            budget.query()
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'],spec['group'],200]))
            if state['checkpoint'] is not None:
                teacher.restore(state['checkpoint']); rng.bit_generator.state=state['rng']
            elif spec.get('fresh_anchor'):
                state['stages'].append(teacher.checkpoint())
            try:
                while len(state['stages']) < 4:
                    if state['index'] >= config['data']['history_max_intervals']:
                        raise LimitReached(f"history {spec['group']} reached configured interval limit; missing stages")
                    old_control, old_rng = copy.deepcopy(state['control']), copy.deepcopy(rng.bit_generator.state)
                    p = history_command(teacher,spec,state['control'],rng)
                    try:
                        interval(teacher,p)
                    except BaseException:
                        state['control']=old_control; rng.bit_generator.state=old_rng
                        raise
                    state['index'] += 1
                    stage = len(state['stages'])
                    soh = teacher.battery.state.soh_Q
                    if soh <= spec['targets'][stage]:
                        lo,hi=[(.95,1),(.90,.95),(.85,.90),(.80,.85)][stage]
                        if not lo <= soh <= hi:
                            raise ValueError('health stage skipped; cannot manufacture a checkpoint')
                        state['stages'].append(teacher.checkpoint())
                        run.log('health_stage',group=spec['group'],stage=stage,soh=soh,**budget.report())
                    if state['index'] % config['data']['save_every_intervals'] == 0:
                        state.update(checkpoint=teacher.checkpoint(),rng=copy.deepcopy(rng.bit_generator.state))
                        save_state(path,state)
            finally:
                state.update(checkpoint=teacher.checkpoint(),rng=copy.deepcopy(rng.bit_generator.state))
                save_state(path,state)
    except LimitReached as exc:
        run.log('paused',reason=str(exc),**budget.report()); return
    write_json(run.path/'complete.json',dict(files={p.name:digest(p) for p in sorted(run.path.glob('history-*.pt'))},groups=30))
    run.log('complete',**budget.report())


def generate(config, histories_path, out, certificate, resume=False):
    check_certificate(certificate)
    source=Path(histories_path)
    complete=read_json(source/'complete.json')
    specs=read_json(source/'groups.json')
    if specs != group_specs(config['seed']):
        raise ValueError('history group configuration differs')
    for name,sha in complete['files'].items():
        if digest(source/name)!=sha:
            raise ValueError('history source changed')
    run=Run(out,config,'hybrid-data',resume,dict(history=digest(source/'complete.json'),native=native_fingerprint(),certificate=digest(certificate)))
    budget=Budget(**config['limits'])
    teacher=make_teacher(config,budget)
    physics=Physics.from_teacher(teacher)
    write_json(run.path/'dataset.json',dict(schema='hybrid-days-v1',physics=vars(physics),groups=specs,
        units=dict(power='AC kW positive discharge',heat='W',temperature='degC',damage='pu',time='s'),fine_columns=FINE_COLUMNS))
    try:
        for spec in specs:
            for stage,cp in enumerate(load_state(source/f"history-{spec['group']:02d}.pt")['stages']):
                for fi,family in enumerate(FAMILIES):
                    stem=f"branch-{spec['group']:02d}-{stage}-{family}"
                    final=run.path/(stem+'.pt')
                    if final.exists():
                        continue
                    progress=run.path/(stem+'.resume.pt')
                    state=load_state(progress) if progress.exists() else dict(checkpoint=cp,rows=[],control={},rng=None,preparation=[],prepared=False)
                    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],spec['group'],stage,fi,300]))
                    if state['rng'] is not None:
                        rng.bit_generator.state=state['rng']
                    teacher.restore(state['checkpoint']); budget.query()
                    try:
                        if not state['prepared']:
                            if 'prepare_target' not in state:
                                if family in ('random','shallow','deep'):
                                    depth=float(rng.uniform(*{'random':(.15,.4),'shallow':(.02,.15),'deep':(.4,.8)}[family]))
                                    low=float(rng.uniform(.1,.9-depth))
                                    state['control'].update(low=low,high=low+depth,target=low+depth)
                                    state['prepare_target']=low
                                else:
                                    state['prepare_target']=(teacher.battery.state.soc if cp['detector']['_direction']==0
                                        else (.75 if (spec['group']+stage)%5==4 else .25))
                            while abs(teacher.battery.state.soc-state['prepare_target'])>.002:
                                if len(state['preparation'])>=16:
                                    raise ValueError('preparation failed to reach its feasible SOC target in 16 intervals')
                                state['preparation'].append(interval(teacher,safe_power(teacher,state['prepare_target'],.8)))
                            state['prepared']=True
                        while len(state['rows'])<192:
                            old_control,old_rng=copy.deepcopy(state['control']),copy.deepcopy(rng.bit_generator.state)
                            p=branch_command(teacher,family,len(state['rows']),state['control'],rng,spec['group']+stage)
                            try:
                                row=interval(teacher,p)
                            except BaseException:
                                state['control']=old_control; rng.bit_generator.state=old_rng
                                raise
                            state['rows'].append(row)
                            if len(state['rows'])%config['data']['save_every_intervals']==0:
                                state.update(checkpoint=teacher.checkpoint(),rng=copy.deepcopy(rng.bit_generator.state))
                                save_state(progress,state)
                    finally:
                        state.update(checkpoint=teacher.checkpoint(),rng=copy.deepcopy(rng.bit_generator.state))
                        save_state(progress,state)
                    days=[day_record(state['rows'][j*96:(j+1)*96],physics,spec['group'],stage,family,j) for j in range(2)]
                    save_state(final,dict(split=spec['split'],days=days,rows=state['rows'],preparation=state['preparation'],source_checkpoint=cp))
                    run.log('branch_committed',group=spec['group'],stage=stage,family=family,
                        curtailed_intervals=sum(r['curtailed'] for r in state['rows']),
                        temperature_min=float(min(r['fine'][:,2].min() for r in state['rows'])),
                        temperature_max=float(max(r['fine'][:,2].max() for r in state['rows'])),**budget.report())
                    progress.unlink(missing_ok=True)
    except LimitReached as exc:
        run.log('paused',reason=str(exc),**budget.report()); return
    files={p.name:digest(p) for p in sorted(run.path.glob('branch-*.pt')) if '.resume.' not in p.name}
    if len(files)!=480:
        raise ValueError('incomplete branch set')
    write_json(run.path/'complete.json',dict(files=files,days=960,metadata=digest(run.path/'dataset.json')))
    run.log('complete',**budget.report())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['histories','generate'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--certificate',required=True)
    parser.add_argument('--histories')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    c=load_config(args.config)
    for key in ('max_seconds','max_steps'):
        value=getattr(args,key)
        if value is not None:
            if value<=0: parser.error('work limit must be positive')
            c['limits'][key]=value
    if args.command=='histories':
        histories(c,args.out,args.certificate,args.resume)
    else:
        if not args.histories: parser.error('--histories required for generate')
        generate(c,args.histories,args.out,args.certificate,args.resume)


if __name__=='__main__':
    main()
