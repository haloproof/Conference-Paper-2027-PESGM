"""idea-1016 2.4/2.5, explicit offline two-stage fitting (never on import)."""
from __future__ import annotations
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import Budget, LimitReached, Run, digest, read_json, seed_all, write_json
from .hybrid import HeatNetwork, HybridModel, InitialState, Physics
from .hybrid_data import load_config, load_state, save_state


def read_days(path):
    path=Path(path)
    complete=read_json(path/'complete.json')
    if complete['days']!=960 or len(complete['files'])!=480:
        raise ValueError('formal training requires the completed 960-day dataset')
    if digest(path/'dataset.json')!=complete['metadata']:
        raise ValueError('dataset metadata changed')
    metadata=read_json(path/'dataset.json')
    splits={'train':[],'validation':[]}
    assignment={g['group']:g['split'] for g in metadata['groups']}
    for name,sha in complete['files'].items():
        if digest(path/name)!=sha:
            raise ValueError('dataset hash mismatch')
        # Test files are hashed for integrity but never loaded for fitting/selection.
        group=int(name.split('-')[1])
        if assignment[group]=='test':
            continue
        branch=load_state(path/name)
        if branch['split']!=assignment[group]:
            raise ValueError('group split mismatch')
        for day in branch['days']:
            if day['group']!=group or len(day['power'])!=96:
                raise ValueError('invalid grouped day')
            splits[branch['split']].append(day)
    if [len(splits[k]) for k in ('train','validation')]!=[640,160]:
        raise ValueError('unexpected training/validation sizes')
    return splits,Physics(**metadata['physics'])


def local_arrays(days):
    x=np.concatenate([d['x'] for d in days])
    y=np.concatenate([d['heat'] for d in days])
    active=x[:,0]!=0
    if not active.any():
        raise ValueError('local heat fit requires nonzero activity')
    return torch.as_tensor(x[active],dtype=torch.float64),torch.as_tensor(y[active],dtype=torch.float64)


def normalizers(days):
    x,y=local_arrays(days)
    t=torch.as_tensor(np.concatenate([d['temperature'] for d in days]),dtype=torch.float64)
    xs=x.std(0,unbiased=False)
    ys=y.std(unbiased=False)
    ts=t.std(unbiased=False)
    return dict(input_center=x.mean(0),input_scale=torch.where(xs>0,xs,torch.ones_like(xs)),
                heat_center_w=y.mean(),heat_scale_w=torch.where(ys>0,ys,torch.ones_like(ys)),
                temperature_scale=float(ts) if ts>0 else 1.)


def initialize(stats,seed=1016):
    seed_all(seed)
    net=HeatNetwork(**{k:v for k,v in stats.items() if k!='temperature_scale'})
    for layer in net.modules():
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    return net


def local_loss(net,x,y):
    return ((net(x)-y)/net.heat_scale_w).square().mean()


def day_losses(model,days,r_t,epsilon_calendar,check=lambda:None):
    temperatures,calendars,soc_errors,cyclic_errors,total_errors=[],[],[],[],[]
    for day in days:
        check()
        power=torch.as_tensor(day['power'],dtype=torch.float64)
        prediction=model(power,InitialState.from_checkpoint(day['initial']))
        target_t=torch.as_tensor(day['temperature'],dtype=torch.float64)
        temperatures.append(((prediction.temperature_c[1:]-target_t)/r_t).square().mean())
        calendars.append(((prediction.calendar_damage-day['calendar'])/epsilon_calendar).square())
        soc_errors.append(float((prediction.soc[1:].detach()-torch.as_tensor(day['soc'])).abs().max()))
        cyclic_errors.append(abs(float(prediction.cyclic.damage.detach())-day['cyclic']))
        total_errors.append(abs(float(prediction.total_damage.detach())-day['calendar']-day['cyclic']))
    return torch.stack(temperatures).mean(),torch.stack(calendars).mean(),dict(
        max_soc_error=max(soc_errors),mean_cyclic_error=float(np.mean(cyclic_errors)),
        mean_total_error=float(np.mean(total_errors)))


def make_optimizer(net,settings,stage):
    return torch.optim.Adam(net.parameters(),lr=settings[stage]['lr'],betas=tuple(settings['betas']),
                            eps=settings['adam_eps'],weight_decay=settings['weight_decay'])


def validate(model,days,xy,stats,settings,stage,check=lambda:None):
    model.eval()
    x,y=xy
    total_q=0.
    with torch.no_grad():
        for start in range(0,len(x),256):
            check()
            batch=x[start:start+256]
            total_q+=float(local_loss(model.heat,batch,y[start:start+256]))*len(batch)
        q=total_q/len(x)
        if stage=='stage1':
            return dict(loss=q,q=q)
        t,c=0.,0.
        max_soc=0.; cyclic=0.; total=0.
        for day in days:
            check()
            lt,lc,metrics=day_losses(model,[day],stats['temperature_scale'],settings['epsilon_calendar'],check)
            t+=float(lt); c+=float(lc)
            max_soc=max(max_soc,metrics['max_soc_error'])
            cyclic+=metrics['mean_cyclic_error']; total+=metrics['mean_total_error']
        t/=len(days); c/=len(days)
        w=settings['weights']
        return dict(loss=w[0]*q+w[1]*t+w[2]*c,q=q,t=t,calendar=c,
                    max_soc_error=max_soc,mean_cyclic_error=cyclic/len(days),mean_total_error=total/len(days))


def update_selection(state,value,min_relative):
    if not math.isfinite(value):
        raise FloatingPointError('nonfinite validation loss')
    best=value<state['best']
    if best:
        state['best']=value
    reference=state['reference']
    if not math.isfinite(reference) or (reference>0 and value<reference and value<=reference*(1-min_relative)):
        state['reference']=value; state['bad_epochs']=0
    else:
        state['bad_epochs']+=1
    return best


def train(config,dataset,out,resume=False,stage_seconds=None):
    settings=config['training']
    for stage in ('stage1','stage2'):
        s=settings[stage]
        if any(s[k]<=0 for k in ('lr','batch_size','epochs','patience')):
            raise ValueError('positive training settings required')
    if settings['epsilon_calendar']<=0 or len(settings['weights'])!=3 or min(settings['weights'])<0:
        raise ValueError('invalid loss scales/weights')
    run=Run(out,config,'hybrid-train',resume,dict(dataset=digest(Path(dataset)/'complete.json')))
    if (run.path/'complete.json').exists():
        raise ValueError('training already complete; use a new run for changed settings')
    splits,physics=read_days(dataset)
    stats=normalizers(splits['train'])
    net=initialize(stats,config['seed'])
    torch.set_num_threads(settings['threads'])
    model=HybridModel(physics,net)
    train_xy=local_arrays(splits['train']); val_xy=local_arrays(splits['validation'])
    write_json(run.path/'data_usage.json',dict(training_days=640,validation_days=160,
        training_curtailed_intervals=sum(int(d['curtailed'].sum()) for d in splits['train']),
        validation_curtailed_intervals=sum(int(d['curtailed'].sum()) for d in splits['validation']),
        policy='retain all labels with command inputs; never replace by delivered power',
        normalization='training only',test_used_for_selection=False,precision='float64'))
    rng=np.random.default_rng(config['seed'])
    latest=run.path/'latest.pt'
    state=load_state(latest) if resume and latest.exists() else None
    if state is not None:
        net.load_state_dict(state['network']); rng.bit_generator.state=state['rng']; torch.set_rng_state(state['torch_rng'])
    for stage in ('stage1','stage2'):
        if state is not None and state['stage']=='stage2' and stage=='stage1':
            continue
        if state is None or state['stage']!=stage:
            if stage=='stage2':
                best_path=run.path/'stage1-best.pt'
                if not best_path.exists():
                    raise ValueError('stage 1 has no validated weights; resume it first')
                net.load_state_dict(load_state(best_path)['network'])
            state=dict(stage=stage,epoch=0,offset=0,order=None,best=float('inf'),reference=float('inf'),
                       bad_epochs=0,finished=False,train_sum=0.,train_count=0)
        optimizer=make_optimizer(net,settings,stage)
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        budget=Budget(stage_seconds or settings['max_stage_seconds'],1,1)
        max_unit_seconds=1.

        def check():
            budget.check()
            if budget.max_seconds-(time.monotonic()-budget.started) < max(2.,1.2*max_unit_seconds):
                raise LimitReached('remaining time reserved for checkpoint; resume this stage')

        def persist():
            state.update(network=copy.deepcopy(net.state_dict()),optimizer=optimizer.state_dict(),
                rng=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),normalizers=stats,
                physics=vars(physics),config=config)
            save_state(latest,state)

        try:
            s=settings[stage]
            while not state['finished'] and state['epoch']<s['epochs']:
                count=len(train_xy[0]) if stage=='stage1' else len(splits['train'])
                if state['order'] is None:
                    state['order']=rng.permutation(count)
                net.train()
                while state['offset']<count:
                    check(); started=time.monotonic()
                    ids=state['order'][state['offset']:state['offset']+s['batch_size']]
                    old_rng=copy.deepcopy(rng.bit_generator.state)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        if stage=='stage1':
                            loss=local_loss(net,train_xy[0][ids],train_xy[1][ids])
                        else:
                            local_ids=rng.integers(len(train_xy[0]),size=s['local_batch_size'])
                            q=local_loss(net,train_xy[0][local_ids],train_xy[1][local_ids])
                            t,c,_=day_losses(model,[splits['train'][int(i)] for i in ids],
                                stats['temperature_scale'],settings['epsilon_calendar'],check)
                            w=settings['weights']; loss=w[0]*q+w[1]*t+w[2]*c
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite training loss')
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                            raise FloatingPointError('nonfinite parameter gradient')
                        check()
                    except LimitReached:
                        rng.bit_generator.state=old_rng
                        optimizer.zero_grad(set_to_none=True)
                        raise
                    optimizer.step()
                    state['offset']+=len(ids)
                    state['train_sum']+=float(loss.detach())*len(ids); state['train_count']+=len(ids)
                    persist()
                    max_unit_seconds=max(max_unit_seconds,time.monotonic()-started)
                metrics=validate(model,splits['validation'],val_xy,stats,settings,stage,check)
                state['epoch']+=1
                if update_selection(state,metrics['loss'],settings['min_relative_improvement']):
                    save_state(run.path/(stage+'-best.pt'),dict(network=copy.deepcopy(net.state_dict()),
                        normalizers=stats,physics=vars(physics),stage=stage,epoch=state['epoch'],validation=metrics))
                run.log('epoch',stage=stage,epoch=state['epoch'],training_loss=state['train_sum']/state['train_count'],
                        validation=metrics,lr=s['lr'],seconds=time.monotonic()-budget.started)
                state.update(offset=0,order=None,train_sum=0.,train_count=0)
                if state['bad_epochs']>=s['patience'] or state['epoch']>=s['epochs']:
                    state['finished']=True
                    state['stop_reason']='early_stop' if state['bad_epochs']>=s['patience'] else 'epoch_limit'
                persist()
        except LimitReached as exc:
            persist()
            run.log('paused',stage=stage,reason=str(exc),validated_best_exists=(run.path/(stage+'-best.pt')).exists())
            return
        run.log('stage_complete',stage=stage,reason=state['stop_reason'],best=state['best'])
    write_json(run.path/'complete.json',dict(best='stage2-best.pt',sha256=digest(run.path/'stage2-best.pt'),
        scope='trained model; independent power-gradient and test-set evaluations not executed by this command'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--max-stage-seconds',type=float)
    args=p.parse_args()
    if args.max_stage_seconds is not None and args.max_stage_seconds<=0:
        p.error('positive time limit required')
    train(load_config(args.config),args.dataset,args.out,args.resume,args.max_stage_seconds)


if __name__=='__main__':
    main()
