"""Export public CSVs and recreate Figures 2-5; no simulations.

Git Bash, from the project root:
  source .venv/Scripts/activate
  timeout 60s python scripts/plot_results.py --max-seconds 55
  timeout 60s python scripts/plot_results.py --from-csv results --out-dir reproduced
  timeout 55s python scripts/plot_results.py --dispatch-only --max-seconds 50 --force
  timeout 60s python scripts/plot_results.py --day 100 --window-days 7 --force
  # Manual opt-in: frozen inference on 365 M1 schedules, then Figure 3 only.
  timeout 600s python scripts/plot_results.py --figure3-only --evaluate-m1 --max-seconds 540 --force

Default typical day: first-year price spread closest to its median (ties: earliest).
This choice is independent of method performance, prediction errors, and failures.
Figure 3 uses M1 days 1-365 as the common trajectory; Figure 5 uses 1 or 7 selected days.
Figures 2-4 follow the publication style; Figure 5 is one compact dual-axis panel.
Figure 5 delegates to plot_dispatch_overlay.py and never runs model inference.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from pypdf import PdfReader
from plot_training import (setup_publication_style, figure_font, figure_size,
                           figure_style, write_caption_style, STYLE_PATH,
                           CASE_STUDY_COLORS)

RUNS = {'M1': 'm0-da-1031', 'M2': 'b1-da-1031', 'M3': 'b2-da-1031',
        'M4': 'b5-v4-eol', 'M5': 'b3-v4-eol'}
COLORS = CASE_STUDY_COLORS
FILES = ['fig2_training.pdf','fig3_degradation_tracking.pdf',
         'fig4_lifecycle_comparison.pdf','fig5_dispatch_comparison.pdf']
CSV_FILES = ['training_results.csv', 'degradation_tracking_results.csv',
             'lifecycle_results.csv', 'dispatch_results.csv']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def saved_reports(paths):
    """Bound disk prefetch to 16 reports while preserving source-file order."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        source = iter(paths)
        pending = deque((p, pool.submit(p.read_bytes)) for p in islice(source, 16))
        while pending:
            path, future = pending.popleft()
            raw = future.result()
            next_path = next(source, None)
            if next_path is not None:
                pending.append((next_path, pool.submit(next_path.read_bytes)))
            yield path, raw


def finite(values, label):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f'Non-finite values in {label}')
    return a


def setup_style():
    return setup_publication_style()


def tidy(ax, grid=True):
    ax.tick_params(direction='out', length=3, width=0.7)
    if grid:
        ax.grid(axis='y', color='#D8D8D8', linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig, path, title, subject):
    number = FILES.index(Path(path).name) + 2
    dimensions = figure_size(number)
    fig.set_size_inches(*dimensions, forward=True)
    fig.savefig(path, format='pdf', metadata={'Title':title, 'Subject':subject,
        'Author':'Pengfei He', 'Creator':'results/plot_results.py (Matplotlib)'})
    plt.close(fig)
    pdf = PdfReader(path)
    rect = pdf.pages[0].mediabox
    if len(pdf.pages) != 1 or not math.isclose(float(rect.width)/float(rect.height),dimensions[0]/dimensions[1],abs_tol=1e-9):
        raise ValueError(f'Expected one page with configured aspect ratio: {path}')


def write_csv(path, fields, rows):
    """Round-trip binary64 values without publication rounding."""
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def tracking_initial_states(days):
    """Recover native day-start pending state without replaying the teacher.

    SOC endpoints determine direction/depth. At SOC saturation, active time
    is shorter than a quarter hour: invert the PREVIOUS day's saved residual
    ledger, then require the native 10 s grid and a lossless ledger round trip.
    No current-day target is used as a predictor input.
    """
    from dad.hybrid import InitialState, Pending
    direction = 0
    start_soc = previous_soc = days[0]['health_start']['native_soc']
    active_upper_h = 0.
    pending = Pending()
    previous_health = None
    for day in days:
        health = day['health_start']
        if previous_health is not None and health != previous_health:
            raise ValueError('M1 native health history is discontinuous')
        qc, qy, residual = health['ledger_components_pu']
        yield InitialState(health['native_soc'], health['temperature_c'], qc, qy, pending)
        for row in day['intervals']:
            soc = row['teacher_state'][0]
            delta = soc - previous_soc
            sign = int(delta > 0) - int(delta < 0)
            if sign:
                # Constant commands have monotone fine-step SOC; check saved extrema.
                lo, hi = row['fine_extrema'][0][0], row['fine_extrema'][1][0]
                if lo < min(previous_soc, soc)-1e-12 or hi > max(previous_soc, soc)+1e-12:
                    raise ValueError('Cannot recover pending depth from nonmonotone interval')
                if direction and sign != direction:
                    start_soc = previous_soc
                    active_upper_h = 0.
                active_upper_h += row['dt_h']
                direction = sign
            previous_soc = soc
        depth = abs(previous_soc - start_soc)
        previous_health = day['health_end']
        _, qy, residual = previous_health['ledger_components_pu']
        if depth == 0:
            if residual != 0: raise ValueError('Nonzero residual with empty half-cycle')
            pending = Pending()
            continue
        stress_coefficient = 100 * math.sqrt(residual * (2*qy+residual) * 2/depth)
        rate = (stress_coefficient/(4.0253*(depth-.6)**3+1.0923)-.0971)/.0630
        if rate <= 0: raise ValueError('Invalid recovered half-cycle rate')
        seconds = 3600*depth/rate
        rounded = round(seconds/10)*10
        if rounded <= 0 or rounded > active_upper_h*3600+1e-6 or abs(seconds-rounded)>1e-5:
            raise ValueError('Recovered active duration does not match native 10 s grid')
        hours = rounded/3600
        k = (.0630*depth/hours+.0971)*(4.0253*(depth-.6)**3+1.0923)
        stress = (k/100)**2*depth/2
        recovered = stress/(math.sqrt(qy*qy+stress)+qy)
        if not math.isclose(recovered,residual,rel_tol=0,abs_tol=1e-14):
            raise ValueError('Recovered pending state disagrees with saved residual')
        pending = Pending(direction,depth,hours)


def export_tracking(days, root, config, target, metadata, guard=lambda: None):
    # Explicit frozen-model inference only; no optimizer, training or teacher replay.
    sys.path.insert(0,str(root/'src'))
    import torch
    from dad.benchmark_b2_tables import AgingTables, blocks, evaluate
    from dad.benchmark_b3_v4_solver import load_model
    if len(days)!=365: raise ValueError('Figure 3 requires all 365 first-year M1 days')
    m2_first,_=load_json(root/'runs'/RUNS['M2']/'days/00001.json')
    m2_solver=m2_first['mpc_windows'][0]['plan']['solver']
    alpha=m2_solver['alpha_eur_per_kwh']
    nominal_kwh=m2_solver['energy_kwh']/m2_first['health_start']['native_soh']
    kappa=6480.
    model_path=root/'models/hybrid-fit-1020/stage2-best.pt'
    model=load_model(model_path)
    torch.set_num_threads(1)
    if not math.isclose(model.physics.energy_kwh,nominal_kwh,abs_tol=1e-12):
        raise ValueError('M1/M5 nominal energy mismatch')
    tables=AgingTables();groups=blocks(96)
    values={m:{'cal':[],'cyc':[]} for m in ['M2','M3','M5','actual']}
    states=tracking_initial_states(days)
    with torch.inference_mode():
        for index,(day,initial) in enumerate(zip(days,states)):
            guard()
            if day['index']!=index or not day['complete_day'] or len(day['mpc_windows'])!=1:
                raise ValueError('First-year M1 alignment changed')
            plan=day['mpc_windows'][0]['plan']
            p=finite(plan['power_ac_kw'],'M1 power');soc=finite(plan['predicted_soc'],'M1 predicted SOC')
            dt=finite([r['dt_h'] for r in day['intervals']],'M1 dt')
            if len(p)!=96 or not np.allclose(dt,0.25,rtol=0,atol=1e-12):
                raise ValueError('Expected 96 quarter-hour intervals')
            capacity=nominal_kwh*day['health_start']['native_soh']
            ch=np.maximum(-p,0);dis=np.maximum(p,0)
            reconstructed=initial.soc+np.cumsum((.91*ch-dis/.97)*dt/capacity)
            if not np.allclose(soc,reconstructed,rtol=0,atol=1e-7):
                raise ValueError('Common SOC convention mismatch')
            estimate=evaluate(tables.select(day['health_start']['native_soh'],capacity),soc,ch,dis,groups)['raw']
            hybrid=model(torch.tensor(p,dtype=torch.float64),initial,ambient_c=config['mpc']['ambient_c'])
            if not np.allclose(hybrid.soc.numpy()[1:],soc,rtol=0,atol=1e-7):
                raise ValueError('M5 shared SOC convention mismatch')
            values['M2']['cal'].append(0.0)
            values['M2']['cyc'].append(float(alpha*np.dot(np.abs(p),dt)/kappa))
            values['M3']['cal'].append(float(np.sum(estimate['cal'])))
            values['M3']['cyc'].append(float(np.sum(estimate['cycl'])))
            values['M5']['cal'].append(float(hybrid.calendar_damage))
            values['M5']['cyc'].append(float(hybrid.cyclic.damage))
            values['actual']['cal'].append(day['damage_components_pu'][0])
            values['actual']['cyc'].append(math.fsum(day['damage_components_pu'][1:]))
            if (index+1)%50==0: print(f'Figure 3: {index+1}/365 M1 schedules evaluated',flush=True)
    metadata.update({'comparison_basis':'M1 saved first-year plans and native day-start states',
        'nodes_per_curve':365,'days':[1,365],'quantity':'Daily total calendar + cyclic increments, not cumulative losses',
        'units':'capacity loss pu','yscale':'symlog with linthresh=1e-5 pu',
        'M2_calendar':'Omitted term represented by zero, not a physical zero-damage claim',
        'M2_cyclic':'alpha * planned AC throughput / kappa',
        'M3':'Unscaled PWL tables on M1 plans and M1 native day-start SOH',
        'M5':'Frozen historical hybrid-fit-1020 inference on M1 plans and native day-start states',
        'model_sha256':digest(model_path.read_bytes()),
        'pending_state':'Direction/depth from historical native SOC endpoints; active duration recovered from previous-day residual ledger and verified on 10 s grid. No current-day labels used.',
        'actual_cyclic':'Saved M1 settled cyclic increment plus signed residual valuation change',
        'interpretation':'actual denotes native SimSES execution, not experimental measurement; planned and delivered power may differ',
        'alpha_eur_per_kwh':alpha,'kappa_eur_per_pu':kappa,'nominal_energy_kwh':nominal_kwh,
        'aging_table_sha256':tables.hashes,
        'analytic_evaluator_sha256':digest((root/'src/dad/benchmark_b2_tables.py').read_bytes()),
        'annual_components_pu':{m:{k:math.fsum(v) for k,v in parts.items()} for m,parts in values.items()}})
    fields=['method','simulation_day','calendar_damage_pu','cyclic_damage_pu','total_damage_pu',
            'calendar_term_omitted','cyclic_definition','common_trajectory_run']
    definitions={'M2':'cost_equivalent_throughput','M3':'unscaled_pwl_prediction',
                 'M5':'frozen_hybrid_on_M1','actual':'settled_plus_signed_residual'}
    write_csv(target,fields,(
        dict(method=m,simulation_day=d+1,calendar_damage_pu=values[m]['cal'][d],
             cyclic_damage_pu=values[m]['cyc'][d],total_damage_pu=values[m]['cal'][d]+values[m]['cyc'][d],
             calendar_term_omitted=int(m=='M2'),cyclic_definition=definitions[m],common_trajectory_run=RUNS['M1'])
        for m in values for d in range(365)))


def plot_tracking_csv(csv_path, target):
    rows=read_csv(csv_path)
    methods=['M2','M3','M5','actual']
    if set(r['method'] for r in rows)!=set(methods) or len(rows)!=1460:
        raise ValueError('Figure 3 CSV requires M2/M3/M5/actual, 365 days each')
    if any(r['common_trajectory_run']!=RUNS['M1'] for r in rows):
        raise ValueError('Figure 3 requires the M1 common trajectory')
    fig,ax=plt.subplots()
    fig.subplots_adjust(left=.20,right=.95,bottom=.25,top=.95)
    styles={'M2':(':',CASE_STUDY_COLORS['M2']),
            'M3':('-.',CASE_STUDY_COLORS['M3']),
            'M5':('--',CASE_STUDY_COLORS['M5']),
            'actual':('-',CASE_STUDY_COLORS['M1'])}
    for method in methods:
        entries=sorted((r for r in rows if r['method']==method),key=lambda r:int(r['simulation_day']))
        if [int(r['simulation_day']) for r in entries]!=list(range(1,366)):
            raise ValueError('Figure 3 CSV has missing/repeated days')
        cal=finite([float(r['calendar_damage_pu']) for r in entries],method+' cal')
        cyc=finite([float(r['cyclic_damage_pu']) for r in entries],method+' cyc')
        total=finite([float(r['total_damage_pu']) for r in entries],method+' total')
        if not np.allclose(total,cal+cyc,rtol=1e-14,atol=0):
            raise ValueError('Total damage differs from calendar + cyclic')
        if any(int(r['calendar_term_omitted'])!=int(method=='M2') for r in entries) or (method=='M2' and np.any(cal!=0)):
            raise ValueError('Omitted calendar convention changed')
        ls,color=styles[method]
        ax.plot(np.arange(1,366),total,ls=ls,color=color,alpha=1.0,lw=1.35 if method!='actual' else 1.8,
                label=method,zorder=4 if method=='M5' else 3)
    ax.set_yscale('symlog',linthresh=1e-5,linscale=1)
    ax.set_ylabel('Daily total\n'+r'degradation $\hat{D}$ (pu)')
    ax.set_xlabel('Simulation day')
    ax.set_xlim(1,365);ax.set_xticks([1,50,100,150,200,250,300,365])
    ax.legend(loc='upper right',ncol=2,frameon=False)
    tidy(ax,False)
    save(fig,target,'Figure 3. Daily total degradation for the first 365 M1 schedules',
         'M2/M3/M5 on common M1 planned power; actual is saved native SimSES execution. Total = calendar + signed cyclic damage.')


def export_lifecycle(series, target):
    fields = ['method','simulation_day','native_soh_pu','daily_revenue_eur',
              'cumulative_revenue_eur','daily_teacher_utility_eur','cumulative_teacher_utility_eur']
    rows=[]
    for method,data in series.items():
        revenue=np.cumsum(data['revenue']);utility=np.cumsum(data['utility'])
        for day in range(len(data['soh'])):
            rows.append(dict(method=method,simulation_day=day,native_soh_pu=data['soh'][day],
                daily_revenue_eur=data['revenue'][day],cumulative_revenue_eur=float(revenue[day]),
                daily_teacher_utility_eur=data['utility'][day],cumulative_teacher_utility_eur=float(utility[day])))
    write_csv(target,fields,rows)


def plot_lifecycle_csv(csv_path, target):
    rows=read_csv(csv_path)
    if set(r['method'] for r in rows) != set(RUNS):
        raise ValueError('Figure 4 CSV must contain methods M1-M5')
    fig, soh_ax=plt.subplots(figsize=figure_size(4))
    utility_ax=soh_ax.twinx()
    fig.subplots_adjust(left=.15,right=.82,bottom=.25,top=.70)
    utility_ax.spines['right'].set_visible(True)
    handles=[]
    for method in RUNS:
        data=sorted((r for r in rows if r['method']==method),key=lambda r:int(r['simulation_day']))
        x=np.asarray([int(r['simulation_day']) for r in data])
        if x.tolist()!=list(range(len(data))) or len(data)<2:
            raise ValueError('Figure 4 CSV must include day zero and consecutive daily endpoints')
        soh=finite([float(r['native_soh_pu']) for r in data],method+' SOH')
        utility=finite([float(r['cumulative_teacher_utility_eur']) for r in data],method+' utility')
        daily=finite([float(r['daily_teacher_utility_eur']) for r in data],method+' daily utility')
        if not np.allclose(utility,np.cumsum(daily),rtol=1e-12,atol=1e-8) or daily[0]!=0:
            raise ValueError('Figure 4 cumulative utility does not match daily values')
        color=CASE_STUDY_COLORS[method]
        width=2.0
        line,=soh_ax.plot(x,100*soh,color=color,alpha=1.0,lw=width,label=method,
                         marker='o',markevery=[-1],ms=4)
        utility_ax.plot(x,utility,color=color,alpha=1.0,lw=width,ls=(0,(5,3)),
                        marker='o',markevery=[-1],ms=4,mfc='white')
        handles.append(line)
    last_day=max(int(r['simulation_day']) for r in rows)
    horizon=500*math.ceil(last_day/500)
    soh_ax.set(ylabel='SoH (%)',ylim=(79,101),xlabel='Simulation day',xlim=(0,horizon))
    soh_ax.set_xticks(np.arange(0,horizon+1,500))
    soh_ax.set_yticks([80,85,90,95,100])
    utility_ax.set_ylabel('Cumulative utility\n(EUR)')
    utility_ax.set_yticks(np.arange(0,500*math.ceil(max(float(r['cumulative_teacher_utility_eur']) for r in rows)/500)+1,500))
    tidy(soh_ax,False)
    tidy(utility_ax,False)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.48,1.025),
               ncol=5,frameon=False,handlelength=1.3,columnspacing=.9,
               handletextpad=.4)
    metric_handles=[Line2D([],[],color='#555555',lw=2,label='SoH (left)'),
                    Line2D([],[],color='#555555',lw=2,ls=(0,(5,3)),
                           label='Utility (right)')]
    fig.legend(handles=metric_handles,loc='upper center',bbox_to_anchor=(.48,.865),
               ncol=2,frameon=False,handlelength=2,columnspacing=1.4)
    save(fig,target,'Lifetime trajectories of SoH and Cumulative utility',
         'All complete days. Native SOH; cumulative actual arbitrage minus kappa times full teacher damage, including signed residual valuation. Each curve ends at retirement.')


def export_dispatch(selected, day_number, window, target):
    flat={m:[r for d in ds for r in d['intervals']] for m,ds in selected.items()}
    dt=finite([r['dt_h'] for r in flat['M5']],'dt')
    edges=np.r_[0,np.cumsum(dt)]
    fields=['simulation_day','interval_in_day','time_start_h','time_end_h','source_start_utc',
            'price_eur_per_mwh']+[m+'_command_ac_kw' for m in RUNS]
    rows=[]
    for i,reference in enumerate(flat['M5']):
        row=dict(simulation_day=day_number+i//96,interval_in_day=i%96,
                 time_start_h=float(edges[i]),time_end_h=float(edges[i+1]),
                 source_start_utc=reference['source_start_utc'],
                 price_eur_per_mwh=reference['price_eur_kwh']*1000)
        row.update({m+'_command_ac_kw':flat[m][i]['command_ac_kw'] for m in RUNS})
        rows.append(row)
    if len(rows)!=window*96:
        raise ValueError('Figure 5 export window mismatch')
    write_csv(target,fields,rows)


def plot_dispatch_csv(csv_path, target):
    """Render the saved price and M1--M5 powers in one dual-axis panel."""
    from plot_dispatch_overlay import plot_dispatch_overlay
    return plot_dispatch_overlay(csv_path, target)


def collect_saved_results(args, root, metadata, guard):
    # M4/M5 dispatch shares the original trained hybrid, not HNN-compare.
    complete,csha=load_json(root/'models/hybrid-fit-1020/complete.json')
    metadata['dispatch_model_source']={'path':'runs/hybrid-fit-1020/complete.json',
        'complete_sha256':csha,'frozen_model_sha256':complete['sha256'],
        'scope':'M4 and M5 dispatch; distinct from the HNN training comparison'}
    price_path=root/'data/dispatch-prices/prices.csv'
    raw_price=price_path.read_bytes()
    with price_path.open(newline='',encoding='utf-8-sig') as f: prices=list(csv.DictReader(f))
    first_year=finite([float(r['price_eur_kwh']) for r in prices[:365*96]],'first-year prices').reshape(365,96)
    spreads=np.ptp(first_year,axis=1)
    chosen=int(np.argmin(np.abs(spreads-np.median(spreads))))+1 if args.day is None else args.day
    if chosen<1: raise ValueError('--day must be at least 1')
    metadata['typical_window']={'day':chosen,'window_days':args.window_days,
        'selection':'closest to median daily price range among first 365 days; earliest tie' if args.day is None else 'explicit --day',
        'selection_independent_of_prediction_error':args.day is None,
        'price_csv_sha256':digest(raw_price),'median_first_year_spread_eur_kwh':float(np.median(spreads)),
        'selected_spread_eur_kwh':float(spreads[(chosen-1)%365]),
        'source_start_local':prices[((chosen-1)%365)*96]['start_local']}
    print(f'Typical window starts on day {chosen}: {metadata["typical_window"]["source_start_local"]}',flush=True)
    series={};selected={};first_year_days=[];m5_config=None
    metadata['lifecycle_utility']='Cumulative sum of daily actual revenue minus 6480 times full teacher damage, including signed residual valuation'
    for method,run_id in RUNS.items():
        guard();folder=root/'runs'/run_id
        summary,ssha=load_json(folder/'summary.json');manifest,msha=load_json(folder/'manifest.json')
        if method=='M5': m5_config=manifest['signature']['config']
        count=summary['completed_days']
        if chosen+args.window_days-1>count: raise ValueError(f'Selected window exceeds {method} completed days')
        paths=sorted((folder/'days').glob('*.json'))
        if len(paths)!=count: raise ValueError(f'{method}: daily report count differs from summary')
        soh=[];revenue=[0.0];utility=[0.0];daily_digest=hashlib.sha256();picked=[];failed=0;last_soh=None
        for day_num,(path,raw) in enumerate(saved_reports(paths),1):
            guard();d=json.loads(raw)
            daily_digest.update(path.name.encode());daily_digest.update(hashlib.sha256(raw).digest())
            if d['index']!=day_num-1 or not d.get('complete_day') or len(d['intervals'])!=96:
                raise ValueError(f'{method}: non-contiguous/incomplete day {path}')
            if last_soh is not None and not math.isclose(d['health_start']['native_soh'],last_soh,abs_tol=1e-10):
                raise ValueError(f'{method}: health discontinuity at day {day_num}')
            if not soh: soh.append(d['health_start']['native_soh'])
            last_soh=d['health_end']['native_soh'];soh.append(last_soh)
            revenue.append(d['revenue_eur']);failed+=int(d['failed'])
            daily_utility=d['revenue_eur']-6480*math.fsum(d['damage_components_pu'])
            if not math.isclose(daily_utility,d['teacher_utility_eur'],rel_tol=1e-9,abs_tol=1e-8):
                raise ValueError('Teacher utility ledger mismatch')
            utility.append(daily_utility)
            if chosen<=day_num<chosen+args.window_days:
                picked.append(d)
            if method=='M1' and day_num<=365: first_year_days.append(d)
            if method=='M5' and (chosen==day_num or day_num<=365):
                model_sha=d['mpc_windows'][0]['plan']['solver']['model_sha256']
                if model_sha!=complete['sha256']: raise ValueError('M5 prediction model differs from training checkpoint')
        finite(soh,method+' SOH');finite(revenue,method+' revenue')
        if not math.isclose(sum(revenue),summary['revenue_eur'],rel_tol=1e-10,abs_tol=1e-7):
            raise ValueError(f'{method}: daily revenue does not reproduce summary')
        if not math.isclose(soh[-1],summary['final_soh'],abs_tol=1e-10):
            raise ValueError(f'{method}: final SOH does not match summary')
        if not math.isclose(math.fsum(utility),summary['teacher_utility_eur'],rel_tol=1e-9,abs_tol=1e-7):
            raise ValueError('Cumulative teacher utility does not reproduce summary')
        if failed!=summary['failed_days']: raise ValueError('Failed-day count mismatch')
        series[method]={'soh':soh,'revenue':revenue,'utility':utility};selected[method]=picked
        metadata['runs'][method]={'run_id':run_id,'days':count,'manifest_sha256':msha,
            'summary_sha256':ssha,'ordered_daily_reports_sha256':daily_digest.hexdigest(),
            'seed':manifest.get('signature',{}).get('config',{}).get('seed'),
            'versions':manifest.get('signature',{}).get('versions'),
            'final_soh':soh[-1],'actual_arbitrage_eur':sum(revenue),'cumulative_teacher_utility_eur':math.fsum(utility),'failed_days_included':failed,
            'retirement_reason':summary.get('reason'),
            'selected_daily_files':[f'{i:05}.json' for i in range(chosen,chosen+args.window_days)]}
        print(f'{method}: {count} days, final SOH {soh[-1]:.8f}, actual arbitrage EUR {sum(revenue):.6f}',flush=True)
    ref=[(r['source_start_utc'],r['price_eur_kwh'],r['dt_h']) for d in selected['M5'] for r in d['intervals']]
    for method,days in selected.items():
        now=[(r['source_start_utc'],r['price_eur_kwh'],r['dt_h']) for d in days for r in d['intervals']]
        if now!=ref: raise ValueError(f'{method}: selected price/time grids differ')
    return series,selected,first_year_days,m5_config,chosen


def main():
    from plot_training import export_training, plot_training_csv
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out-dir',type=Path,default=None)
    parser.add_argument('--from-csv',type=Path,help='Replot the four public CSVs without any original run/data folders')
    parser.add_argument('--day',type=int,default=None,help='1-based simulation day; default: median first-year price spread')
    parser.add_argument('--window-days',type=int,choices=[1,7],default=1,help='Figure 5 window; Figure 3 always uses days 1-365')
    parser.add_argument('--max-seconds',type=float,default=55)
    parser.add_argument('--figure3-only',action='store_true',help='Update only Figure 3 and its CSV/provenance')
    parser.add_argument('--dispatch-only',action='store_true',help='Update only Figures 4-5, their CSVs and provenance from saved reports; no model evaluation')
    parser.add_argument('--evaluate-m1',action='store_true',help='Explicitly evaluate the frozen M5 model on all 365 saved M1 schedules')
    parser.add_argument('--force',action='store_true',help='Explicitly allow replacing existing figure/data outputs')
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:
        parser.error('--max-seconds must be finite and positive')
    if args.from_csv and (args.day is not None or args.window_days!=1):
        parser.error('--from-csv uses the saved CSV window; omit --day/--window-days')
    if args.dispatch_only and (args.figure3_only or args.evaluate_m1):
        parser.error('--dispatch-only cannot evaluate or update Figure 3')
    if not args.dispatch_only:
        write_caption_style()
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>args.max_seconds:
            raise TimeoutError(f'Figure build exceeded {args.max_seconds:g} s; final outputs were not replaced')
    root=args.root.resolve();out=(args.out_dir or root/'results').resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.evaluate_m1 and (not args.figure3_only or args.from_csv):
        parser.error('--evaluate-m1 requires --figure3-only and cannot use --from-csv')
    if args.dispatch_only:
        names=[FILES[2],FILES[3],CSV_FILES[2],CSV_FILES[3],'figure_metadata.json']
        for name in names:
            if (out/name).exists() and not args.force:
                raise FileExistsError(f'{out/name} exists; use --force for intentional replacement')
        metadata=load_json(out/'figure_metadata.json')[0] if (out/'figure_metadata.json').exists() else {}
        metadata['method_mapping']=dict(RUNS)
        metadata['input_mode']='retained_figures_2_3_and_saved_runs_to_figures_4_5'
        metadata['data_scope']='Figures 4-5: read-only aggregation of saved dispatch reports. Figures 2-3 retained unchanged. No model, teacher, training or optimizer execution in this update.'
        metadata['runs']={}
        setup_style()
        with tempfile.TemporaryDirectory(prefix='.dispatch-figure-build-',dir=out) as temp:
            dest=Path(temp)
            if args.from_csv:
                metadata['input_mode']='retained_figures_2_3_and_public_csv_to_figures_4_5'
                metadata['data_scope']='Figures 4-5 plotted from public CSVs. Figures 2-3 retained unchanged; no models or private runs accessed.'
                source=args.from_csv.resolve()
                for name in CSV_FILES[2:]:
                    (dest/name).write_bytes((source/name).read_bytes())
            else:
                series,selected,_,_,chosen=collect_saved_results(args,root,metadata,guard)
                guard();export_lifecycle(series,dest/CSV_FILES[2])
                guard();export_dispatch(selected,chosen,args.window_days,dest/CSV_FILES[3])
            guard();plot_lifecycle_csv(dest/CSV_FILES[2],dest/FILES[2])
            guard();plot_dispatch_csv(dest/CSV_FILES[3],dest/FILES[3])
            guard()
            metadata.setdefault('figure_inches', {})['fig5']=[figure_style()['canvas_width_inches'], figure_style()['canvas_width_inches']*.4]
            metadata['dispatch_figures_updated_utc']=datetime.now(timezone.utc).isoformat()
            metadata['dispatch_figures_script_sha256']=digest(Path(__file__).read_bytes())
            metadata['dispatch_figures_computation']=('Public CSV plotting only; no model, teacher, training or optimization execution' if args.from_csv else 'Saved daily reports only; no model, teacher, training or optimization execution')
            metadata['dispatch_figures_elapsed_seconds']=time.monotonic()-started
            metadata['dispatch_figures_command']=sys.argv
            for name in names[:-1]:
                metadata.setdefault('outputs',{})[name]={'sha256':digest((dest/name).read_bytes()),'bytes':(dest/name).stat().st_size}
                if name in CSV_FILES[2:]:
                    metadata.setdefault('csv_input_sha256',{})[name]=digest((dest/name).read_bytes())
            (dest/names[-1]).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            for name in names: os.replace(dest/name,out/name)
        print(f'Updated Figures 4-5 only in {out}; elapsed {time.monotonic()-started:.2f} s',flush=True)
        return
    if args.figure3_only:
        names=[FILES[1],CSV_FILES[1],'figure_metadata.json']
        for name in names:
            if (out/name).exists() and not args.force:
                raise FileExistsError(f'{out/name} exists; use --force for intentional replacement')
        setup_style()
        metadata=load_json(out/'figure_metadata.json')[0] if (out/'figure_metadata.json').exists() else {}
        tracking={}
        with tempfile.TemporaryDirectory(prefix='.figure3-build-',dir=out) as temp:
            dest=Path(temp)
            if args.evaluate_m1:
                source=root/'runs'/RUNS['M1']
                manifest,msha=load_json(source/'manifest.json')
                days=[];source_hash=hashlib.sha256()
                for n in range(1,366):
                    guard();path=source/'days'/f'{n:05}.json';raw=path.read_bytes()
                    source_hash.update(path.name.encode());source_hash.update(hashlib.sha256(raw).digest())
                    days.append(json.loads(raw))
                export_tracking(days,root,manifest['signature']['config'],dest/CSV_FILES[1],tracking,guard)
                tracking.update({'source_run':RUNS['M1'],'manifest_sha256':msha,
                    'ordered_daily_reports_sha256':source_hash.hexdigest()})
            else:
                source=(args.from_csv or root/'results').resolve()
                (dest/CSV_FILES[1]).write_bytes((source/CSV_FILES[1]).read_bytes())
                tracking=metadata.get('first_year_tracking',{})
            guard();plot_tracking_csv(dest/CSV_FILES[1],dest/FILES[1]);guard()
            metadata['first_year_tracking']=tracking
            metadata['figure3_updated_utc']=datetime.now(timezone.utc).isoformat()
            metadata['figure3_script_sha256']=digest(Path(__file__).read_bytes())
            metadata['figure3_computation']='Frozen M5 forward evaluation on saved M1 plans; no training, optimization or teacher replay' if args.evaluate_m1 else 'Public CSV plotting only'
            metadata['figure3_versions']={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__}
            if args.evaluate_m1:
                import importlib.metadata
                metadata['figure3_versions']['torch']=importlib.metadata.version('torch')
                metadata['figure3_versions']['simses']=importlib.metadata.version('simses')
            metadata['figure3_elapsed_seconds']=time.monotonic()-started
            metadata.setdefault('csv_units',{})[CSV_FILES[1]]='Daily calendar, signed cyclic, and total loss in pu; common M1 trajectory'
            for name in names[:2]:
                metadata.setdefault('outputs',{})[name]={'sha256':digest((dest/name).read_bytes()),'bytes':(dest/name).stat().st_size}
            (dest/names[2]).write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            for name in names: os.replace(dest/name,out/name)
        print(f'Updated Figure 3 PDF/CSV in {out}; elapsed {time.monotonic()-started:.2f} s',flush=True)
        return
    output_files=FILES+CSV_FILES
    for name in output_files+['figure_metadata.json']:
        if (out/name).exists() and not args.force:
            raise FileExistsError(f'{out/name} exists; use --force for an intentional replacement')
    font=setup_style()
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':digest(Path(__file__).read_bytes()),
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'font':{'family':font['family']},'figure_inches':{f'fig{i}':(list(figure_size(i)) if i != 5 else [figure_size(i)[0], figure_size(i)[0]*.4]) for i in range(2,6)},
        'typography':figure_style(),'style_config_sha256':digest(STYLE_PATH.read_bytes()),
        'method_mapping':RUNS,'training':{},'runs':{},'first_year_tracking':{},
        'data_scope':'CSV-derived figures. Source mode reads saved runs plus analytic M2/M3 tables; CSV mode requires only published CSV files. No training, neural inference, teacher calls, or dispatch optimization.',
        'raw_data_policy':'All complete days retained, including execution-check failures; no smoothing, clipping, resampling or post-retirement extrapolation.',
        'csv_mapping':dict(zip(FILES,CSV_FILES)),
        'csv_units':{'degradation_tracking_results.csv':'Daily calendar/cyclic loss in pu; M2 calendar omitted; signed teacher cyclic ledger retained',
                     'lifecycle_results.csv':'SOH in pu, revenue/utility in EUR; includes initial day zero; utility = revenue - 6480*full teacher damage',
                     'dispatch_results.csv':'Time in h, price in EUR/MWh, commanded AC power in kW, positive = discharge; time_start/end define stair edges'}}
    with tempfile.TemporaryDirectory(prefix='.figure-build-',dir=out) as temp:
        dest=Path(temp)
        if args.from_csv:
            source=args.from_csv.resolve()
            for name in CSV_FILES:
                guard();raw=(source/name).read_bytes()
                (dest/name).write_bytes(raw)
            metadata['input_mode']='public_csv_only'
            metadata['csv_input_sha256']={n:digest((dest/n).read_bytes()) for n in CSV_FILES}
        else:
            series,selected,first_year_days,m5_config,chosen=collect_saved_results(args,root,metadata,guard)
            guard();export_training(root,dest,metadata['training'])
            guard();(dest/CSV_FILES[1]).write_bytes((root/'results'/CSV_FILES[1]).read_bytes())
            previous=load_json(root/'results/figure_metadata.json')[0]
            metadata['first_year_tracking']=previous.get('first_year_tracking',{})
            guard();export_lifecycle(series,dest/CSV_FILES[2])
            guard();export_dispatch(selected,chosen,args.window_days,dest/CSV_FILES[3])
            metadata['input_mode']='saved_runs_to_public_csv'
        guard();plot_training_csv(dest/CSV_FILES[0],dest/FILES[0],metadata['training'])
        setup_style()  # Figure 2 may set its own defaults.
        guard();plot_tracking_csv(dest/CSV_FILES[1],dest/FILES[1])
        guard();plot_lifecycle_csv(dest/CSV_FILES[2],dest/FILES[2])
        guard();plot_dispatch_csv(dest/CSV_FILES[3],dest/FILES[3])
        guard()
        metadata['elapsed_seconds']=time.monotonic()-started
        metadata['outputs']={n:{'sha256':digest((dest/n).read_bytes()),'bytes':(dest/n).stat().st_size} for n in output_files}
        (dest/'figure_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for name in output_files+['figure_metadata.json']: os.replace(dest/name,out/name)
    print(f'Created four public CSVs and four vector PDFs (Figures 2-4 4:1.6; Figure 5 4:3) in {out}; elapsed {metadata["elapsed_seconds"]:.2f} s',flush=True)


if __name__=='__main__':
    main()
