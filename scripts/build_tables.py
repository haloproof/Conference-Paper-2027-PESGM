"""Export six publication tables from existing daily reports; never run models.

Uses Python standard-library modules only.
Git Bash from the project root:
  python scripts/build_tables.py --max-seconds 55
  python scripts/build_tables.py --max-seconds 55 --force
CSV only. NA means unavailable estimation (M1) or payback not reached.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

METHODS = {
    'M1': ('m0-da-1031', []),
    'M2': ('b1-da-1031', []),
    'M3': ('b2-da-1031', []),
    # Proposed methods use authoritative saved runs rather than older Markdown summaries.
    'M4': ('b5-v4-eol', []),
    'M5': ('b3-v4-eol', []),
}
KAPPA = 6480.0
ANNUAL = ['simulation_year', 'charge_energy_kwh', 'discharge_energy_kwh',
    'arbitrage_revenue_eur', 'operating_days', 'calendar_aging_pp',
    'settled_cyclic_aging_pp', 'end_soh_pct', 'teacher_utility_eur',
    'predicted_total_damage_pp', 'teacher_total_damage_pp', 'wape_pct']
COMPARISON = ['method', 'arbitrage_revenue_eur', 'teacher_utility_eur',
              'operating_days', 'payback_days', 'mean_decision_time_s']
NOTES = '''PUBLIC SUPPLEMENTARY TABLES: DEFINITIONS AND REPRODUCTION

M1: arbitrage without degradation cost.
M2: throughput-proportional degradation cost.
M3: piecewise-linear calendar and cyclic aging costs.
M4: proposed health-scaled throughput-constrained dispatch with external hybrid-model assessment (b5-v4).
M4 evaluates ten pure-arbitrage MILP candidates: one uncapped pure plan, then
Q <= alpha Q_pure for alpha = 0.1, ..., 0.9; all caps share the pure-plan reference.
M5: proposed health-scaled hybrid-estimator/subspace projected-gradient dispatch (b3-v4).

Each M1-results through M5-results CSV contains one row per simulation year.
Year y contains days 365(y-1)+1 through min(365y, retirement day). The last year
is not annualized. operating_days gives the actual number of complete days.
All complete days, including recorded execution-check failures, are retained.
The historical German/Luxembourg price year (Sep 2025-Aug 2026) is repeated.
These are deterministic perfect-foresight simulation years, not future prices.

Column definitions (units are also encoded in the CSV headers):
charge_energy_kwh / discharge_energy_kwh: teacher-delivered AC energy, positive magnitudes.
arbitrage_revenue_eur: actual sale revenue minus charging expenditure, before investment or depreciation.
calendar_aging_pp: annual native calendar capacity loss.
settled_cyclic_aging_pp: annual native settled cyclic capacity loss, excluding pending half-cycle valuation.
end_soh_pct: native capacity SOH at the last completed day of the year.
teacher_utility_eur: actual arbitrage minus 6480 EUR/pu times teacher total damage.
predicted_total_damage_pp: annual estimated damage on the selected planned schedules; see method definitions below.
teacher_total_damage_pp: annual calendar + settled cyclic loss + signed change of pending half-cycle valuation.
pp means percentage points of nominal capacity: 1 pp = 0.01 pu. SOH (%) = 100 SOH (pu).
Pending half-cycle valuation is included in total damage and cost, but not in native SOH.
Negative valuation changes are retained without clipping.

M1 has no degradation estimator: predicted damage and annual WAPE are NA (not zero).
M2 predicted damage is COST-EQUIVALENT damage: throughput cost / 6480, not a separate physical total-damage model.
M3 uses the UNSCALED piecewise-linear damage estimate; its cost is 6480 times that estimate.
M3's scaled optimization penalty is not used as a damage estimate or in the error columns.
M4 and M5 use the frozen hybrid model's total damage (calendar + net cyclic, including residual valuation).
Thus the error columns compare method-specific cost estimates with the same teacher total-cost definition.
They also include differences between planned and actually delivered power.

Let C_d = 6480 D_teacher,d and C_hat,d = 6480 D_hat,d (EUR).
Annual WAPE (%) = 100 sum_d |C_hat,d - C_d| / sum_d C_d within that year.
Lifecycle relative error (%) = 100 (sum_d C_hat,d - sum_d C_d) / sum_d C_d.
Every recorded daily teacher cost is strictly positive; no day is excluded or denominator floored.
Relative error is signed: negative means underestimation, positive means overestimation.
It compares cumulative costs, rather than averaging daily percentage errors.

comparison-results has five methods and five lifecycle metrics:
arbitrage_revenue_eur: cumulative actual AC arbitrage to retirement, before installed investment.
teacher_utility_eur: cumulative actual arbitrage minus 6480 EUR/pu times teacher total damage, including signed residual valuation.
operating_days: complete days to first end-of-day native SOH <= 80%; no 2920-day truncation is applied to the completed retirement runs.
payback_days: elapsed simulation days when actual cumulative arbitrage first recovers the installed investment of 2880 EUR; interpolated within the crossing interval. NA means payback was not reached before retirement.
mean_decision_time_s: arithmetic mean of recorded daily decision elapsed time (seconds/day).
This time covers each method's recorded solver/search and associated model evaluation;
it excludes teacher execution, program startup, and time between continuation stages.
It is not COPT's internal CPU time or whole-program wall time.

CSV uses UTF-8, one header row, decimal points and no thousands separators.
NA marks unavailable annual estimation for M1 and payback not reached for M1-M3. M2-M5 have numeric WAPE in every annual row.
The wape_pct column contains annual WAPE expressed as a percentage (not a fraction).
CSV values are rounded to ten decimal places.
The separate generation metadata records source hashes and run provenance for local verification.
The five internal Markdown reports are not included in these public tables.
'''


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw=path.read_bytes()
    return json.loads(raw),sha(raw)


def saved_reports(paths):
    with ThreadPoolExecutor(max_workers=8) as pool:
        source=iter(paths)
        pending=deque((p,pool.submit(p.read_bytes)) for p in islice(source,16))
        while pending:
            p,f=pending.popleft();raw=f.result();next_p=next(source,None)
            if next_p is not None: pending.append((next_p,pool.submit(next_p.read_bytes)))
            yield p,raw


def close(a,b,label,atol=1e-7):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-9,abs_tol=atol):
        raise ValueError(f'{label}: {a} != {b}')


def aggregate(rows,year):
    total=lambda k:math.fsum(r[k] for r in rows)
    estimated=None if rows[0]['pred'] is None else total('pred')
    errors=None if estimated is None else [abs(r['pred']-r['cost']) for r in rows]
    return dict(zip(ANNUAL,[year,total('charge'),total('discharge'),total('revenue'),len(rows),
        100*total('calendar'),100*total('cyclic'),100*rows[-1]['soh'],total('utility'),
        None if estimated is None else 100*estimated/KAPPA,100*total('damage'),
        None if errors is None else 100*math.fsum(errors)/total('cost')]))


def markdown_tables(text):
    groups=[];current=[]
    for line in text.splitlines()+['']:
        if line.strip().startswith('|'):
            cells=[s.strip() for s in line.strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',s) for s in cells): current.append(cells)
        elif current:
            groups.append(current);current=[]
    return groups


def check_report(method,text,annual):
    tables=markdown_tables(text)
    checks=0
    def compare(value,printed,factor=1):
        nonlocal checks
        printed=printed.replace(',','').replace('**','').replace('%','').strip()
        number=float(printed)
        decimals=len(printed.split('.')[1]) if '.' in printed else 0
        if abs(value/factor-number)>0.5001*10**(-decimals)+1e-8:
            raise ValueError(f'{method}: annual value differs from report: {value/factor} vs {printed}')
        checks+=1
    def year_rows(table):
        return [r for r in table[1:] if re.search(r'[1-8]',r[0]) and '累计' not in r[0]]
    energy=next(t for t in tables if any('充电' in c for c in t[0]) and '指标' not in t[0][0])
    yearly=year_rows(energy)
    if len(yearly)!=len(annual): raise ValueError(f'{method}: year count differs from report')
    for a,r in zip(annual,yearly):
        if method=='M1':
            compare(a['operating_days'],r[1]);q=r[2].split('/');z=r[3].split('/')
            compare(a['charge_energy_kwh'],q[0]);compare(a['discharge_energy_kwh'],q[1])
            compare(a['calendar_aging_pp'],z[0]);compare(a['settled_cyclic_aging_pp'],z[1]);compare(a['arbitrage_revenue_eur'],r[6])
        elif method=='M2':
            compare(a['operating_days'],r[1]);compare(a['charge_energy_kwh'],r[2],1000)
            compare(a['discharge_energy_kwh'],r[3],1000);compare(a['arbitrage_revenue_eur'],r[5]);compare(a['end_soh_pct'],r[6])
        elif method=='M3':
            for key,col in [('charge_energy_kwh',1),('discharge_energy_kwh',2),('calendar_aging_pp',3),
                            ('settled_cyclic_aging_pp',4),('end_soh_pct',5),('arbitrage_revenue_eur',6)]: compare(a[key],r[col])
        else:
            compare(a['charge_energy_kwh'],r[1]);compare(a['discharge_energy_kwh'],r[2]);compare(a['arbitrage_revenue_eur'],r[5])
    if method!='M1':
        errors=[t for t in tables if any('WAPE' in c for c in t[0])][-1]
        for a,r in zip(annual,year_rows(errors)): compare(a['wape_pct'],r[-1])
    return checks


def export_values(rows,headers):
    return [[('NA' if r[h] is None else round(r[h],10) if isinstance(r[h],float) else r[h]) for h in headers] for r in rows]


def write_table(folder,name,headers,rows):
    values=export_values(rows,headers)
    with (folder/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(headers);writer.writerows(values)
    with (folder/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        saved=list(csv.reader(stream))
    if saved[0]!=headers or len(saved)!=len(values)+1: raise ValueError('CSV schema/row count mismatch')
    for expected,actual in zip(values,saved[1:]):
        if len(actual)!=len(headers): raise ValueError('CSV field count mismatch')
        for x,y in zip(expected,actual):
            if isinstance(x,str):
                if x!=y: raise ValueError('CSV identifier/NA mismatch')
            else: close(x,float(y),'CSV numeric round-trip',1e-10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--max-seconds',type=float,default=55)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args();started=time.monotonic()
    def guard():
        if time.monotonic()-started>args.max_seconds: raise TimeoutError('Table export exceeded its time limit')
    root=args.root.resolve();out=(args.out_dir or root/'results').resolve();out.mkdir(parents=True,exist_ok=True)
    names=[m+'-results' for m in METHODS]+['comparison-results']
    outputs=[n+'.csv' for n in names]+['tables-notes.txt','table_metadata.json']
    if not args.force:
        for n in outputs:
            if (out/n).exists(): raise FileExistsError(f'{out/n} exists; use --force for intentional replacement')
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__).read_bytes()),
        'versions':{'python':platform.python_version()},'command':sys.argv,
        'kappa_eur_per_pu':KAPPA,'year_days':365,'annual_metric':'WAPE','lifecycle_metric':'actual utility, operating life, cash payback and decision time',
        'scope':'Read-only aggregation of saved results. No model, teacher, training, or optimizer execution.',
        'runs':{}}
    annual_tables={};comparison=[]
    for method,(run_id,report_names) in METHODS.items():
        guard();folder=root/'runs'/run_id
        report=next((root/n for n in report_names if (root/n).is_file()),None)
        if report is None and report_names: raise FileNotFoundError(report_names)
        report_raw=report.read_bytes() if report is not None else None
        summary,summary_sha=read_json(folder/'summary.json')
        manifest,manifest_sha=read_json(folder/'manifest.json')
        config=manifest['signature']['config'];configured=config.get('aging',{}).get('kappa_eur_per_pu',KAPPA)
        close(configured,KAPPA,'Common damage price')
        paths=sorted((folder/'days').glob('*.json'))
        if len(paths)!=summary['completed_days']: raise ValueError('Daily report count mismatch')
        records=[];digest=hashlib.sha256();previous=None;failed=0
        for i,(path,raw) in enumerate(saved_reports(paths)):
            guard();d=json.loads(raw);digest.update(path.name.encode());digest.update(hashlib.sha256(raw).digest())
            if d['index']!=i or not d['complete_day'] or len(d['intervals'])!=96: raise ValueError('Incomplete/out-of-order day')
            if previous is not None and d['health_start']!=previous: raise ValueError('Discontinuous health ledger')
            previous=d['health_end'];damage=math.fsum(d['damage_components_pu']);cost=d['teacher_aging_cost_eur']
            close(cost,KAPPA*damage,'Teacher damage-cost ledger')
            if cost<=0: raise ValueError('Recorded daily teacher cost must remain positive; no clipping/exclusion permitted')
            close(d['teacher_utility_eur'],d['revenue_eur']-cost,'Teacher utility')
            elapsed=math.fsum(w['plan']['solver']['elapsed_seconds'] for w in d['mpc_windows'])
            close(elapsed,d['solver_seconds'],'Decision elapsed time')
            if method=='M1':
                if d['aging_prediction_available']: raise ValueError('M1 prediction policy changed')
                pred=None
            elif method=='M3': pred=KAPPA*d['unscaled_table_damage_pu']
            else:
                pred=math.fsum(w['plan']['predicted_aging_cost_eur'] for w in d['mpc_windows'])
                if method in ('M4','M5'): close(pred,KAPPA*math.fsum(d['predicted_damage_components_pu']),'Hybrid prediction')
            records.append({'charge':d['delivered_charge_kwh'],'discharge':d['delivered_discharge_kwh'],
                'revenue':d['revenue_eur'],'calendar':d['damage_components_pu'][0],'cyclic':d['damage_components_pu'][1],
                'soh':previous['native_soh'],'utility':d['teacher_utility_eur'],'damage':damage,'cost':cost,
                'pred':pred,'seconds':elapsed})
            failed+=int(d['failed'])
        total=lambda k: math.fsum(r[k] for r in records)
        for key,skey in [('charge','delivered_charge_kwh'),('discharge','delivered_discharge_kwh'),
            ('revenue','revenue_eur'),('cost','teacher_aging_cost_eur'),('utility','teacher_utility_eur')]: close(total(key),summary[skey],method+' '+key)
        close(records[-1]['soh'],summary['final_soh'],method+' SOH')
        close(total('calendar')+total('cyclic'),1-records[-1]['soh'],method+' native capacity ledger')
        if failed!=summary['failed_days']: raise ValueError('Failure-day count mismatch')
        annual=[aggregate(records[i:i+365],i//365+1) for i in range(0,len(records),365)]
        report_checks=check_report(method,report_raw.decode('utf-8-sig'),annual) if report_raw is not None else 0
        estimated_cost=None if method=='M1' else total('pred')
        relative_error=None if estimated_cost is None else 100*(estimated_cost-total('cost'))/total('cost')
        row=dict(zip(COMPARISON,[method,total('revenue'),total('utility'),len(records),
                                 summary['payback_days'],total('seconds')/len(records)]))
        if 'decision_timing' in summary: close(row['mean_decision_time_s'],summary['decision_timing']['mean_seconds'],'Mean decision time')
        comparison.append(row);annual_tables[method]=annual
        provenance['runs'][method]={'run_id':run_id,'source_report':report.name if report is not None else None,
            'source_report_sha256':sha(report_raw) if report_raw is not None else None,
            'summary_sha256':summary_sha,'manifest_sha256':manifest_sha,'ordered_daily_reports_sha256':digest.hexdigest(),
            'seed':config.get('seed'),'versions':manifest['signature'].get('versions'),'days':len(records),
            'years':len(annual),'last_year_days':annual[-1]['operating_days'],'failed_days_included':failed,
            'report_rounded_values_checked':report_checks,'minimum_daily_teacher_cost_eur':min(r['cost'] for r in records),
            'lifecycle':row,'total_teacher_damage_pp':100*total('damage')}
        print(f'{method}: {len(annual)} annual rows; {len(records)} days; estimated cost={estimated_cost}; relative error={relative_error}; mean time={row["mean_decision_time_s"]:.9f} s',flush=True)
    with tempfile.TemporaryDirectory(prefix='.table-build-',dir=out) as tmp:
        dest=Path(tmp)
        for method,rows in annual_tables.items(): guard();write_table(dest,method+'-results',ANNUAL,rows)
        write_table(dest,'comparison-results',COMPARISON,comparison);guard()
        (dest/'tables-notes.txt').write_text(NOTES,encoding='utf-8')
        provenance['elapsed_seconds']=time.monotonic()-started
        provenance['outputs']={n:{'sha256':sha((dest/n).read_bytes()),'bytes':(dest/n).stat().st_size} for n in outputs if n!='table_metadata.json'}
        (dest/'table_metadata.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for n in outputs: os.replace(dest/n,out/n)
    print(f'Created six CSV tables in {out}; {provenance["elapsed_seconds"]:.2f} s',flush=True)


if __name__=='__main__': main()
