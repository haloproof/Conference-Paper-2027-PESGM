"""UTC-day price audit and explicit historical replay mapping (no simulation)."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_json, digest
from .prices import load_prepared

SPLITS=Path(__file__).resolve().parents[2]/'configs/market_splits.json'


def utc_instant(value):
    """Explicit UTC instant on the physical 15-minute grid; no midnight requirement."""
    t=pd.Timestamp(value)
    if t.tzinfo is None or t.utcoffset().total_seconds()!=0 or t!=t.floor('15min'):
        raise ValueError('15-minute UTC boundary with explicit offset required')
    return t.tz_convert('UTC')


def utc_day(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None or t.utcoffset().total_seconds()!=0 or t!=t.normalize():
        raise ValueError('UTC midnight with explicit offset required')
    return t.tz_convert('UTC')


def partitions(lo,hi,splits):
    hits=[]; cursor=lo
    for name,block in sorted(((k,v) for k,v in splits.items() if k!='basis'),key=lambda kv:kv[1][0]):
        a,b=map(pd.Timestamp,block)
        left,right=max(a,lo),min(b,hi)
        if left<right:
            if left!=cursor: raise ValueError('price interval outside frozen market partitions')
            hits.append(name); cursor=right
    if cursor!=hi: raise ValueError('price interval outside frozen market partitions')
    return hits


def coverage(path):
    frame,audit=load_prepared(path)
    lo=frame.start_utc.iloc[0].ceil('D'); hi=frame.end_utc.iloc[-1].floor('D')
    source=Path(audit.get('source','__source_unavailable__'))
    source_verified=source.is_file() and audit.get('sha256')==digest(source)
    if source.is_file() and not source_verified: raise ValueError('raw source hash differs from audit')
    report=dict(raw_source_hash_verified=bool(source_verified),source_sha256=audit.get('sha256'),prices_sha256=digest(path),audit_sha256=digest(Path(path).with_name('audit.json')),
        rows=len(frame),start_utc=str(frame.start_utc.iloc[0]),end_utc=str(frame.end_utc.iloc[-1]),
        complete_utc_days=max(0,(hi-lo).days),first_complete_day=str(lo),complete_days_end=str(hi),
        negative_intervals=int((frame.price_eur_kwh<0).sum()),zero_intervals=int((frame.price_eur_kwh==0).sum()),
        min_eur_kwh=float(frame.price_eur_kwh.min()),max_eur_kwh=float(frame.price_eur_kwh.max()),
        missing=0,duplicates=0,interval_minutes=15,day_boundary='96 consecutive intervals anchored at scenario.start_utc',
        repeat_year_days=365,selected_local_start=audit.get('selected_local_start'),
        selected_local_end_exclusive=audit.get('selected_local_end_exclusive'),
        source_timezone=audit['timezone'],timezone_basis=audit['timezone_basis'],
        timezone_independently_verified=False,
        note='Source timezone provenance is preserved from audit; no independent timezone inference.')
    return frame,report


def validate_scenario(spec,frame):
    if spec['evaluation_scope']!='frozen_cross_partition_no_feedback':
        raise ValueError('explicit frozen cross-partition evaluation scope required')
    if spec['price_scenario'] not in ['historical','repeat_history_year']:
        raise ValueError('explicit historical or repeat_history_year scenario required')
    start=utc_instant(spec['start_utc'])
    if spec['price_scenario']=='repeat_history_year':
        lo,hi=utc_instant(spec['repeat_start_utc']),utc_instant(spec['repeat_end_utc'])
        if hi-lo!=pd.Timedelta(days=365): raise ValueError('repeat scenario requires exactly 365 elapsed 24-hour days')
        if lo<frame.start_utc.iloc[0] or hi>frame.end_utc.iloc[-1]:
            raise ValueError('repeat source year unavailable')
        partitions(lo,hi,read_json(SPLITS))
    else:
        if not frame.start_utc.iloc[0]<=start<frame.end_utc.iloc[-1]:
            raise ValueError('start outside historical price coverage')
    return start


def day_prices(frame,spec,index):
    simulated=utc_instant(spec['start_utc'])+pd.Timedelta(days=index)
    source=simulated
    if spec['price_scenario']=='repeat_history_year':
        source=utc_instant(spec['repeat_start_utc'])+pd.Timedelta(days=index%365)
    end=source+pd.Timedelta(days=1)
    if source<frame.start_utc.iloc[0] or end>frame.end_utc.iloc[-1]: return None,None
    part=frame[(frame.start_utc>=source)&(frame.start_utc<end)].copy().reset_index(drop=True)
    if len(part)!=96 or part.start_utc.iloc[0]!=source or part.end_utc.iloc[-1]!=end:
        raise ValueError('not a complete 96-interval simulation day')
    blocks=partitions(source,end,read_json(SPLITS))
    offset=simulated-source
    part.start_utc+=offset; part.end_utc+=offset
    mapping=dict(date_utc=str(simulated),end_utc=str(simulated+pd.Timedelta(days=1)),
        source_date_utc=str(source),source_end_utc=str(end),scenario=spec['price_scenario'],
        market_partitions=blocks,repeat_cycle=index//365 if spec['price_scenario']=='repeat_history_year' else None,
        price_min_eur_kwh=float(part.price_eur_kwh.min()),price_max_eur_kwh=float(part.price_eur_kwh.max()),
        zero_intervals=int((part.price_eur_kwh==0).sum()),negative_intervals=int((part.price_eur_kwh<0).sum()))
    return part,mapping
