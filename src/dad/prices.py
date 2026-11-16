"""SMARD export auditing: never infer a timezone from the market column."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, write_json


def read_raw(path):
    frame = pd.read_csv(path, sep=';', encoding='utf-8-sig', dtype=str)
    column = 'Germany/Luxembourg [€/MWh] Original resolutions'
    start = pd.to_datetime(frame['Start date'], format='%b %d, %Y %I:%M %p')
    end = pd.to_datetime(frame['End date'], format='%b %d, %Y %I:%M %p')
    price = pd.to_numeric(frame[column].str.replace(',', '', regex=False), errors='raise')/1000
    if not np.isfinite(price).all():
        raise ValueError('missing/nonfinite price')
    return start, end, price


def audit(path):
    start, end, price = read_raw(path)
    return dict(source=str(path), sha256=digest(path), rows=len(price),
        first_local=str(start.iloc[0]), last_end_local=str(end.iloc[-1]),
        duplicate_start_text=int(start.duplicated().sum()),
        non_15min_start_gaps=int((start.diff().dropna()!=pd.Timedelta(minutes=15)).sum()),
        non_15min_duration=int(((end-start)!=pd.Timedelta(minutes=15)).sum()),
        negative_prices=int((price<0).sum()), min_eur_kwh=float(price.min()),
        max_eur_kwh=float(price.max()), timezone='UNCONFIRMED',
        source_attribution='SMARD / Bundesnetzagentur, https://www.smard.de/',
        resolution_note='15 minute export grid; repeated prices do not establish native product resolution')


def prepare(path, timezone, out, *, start_local='2025-09-01', end_local='2026-09-01'):
    if not timezone:
        raise ValueError('source timezone is required; Germany does not identify export timezone')
    start, end, price = read_raw(path)
    # Infer folds only from repeated timestamps; nonexistent wall times must fail.
    s = pd.DatetimeIndex(start).tz_localize(timezone, ambiguous='infer', nonexistent='raise').tz_convert('UTC')
    # End at an autumn fold can be ambiguous even where start is not. Reconstruct
    # physical 15-min ends, then verify every original end text in the source zone.
    e = s + pd.Timedelta(minutes=15)
    local_end=e.tz_convert(timezone).tz_localize(None)
    local_start=s.tz_convert(timezone).tz_localize(None)
    mismatch=local_end.values!=end.values
    nominal=(end-start)==pd.Timedelta(minutes=15)
    offset_change=(local_end-local_start)!=pd.Timedelta(minutes=15)
    if np.any(mismatch & ~(nominal.to_numpy() & offset_change)):
        raise ValueError('end timestamps conflict with timezone / 15-minute physical grid')
    if s.has_duplicates or not s.is_monotonic_increasing or not np.all(np.diff(s.as_unit('ns').asi8)==900_000_000_000):
        raise ValueError('duplicate, missing or unordered physical price intervals')
    selected=(start>=pd.Timestamp(start_local)) & (start<pd.Timestamp(end_local))
    if not selected.any():raise ValueError('empty local-date price window')
    frame=pd.DataFrame(dict(start_utc=s, end_utc=e, price_eur_kwh=price, dt_h=0.25))
    frame['start_local']=s.tz_convert(timezone).astype(str)
    frame['end_local']=e.tz_convert(timezone).astype(str)
    frame=frame.loc[selected].reset_index(drop=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out/'prices.csv', index=False)
    report = audit(path)
    report.update(selected_local_start=str(pd.Timestamp(start_local)),
                  selected_local_end_exclusive=str(pd.Timestamp(end_local)),
                  selected_rows=len(frame),excluded_rows=len(price)-len(frame),
                  simulation_day_intervals=96,repeat_year_days=365,
                  simulation_calendar='365 consecutive 24-hour days; repeat 35040 intervals; no leap-day insertion',
                  timezone=timezone, timezone_basis='explicit user-supplied assumption; not inferred',
                  reconstructed_dst_end_labels=int(mismatch.sum()),
                  end_label_rule='At offset transitions only, nominal wall-clock end labels are replaced by start_UTC+15min; raw preserved',
                  processed_sha256=digest(out/'prices.csv'))
    write_json(out/'audit.json', report)
    return report


def load_prepared(path):
    """Require explicit timezone provenance and validate the physical grid again."""
    from .common import read_json
    from zoneinfo import ZoneInfo
    path=Path(path)
    report=read_json(path.with_name('audit.json'))
    zone=report.get('timezone')
    if not zone or zone=='UNCONFIRMED' or not report.get('timezone_basis'):
        raise ValueError('price source timezone must be explicitly supplied and audited')
    ZoneInfo(zone)
    if report.get('processed_sha256')!=digest(path):
        raise ValueError('prepared price hash differs from timezone audit')
    frame=pd.read_csv(path)
    required=['start_utc','end_utc','price_eur_kwh','dt_h']
    if frame.empty or any(k not in frame for k in required):
        raise ValueError('empty/incomplete prepared prices')
    for key in ['start_utc','end_utc']:
        if not frame[key].astype(str).str.contains(r'(?:Z|[+-]\d{2}:\d{2})$',regex=True).all():
            raise ValueError('prepared timestamps require explicit offsets')
        frame[key]=pd.to_datetime(frame[key],utc=True,errors='raise')
        if frame[key].isna().any(): raise ValueError('missing price timestamps')
    numbers=frame[['price_eur_kwh','dt_h']].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or not np.all(numbers[:,1]==0.25):
        raise ValueError('finite prices and audited 15-minute durations required')
    if not ((frame.end_utc-frame.start_utc)==pd.Timedelta(minutes=15)).all():
        raise ValueError('price interval duration mismatch')
    if frame.start_utc.duplicated().any() or not frame.start_utc.is_monotonic_increasing:
        raise ValueError('duplicate or unordered prices')
    if len(frame)>1 and not np.array_equal(frame.end_utc.iloc[:-1].values,frame.start_utc.iloc[1:].values):
        raise ValueError('missing/overlapping price intervals')
    return frame,report
