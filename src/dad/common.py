from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize first: an invalid value must never affect the committed checkpoint.
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    tmp = None
    try:
        # Same directory for atomic replacement; unique names avoid stale .tmp files.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            f.write(payload)
        # Windows readers/scanners can briefly deny replacement. Never delete the
        # destination as a workaround: the last committed state must survive.
        delays = (0.05, 0.1, 0.2, 0.4, 0.8)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                if attempt == len(delays):
                    raise PermissionError(
                        f'Cannot atomically replace {path} after 6 attempts; '
                        'the previous checkpoint is preserved. Close readers or '
                        'other writers using this run, then resume.'
                    ) from exc
                time.sleep(delays[attempt])
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def environment():
    import sys
    import torch
    import simses
    import importlib.util
    optional=['coptpy'] if importlib.util.find_spec('coptpy') is not None else []
    return dict(python=sys.version, executable=sys.executable, simses_path=simses.__file__,
                versions={p: md.version(p) for p in ['simses', 'torch', 'numpy', 'scipy', 'pandas']+optional},
                platform=platform.platform(), cpu=platform.processor(), cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


class LimitReached(RuntimeError):
    pass


@dataclass
class Budget:
    max_seconds: float
    max_steps: int
    max_queries: int
    steps: int = 0
    queries: int = 0

    def __post_init__(self):
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started >= self.max_seconds:
            raise LimitReached('wall time limit reached; resume from last committed checkpoint')

    def step(self):
        self.check()
        if self.steps >= self.max_steps:
            raise LimitReached('integration step limit reached')
        self.steps += 1

    def query(self):
        self.check()
        if self.queries >= self.max_queries:
            raise LimitReached('teacher query limit reached')
        self.queries += 1

    def report(self):
        return dict(steps=self.steps, queries=self.queries, seconds=time.monotonic()-self.started)


class Run:
    def __init__(self, path, config, command, resume=False, inputs=None):
        self.path = Path(path)
        self.console_output = config.get('continuous', {}).get('console_mode') != 'daily'
        code_root=Path(__file__).parent
        source_hash=canonical({p.name:digest(p) for p in sorted(code_root.glob('*.py'))})
        signature = dict(config={k:v for k,v in config.items() if k!='limits'},
                         command=command, inputs=inputs or {}, source_hash=source_hash)
        if resume:
            old = read_json(self.path / 'manifest.json')
            if old['signature'] != signature:
                raise ValueError('resume requires identical config, command and input hashes')
        else:
            self.path.mkdir(parents=True, exist_ok=False)
            write_json(self.path / 'manifest.json', dict(signature=signature, environment=environment(),
                       run_id=self.path.name, created=time.time(),initial_work_limits=config['limits']))
        self.log('start', resume=resume, work_limits=config['limits'])

    def log(self, event, **values):
        row = dict(time=time.time(), event=event, **values)
        with (self.path / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if self.console_output:
            print(json.dumps(row), flush=True)


def load_config(path):
    c = read_json(path)
    if c['kind'] not in ['smoke', 'production']:
        raise ValueError('explicit smoke/production kind required')
    t = c['teacher']
    for k in ['series', 'parallel', 'internal_dt_s', 'power_kw']:
        if t[k] <= 0:
            raise ValueError(f'{k} must be positive')
    for k in ['eta_charge', 'eta_discharge', 'cooling_area_fraction']:
        if not 0 < t[k] <= 1:
            raise ValueError(f'{k} must be in (0, 1]')
    from .contract import CONTRACT
    sustained_delivery=c['model'].get('delivery_contract')==CONTRACT and c['model'].get('state_update')=='electrothermal_memory_v7'
    soc_ok=(0 <= t['soc_min'] < t['initial_soc'] < t['soc_max'] <= 1) if sustained_delivery else (0 < t['soc_min'] < t['initial_soc'] < t['soc_max'] < 1)
    if not soc_ok:
        raise ValueError('invalid SOC limits')
    if c['data']['horizon'] < 2 or c['data']['dt_h']*3600 < t['internal_dt_s']:
        raise ValueError('invalid time grid')
    if c['data'].get('strategy')=='coverage-v2':
        raise ValueError('coverage-v2 was retired; no generator is available for that strategy')
    if c['model'].get('training_version') == 2:
        raise ValueError('training_version 2 was retired; no trainer is available for that version')
    if any(x <= 0 for x in c['limits'].values()):
        raise ValueError('all work limits must be positive')
    if c['kind'] == 'smoke' and (c['limits']['max_seconds'] > 55 or
            c['limits']['max_steps'] > 1000 or c['limits']['max_queries'] > 3):
        raise ValueError('smoke budget exceeds authorization')
    return c


def seed_all(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
