"""Version-pinned SimSES adapter. No replacement electrochemical/aging equations."""
from __future__ import annotations

import copy
import importlib.metadata as md
from dataclasses import asdict
from pathlib import Path

import numpy as np
from simses.battery.battery import Battery
from simses.battery.state import BatteryState
from simses.degradation.cycle_detector import HalfCycle
from simses.degradation.state import DegradationState
from simses.model.cell.sony_lfp import SonyLFP
from simses.thermal.ambient import AmbientThermalModel

from .common import Budget, canonical, digest
from .power_limits import ac_to_dc_kw, power_margin

SCHEMA = 'simses-2.1.1-residual-ledger-v1'


def native_fingerprint():
    import simses
    root = Path(simses.__file__).parent
    return canonical({str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
                      if p.suffix in ['.py', '.csv']})


def terminal_soc_ok(error, config):
    tolerance=config['terminal_soc_tolerance']
    return bool(np.isfinite(error) and (tolerance is None or error<=tolerance))


class Teacher:
    def __init__(self, config, budget: Budget):
        if md.version('simses') != '2.1.1':
            raise RuntimeError('audited teacher requires simses==2.1.1')
        self.config = copy.deepcopy(config)
        self.budget = budget
        self.battery = Battery(SonyLFP(), (config['series'], config['parallel']),
            dict(start_soc=config['initial_soc'], start_T=config['initial_temperature_c']),
            soc_limits=(config['soc_min'], config['soc_max']), degradation=True,
            effective_cooling_area=config['cooling_area_fraction'])
        self.thermal = AmbientThermalModel(config['initial_temperature_c'], [self.battery])
        self.time_s = 0.0
        # Forward Euler stability bound is necessary, not a convergence proof.
        if config['internal_dt_s'] > 0.1*self.battery.thermal_capacity*self.battery.thermal_resistance:
            raise ValueError('thermal timestep exceeds conservative RC/10 bound')

    def checkpoint(self):
        d = self.battery.degradation
        detector = copy.deepcopy(vars(d.cycle_detector))
        detector['last_cycle'] = asdict(detector['last_cycle']) if detector['last_cycle'] else None
        return dict(schema=SCHEMA, config=self.config, time_s=self.time_s,
                    battery=asdict(self.battery.state), degradation=asdict(d.state),
                    detector=detector, ambient_c=self.thermal.T_ambient)

    def restore(self, checkpoint):
        if checkpoint['schema'] != SCHEMA or checkpoint['config'] != self.config:
            raise ValueError('incompatible teacher checkpoint')
        self.battery.state = BatteryState(**checkpoint['battery'])
        self.battery.degradation.state = DegradationState(**checkpoint['degradation'])
        detector = copy.deepcopy(checkpoint['detector'])
        detector['last_cycle'] = HalfCycle(**detector['last_cycle']) if detector['last_cycle'] else None
        vars(self.battery.degradation.cycle_detector).clear()
        vars(self.battery.degradation.cycle_detector).update(detector)
        self.time_s = checkpoint['time_s']
        self.thermal.T_ambient = checkpoint['ambient_c']

    def fork(self):
        result = copy.deepcopy(self)
        result.budget = self.budget  # shared budget across all candidates
        return result

    def features(self):
        cp = self.checkpoint()
        # All battery fields + aging components + all detector sufficient statistics.
        # last_cycle is diagnostic only; native step overwrites it before use.
        return np.asarray(list(cp['battery'].values()) + list(cp['degradation'].values()) +
            [v for k, v in cp['detector'].items() if k != 'last_cycle'], dtype=float)

    def observable(self):
        s = self.battery.state
        return np.array([s.soc, s.T, s.v, -s.i])

    def ledger(self):
        # No closing/reversal step and no writes: native formula on an independent copy.
        d = copy.deepcopy(self.battery.degradation)
        hc = d.cycle_detector._make_half_cycle()
        residual = d.cyclic.update_capacity(copy.copy(self.battery.state), hc, d.state.qloss_cyc)
        return np.array([d.state.qloss_cal, d.state.qloss_cyc, residual])

    def ac_to_native(self, p):
        dc = ac_to_dc_kw(p,self.config)
        return -1000*dc

    def delivered_ac(self):
        dc = -self.battery.state.power/1000
        return dc*self.config['eta_discharge'] if dc >= 0 else dc/self.config['eta_charge']

    def current_limits(self):
        b=self.battery
        rate=self.config.get('max_c_rate')
        # Instantaneous current / available Ah, separate from native half-cycle rate.
        cap=float('inf') if rate is None else rate*b.capacity(b.state)
        return min(b.max_charge_current,cap),min(b.max_discharge_current,cap)

    def margins(self, commanded):
        s, b, c = self.battery.state, self.battery, self.config
        charge_limit,discharge_limit=self.current_limits()
        margins = np.array([s.soc-c['soc_min'], c['soc_max']-s.soc,
            s.T-c['temperature_min_c'], c['temperature_max_c']-s.T,
            s.v-b.min_voltage, b.max_voltage-s.v,
            charge_limit-s.i, discharge_limit+s.i,
            c['power_tolerance_kw']-abs(self.delivered_ac()-commanded),
            power_margin(commanded,-s.power/1000,c)])
        if c.get('report_only'):
            # Temperature is reported, not a project-added acceptance boundary.
            # Current/DC diagnostics retain their values but never select plans.
            margins[2:4]=0.
        return margins

    def rollout(self, power, ambient, dt_h, *, commit=False, terminal_target=None, record_checkpoints=False, resume_state=None, checkpoint_sink=None):
        power, ambient, dt_h = np.broadcast_arrays(np.asarray(power, float),
            np.asarray(ambient, float), np.asarray(dt_h, float))
        if power.ndim != 1 or not len(power) or not np.isfinite([power, ambient, dt_h]).all() or (dt_h <= 0).any():
            raise ValueError('finite nonempty 1-D controls and positive durations required')
        if resume_state is None:
            self.budget.query()
        obj = self if commit else self.fork()
        start = obj.ledger()
        c0, x0 = obj.features(), obj.observable()
        states, delivered, extrema, checkpoints = [], [], [], []
        min_margin = np.full(10, np.inf)
        first=0
        if resume_state is not None:
            if not all(np.array_equal(resume_state[k],v) for k,v in [('u',power),('ambient',ambient),('dt',dt_h)]):
                raise ValueError('replay controls changed on resume')
            obj.restore(resume_state['checkpoint'])
            start=resume_state['start']; c0=resume_state['c0']; x0=resume_state['x0']
            states=list(resume_state['states']); delivered=list(resume_state['delivered'])
            extrema=list(resume_state['extrema']); checkpoints=list(resume_state['checkpoints'])
            min_margin=resume_state['min_margin'].copy(); first=len(states)
        def persist():
            if checkpoint_sink:
                checkpoint_sink(dict(u=power.copy(),ambient=ambient.copy(),dt=dt_h.copy(),
                    checkpoint=obj.checkpoint(),start=start,c0=c0,x0=x0,
                    states=list(states),delivered=list(delivered),extrema=list(extrema),
                    checkpoints=list(checkpoints),min_margin=min_margin.copy()))
        persist()
        for p, ta, hours in zip(power[first:], ambient[first:], dt_h[first:]):
            remaining, integral = float(hours*3600), 0.0
            fine = []
            obj.thermal.T_ambient = float(ta)
            while remaining > 1e-8:
                obj.budget.step()
                step = min(remaining, obj.config['internal_dt_s'])
                obj.battery.step(obj.ac_to_native(float(p)), step)
                obj.thermal.step(step)
                obj.time_s += step
                integral += obj.delivered_ac()*step
                margin = obj.margins(p)
                if not np.isfinite(margin).all() or not np.isfinite(obj.observable()).all():
                    raise FloatingPointError('nonfinite native state')
                min_margin = np.minimum(min_margin, margin)
                fine.append(obj.observable())
                remaining -= step
            fine = np.asarray(fine)
            states.append(obj.observable())
            delivered.append(integral/(hours*3600))
            extrema.append(np.stack([fine.min(0), fine.max(0)]))
            if record_checkpoints:
                checkpoints.append(obj.checkpoint())
            persist()
        end = obj.ledger()
        damage = end-start  # [nonnegative calendar, nonnegative settled, SIGNED residual correction]
        if (damage[:2] < -1e-12).any():
            raise ArithmeticError('negative settled loss; never clip inconsistent labels')
        target = c0[5] if terminal_target is None else terminal_target  # BatteryState.soc index
        terminal_error = abs(obj.battery.state.soc-target)
        feasible = bool((min_margin >= -1e-9).all() and terminal_soc_ok(terminal_error,self.config))
        return dict(c0=c0, x0=x0, u=power.copy(), ambient=ambient.copy(), dt=dt_h.copy(),
                    x=np.asarray(states), damage=damage, delivered=np.asarray(delivered),
                    extrema=np.asarray(extrema), margins=min_margin, feasible=feasible,
                    terminal_error=terminal_error, checkpoint=obj.checkpoint(), checkpoints=checkpoints)
