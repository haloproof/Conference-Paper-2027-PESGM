"""Estimator comparison models and frozen, original-unit metrics.

No file loading, teacher integration, training loop, plotting, or work on import.
The caller supplies training-only days for normalization and explicit evaluation
splits. Signed daily damage labels and predictions are never clipped.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .common import seed_all
from .hybrid import (CyclicResult, HybridModel, InitialState, Pending, Physics,
                     cyclic_path, root_increment, soc_path)
from .hybrid_train import initialize, local_arrays, normalizers


METHODS = ('HNN-compare', 'SOC-only', 'SOC NN', 'End-to-End NN')
MLP_METHODS = ('SOC NN', 'End-to-End NN')


def _nonthermal_initial(checkpoint, *, pending=False):
    """Use only nonthermal state; the carrier's temperature field is unused."""
    battery, degradation = checkpoint['battery'], checkpoint['degradation']
    history = Pending()
    if pending:
        detector = checkpoint['detector']
        history = Pending(detector['_direction'],
                          abs(detector['_prev_soc'] - detector['_start_soc']),
                          detector['_elapsed_time'] / 3600)
    return InitialState(battery['soc'], 0.0, degradation['qloss_cal'],
                        degradation['qloss_cyc'], history)


def daily_features(method, physics, day):
    """97 inferred SOC nodes or 96 command powers, without teacher endpoints."""
    power = torch.as_tensor(day['power'], dtype=torch.float64)
    if power.shape != (96,) or not torch.isfinite(power).all():
        raise ValueError('MLP input requires 96 finite command powers')
    if method == 'End-to-End NN':
        return power
    if method == 'SOC NN':
        return soc_path(power, _nonthermal_initial(day['initial']), physics)
    raise ValueError('daily_features is defined only for the two MLPs')


def total_label(day):
    # cyclic already includes the signed end-minus-start residual correction.
    value = torch.as_tensor(day['calendar'], dtype=torch.float64) + torch.as_tensor(
        day['cyclic'], dtype=torch.float64)
    if value.ndim or not torch.isfinite(value):
        raise ValueError('finite scalar daily total label required')
    return value


@dataclass
class SOCOnlyResult:
    soc: torch.Tensor
    calendar_damage: torch.Tensor
    cyclic: CyclicResult

    @property
    def total_damage(self):
        return self.calendar_damage + self.cyclic.damage


class SOCOnly(nn.Module):
    """Zero-parameter ablation: delete only the calendar temperature factor."""
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, power, initial, dt_h=0.25):
        ph = self.physics
        soc = soc_path(power, initial, ph, dt_h)
        dt = torch.as_tensor(dt_h, dtype=power.dtype, device=power.device).expand_as(power)
        stress = power.sum() * 0
        for i in range(len(power)):
            steps = float(dt[i].detach()) * 3600 / ph.internal_dt_s
            count = round(steps)
            if count < 1 or not math.isclose(steps, count, abs_tol=1e-7):
                raise ValueError('interval must contain an integer number of native steps')
            j = torch.arange(1, count + 1, dtype=power.dtype, device=power.device)
            fine_soc = soc[i] + j / count * (soc[i + 1] - soc[i])
            g = 1.2571e-5 * (2.8575 * (fine_soc - .5)**3 + .60225)
            stress = stress + torch.where(g > 0, g.square(), torch.zeros_like(g)).sum() * ph.internal_dt_s
        q = torch.as_tensor(initial.q_calendar, dtype=power.dtype, device=power.device)
        return SOCOnlyResult(soc, root_increment(q, stress),
                             cyclic_path(soc, dt, initial.q_cyclic, initial.pending))


class DailyMLP(nn.Module):
    """Whole-day vector to one signed daily total-damage prediction in pu."""
    def __init__(self, method, physics, stats):
        super().__init__()
        if method not in MLP_METHODS:
            raise ValueError('unknown daily MLP method')
        self.method, self.physics = method, physics
        width = 97 if method == 'SOC NN' else 96
        for key in ('input_center', 'input_scale', 'target_center', 'target_scale'):
            self.register_buffer(key, torch.as_tensor(stats[key], dtype=torch.float64).detach().clone())
        if self.input_center.shape != (width,) or self.input_scale.shape != (width,):
            raise ValueError('normalization dimensions do not match the selected input')
        if self.target_center.ndim or self.target_scale.ndim:
            raise ValueError('scalar target normalization required')
        if not all(torch.isfinite(b).all() for b in self.buffers()):
            raise ValueError('normalization must be finite')
        if (self.input_scale <= 0).any() or self.target_scale <= 0:
            raise ValueError('normalization scales must be positive')
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.Tanh(),
                                    nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1)).double()

    def forward(self, features):
        if features.shape[-1] != self.input_center.numel():
            raise ValueError('wrong whole-day input dimension')
        z = self.layers((features - self.input_center) / self.input_scale).squeeze(-1)
        return self.target_center + self.target_scale * z


def build_model(method, physics, train_days, seed=1016):
    """Fit normalization exclusively to the explicitly supplied training days."""
    if method not in METHODS:
        raise ValueError('unknown estimator comparison method')
    if method == 'SOC-only':
        return SOCOnly(physics), {}
    if not train_days:
        raise ValueError('nonempty training days required')
    if method == 'HNN-compare':
        stats = normalizers(train_days)
        return HybridModel(physics, initialize(stats, seed)), stats
    with torch.no_grad():
        features = torch.stack([daily_features(method, physics, day) for day in train_days])
        labels = torch.stack([total_label(day) for day in train_days])
        input_std, target_std = features.std(0, unbiased=False), labels.std(unbiased=False)
        zero_target = bool(target_std == 0)
        stats = dict(input_center=features.mean(0),
                     input_scale=torch.where(input_std > 0, input_std, torch.ones_like(input_std)),
                     target_center=labels.mean(),
                     target_scale=torch.as_tensor(1e-6, dtype=torch.float64) if zero_target else target_std,
                     target_zero_variance=zero_target)
    seed_all(seed)
    model = DailyMLP(method, physics, stats)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model, stats


def _physical_prediction(model, day):
    power = torch.as_tensor(day['power'], dtype=torch.float64)
    initial = (_nonthermal_initial(day['initial'], pending=True) if isinstance(model, SOCOnly)
               else InitialState.from_checkpoint(day['initial']))
    return model(power, initial)


def predict_daily(model, day):
    if isinstance(model, DailyMLP):
        return model(daily_features(model.method, model.physics, day))
    if isinstance(model, (HybridModel, SOCOnly)):
        return _physical_prediction(model, day).total_damage
    raise TypeError('unsupported estimator comparison model')


def daily_batch_loss(model, days):
    if not isinstance(model, DailyMLP) or not days:
        raise ValueError('daily_batch_loss requires an MLP and a nonempty daily batch')
    features = torch.stack([daily_features(model.method, model.physics, day) for day in days])
    targets = torch.stack([total_label(day) for day in days])
    return ((model(features) - targets) / model.target_scale).square().mean()


def _error_summary(errors, unit='pu^2', target='daily_total_degradation'):
    values = torch.stack([e.detach().to(dtype=torch.float64) for e in errors])
    if not len(values) or not torch.isfinite(values).all():
        raise FloatingPointError('nonempty finite raw residuals required')
    sse = float(values.square().sum())
    return dict(raw_sse=sse, count=len(values), raw_mse=sse / len(values),
                raw_mae=float(values.abs().mean()), unit=unit, target=target,
                raw_mae_unit='W' if unit == 'W^2' else 'pu')


def _component_metrics(calendar_errors, cyclic_errors):
    cal, cyc = _error_summary(calendar_errors), _error_summary(cyclic_errors)
    return dict(calendar_raw_mse=cal['raw_mse'], calendar_raw_mae=cal['raw_mae'],
                cyclic_raw_mse=cyc['raw_mse'], cyclic_raw_mae=cyc['raw_mae'])


def hnn_day_losses(model, days, stats, epsilon_calendar, check=lambda: None):
    """Original differentiable temperature/calendar objectives plus raw metrics."""
    if not days or epsilon_calendar <= 0 or stats['temperature_scale'] <= 0:
        raise ValueError('nonempty days and positive loss scales required')
    temperatures, calendars, errors, cal_errors, cyc_errors = [], [], [], [], []
    max_soc = 0.0
    for day in days:
        check()
        prediction = _physical_prediction(model, day)
        target_t = torch.as_tensor(day['temperature'], dtype=torch.float64)
        temperatures.append(((prediction.temperature_c[1:] - target_t) / stats['temperature_scale']).square().mean())
        cal_error = prediction.calendar_damage - day['calendar']
        cyc_error = prediction.cyclic.damage - day['cyclic']
        calendars.append((cal_error / epsilon_calendar).square())
        cal_errors.append(cal_error)
        cyc_errors.append(cyc_error)
        errors.append(prediction.total_damage - total_label(day))
        max_soc = max(max_soc, float((prediction.soc[1:].detach() - torch.as_tensor(
            day['soc'], dtype=torch.float64)).abs().max()))
    metrics = _error_summary(errors)
    metrics.update(_component_metrics(cal_errors, cyc_errors), max_soc_error=max_soc)
    return torch.stack(temperatures).mean(), torch.stack(calendars).mean(), metrics


def _heat_metrics(model, days, check):
    x, labels = local_arrays(days)
    sse, absolute, count = 0.0, 0.0, 0
    for start in range(0, len(x), 256):
        check()
        residual = (model.heat(x[start:start + 256]) - labels[start:start + 256]).to(torch.float64)
        if not torch.isfinite(residual).all():
            raise FloatingPointError('nonfinite heat residual')
        sse += float(residual.square().sum())
        absolute += float(residual.abs().sum())
        count += residual.numel()
    raw_mse = sse / count
    return dict(objective=raw_mse / float(model.heat.heat_scale_w)**2,
                raw_sse=sse, count=count, raw_mse=raw_mse, raw_mae=absolute / count,
                unit='W^2', raw_mae_unit='W', target='equivalent_heat', objective_unit='1')


def evaluate(model, method, days, stats, settings, stage, check=lambda: None):
    """Frozen whole-split metrics; restores all module modes, even on timeout.

    settings is the training settings mapping, including weights and
    epsilon_calendar for HNN stage 2. No optimizer or early-stop state changes.
    """
    if method not in METHODS or not days:
        raise ValueError('known method and nonempty evaluation days required')
    if stage == 'stage1' and method != 'HNN-compare':
        raise ValueError('only HNN has heat pretraining')
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            if method == 'HNN-compare':
                heat = _heat_metrics(model, days, check)
                t, c, result = hnn_day_losses(model, days, stats, settings['epsilon_calendar'], check)
                q, t, c = heat['objective'], float(t), float(c)
                w = settings['weights']
                result.update(objective=q if stage == 'stage1' else w[0] * q + w[1] * t + w[2] * c,
                              objective_unit='1', q=q, t=t, calendar=c,
                              heat_metrics=heat)
                return result
            errors, calendar_errors, cyclic_errors = [], [], []
            for day in days:
                check()
                if method == 'SOC-only':
                    prediction = _physical_prediction(model, day)
                    errors.append(prediction.total_damage - total_label(day))
                    calendar_errors.append(prediction.calendar_damage - day['calendar'])
                    cyclic_errors.append(prediction.cyclic.damage - day['cyclic'])
                else:
                    errors.append(predict_daily(model, day) - total_label(day))
            result = _error_summary(errors)
            if method == 'SOC-only':
                result.update(_component_metrics(calendar_errors, cyclic_errors),
                              objective=result['raw_mse'], objective_unit='pu^2')
            else:
                result.update(objective=result['raw_mse'] / float(model.target_scale)**2,
                              objective_unit='1')
            return result
    finally:
        for module, was_training in modes:
            module.training = was_training
