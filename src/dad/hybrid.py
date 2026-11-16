"""idea-1016 section 1. AC kW, hours, Celsius, W and capacity loss in p.u.

Single-window differentiable model; no data generation, fitting or dispatch.
Gradients are local to a fixed nonzero-power/half-cycle structure. Zero power
is an exact idle branch, not a smooth approximation to an active interval.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

SOC_ETA_CHARGE = 0.91
SOC_ETA_DISCHARGE = 0.97


def _value(value):
    # Only branch decisions/validation detach; arithmetic retains its graph.
    return value.detach().item() if isinstance(value, Tensor) else float(value)


@dataclass(frozen=True)
class Physics:
    energy_kwh: float
    resistance_k_per_w: float
    capacity_j_per_k: float
    eta_charge: float = SOC_ETA_CHARGE  # SOC coefficient, not converter efficiency
    eta_discharge: float = SOC_ETA_DISCHARGE
    internal_dt_s: float = 10.0

    def __post_init__(self):
        values = vars(self).values()
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('physical constants must be finite and positive')
        if max(self.eta_charge, self.eta_discharge) > 1:
            raise ValueError('efficiencies must be <= 1')
        if not 0 < self.alpha < 1:
            raise ValueError('thermal Euler step must have 0 < alpha < 1')

    @property
    def alpha(self):
        return 1 - self.internal_dt_s / (self.resistance_k_per_w * self.capacity_j_per_k)

    @classmethod
    def from_teacher(cls, teacher, *, soc_eta_charge=SOC_ETA_CHARGE,
                     soc_eta_discharge=SOC_ETA_DISCHARGE):
        """Read native constants with confirmed SOC coefficients; teacher is unchanged."""
        b, c = teacher.battery, teacher.config
        return cls(b.nominal_energy_capacity / 1000, b.thermal_resistance,
                   b.thermal_capacity,
                   soc_eta_charge, soc_eta_discharge,
                   c['internal_dt_s'])


@dataclass(frozen=True)
class Pending:
    direction: int = 0
    depth: float | Tensor = 0.0
    hours: float = 0.0


@dataclass(frozen=True)
class InitialState:
    soc: float | Tensor
    temperature_c: float | Tensor
    q_calendar: float | Tensor = 0.0
    q_cyclic: float | Tensor = 0.0  # settled only; never includes residual
    pending: Pending = Pending()

    @classmethod
    def from_checkpoint(cls, checkpoint):
        from .teacher import SCHEMA
        if checkpoint['schema'] != SCHEMA:
            raise ValueError('unsupported native checkpoint schema')
        b, d, h = (checkpoint[k] for k in ('battery', 'degradation', 'detector'))
        return cls(b['soc'], b['T'], d['qloss_cal'], d['qloss_cyc'],
                   Pending(h['_direction'], abs(h['_prev_soc'] - h['_start_soc']),
                           h['_elapsed_time'] / 3600))


class HeatNetwork(nn.Module):
    """129 trainable parameters. Normalizers are explicit, saved buffers.

    Supply training-only statistics when fitting later; identity normalization
    is acceptable for synthetic tests but is not a fitted model.
    """
    def __init__(self, input_center, input_scale, heat_center_w=0.0, heat_scale_w=1.0):
        super().__init__()
        for name, value in [('input_center', input_center), ('input_scale', input_scale),
                            ('heat_center_w', heat_center_w), ('heat_scale_w', heat_scale_w)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).clone())
        if self.input_center.shape != (5,) or self.input_scale.shape != (5,):
            raise ValueError('five input normalization constants required')
        if self.heat_center_w.ndim or self.heat_scale_w.ndim:
            raise ValueError('heat normalization must be scalar')
        if not all(torch.isfinite(b).all() for b in self.buffers()):
            raise ValueError('normalization must be finite')
        if (self.input_scale <= 0).any() or self.heat_scale_w <= 0:
            raise ValueError('normalization scales must be positive')
        self.layers = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 8),
                                    nn.Tanh(), nn.Linear(8, 1)).double()

    def forward(self, x):
        return self.heat_center_w + self.heat_scale_w * self.layers(
            (x - self.input_center) / self.input_scale).squeeze(-1)


def _scalar(value, like):
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _validate_pending(pending):
    d, t = _value(pending.depth), pending.hours
    if pending.direction not in (-1, 0, 1) or not math.isfinite(d + t) or d < 0 or t < 0:
        raise ValueError('invalid pending half-cycle')
    if (pending.direction == 0 and (d != 0 or t != 0)) or (pending.direction != 0 and (d <= 0 or t <= 0)):
        raise ValueError('pending direction/depth/duration are inconsistent')


def cyclic_stress(depth: Tensor, hours: float) -> Tensor:
    """Phi in squared p.u.; exact empty branch avoids 0/0."""
    if _value(depth) == 0:
        return depth * 0
    if hours <= 0 or _value(depth) < 0:
        raise ValueError('active half-cycle needs positive depth and duration')
    k = (0.0630 * depth / hours + 0.0971) * (4.0253 * (depth - 0.6)**3 + 1.0923)
    # Native Sony implementation returns zero for nonpositive stress.
    return (k / 100)**2 * depth / 2 if _value(k) > 0 else depth * 0


def root_increment(q: Tensor, stress: Tensor) -> Tensor:
    """sqrt(q^2+stress)-q without cancellation or artificial epsilon."""
    if _value(q) < 0 or _value(stress) < 0:
        raise ValueError("root increment requires nonnegative q and stress")
    if _value(q) == 0 and _value(stress) == 0:
        return (q + stress) * 0  # exact empty branch; no artificial smoothing
    return stress / (torch.sqrt(q*q + stress) + q)


@dataclass
class CyclicResult:
    damage: Tensor
    settled: Tensor
    residual_start: Tensor
    residual_end: Tensor
    pending: Pending


def cyclic_path(soc: Tensor, dt_h, q0=0.0, pending=Pending()) -> CyclicResult:
    """SOC includes the initial node. Durations exclude idle automatically.

    Topology selection uses scalar signs; active depths retain their tensor
    graph. Can also replay native SOC endpoints for section 2.1 diagnosis.
    """
    _validate_pending(pending)
    durations = torch.as_tensor(dt_h, dtype=soc.dtype).expand(len(soc) - 1).tolist()
    if soc.ndim != 1 or len(soc) < 2 or not torch.isfinite(soc).all():
        raise ValueError('finite SOC nodes required')
    if any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError('positive interval durations required')
    q = _scalar(q0, soc)
    if not torch.isfinite(q) or _value(q) < 0:
        raise ValueError('settled cyclic loss must be nonnegative')
    direction, depth, hours = pending.direction, _scalar(pending.depth, soc), pending.hours
    start_phi = cyclic_stress(depth, hours)
    completed = soc.sum() * 0
    for delta, duration in zip(soc.diff(), durations):
        sign = int(torch.sign(delta))
        if sign == 0:
            continue
        if direction and sign != direction:
            completed = completed + cyclic_stress(depth, hours)
            depth, hours = delta.abs(), duration
        else:
            depth, hours = depth + delta.abs(), hours + duration
        direction = sign
    end_phi = cyclic_stress(depth, hours)
    settled = q + root_increment(q, completed)
    r0 = root_increment(q, start_phi)
    rn = root_increment(settled, end_phi)
    # Rationalize the signed difference, including decreasing residuals.
    numerator = completed + end_phi - start_phi
    denominator = torch.sqrt(q*q + completed + end_phi) + torch.sqrt(q*q + start_phi)
    damage = numerator / denominator if _value(denominator) > 0 else numerator * 0
    return CyclicResult(damage, settled, r0, rn, Pending(direction, depth, hours))


def soc_path(power: Tensor, initial: InitialState, physics: Physics, dt_h=0.25):
    """Fixed window-start energy approximation; no clipping or native curtailment."""
    if power.ndim != 1 or not len(power) or not torch.isfinite(power).all():
        raise ValueError('nonempty finite one-dimensional AC power required')
    if not power.is_floating_point():
        raise ValueError('floating-point power required')
    dt = _scalar(dt_h, power).expand_as(power)
    if not torch.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError('positive finite durations required')
    qc, qy = _scalar(initial.q_calendar, power), _scalar(initial.q_cyclic, power)
    s0 = _scalar(initial.soc, power)
    if not torch.isfinite(torch.stack((s0, qc, qy))).all() or min(_value(qc), _value(qy)) < 0 or _value(qc+qy) >= 1:
        raise ValueError('invalid initial health/SOC')
    energy = physics.energy_kwh * (1 - qc - qy)
    increment = dt / energy * (physics.eta_charge * (-power).clamp_min(0)
                               - power.clamp_min(0) / physics.eta_discharge)
    return torch.cat((s0.reshape(1), s0 + increment.cumsum(0)))


@dataclass
class HybridResult:
    soc: Tensor  # N+1 nodes
    temperature_c: Tensor  # N+1 nodes
    heat_w: Tensor  # N effective constant heat values
    calendar_damage: Tensor
    cyclic: CyclicResult

    @property
    def total_damage(self):
        return self.calendar_damage + self.cyclic.damage


class HybridModel(nn.Module):
    def __init__(self, physics: Physics, heat: HeatNetwork):
        super().__init__()
        self.physics, self.heat = physics, heat

    def forward(self, power: Tensor, initial: InitialState, ambient_c=25.0, dt_h=0.25):
        ph = self.physics
        soc = soc_path(power, initial, ph, dt_h)
        dt = _scalar(dt_h, power).expand_as(power)
        ambient = _scalar(ambient_c, power).expand_as(power)
        temperature = _scalar(initial.temperature_c, power)
        if not torch.isfinite(ambient).all() or not torch.isfinite(temperature):
            raise ValueError('finite temperatures required')
        qc, qy = _scalar(initial.q_calendar, power), _scalar(initial.q_cyclic, power)
        temperatures, heats = [temperature], []
        stress = power.sum() * 0
        for i, p in enumerate(power):
            steps = _value(dt[i]) * 3600 / ph.internal_dt_s
            m = round(steps)
            if m < 1 or not math.isclose(steps, m, abs_tol=1e-7):
                raise ValueError('interval must be an integer number of native Euler steps')
            heat = p * 0 if _value(p) == 0 else self.heat(torch.stack((p, soc[i], temperature, qc, qy)))
            j = torch.arange(m, dtype=power.dtype, device=power.device)
            alpha_j = ph.alpha**j
            fine_t = alpha_j * temperature + (1-alpha_j) * (ambient[i] + ph.resistance_k_per_w * heat)
            fine_s = soc[i] + (j+1)/m * (soc[i+1]-soc[i])
            if (fine_t <= -273.15).any():
                raise ValueError('temperature below absolute zero')
            g = 1.2571e-5 * torch.exp(-17126 / 8.3144598 * (1/(fine_t+273.15) - 1/298.15)) * (2.8575*(fine_s-.5)**3 + .60225)
            stress = stress + torch.where(g > 0, g.square(), torch.zeros_like(g)).sum() * ph.internal_dt_s
            a = ph.alpha**m
            temperature = a*temperature + (1-a)*(ambient[i] + ph.resistance_k_per_w*heat)
            temperatures.append(temperature)
            heats.append(heat)
        return HybridResult(soc, torch.stack(temperatures), torch.stack(heats),
                            root_increment(qc, stress), cyclic_path(soc, dt, qy, initial.pending))
