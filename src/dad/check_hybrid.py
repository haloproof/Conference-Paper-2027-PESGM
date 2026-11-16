"""Bounded section 2.1 check only: three 45-minute windows, 810 native steps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import Budget, digest, environment, read_json, seed_all
from .hybrid import (InitialState, Physics, cyclic_path, soc_path,
                     SOC_ETA_CHARGE, SOC_ETA_DISCHARGE)
from .power_limits import ac_power_bounds
from .teacher import Teacher, native_fingerprint


def check(max_seconds=45.0, *, eta_charge=SOC_ETA_CHARGE, eta_discharge=SOC_ETA_DISCHARGE):
    if not 0 < max_seconds <= 50:
        raise ValueError('short-check budget must be in (0, 50] seconds')
    seed_all(1016)
    # Reuse physical parameters only; never execute the production benchmark.
    config_path = Path('configs/benchmark-b1-da-base.json')
    config = read_json(config_path)['teacher']
    budget = Budget(max_seconds=max_seconds, max_steps=810, max_queries=3)
    teacher = Teacher(config, budget)
    physics = Physics.from_teacher(teacher, soc_eta_charge=eta_charge,
                                   soc_eta_discharge=eta_discharge)
    rows = []
    for name, commands in [('charge_idle_charge', [-6., 0., -2.]),
                           ('inherited_reversal_idle_discharge', [3., 0., 3.]),
                           ('two_reversals_idle', [-1., 6., 0.])]:
        before = teacher.checkpoint()
        initial = InitialState.from_checkpoint(before)
        p = torch.tensor(commands, dtype=torch.float64)
        approx_soc = soc_path(p, initial, physics)
        approx_cyc = cyclic_path(approx_soc, .25, initial.q_cyclic, initial.pending)
        result = teacher.rollout(commands, 25., .25, commit=True, record_checkpoints=True)
        native_soc = torch.tensor([initial.soc, *result['x'][:, 0]], dtype=torch.float64)
        replay = cyclic_path(native_soc, .25, initial.q_cyclic, initial.pending)
        native_cyc = float(result['damage'][1] + result['damage'][2])
        native_pending = InitialState.from_checkpoint(result['checkpoint']).pending
        # This replay isolates accounting only when delivery is uncurtailed.
        delivery_error = float(np.max(np.abs(result['delivered'] - commands)))
        replay_ok = (delivery_error < 1e-8 and
                     abs(replay.pending.hours - native_pending.hours) < 1e-10 and
                     replay.pending.direction == native_pending.direction)
        if not replay_ok:
            raise AssertionError('tiny witness was curtailed or its activity duration changed')
        if abs(float(replay.damage) - native_cyc) > 1e-12:
            raise AssertionError('native-SOC half-cycle accounting mismatch')
        rows.append(dict(name=name, power_ac_kw=commands, initial_checkpoint=before,
                         final_checkpoint=result['checkpoint'], teacher_soc=native_soc.tolist(),
                         approximate_soc=approx_soc.tolist(),
                         max_abs_soc_error_pu=float((approx_soc-native_soc).abs().max()),
                         cyclic_teacher_pu=native_cyc, cyclic_approximate_pu=float(approx_cyc.damage),
                         cyclic_error_pu=float(approx_cyc.damage)-native_cyc,
                         native_soc_accounting_error_pu=float(replay.damage)-native_cyc,
                         max_mean_delivery_error_kw=delivery_error,
                         min_native_delivery_margin_kw=float(result['margins'][-2]),
                         pending_duration_error_h=replay.pending.hours-native_pending.hours,
                         teacher_feasible=bool(result['feasible'])))
        budget.check()
    return dict(scope='idea-1016 section 2.1; no fitted heat network or total-error certification',
                seed=1016, dt_h=.25, environment=environment(), physics=vars(physics),
                coefficient_role='physics eta values are SOC effective coefficients, not native converter efficiencies',
                teacher_converter_efficiency=dict(charge=config['eta_charge'], discharge=config['eta_discharge']),
                power_bounds_ac_kw=ac_power_bounds(config),
                config_sha256=digest(config_path), native_fingerprint=native_fingerprint(),
                source_sha256={n: digest(Path(__file__).with_name(n)) for n in
                               ['hybrid.py', 'check_hybrid.py', 'teacher.py']},
                budget=budget.report(), windows=rows,
                exceeds_total_error_budget_on_witness=any(abs(r['cyclic_error_pu']) > 1e-6 for r in rows),
                limitations=['Only three linked short windows at one native health history.',
                             'No claim about full-day errors, health coverage or trained accuracy.',
                             'SOH_R hidden-history heat ambiguity remains untested; full native checkpoints retained.',
                             'Fixed nominal-energy SOC approximation is unchanged; heat fitting cannot repair cyclic errors.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new JSON file; refuses overwrite')
    parser.add_argument('--max-seconds', type=float, default=45.)
    parser.add_argument('--eta-charge', type=float, default=SOC_ETA_CHARGE, help='SOC effective charge coefficient only')
    parser.add_argument('--eta-discharge', type=float, default=SOC_ETA_DISCHARGE, help='SOC effective discharge coefficient only')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output exists; choose a new report name')
    report = check(args.max_seconds, eta_charge=args.eta_charge, eta_discharge=args.eta_discharge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    for row in report['windows']:
        print(f"{row['name']}: SOC error={row['max_abs_soc_error_pu']:.6g}, "
              f"cyclic error={row['cyclic_error_pu']:.6g} p.u.")
    print(f"Saved {args.out}; {report['budget']['queries']} trajectories, {report['budget']['steps']} steps")


if __name__ == '__main__':
    main()
