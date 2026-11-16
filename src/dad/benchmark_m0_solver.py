"""M0-DA: B1's audited energy-bucket MILP with alpha hard-coded to zero.

No teacher, aging table, surrogate, SOH prediction, or terminal inventory value.
The caller supplies actual day-start capacity. SOC coefficients match B3; historical 0.90/0.90 checkpoints
are not compatible with this version.
"""
import numpy as np
from .benchmark_b1_solver import B1Solver


class M0Solver(B1Solver):
    def solve(self,prices,*,initial_soc,energy_kwh,charge_cap,discharge_cap,
              eta_charge=.91,eta_discharge=.97,dt_h=.25,max_seconds=None,soc_min=.1,soc_max=.9,
              start_interval=0,day_end_soc=None,toy=False):
        prices=np.asarray(prices,float)
        if prices.ndim!=1 or not np.isfinite(prices).all():
            raise ValueError('finite 1-D M0 price sequence required')
        if type(start_interval) is not int or start_interval<0:
            raise ValueError('nonnegative integer interval index required')
        if toy:
            if not 1<=len(prices)<=8 or self.options['node_limit']>10 or self.options['max_seconds']>5:
                raise ValueError('M0 toy interface exceeds short-test budget')
        elif len(prices)!=96 or start_interval%96:
            raise ValueError('M0-DA requires 96 prices from one 24-hour simulation day only')
        if (dt_h,eta_charge,eta_discharge,soc_min,soc_max)!=(.25,.91,.97,.1,.9) or day_end_soc is not None:
            raise ValueError('M0 requires 15min, SOC coefficients .91/.97, SOC .1.. .9 and no terminal target')
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds<=0):
            raise ValueError('positive finite solve allowance required')
        plan=super().solve(prices,initial_soc=initial_soc,energy_kwh=energy_kwh,
            charge_cap=charge_cap,discharge_cap=discharge_cap,eta_charge=eta_charge,eta_discharge=eta_discharge,alpha=0.,
            dt_h=.25,max_seconds=max_seconds,soc_min=.1,soc_max=.9,
            start_interval=start_interval,day_end_soc=None)
        plan['solver'].update(method='M0-DA',objective='day-ahead arbitrage only',
            aging_prediction_available=False,terminal_inventory_value=False,toy_interface_only=toy)
        if plan['success']:
            if plan['predicted_aging_cost_eur']!=0 or plan['predicted_objective_eur']!=plan['predicted_revenue_eur']:
                raise ArithmeticError('M0 objective must equal planned revenue exactly')
            plan['optimizer_aging_penalty_eur']=0.
            plan['aging_prediction_available']=False
        return plan
