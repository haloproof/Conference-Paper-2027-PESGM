"""Pure B2-DA COPT MILP: no teacher, surrogate, classifier, or terminal value."""
import time
import numpy as np
from .benchmark_b1_solver import B1Solver, validate_solution
from .benchmark_b2_tables import AgingTables, blocks, evaluate


def add_pwl(model,cp,coordinate,x,y):
    """Actual SOS2 equality, not the convex hull of nonadjacent points.

    y is expressed in EUR inside COPT to avoid tiny pu equality coefficients.
    """
    weights=[model.addVar(lb=0,ub=1) for _ in x]
    model.addConstr(cp.quicksum(weights)==1)
    model.addConstr(cp.quicksum(w*float(v) for w,v in zip(weights,x))==coordinate)
    model.addSOS(cp.COPT.SOS_TYPE2,weights,list(map(float,x)))
    return cp.quicksum(w*float(v) for w,v in zip(weights,y)),weights


def check_weights(weights):
    w=np.array([v.x for v in weights]);active=np.flatnonzero(w>1e-7)
    if np.any(w<-1e-7) or abs(w.sum()-1)>1e-7 or len(active)>2 or (len(active)==2 and active[1]-active[0]!=1):
        raise ArithmeticError('SOS2 adjacency/convexity verification failed')


class B2Solver(B1Solver):
    def __enter__(self):
        super().__enter__();self.tables=AgingTables();return self

    def solve(self,prices,*,initial_soc,energy_kwh,native_soh,charge_cap,discharge_cap,
              eta_charge,eta_discharge,dt_h=.25,max_seconds=None,soc_min=.1,soc_max=.9,
              start_interval=0,day_end_soc=None,toy=False):
        cp=self.cp;prices=np.asarray(prices,float);n=len(prices)
        groups=blocks(n,toy=toy)
        if not toy and (type(start_interval) is not int or start_interval%96):
            raise ValueError('B2 day must begin at a 96-interval simulation-day boundary')
        if dt_h!=.25 or (eta_charge,eta_discharge)!=(.91,.97) or (soc_min,soc_max)!=(.1,.9) or day_end_soc is not None:
            raise ValueError('B2 requires 15min, SOC coefficients .91/.97, SOC .1.. .9 and no terminal target')
        if prices.shape!=(n,) or not np.isfinite(prices).all() or not np.isfinite([initial_soc,energy_kwh,charge_cap,discharge_cap,native_soh]).all():
            raise ValueError('finite B2 inputs required')
        if not 0<=initial_soc<=1 or min(energy_kwh,charge_cap,discharge_cap)<=0 or not .8<native_soh<=1:
            raise ValueError('invalid B2 physical state or retired native SOH')
        curves=self.tables.select(native_soh,energy_kwh)
        # Deliberately conservative: even 4 hours continuously at either AC cap fits.
        if 4*max(charge_cap,discharge_cap)>curves['cycl']['x'][-1]+1e-9:
            raise ValueError('4h throughput bound exceeds author table coverage')
        seconds=min(self.options['max_seconds'],max_seconds if max_seconds is not None else self.options['max_seconds'])
        if not np.isfinite(seconds) or seconds<=0:raise ValueError('positive finite solve budget required')
        began=time.monotonic()
        with self.env.createModel('B2_DA_calendar_cyclic_SOS2') as model:
            params=dict(Logging=0,Threads=self.options['threads'],NodeLimit=self.options['node_limit'],
                RelGap=self.options['relative_gap'],AbsGap=self.options['absolute_gap_eur'],FeasTol=1e-9,IntTol=1e-9)
            if toy:params.update(NodeLimit=min(10,self.options['node_limit']),BarIterLimit=10)
            for k,v in params.items():model.setParam(k,v)
            c=[model.addVar(lb=0,ub=charge_cap) for _ in prices]
            d=[model.addVar(lb=0,ub=discharge_cap) for _ in prices]
            b=[model.addVar(vtype=cp.COPT.BINARY) for _ in prices]
            z=[model.addVar(lb=.1,ub=.9) for _ in prices]
            penalties=[];sets=[]
            for t in range(n):
                model.addConstr(c[t]<=charge_cap*b[t]);model.addConstr(d[t]<=discharge_cap*(1-b[t]))
                model.addConstr(z[t]==(initial_soc if t==0 else z[t-1])+.25*(eta_charge*c[t]-d[t]/eta_discharge)/energy_kwh)
                expr,w=add_pwl(model,cp,z[t],curves['cal']['x'],6480*curves['cal']['scaled'])
                penalties.append(expr);sets.append(w)
            for g in groups:
                for power in (c,d):
                    energy=cp.quicksum(.25*power[t] for t in g)
                    expr,w=add_pwl(model,cp,energy,curves['cycl']['x'],6480*curves['cycl']['scaled'])
                    penalties.append(expr);sets.append(w)
            model.setObjective(cp.quicksum(.25*prices[t]*(d[t]-c[t]) for t in range(n))-cp.quicksum(penalties),cp.COPT.MAXIMIZE)
            remaining=seconds-(time.monotonic()-began)
            if remaining<=0:raise TimeoutError('B2 construction exhausted solve allowance')
            model.setParam('TimeLimit',remaining);model.solve()
            status=int(model.getAttr('Status'));has=bool(model.getAttr('HasSol'))
            info=dict(backend='copt',version='8.0.7',status_code=status,has_incumbent=has,
                optimal_to_tolerance=status==cp.COPT.OPTIMAL,elapsed_seconds=time.monotonic()-began,
                solver_seconds=float(model.getAttr('SolvingTime')),decision_intervals=n,
                initial_soc=float(initial_soc),energy_kwh=float(energy_kwh),native_soh=float(native_soh),
                eta_charge=eta_charge,eta_discharge=eta_discharge,kappa_eur_per_pu=6480,throughput_penalty_eur_per_kwh=0,
                soc_min=.1,soc_max=.9,start_interval=start_interval,day_end_soc=None,endpoint_constraints=[],
                cyclic_blocks=groups,sos2_sets=len(sets),toy_interface_only=toy)
            if not has or status not in (cp.COPT.OPTIMAL,cp.COPT.TIMEOUT,cp.COPT.NODELIMIT):
                return dict(success=False,solver=info,reason='no usable B2 integer solution; no fallback')
            for w in sets:check_weights(w)
            plan=validate_solution([v.x for v in c],[v.x for v in d],[v.x for v in b],[v.x for v in z],
                prices,.25,initial_soc,energy_kwh,charge_cap,discharge_cap,eta_charge,eta_discharge,0,.1,.9)
            damage=evaluate(curves,plan['predicted_soc'],plan['charge_ac_kw'],plan['discharge_ac_kw'],groups)
            aging=6480*damage['scaled']['total_pu']
            plan.update(predicted_aging_cost_eur=aging,predicted_objective_eur=plan['predicted_revenue_eur']-aging,
                unscaled_table_damage_pu=damage['raw']['total_pu'],scaled_aging_penalty_eur=aging,
                table_damage=damage,table_selection=curves['audit'])
            if not np.isclose(model.getAttr('ObjVal'),plan['predicted_objective_eur'],atol=1e-7,rtol=1e-8):
                raise ArithmeticError('B2 objective disagrees with independent adjacent interpolation')
            for attr,key in [('ObjBound','bound_eur'),('BestGap','relative_gap')]:
                value=float(model.getAttr(attr));info[key]=value if np.isfinite(value) and abs(value)<1e20 else None
            return dict(success=True,solver=info,**plan)
