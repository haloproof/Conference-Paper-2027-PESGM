"""Collath model (i), with exclusive AC charge/discharge and COPT.

Pure optimizer interface: no teacher object, neural model or rollout callbacks.
Positive returned AC power means discharge. Optional UTC day-end SOC control.
"""
import importlib.metadata as md
import time
import numpy as np


def day_end_constraints(start_interval, horizon, energy_kwh, charge_cap,
                        discharge_cap, eta_charge, eta_discharge, dt_h, policy):
    """Midnight endpoint bounds, plus reachability to the next unseen midnight.

    Indices refer to interval endpoints. No target at arbitrary MPC endpoints.
    Reachability uses the same frozen-capacity linear model as the optimizer.
    """
    if policy is None:return []
    if dt_h!=.25 or type(start_interval) is not int or start_interval<0:
        raise ValueError('day-end control requires UTC-aligned 15 min indices')
    if set(policy)!= {'target','tolerance'} or policy['target']!=.5 or policy['tolerance']!=.025:
        raise ValueError('day-end SOC policy must match V7: 0.5 +/- 0.025')
    lower=policy['target']-policy['tolerance'];upper=policy['target']+policy['tolerance']
    end=start_interval+horizon
    bounds=[dict(index=k-start_interval-1,lower=lower,upper=upper,kind='utc_midnight')
            for k in range((start_interval//96+1)*96,end+1,96)]
    if end%96:
        hours=(96-end%96)*dt_h
        bounds.append(dict(index=horizon-1,
            lower=lower-eta_charge*charge_cap*hours/energy_kwh,
            upper=upper+discharge_cap*hours/(eta_discharge*energy_kwh),
            kind='next_midnight_reachability'))
    return bounds


def validate_solution(charge, discharge, mode, soc, prices, dt_h, initial_soc,
                      energy_kwh, charge_cap, discharge_cap, eta_charge, eta_discharge, alpha,
                      soc_min=.1, soc_max=.9, endpoint_constraints=()):
    c,d,b,z=[np.asarray(v,float) for v in (charge,discharge,mode,soc)]
    prices=np.asarray(prices,float)
    tol=1e-7
    if any(v.shape!=prices.shape for v in (c,d,b,z)) or not np.isfinite([c,d,b,z]).all():
        raise ValueError('invalid B1 solution shape/values')
    if np.any(abs(b-np.rint(b))>tol) or np.any(b<-tol) or np.any(b>1+tol):
        raise ValueError('B1 mode is not binary')
    if np.any(c<-tol) or np.any(d<-tol) or np.any(c>charge_cap*b+tol) or np.any(d>discharge_cap*(1-b)+tol):
        raise ValueError('B1 power or exclusivity violation')
    expected=initial_soc+np.cumsum(eta_charge*c-d/eta_discharge)*dt_h/energy_kwh
    if not np.allclose(expected,z,rtol=0,atol=tol) or np.any(z<soc_min-tol) or np.any(z>soc_max+tol):
        raise ValueError('B1 SOC balance/bounds violation')
    for bound in endpoint_constraints:
        if not bound['lower']-tol<=z[bound['index']]<=bound['upper']+tol:
            raise ValueError('B1 day-end SOC/reachability violation')
    u=d-c
    revenue=float(np.dot(prices,u)*dt_h)
    aging=float(alpha*np.sum(c+d)*dt_h)
    return dict(power_ac_kw=u.tolist(),charge_ac_kw=c.tolist(),discharge_ac_kw=d.tolist(),
                predicted_soc=z.tolist(),mode=b.tolist(),predicted_revenue_eur=revenue,
                predicted_aging_cost_eur=aging,predicted_objective_eur=revenue-aging,
                max_simultaneous_kw=float(np.minimum(c,d).max()))


class B1Solver:
    """One licensed COPT environment per foreground run; a new model per window."""
    def __init__(self, options):
        self.options=dict(options)
        self.env=None

    def __enter__(self):
        if md.version('coptpy')!='8.0.7':
            raise RuntimeError('B1 requires tested coptpy==8.0.7 in project .venv')
        import coptpy as cp
        self.cp=cp
        settings=cp.EnvrConfig();settings.set('nobanner','1')
        self.env=cp.Envr(settings)
        return self

    def __exit__(self,*args):
        if self.env is not None:self.env.__exit__(*args)

    def solve(self,prices,*,initial_soc,energy_kwh,charge_cap,discharge_cap,
              eta_charge,eta_discharge,alpha,dt_h=.25,max_seconds=None,soc_min=.1,soc_max=.9,
              start_interval=0,day_end_soc=None):
        cp=self.cp;prices=np.asarray(prices,float);n=len(prices)
        numbers=[initial_soc,energy_kwh,charge_cap,discharge_cap,eta_charge,eta_discharge,alpha,dt_h,soc_min,soc_max]
        if prices.ndim!=1 or not n or not np.isfinite(prices).all() or not np.isfinite(numbers).all():
            raise ValueError('finite B1 inputs required')
        if not (0<=initial_soc<=1 and energy_kwh>0 and min(charge_cap,discharge_cap,dt_h)>0 and 0<eta_charge<=1 and 0<eta_discharge<=1 and alpha>=0):
            raise ValueError('invalid B1 physical inputs')
        if not 0<=soc_min<soc_max<=1:
            raise ValueError('invalid B1 scheduling SOC bounds')
        endpoints=day_end_constraints(start_interval,n,energy_kwh,charge_cap,
                                     discharge_cap,eta_charge,eta_discharge,dt_h,day_end_soc)
        seconds=self.options['max_seconds'] if max_seconds is None else min(max_seconds,self.options['max_seconds'])
        if seconds<=0:raise ValueError('positive B1 solve allowance required')
        began=time.monotonic()
        with self.env.createModel('B1_throughput_MPC') as model:
            params=dict(Logging=0,Threads=self.options['threads'],NodeLimit=self.options['node_limit'],
                        RelGap=self.options['relative_gap'],AbsGap=self.options['absolute_gap_eur'],
                        FeasTol=1e-9,IntTol=1e-9)
            for name,value in params.items():model.setParam(name,value)
            c=[model.addVar(lb=0,ub=charge_cap) for _ in range(n)]
            d=[model.addVar(lb=0,ub=discharge_cap) for _ in range(n)]
            b=[model.addVar(vtype=cp.COPT.BINARY) for _ in range(n)]
            z=[model.addVar(lb=soc_min,ub=soc_max) for _ in range(n)]
            for bound in endpoints:
                model.addConstr(z[bound['index']]>=bound['lower'])
                model.addConstr(z[bound['index']]<=bound['upper'])
            for t in range(n):
                model.addConstr(c[t]<=charge_cap*b[t])
                model.addConstr(d[t]<=discharge_cap*(1-b[t]))
                model.addConstr(z[t]==(initial_soc if t==0 else z[t-1])+
                                dt_h*(eta_charge*c[t]-d[t]/eta_discharge)/energy_kwh)
            model.setObjective(cp.quicksum(dt_h*(prices[t]*(d[t]-c[t])-alpha*(c[t]+d[t]))
                                          for t in range(n)),cp.COPT.MAXIMIZE)
            remaining=seconds-(time.monotonic()-began)
            if remaining<=0:raise TimeoutError('B1 model construction exhausted solve allowance')
            model.setParam('TimeLimit',remaining)
            model.solve()
            status=int(model.getAttr('Status'));has=bool(model.getAttr('HasSol'))
            info=dict(backend='copt',version=md.version('coptpy'),status_code=status,has_incumbent=has,
                      optimal_to_tolerance=status==cp.COPT.OPTIMAL,elapsed_seconds=time.monotonic()-began,
                      solver_seconds=float(model.getAttr('SolvingTime')),decision_intervals=n,
                      initial_soc=float(initial_soc),energy_kwh=float(energy_kwh),
                      eta_charge=eta_charge,eta_discharge=eta_discharge,alpha_eur_per_kwh=alpha,
                      soc_min=soc_min,soc_max=soc_max,
                      start_interval=start_interval,day_end_soc=day_end_soc,
                      endpoint_constraints=endpoints)
            if not has or status not in (cp.COPT.OPTIMAL,cp.COPT.TIMEOUT,cp.COPT.NODELIMIT):
                return dict(success=False,solver=info,reason='no usable integer solution; no fallback')
            plan=validate_solution([v.x for v in c],[v.x for v in d],[v.x for v in b],[v.x for v in z],
                                   prices,dt_h,initial_soc,energy_kwh,charge_cap,discharge_cap,eta_charge,eta_discharge,alpha,
                                   soc_min,soc_max,endpoints)
            for attr,key in [('ObjBound','bound_eur'),('BestGap','relative_gap')]:
                value=float(model.getAttr(attr))
                info[key]=value if np.isfinite(value) and abs(value)<1e20 else None
            if not np.isclose(float(model.getAttr('ObjVal')),plan['predicted_objective_eur'],atol=1e-7,rtol=1e-8):
                raise ValueError('B1 objective verification failed')
            return dict(success=True,solver=info,**plan)
