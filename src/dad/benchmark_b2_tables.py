"""Author model (iii) tables, with the original health-column offset and scaling.

Audit against Collath 2023 pp. 5,6,12 and benchmark.md sections 5,8,11:
* Paper separates calendar/cyclic health; author main.py passes total native SOH.
* Reference .95 actually selects representative .949375, not an exact 5% loss.
* Paper Eq.16 uses remaining capacity; source objective uses initial capacity,
  while its cost logs use remaining capacity. We use fixed 6480 EUR/pu throughout.
* Calendar ordinates are 15-minute increments; cyclic ordinates are per direction
  per 4-hour block. Cyclic x is e_AC / current energy, NOT FEC or C-rate.
* CSV x reaches 8 (Fig.4 shows about 4). The .96 sample agrees with a 2-hour
  direction-rate assumption; this is inference, not an available generator.
* Source's very-low-SOH branch selects label column zero (k=-1 then k+1=0).
  Reject that unsupported domain explicitly; production retires at .8 anyway.
* Original fixed-25C intraday MPC / SimSES 1.3.2 is adapted to daily commitments
  and dynamic-temperature SimSES 2.1.1. Tables are retained, not regenerated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import digest

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT/'aging-aware-MPC/data/linearizations'
HASHES = dict(cal='e2d48378ef7f0714948b6ee4f8ba804eef408adcac7e8d480bbc4be0fcc6ea8f',
              cycl='7996fc43e44f9a79afa6ba51751f471e507d0910110309943a4a4df310fb7e0a')


class AgingTables:
    def __init__(self):
        self.data={}; self.hashes={}
        for key,rows in [('cal',13),('cycl',58)]:
            p=TABLE_DIR/f'coefficients_linear_{key}_deg_LFP.csv'
            self.hashes[key]=digest(p)
            if self.hashes[key]!=HASHES[key]:raise ValueError('author aging table hash changed')
            f=pd.read_csv(p)
            if f.shape!=(rows,230) or list(f.iloc[:2,0])!=['SOH','SOH_lower_limit']:
                raise ValueError('unexpected author CSV layout')
            a=f.iloc[:,1:].to_numpy(float)
            if not np.isfinite(a).all() or np.any(a<0) or not np.all(np.diff(a[1])<0):
                raise ValueError('invalid author coefficients/health ordering')
            y=a[2:] if key=='cal' else a[30:58]
            if np.any(np.diff(y,axis=0)<-1e-15):raise ValueError('nonmonotonic damage ordinates')
            self.data[key]=a
        if not np.array_equal(self.data['cal'][:2],self.data['cycl'][:2]):
            raise ValueError('health axes disagree')
        x=self.data['cycl'][2:30]
        if not np.all(x==x[:,[0]]) or x[0,0]!=0 or x[-1,0]!=8 or np.any(np.diff(x[:,0])<=0):
            raise ValueError('invalid cyclic throughput axis')

    def column(self,soh,key='cal'):
        """Return zero-based numeric column; equivalent to author's k+1 including label."""
        lower=self.data[key][1]
        if not np.isfinite(soh) or not 0<=soh<=1:raise ValueError('invalid native SOH')
        if soh<=lower[-2]:
            raise ValueError('author low-SOH branch selects a label, not a numeric column')
        if soh>lower[0]:return 1
        found=np.flatnonzero((lower[:-1]>=soh)&(soh>lower[1:]))
        if len(found)!=1:raise ValueError('ambiguous author health index')
        return int(found[0]+1)

    def select(self,soh,energy_kwh):
        if not np.isfinite(energy_kwh) or energy_kwh<=0:raise ValueError('positive current capacity required')
        result={}; audit=dict(native_soh=float(soh),energy_kwh=float(energy_kwh),
            scaled=True,reference_soh=.95,health_input='native total SOH for both tables',table_sha256=self.hashes)
        for key in ('cal','cycl'):
            a=self.data[key];j=self.column(soh,key);r=self.column(.95,key)
            y=a[2:,j] if key=='cal' else a[30:58,j]
            reference=a[-1,r];factor=float(reference/y[-1])
            x=np.linspace(0,1,11) if key=='cal' else a[2:30,j]*energy_kwh
            result[key]=dict(x=x.copy(),raw=y.copy(),scaled=y*factor)
            audit[key]=dict(numeric_column=j,csv_column=j+1,representative_soh=float(a[0,j]),
                reference_numeric_column=r,reference_representative_soh=float(a[0,r]),
                scaling_factor=factor,raw_endpoint=float(y[-1]),scaled_endpoint=float(reference),
                x_range=[float(x[0]),float(x[-1])],x_unit='SOC pu' if key=='cal' else 'AC kWh per direction per 4h',
                y_unit='capacity loss pu per 15min' if key=='cal' else 'capacity loss pu per direction per 4h')
        result['audit']=audit
        return result


def blocks(n,*,toy=False):
    if toy:
        if type(n) is not int or not 1<=n<=8:raise ValueError('toy permits at most 8 intervals')
        return [list(range(n))]  # interface fixture ONLY, not a physical shortened aging model
    if n!=96:raise ValueError('B2-DA requires exactly 96 intervals')
    return [list(range(i,i+16)) for i in range(0,96,16)]


def evaluate(curves,soc,charge,discharge,groups):
    """Independent adjacent-segment evaluation; never extrapolate or clip damage."""
    coordinates=dict(cal=np.asarray(soc,float),
        cycl=np.array([[sum(charge[i] for i in g)*.25,sum(discharge[i] for i in g)*.25] for g in groups]))
    out={}
    for name in ('raw','scaled'):
        parts={}
        for key,values in coordinates.items():
            curve=curves[key];x=curve['x']
            if np.any(values<x[0]-1e-7) or np.any(values>x[-1]+1e-7):raise ValueError('PWL domain exceeded')
            parts[key]=np.interp(values,x,curve[name]).tolist()
        parts['total_pu']=float(np.sum(parts['cal'])+np.sum(parts['cycl']))
        out[name]=parts
    out['block_direction_energy_kwh']=coordinates['cycl'].tolist()
    return out
