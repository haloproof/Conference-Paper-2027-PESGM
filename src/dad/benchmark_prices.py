"""Continuous 15-minute price windows, independent of daily reporting boundaries."""
import numpy as np
import pandas as pd
from .dispatch_continuous_prices import coverage, utc_instant


class PriceWindows:
    def __init__(self,frame,spec):
        self.spec=spec;self.start=utc_instant(spec['start_utc'])
        self.repeat=spec['price_scenario']=='repeat_history_year'
        if spec['price_scenario'] not in ('historical','repeat_history_year'):
            raise ValueError('explicit historical/repeat_history_year prices required')
        if self.repeat:
            a,b=utc_instant(spec['repeat_start_utc']),utc_instant(spec['repeat_end_utc'])
            if b-a!=pd.Timedelta(days=365):raise ValueError('repeat source must be exactly 365 elapsed 24-hour days')
            frame=frame[(frame.start_utc>=a)&(frame.start_utc<b)].copy()
            if len(frame)!=365*96 or frame.start_utc.iloc[0]!=a or frame.end_utc.iloc[-1]!=b:
                raise ValueError('incomplete repeat price year')
            if self.start!=a:raise ValueError('repeat simulation must start at the source-year anchor')
            self.offset=0
        else:
            matches=np.flatnonzero((frame.start_utc==self.start).to_numpy())
            if len(matches)!=1:raise ValueError('start not in prepared historical prices')
            self.offset=int(matches[0])
        self.frame=frame.reset_index(drop=True)
        self.values=self.frame.price_eur_kwh.to_numpy(float)

    def window(self,index,horizon):
        if type(index) is not int or index<0 or type(horizon) is not int or horizon<1:
            raise ValueError('invalid price window indices')
        indices=self.offset+index+np.arange(horizon)
        if self.repeat:indices%=len(self.values)
        elif indices[-1]>=len(self.values):
            raise ValueError('historical prices do not cover full MPC lookahead; no horizon truncation')
        return self.values[indices].copy(),[str(self.frame.start_utc.iloc[int(i)]) for i in indices]

    def timestamp(self,index):
        return str(self.start+pd.Timedelta(minutes=15*index))


def load_windows(path,spec,horizon,execute):
    frame,audit=coverage(path)
    windows=PriceWindows(frame,spec)
    windows.window(0,horizon)
    windows.window(spec['max_days']*96-execute,horizon)
    return windows,audit
