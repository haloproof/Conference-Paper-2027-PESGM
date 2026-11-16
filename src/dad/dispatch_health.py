"""Native SOH and residual valuation remain distinct; no simulation here."""
import numpy as np


def finite_tree(value):
    if isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,(list,tuple,np.ndarray)):
        for v in value: finite_tree(v)
    elif isinstance(value,(int,float,np.number)) and not np.isfinite(value):
        raise FloatingPointError('nonfinite native checkpoint or damage ledger')


def health(teacher,checkpoint):
    finite_tree(checkpoint)
    obj=teacher.fork(); obj.restore(checkpoint)
    ledger=obj.ledger(); finite_tree(ledger)
    state=checkpoint['battery']
    if not np.isclose(state['soh_Q'],1-ledger[:2].sum(),atol=1e-10,rtol=0):
        raise ArithmeticError('native SOH does not match settled degradation')
    return dict(native_soc=float(state['soc']),temperature_c=float(state['T']),
        native_soh=float(state['soh_Q']),time_s=float(checkpoint['time_s']),
        ledger_components_pu=ledger.tolist(),ledger_total_damage_pu=float(ledger.sum()),
        residual_valued_soh=float(1-ledger.sum()),
        retirement_basis='native_soh: excludes uncommitted residual half-cycle valuation')
