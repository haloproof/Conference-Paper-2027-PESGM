"""Post-execution checks independent of the optimizer and teacher acceptance policy."""
import numpy as np

LABELS=('soc_lower','soc_upper','temperature_lower','temperature_upper','voltage_lower',
        'voltage_upper','charge_current','discharge_current','power_delivery','power_limit')


def execution_checks(result,teacher):
    """Use real fine-step extrema even when older report_only margins mask T."""
    ext=np.asarray(result['extrema'],float)
    margins=np.asarray(result['margins'],float).copy()
    if ext.ndim!=3 or ext.shape[1:]!=(2,4) or not np.isfinite(ext).all():
        raise ValueError('missing/nonfinite fine-step state extrema')
    lo=ext[:,0,:].min(0);hi=ext[:,1,:].max(0)
    config=teacher.config
    margins[2]=lo[1]-config['temperature_min_c']
    margins[3]=config['temperature_max_c']-hi[1]
    if margins.shape!=(10,) or not np.isfinite(margins).all():
        raise ValueError('invalid execution margins')
    findings=[name for name,value in zip(LABELS,margins) if value < -1e-9]
    state_findings=[x for x in findings if x not in ('power_delivery','power_limit')]
    native_temp=[float(teacher.battery.min_temperature),float(teacher.battery.max_temperature)]
    return dict(passed=not findings,findings=findings,state_findings=state_findings,
                margins=dict(zip(LABELS,margins.tolist())),
                fine_state_ranges=dict(zip(('soc','temperature_c','voltage_v','current_a_positive_discharge'),
                                           np.stack((lo,hi),1).tolist())),
                temperature_check_enforced=True,temperature_policy_c=[config['temperature_min_c'],config['temperature_max_c']],
                native_temperature_bounds_c=native_temp,
                native_temperature_violation=bool(lo[1]<native_temp[0]-1e-9 or hi[1]>native_temp[1]+1e-9),
                max_delivery_error_kw=max(0.,config['power_tolerance_kw']-margins[8]),
                role='post-execution diagnosis only; never selects or repairs a plan')
