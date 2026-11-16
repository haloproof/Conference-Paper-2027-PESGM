import copy

import numpy as np
import pytest
# SimSES 2.1.1 requires battery initialization before degradation submodules.
from dad.teacher import SCHEMA, Teacher, native_fingerprint
from simses.degradation.cycle_detector import HalfCycleDetector
from simses.model.degradation.sony_lfp_cyclic import SonyLFPCyclicDegradation

from dad.common import Budget, digest, load_config, write_json


@pytest.mark.teacher
def test_native_accounting_restore_convergence():
    c=load_config('configs/smoke.json')
    budget=Budget(**c['limits'])
    teacher=Teacher(c['teacher'],budget)
    original=teacher.checkpoint()
    u=np.array([0.,-2.,-2.,0.,1.8,1.8,0.,-100.])
    dt=np.full(len(u),60/3600)
    # Trajectory 1: rest, monotone charge, rest, reversal, discharge, reversal.
    # Resume the same first trajectory at a committed interval, with no new query.
    from dad.common import LimitReached
    saved=[]
    def interrupt(state):
        saved[:]=[copy.deepcopy(state)]
        if len(state['states'])==3:
            raise LimitReached('audit replay checkpoint interruption')
    with pytest.raises(LimitReached):
        teacher.rollout(u,25.,dt,record_checkpoints=True,checkpoint_sink=interrupt)
    full=teacher.rollout(u,25.,dt,record_checkpoints=True,resume_state=saved[0])
    assert teacher.checkpoint()==original
    assert full['x'][2,0]>0.5
    assert full['x'][5,0]<full['x'][3,0]
    assert np.ptp(full['x'][:,1])>0
    assert np.max(np.abs(full['delivered'][:-1]-u[:-1]))<1e-8
    assert full['margins'][-2]<0 and not full['feasible']
    assert abs(full['delivered'][-1])<abs(u[-1]), 'native clipping must not masquerade as feasible command'
    assert full['damage'][0]>0 and full['damage'][1]>0
    # Trajectory 2: JSON-restored mid-half-cycle history, crossing next reversal.
    import json
    teacher.restore(json.loads(json.dumps(full['checkpoints'][1])))
    suffix=teacher.rollout(u[2:],25.,dt[2:])
    assert suffix['checkpoint']==full['checkpoint']
    assert np.array_equal(suffix['x'],full['x'][2:])
    # Every cut uses the same ledger; no flush is committed to the execution state.
    teacher.restore(original)
    last=teacher.ledger()
    increments=[]
    for checkpoint in full['checkpoints']:
        teacher.restore(checkpoint)
        before=teacher.checkpoint()
        now=teacher.ledger()
        assert before==teacher.checkpoint()
        increments.append(now-last)
        last=now
    np.testing.assert_allclose(np.sum(increments,axis=0),full['damage'],atol=1e-15)
    # Trajectory 3: same path at dt/2, not another campaign.
    fine_config=copy.deepcopy(c['teacher']); fine_config['internal_dt_s']=5.
    fine=Teacher(fine_config,budget).rollout(u,25.,dt)
    error=np.max(np.abs(full['x']-fine['x']),axis=0)
    assert error[0]<1e-5 and error[1]<0.02 and error[2]<0.1 and error[3]<0.01
    assert budget.queries==3 and budget.steps==180
    # Algebraic detector witness: slower continuation can reduce residual valuation.
    detector=HalfCycleDetector(0.5)
    detector.step(0.51,36.)
    cyclic=SonyLFPCyclicDegradation()
    q1=cyclic.update_capacity(teacher.battery.state,detector._make_half_cycle(),0.)
    detector.step(0.51001,360.)
    q2=cyclic.update_capacity(teacher.battery.state,detector._make_half_cycle(),0.)
    assert q2-q1<0, 'signed residual correction cannot be a nonnegative-only label'
    # Native sign/unit conversion and overcommand detection, without extra integration.
    assert teacher.ac_to_native(1.)<0 and teacher.ac_to_native(-1.)>0
    assert teacher.margins(1e6)[-2]<0
    write_json('reports/teacher_certificate.json',dict(passed=True,schema=SCHEMA,
        adapter_sha256=digest(__import__('inspect').getfile(Teacher)),
        native_fingerprint=native_fingerprint(),trajectories=budget.queries,integration_steps=budget.steps,
        convergence_max_abs_state=error.tolist(),negative_residual_witness=q2-q1,
        thermal_check='10 s vs 5 s on one 8-minute path; no universal convergence claim',
        budget=budget.report()))
