import copy
import numpy as np
import pytest
from tensorfvm.fourth_round_audit import local_case,pop,feedback_case,unqualified

def fixture():
    f=np.ones((9,2,2));return dict(initial={'f':f.tolist()},final={'f':f.tolist(),'history':[dict(wall_impulse_on_solid=[0,0],conversion_impulse_on_solid=[0,0],global_mass_rescale=False,maximum_transfer_radius=4)]},restart_bitwise=True)
def test_raw_population_mass_and_momentum():
    r=local_case(fixture());assert r['mass_residual_lattice']==0 and r['momentum_residual_lattice']==0

def test_momentum_tampering_rejected():
    r=fixture();r['final']['history'][0]['conversion_impulse_on_solid']=[.01,0]
    with pytest.raises(ValueError):local_case(r)

def test_global_rescale_rejected():
    r=fixture();r['final']['history'][0]['global_mass_rescale']=True
    with pytest.raises(ValueError):local_case(r)

def test_physical_badge_rejected():
    r=fixture();r['physical_accuracy_qualified']=True
    with pytest.raises(ValueError):local_case(r)

def test_negative_distribution_rejected():
    r=fixture();r['final']['f'][0][0][0]=-1
    with pytest.raises(ValueError):local_case(r)

def test_stale_feedback_pose_rejected():
    r={'physical_accuracy_qualified':False,'flexible_suboff_qualified':False,'energy_accuracy_qualified':False,'state':{'body_mass_kg':1,'initial_center_m':[0,0],'fluid':{'config':{}},'history':[{'exchange':{'physical_accuracy_qualified':False,'time_start_s':0,'center_start_m':[1,0],'velocity_held_m_s':[0,0]}}]}}
    with pytest.raises(ValueError,match='reconstructed'):feedback_case(r)
