import copy
import numpy as np
import pytest
from tensorfvm.passive_feedback_audit import energy,case
from tensorfvm.fourth_round_audit import unqualified,require

def uniform():
    weights=np.array([4/9]+[1/9]*4+[1/36]*4);f=weights[:,None,None]*np.ones((9,4,4))
    return {'fluid':{'config':dict(density_kg_m3=1000,thickness_m=.2,dx_m=.01,dt_s=.001),'solver':{'f':f.tolist(),'center':[1,1],'config':dict(nx=4,ny=4,radius=.1)}},'velocity_m_s':[0,0],'body_mass_kg':5}
def test_uniform_reference_has_zero_energy():assert abs(energy(uniform())['total_J'])<1e-12
def test_body_kinetic_energy_exact():
    r=uniform();r['velocity_m_s']=[.03,0];assert abs(energy(r)['kinetic_J']-.00225)<1e-12

def test_density_compression_energy_positive():
    r=uniform();f=np.asarray(r['fluid']['solver']['f']);f[:,0,0]*=2;r['fluid']['solver']['f']=f.tolist();assert energy(r)['density_free_energy_J']>0

def test_negative_raw_population_rejected():
    r=uniform();r['fluid']['solver']['f'][0][0][0]=-1
    with pytest.raises(ValueError):energy(r)
def test_false_qualification_rejected():
    with pytest.raises(ValueError):unqualified({'physical_accuracy_qualified':True})

def test_rejected_candidate_within_budget_rejected():
    s=uniform();r={'physical_accuracy_qualified':False,'completed_requested_duration':False,'initial':{'solver':s},'accepted':{'solver':s,'initial_energy_J':0.,'relative_tolerance':.001,'absolute_tolerance_J':1e-12},'rejected_candidate':s}
    with pytest.raises(ValueError,match='rejection without excess'):case(r)
