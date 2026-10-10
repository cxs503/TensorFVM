"""Numerical operator contracts and actual CPU/CUDA consistency."""
import numpy as np
import torch
import pytest
from tensorfvm.solver import SolverConfig
from tensorfvm.curved_channel import reference
from tensorfvm.sparse_simple import SparseBodyFittedSolver


def test_full_curved_simple_matches_continuous_pressure_velocity():
    torch.set_num_threads(1)
    c=SolverConfig(mesh_type='curved-channel',nx=32,ny=16,length=2,height=1,cylinder_radius=None,inlet_profile='parabolic',reynolds=1,tolerance=1e-6,max_iterations=500)
    s=SparseBodyFittedSolver(c);s.coupled_coarse_shape=(16,8);u,p,f=reference(s.mesh.centers.reshape(-1,2),c);s.momentum_body_force=f
    for _ in range(c.max_iterations):
        s.step()
        if s.converged:break
    assert s.converged
    assert torch.linalg.vector_norm(s.velocity-u)/torch.linalg.vector_norm(u)<.03
    assert torch.linalg.vector_norm(s.p-p)/torch.linalg.vector_norm(p)<.03
    assert all(h['true_residual']<=h['target']*1.05 for h in s.linear_history)
    assert float((s.T.square().sum(-1)).max())>1e-6
    assert all(q['operator_relative_defect']<2e-11 and q['linear_true_residual']<1e-9 for q in s.coarse_history)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA required')
def test_stretched_sa_true_pressure_operator_and_device_equivalence():
    torch.set_num_threads(1);values=[]
    for device in ['cpu','cuda']:
        c=SolverConfig(mesh_type='flat-plate',turbulence_model='spalart-allmaras',nx=32,ny=28,length=1,height=.2,cylinder_radius=None,reynolds=100000,device=device,velocity_relaxation=.3,pressure_relaxation=.2,turbulence_relaxation=.3,pseudo_time_step=.02)
        s=SparseBodyFittedSolver(c)
        for _ in range(5):s.step()
        assert all(h['actual_device'].startswith(device) for h in s.linear_history)
        values.append(s.p.detach().cpu().numpy())
    assert np.linalg.norm(values[0]-values[1])/np.linalg.norm(values[0])<=1e-6
