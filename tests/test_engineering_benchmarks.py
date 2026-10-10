import numpy as np
import pytest
import torch
from tensorfvm.annular import AnnularConfig,solve
from tensorfvm.verification.engineering_audit import annular_metrics
from tensorfvm.wall_functions import friction_velocity,wall_traction,u_plus_at_y_plus


def test_annular_periodic_mesh_and_full_diffusion_balance():
    torch.set_num_threads(1);r=solve(AnnularConfig(ntheta=64,nr=16));v=annular_metrics(r['fields'],r['config'])
    assert v['velocity_volume_l2']<.03
    assert v['true_linear_residual']<2e-10
    assert r['diagnostics']['full_matrix_unknowns']==64*16


def test_corrupt_annular_face_flux_rejected():
    torch.set_num_threads(1);r=solve(AnnularConfig(ntheta=32,nr=8));r['fields']['axial_momentum_flux_n_m'][0]+=.01
    with pytest.raises(ValueError,match='independent reconstruction mismatch'):annular_metrics(r['fields'],r['config'])


def test_spalding_inverse_viscous_buffer_and_log_regions():
    yp=np.array([.01,.1,1.,5.,30.,100.,1000.,1e5]);ut=.05;nu=1e-5;up=u_plus_at_y_plus(yp)
    recovered=friction_velocity(up*ut,yp*nu/ut,nu)
    assert np.max(abs(recovered-ut)/ut)<1e-10
    t=wall_traction(np.column_stack((up*ut,np.zeros(len(up)))),yp*nu/ut,1000.,.01)
    assert np.max(abs(t['traction_pa'][:,0]+1000*ut**2))<1e-9
    assert np.all(t['traction_pa'][:,0]<0)


def test_wall_law_zero_speed_and_invalid_distance():
    assert friction_velocity(0.,1e-3,1e-5)==0
    with pytest.raises(ValueError):friction_velocity(1.,0.,1e-5)
    with pytest.raises(ValueError):wall_traction([float('nan'),0.],.001,1.,.001)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_full_csr_cuda_matches_cpu_without_fallback():
    torch.set_num_threads(1);c=AnnularConfig(ntheta=64,nr=16)
    cpu=solve(c,'cpu','torch-cg');gpu=solve(c,'cuda','torch-cg')
    assert gpu['diagnostics']['device'].startswith('cuda')
    assert gpu['diagnostics']['silent_cpu_solver_fallback'] is False
    assert np.max(abs(cpu['fields']['axial_velocity_m_s']-gpu['fields']['axial_velocity_m_s']))<1e-10
    assert annular_metrics(gpu['fields'],gpu['config'])['true_linear_residual']<2e-10


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_cuda_wall_traction_is_dimensional_resisting_force():
    from tensorfvm.wall_functions import torch_wall_traction
    yp=np.array([.1,1.,30.,100.,1000.]);up=u_plus_at_y_plus(yp)
    v=torch.from_numpy(np.column_stack((up*.05,np.zeros(5)))).cuda()
    t=torch_wall_traction(v,torch.from_numpy(yp*1e-5/.05).cuda(),1000.,.01)
    assert t['traction_pa'].device.type=='cuda'
    assert float((t['traction_pa'][:,0]+2.5).abs().max())<1e-9
