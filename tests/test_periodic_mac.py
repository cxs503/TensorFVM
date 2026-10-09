"""Staggered momentum/energy invariants, solver failure and hardware scope."""
import json
from pathlib import Path
import pytest
import torch
from tensorfvm.periodic_mac import PeriodicMACConfig,PeriodicMACSolver
from tensorfvm.runtime import DistributedRuntime
from scripts.run_periodic_mac import random_curl,numerical_qualified


def build(device='cpu',**kwargs):
    return PeriodicMACSolver(PeriodicMACConfig(nx=8,ny=8,nz=8,device=device,**kwargs))


def test_periodic_face_divergence_gradient_are_adjoint_and_checkerboard_resolved():
    s=build();g=torch.Generator().manual_seed(91)
    u=torch.randn(s.velocity.shape,generator=g,dtype=torch.float64)
    p=torch.randn(s.shape,generator=g,dtype=torch.float64)
    assert abs(float((p*s.divergence(u)).sum()+(u*s.gradient(p)).sum()))<1e-12
    k,j,i=torch.meshgrid(torch.arange(8),torch.arange(8),torch.arange(8),indexing='ij')
    checker=(-1.)**(k+j+i)
    assert float(s.gradient(checker).abs().min())>0


def test_projection_is_orthogonal_nonincreasing_energy_and_keeps_momentum():
    s=build();g=torch.Generator().manual_seed(2)
    u=torch.randn(s.velocity.shape,generator=g,dtype=torch.float64)
    corrected,p,r=s.project(u)
    assert r['pressure_converged'] and r['max_divergence_s_inv']<1e-10
    assert float(s.kinetic_energy(corrected))<float(s.kinetic_energy(u))
    torch.testing.assert_close(s.momentum(u),s.momentum(corrected),rtol=0,atol=1e-12)
    removed=u-corrected
    defect=float(s.kinetic_energy(u)-s.kinetic_energy(corrected)-s.kinetic_energy(removed))
    assert abs(defect)<1e-10
    assert abs(float(p.mean()))<1e-12


def test_variable_mu_complete_stress_has_discrete_negative_work():
    s=build();random_curl(s)
    normal,shear,edge=s.full_stress(s.velocity)
    loss=0.
    for a in range(3):
        strain=(s.velocity[a]-torch.roll(s.velocity[a],1,dims=-(a+1)))/s.spacing[a]
        loss+=float((2*s.dynamic_viscosity*strain.square()).sum())
        for b in range(a+1,3):
            strain=(torch.roll(s.velocity[a],-1,dims=-(b+1))-s.velocity[a])/s.spacing[b]+(torch.roll(s.velocity[b],-1,dims=-(a+1))-s.velocity[b])/s.spacing[a]
            loss+=float((edge[a,b]*strain.square()).sum())
    power=float((s.velocity*s.stress_divergence((normal,shear,edge))).sum())
    assert loss>0 and abs(power+loss)<1e-12


def test_dual_cv_convection_zero_work_and_shared_flux_momentum():
    s=build();random_curl(s)
    s.velocity += torch.tensor([.15,-.08,.03],dtype=torch.float64)[:,None,None,None]
    force=s.convective_divergence(s.convective_fluxes(s.velocity))
    assert float(force.sum((1,2,3)).abs().max())<1e-12
    assert abs(float((s.velocity*force).sum()))<1e-12


def test_actual_midpoint_step_meets_energy_momentum_and_mass_gates():
    s=build();random_curl(s)
    s.velocity += torch.tensor([.15,-.08,.03],dtype=torch.float64)[:,None,None,None]
    for _ in range(3):s.step()
    assert numerical_qualified(s.history)
    assert s.history[-1]['kinetic_after_J']<s.history[0]['kinetic_before_J']
    assert s.last_raw['accepted'] is True


def test_true_nonlinear_failure_rejects_without_advancing():
    s=build(time_step_s=.5,nonlinear_iterations=4);initial=random_curl(s)
    with pytest.raises(RuntimeError):s.step()
    assert s.last_raw['accepted'] is False
    assert s.last_raw['record']['nonlinear_residual_m_s']>s.config.nonlinear_tolerance
    assert torch.equal(s.velocity,initial) and s.time==0 and s.history==[]


def test_distributed_is_explicitly_rejected_not_replicated():
    runtime=DistributedRuntime(torch.device('cpu'),0,2,0,False)
    with pytest.raises(NotImplementedError):PeriodicMACSolver(PeriodicMACConfig(),runtime)
    assert PeriodicMACSolver.distributed_supported is False


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_gpu_midpoint_matches_cpu_face_momentum():
    a,b=build(),build('cuda:0');random_curl(a);random_curl(b)
    for _ in range(3):a.step();b.step()
    torch.testing.assert_close(a.velocity,b.velocity.cpu(),rtol=1e-11,atol=1e-12)
    assert numerical_qualified(a.history) and numerical_qualified(b.history)


def test_published_refinement_gates_preserve_coarse_mms_failure():
    study=json.loads((Path(__file__).resolve().parents[1]/'docs/periodic-mac/study.json').read_text())
    assert study['refinement_order_passed']
    assert study['variable_mu_mms'][0]['error_3_percent_passed'] is False
    assert study['mms_all_grids_3_percent_passed'] is False
    assert study['mms_finest_grid_3_percent_passed'] is True
    assert study['distributed_supported'] is False and study['physical_accuracy_qualified'] is False
