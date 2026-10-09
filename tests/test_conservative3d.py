"""Conservative cell momentum/full-stress subproblem qualifications."""
import math,tempfile
from pathlib import Path
import pytest
import torch
from tensorfvm.conservative3d import ConservativeCylinder3DSolver
from tensorfvm.solver3d import Cylinder3DConfig
from tensorfvm.runtime import DistributedRuntime
from scripts.run_conservative_transport import manufactured


def build(device='cpu',runtime=None):
    return ConservativeCylinder3DSolver(Cylinder3DConfig(nx=32,ny=24,nz=4,max_steps=3,
        pressure_iterations=300,pressure_relative_tolerance=1e-12,time_step=.001,device=device),runtime)


def test_variable_mu_transpose_stress_manufactured_polynomial():
    s=build();x,y,z=s.mesh.centers()
    v=torch.stack((x*y,x*x,torch.zeros_like(x)),-1)
    computed=s.vector_divergence(s.stress_fluxes(v,1+x))
    exact=torch.stack((2*y,3+6*x,torch.zeros_like(x)),-1)
    mask=(x>4*s.mesh.dx)&(x<s.config.length-4*s.mesh.dx)&(y>4*s.mesh.dy)&(y<s.config.height-4*s.mesh.dy)&((x-s.config.cylinder_x)**2+(y-s.config.cylinder_y)**2>(s.config.cylinder_radius+4*max(s.mesh.dx,s.mesh.dy))**2)
    assert torch.count_nonzero(mask)>0
    torch.testing.assert_close(computed[mask],exact[mask],rtol=0,atol=1e-11)
    # mu*Laplacian would miss the x component and yield 2*mu in y.
    assert float(computed[mask][:,0].abs().min())>1


def test_smooth_fixed_region_manufactured_refines_second_order():
    e=[manufactured(n)['relative_l2_error'] for n in (32,64,128)]
    assert e[2] < e[1] < e[0]
    assert .20 < e[1]/e[0] < .30
    assert .20 < e[2]/e[1] < .30


def test_boundary_momentum_and_wall_force_are_actual_step_terms():
    s=build()
    for _ in range(3):
        h=s.step();r=s.last_transport
        assert h['momentum_ledger_residual_Ns']<1e-9
        assert h['boundary_momentum_residual_Ns']<1e-9
        assert h['continuity']<1e-6 and h['pressure_converged']==1
        expected=r['momentum_before']+s.config.time_step*(r['exterior_convective_force_on_fluid']+r['exterior_viscous_force_on_fluid']+r['exterior_pressure_force_on_fluid']-r['body_force'])
        torch.testing.assert_close(r['momentum_after'],expected,rtol=0,atol=1e-9)
        force=sum((w['total_force_on_body_N'].sum(0) for w in r['wall_facets']))
        torch.testing.assert_close(force,r['body_force'],rtol=0,atol=1e-12)
        for w in r['wall_facets']:
            torch.testing.assert_close(w['pressure_force_on_body_N'],w['pressure_pa'][:,None]*w['normal_fluid']*w['area_m2'])
            torch.testing.assert_close(w['viscous_force_on_body_N'],w['viscous_traction_on_body_pa']*w['area_m2'])
    # Hybrid scheme must expose the remaining incompatible center/face state.
    assert h['center_velocity_divergence']>.01
    assert h['hypothetical_reconstruction_defect_Ns']>.1


def test_shared_convective_fluxes_use_signed_upwind_face_velocity():
    s=build();v=torch.randn_like(s.velocity);faces=[torch.randn_like(u) for u in s.face_velocity]
    flux=s.convective_fluxes(v,faces)
    q=faces[0][...,1:-1]
    wanted=s.config.density*q[...,None]*torch.where((q>=0)[...,None],v[...,:-1,:],v[...,1:,:])
    torch.testing.assert_close(flux[0][...,1:-1,:],wanted,rtol=0,atol=0)


def _worker(rank,init_file,out):
    torch.distributed.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    try:
        s=build(runtime=DistributedRuntime.discover('cpu'))
        _,y,z=s.mesh.centers();s.velocity[...,1]+=.02*torch.sin(2*math.pi*z/s.config.span)*torch.sin(math.pi*y/s.config.height)
        for _ in range(3):s.step()
        fields=[]
        for field in [s.velocity,s.pressure,*s.face_velocity]:
            parts=[torch.empty_like(field) for _ in range(2)];torch.distributed.all_gather(parts,field);fields.append(torch.cat(parts))
        if rank==0:torch.save(dict(fields=fields,force=s.last_transport['body_force'],history=s.history),out)
    finally:torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(),reason='Gloo unavailable')
def test_distributed_real_transport_matches_single_rank_with_span_variation():
    s=build();_,y,z=s.mesh.centers();s.velocity[...,1]+=.02*torch.sin(2*math.pi*z/s.config.span)*torch.sin(math.pi*y/s.config.height)
    for _ in range(3):s.step()
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_worker,args=(tmp+'/init',tmp+'/out'),nprocs=2,join=True)
        d=torch.load(tmp+'/out',weights_only=True)
    for a,b in zip([s.velocity,s.pressure,*s.face_velocity],d['fields']):
        assert float((a-b).abs().max()) < 2e-11*max(1.,float(a.abs().max()))
    torch.testing.assert_close(s.last_transport['body_force'],d['force'],rtol=2e-11,atol=1e-10)
    assert all(h['pressure_converged']==1 and h['boundary_momentum_residual_Ns']<1e-9 for h in d['history'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_cuda_transport_matches_cpu():
    a,b=build(),build('cuda:0')
    for _ in range(3):a.step();b.step()
    torch.testing.assert_close(a.velocity,b.velocity.cpu(),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(a.last_transport['body_force'],b.last_transport['body_force'].cpu(),rtol=1e-10,atol=1e-10)
    assert b.history[-1]['boundary_momentum_residual_Ns']<1e-9
