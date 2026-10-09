"""Discrete mass/pressure qualifications, deliberately separate from LES accuracy."""
import torch
from tensorfvm.solver3d import Cylinder3DSolver, Cylinder3DConfig
from tensorfvm.benchmark_cylinder3d import smoke_qualified


def solver(device='cpu'):
    return Cylinder3DSolver(Cylinder3DConfig(nx=32, ny=24, nz=4, max_steps=3,
        pressure_iterations=150, time_step=.001, device=device))


def test_checkerboard_is_not_pressure_null_mode():
    s = solver()
    k,j,i = torch.meshgrid(torch.arange(4),torch.arange(24),torch.arange(32),indexing='ij')
    p = ((-1.)**(k+j+i)).masked_fill(~s.mesh.fluid,0)
    assert torch.linalg.vector_norm(s._pressure_operator(p)) > 100
    assert all(float(x.abs().max()) > 0 for x in s._face_gradient(p))


def test_projection_closes_each_cell_and_solid_faces():
    s = solver()
    for _ in range(3):
        metric = s.step()
        assert metric['pressure_converged'] == 1
        assert metric['continuity'] < 1e-6
        assert metric['boundary_mass_imbalance'] < 1e-6
        u,v,w = s.face_velocity
        assert torch.count_nonzero(u[...,1:-1][~s._face_x]) == 0
        assert torch.count_nonzero(v[:,1:-1][~s._face_y]) == 0
        assert torch.count_nonzero(w[~s._face_z]) == 0
        divergence = s._face_divergence(s.face_velocity)
        external = (u[...,-1]-u[...,0]).sum()*s.mesh.dy*s.mesh.dz
        assert abs(float(divergence.sum()*s.mesh.dx*s.mesh.dy*s.mesh.dz-external)) < 1e-12
    # Center reconstruction is diagnostic, not the divergence-free flux field.
    assert s.history[-1]['center_velocity_divergence'] > 1e-3


def test_pressure_true_residual_and_gradient_projection():
    s=solver();s.step()
    target=-s.config.density/s.config.time_step*s._face_divergence(s.last_tentative_faces)
    true=float(torch.linalg.vector_norm(target-s._pressure_operator(s.pressure)))
    assert abs(true-s.last_pressure_solve.final_residual) < 1e-10
    for predicted,projected,g in zip(s.last_tentative_faces,s.face_velocity,s._face_gradient(s.pressure)):
        torch.testing.assert_close(projected,predicted-s.config.time_step/s.config.density*g,rtol=0,atol=0)


def test_smoke_rejects_earlier_failure_or_continuity_even_if_final_good():
    good=dict(cfl=.1,pressure_converged=1,continuity=1e-9,boundary_mass_imbalance=1e-9)
    assert smoke_qualified([good])
    for change in [dict(pressure_converged=0),dict(continuity=.01),dict(boundary_mass_imbalance=.01),dict(cfl=1.01)]:
        assert not smoke_qualified([dict(good,**change),good])
    assert not smoke_qualified([])
