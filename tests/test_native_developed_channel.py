import numpy as np
import pytest
from tensorfvm.verification.native_developed_channel import developed_channel_flow


def test_pressure_only_channel_has_no_body_force_and_refines_velocity():
    errors=[]
    for n in (8,16):
        m,s,info=developed_channel_flow(n,nu=.1,length=.1,span=.1)
        r=s.solve_newton(tolerance=1e-9)
        y=m.cell_centers[:,1];exact=(y-.5*y*y)/.1
        error=np.sqrt(np.sum(m.cell_volumes*(r['velocity'][:,0]-exact)**2)/np.sum(m.cell_volumes*exact**2))
        errors.append(error)
        assert r['numerically_converged']
        np.testing.assert_array_equal(s.body_force,0.)
        np.testing.assert_allclose(r['pressure'],.1-m.cell_centers[:,0],atol=2e-11)
        np.testing.assert_allclose(r['total_traction_integrated'][info['wall'],0].sum(),-.01,atol=2e-12)
        assert abs(r['energy_balance_error'])<1e-8
    assert errors[-1]<.01 and errors[-1]<errors[0]/3


def test_degenerate_extrusion_rejected_before_solve():
    with pytest.raises(ValueError,match='rigid motion'):
        developed_channel_flow(8,axial=1,spanwise=1)


def test_simplec_zero_pressure_recovers_independent_coupled_momentum():
    """Exercise stretched variable-viscosity momentum, without seeded pressure."""
    from tensorfvm.verification.native_developed_channel import developed_channel_rans
    from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
    m,s,_=developed_channel_rans(16,axial=2,length=.1,span=.1)
    s._refresh_viscosity();f=s.flow
    f.initialize_stokes();expected_u=f.state.copy();expected_p=f.p.copy()
    f.A=f.viscous.matrix;f.rhs=(f.load-f.viscous.full_matrix@f.prescribed)[f.viscous.free]
    f.momentum_relaxation_basis='diagonal';f.p.fill(0.)
    NativeHybridStokes3D.solve(f,algorithm='simplec',relaxation=1.,pressure_relaxation=1.,tolerance=1e-11)
    assert f.converged and len(f.history)>1
    assert np.linalg.norm(f.state-expected_u)/np.linalg.norm(expected_u)<1e-8
    np.testing.assert_allclose(f.p,expected_p,atol=2e-7,rtol=0.)
