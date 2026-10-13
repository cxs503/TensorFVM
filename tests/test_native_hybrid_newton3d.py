"""Verify the actual nonlinear Jacobian and accepted-state safeguards."""
import numpy as np
import pytest
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.verification.native_hybrid_ns import potential_solver,kovasznay_solver


@pytest.mark.parametrize('order',[1,2])
@pytest.mark.parametrize('open_boundary',[False,True])
def test_full_inertia_jacobian_matches_independent_residual_differences(order,open_boundary):
    m=pressure_mesh(2);rng=np.random.default_rng(17)
    fixed=m.neighbor<0
    if open_boundary:fixed[np.flatnonzero(fixed)[::2]]=False
    s=NativeHybridNavierStokes3D(m,fixed,0.,density=1.7,convection_order=order,
                                backflow_velocity=[-.4,.3,.6])
    x=rng.normal(size=len(s.state));v=rng.normal(size=len(x));h=1e-6
    assert np.min(np.abs(s.volume_flux(x)))>1e-5
    def residual(x):
        a,b=s._current_system(x)
        return a@x+b
    jac=s.viscous.full_matrix+s.convection.hybrid_jacobian(x,s.face_mean.matrix)
    difference=(residual(x+h*v)-residual(x-h*v))/(2*h)
    np.testing.assert_allclose(jac@v,difference,rtol=2e-7,atol=2e-8)
    # Derivative construction must not mutate the accepted state.
    np.testing.assert_array_equal(s.state,s.prescribed)


@pytest.mark.parametrize('mu',[.1,.001])
def test_unforced_three_dimensional_newton_converges_and_conserves(mu):
    m,s,e=potential_solver(2,mu)
    r=s.solve_newton(max_iterations=40,tolerance=1e-9)
    assert r['numerically_converged'],r['newton_history'][-1]
    assert r['newton_failure'] is None
    assert r['energy_balance_error']<1e-8
    assert r['local_momentum_balance_max']<1e-8
    assert r['global_momentum_balance_max']<1e-8
    for row in r['newton_history']:
        assert row['accepted']
        if max(row[k] for k in ('current_momentum_residual','divergence_max','velocity_change','pressure_change'))>=1e-9:
            assert row['merit_after']<=(1-1e-4*row['step_length'])*row['merit_before']
    np.testing.assert_array_equal(s.state[np.setdiff1d(np.arange(len(s.state)),s.viscous.free)],s.prescribed[np.setdiff1d(np.arange(len(s.state)),s.viscous.free)])


def test_newton_and_picard_solve_same_nonpotential_flow():
    answers=[]
    for method in ('picard','newton'):
        _,s,_=kovasznay_solver(3,symmetry_extrusion=True,coupled_linear_method='reused_lu')
        if method=='newton':r=s.solve_newton(max_iterations=50,tolerance=1e-9)
        else:r=s.solve(algorithm='coupled',max_outer_iterations=100,outer_relaxation=.7,tolerance=1e-9)
        assert r['numerically_converged']
        answers.append(r)
    for key in ('velocity','pressure','face_velocity','flux'):
        np.testing.assert_allclose(answers[0][key],answers[1][key],atol=2e-8)


def test_failed_line_search_preserves_accepted_velocity_and_pressure(monkeypatch):
    _,s,_=potential_solver(2,.1);old=s.state.copy();p=s.p.copy()
    original=s._coupled_linear
    # Deliberately invert an accurate Newton direction; this is ascent.
    def wrong(matrix,rhs):
        step,error=original(matrix,rhs)
        return -step,error
    monkeypatch.setattr(s,'_coupled_linear',wrong)
    r=s.solve_newton(max_backtracks=4)
    assert not r['numerically_converged']
    assert r['newton_failure']=='line_search_failed'
    np.testing.assert_array_equal(s.state,old);np.testing.assert_array_equal(s.p,p)


def test_iteration_limit_and_invalid_parameters_never_qualify_convergence():
    _,s,_=potential_solver(2,.001)
    for options in (dict(tolerance=0),dict(max_iterations=0),dict(max_backtracks=-1)):
        with pytest.raises(ValueError):s.solve_newton(**options)
    r=s.solve_newton(max_iterations=1)
    assert not r['numerically_converged']
    assert r['newton_failure']=='iteration_limit'
    assert not r['physical_benchmark_passed']


def test_native_stokes_initialization_enables_low_viscosity_newton_without_exact_seed():
    m,s,e=potential_solver(4,.001)
    init=s.initialize_stokes()
    assert init['converged'] and init['divergence_max']<1e-9
    assert not s.nonlinear_converged
    assert np.max(np.abs(s.state[:3*s.N].reshape(-1,3)-e['velocity']))>1e-4
    r=s.solve_newton(max_iterations=20,tolerance=1e-9)
    assert r['numerically_converged']
    assert r['global_momentum_balance_max']<1e-8
    assert r['energy_balance_error']<1e-8
    assert not s.body_force.any()


def test_already_solved_state_and_failed_linear_step_do_not_change_state(monkeypatch):
    _,s,_=potential_solver(2,.1)
    s.initialize_stokes();assert s.solve_newton(tolerance=1e-10)['numerically_converged']
    old=s.state.copy();p=s.p.copy()
    r=s.solve_newton(tolerance=1e-9)
    assert r['numerically_converged'] and r['newton_history'][0]['iteration']==0
    np.testing.assert_array_equal(old,s.state);np.testing.assert_array_equal(p,s.p)
    s.state[s.viscous.free[0]]+=.1;old=s.state.copy()
    def failed(*args):raise RuntimeError('forced linear failure')
    monkeypatch.setattr(s,'_coupled_linear',failed)
    with pytest.raises(RuntimeError,match='forced linear failure'):s.solve_newton()
    np.testing.assert_array_equal(old,s.state);np.testing.assert_array_equal(p,s.p)


def test_restart_can_require_complete_correction_even_at_small_residual(monkeypatch):
    _,s,_=potential_solver(2,.1)
    s.initialize_stokes();assert s.solve_newton(tolerance=1e-10)['numerically_converged']
    original=s._coupled_linear;calls=[]
    def count(a,b):
        calls.append(b.copy());return original(a,b)
    monkeypatch.setattr(s,'_coupled_linear',count)
    result=s.solve_newton(tolerance=1e-9,require_initial_correction=True)
    assert result['numerically_converged'] and result['require_initial_correction']
    assert len(calls)==1
    row=result['newton_history'][0]
    assert row['iteration']==1 and row['step_length']==1.
    assert row['velocity_change']<1e-9 and row['pressure_change']<1e-9
    assert row['true_linear_relative_residual']<1e-9
    old=s.state.copy();old_p=s.p.copy()
    def failed(*args):raise RuntimeError('terminal correction failed')
    monkeypatch.setattr(s,'_coupled_linear',failed)
    with pytest.raises(RuntimeError,match='terminal correction failed'):
        s.solve_newton(require_initial_correction=True)
    assert not s.nonlinear_converged
    np.testing.assert_array_equal(s.state,old);np.testing.assert_array_equal(s.p,old_p)


def test_damping_cannot_turn_large_full_correction_into_convergence(monkeypatch):
    _,s,_=potential_solver(2,.1)
    s.initialize_stokes();assert s.solve_newton(tolerance=1e-10)['numerically_converged']
    old=s.state.copy();old_p=s.p.copy()
    def large_direction(matrix,rhs):
        direction=np.zeros_like(rhs);direction[0]=1e-5
        return direction,0.
    monkeypatch.setattr(s,'_coupled_linear',large_direction)
    result=s.solve_newton(max_iterations=1,max_backtracks=25,tolerance=1e-9,require_initial_correction=True)
    assert not result['numerically_converged']
    assert result['newton_history'][0]['full_velocity_change']>1e-9
    assert result['newton_failure']=='line_search_failed'
    np.testing.assert_array_equal(s.state,old);np.testing.assert_array_equal(s.p,old_p)
