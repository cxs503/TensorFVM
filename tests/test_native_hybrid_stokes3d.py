import numpy as np
import pytest

from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra
from tests.test_native_hybrid_viscous3d import thin_mixed_mesh
from tests.test_native_rans3d import box


@pytest.mark.parametrize('components', [1, 3])
@pytest.mark.parametrize('factory', [two_tetrahedra, thin_mixed_mesh])
def test_hybrid_mixed_boundary_affine_traction_and_energy(components, factory):
    m = factory(); boundary = m.neighbor < 0
    fixed = boundary.copy(); free_face = np.flatnonzero(boundary)[-1]; fixed[free_face] = False
    gradient = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])[:components]
    value = lambda x: 1.2+x @ gradient.T
    stress = gradient if components == 1 else gradient+gradient.T-2/3*np.trace(gradient)*np.eye(3)
    normal = m.area_vectors/np.linalg.norm(m.area_vectors, axis=1)[:, None]
    op = NativeHybridViscous3D(m, 2.3, components=components, dirichlet_mask=fixed)
    r = op.solve(boundary_values=value(m.face_centers), boundary_traction=2.3*normal @ stress.T)
    np.testing.assert_allclose(r['cell_values'], value(m.cell_centers), atol=2e-10)
    np.testing.assert_allclose(r['face_flux'], 2.3*m.area_vectors @ stress.T, atol=2e-10)
    assert r['energy_balance_error'] < 1e-9


@pytest.mark.parametrize('algorithm', ['coupled', 'simple', 'simplec'])
@pytest.mark.parametrize('factory', [two_tetrahedra, thin_mixed_mesh])
def test_hybrid_flow_affine_mixed_total_traction_and_pressure(algorithm, factory):
    m = factory(); b = m.neighbor < 0; fixed = b.copy(); fixed[np.flatnonzero(b)[-1]] = False
    gradient = np.array([[.2, .4, .1], [-.3, -.1, .2], [.1, .1, -.1]])
    bd = .3+m.face_centers @ gradient.T
    stress = gradient+gradient.T
    normal = m.area_vectors/np.linalg.norm(m.area_vectors, axis=1)[:, None]
    traction = normal @ stress.T-.7*normal
    s = NativeHybridStokes3D(m, fixed, bd, boundary_traction=traction)
    r = s.solve(algorithm=algorithm, tolerance=1e-11, max_iterations=8000)
    assert r['numerically_converged'], r['history'][-1]
    np.testing.assert_allclose(r['velocity'], .3+m.cell_centers @ gradient.T, atol=2e-7)
    np.testing.assert_allclose(r['pressure'], .7, atol=2e-7)
    np.testing.assert_allclose(r['total_traction_integrated'],
                               traction*np.linalg.norm(m.area_vectors, axis=1)[:, None], atol=2e-7)
    assert r['energy_balance_error'] < 1e-8
    assert r['interface_traction_defect_max'] < 1e-8
    assert not r['physical_benchmark_passed']


def pressure_channel(n):
    m = box(n); x, y, _ = m.face_centers.T
    boundary = m.neighbor < 0
    ends = boundary & ((x < 1e-10) | (x > 1-1e-10))
    velocity = np.zeros((len(x), 3)); velocity[:, 0] = y*(1-y)
    normal = m.area_vectors/np.linalg.norm(m.area_vectors, axis=1)[:, None]
    traction = -2*(1-x[:, None])*normal
    traction[:, 1] += (1-2*y)*normal[:, 0]
    return m, NativeHybridStokes3D(m, boundary & ~ends, velocity, boundary_traction=traction)


def test_pressure_driven_channel_refines_both_velocity_and_pressure():
    errors = []
    for n in (2, 4, 8):
        m, solver = pressure_channel(n)
        r = solver.solve(algorithm='coupled', tolerance=1e-10)
        assert r['numerically_converged']
        x, y, _ = m.cell_centers.T
        exact_u = np.zeros((len(x), 3)); exact_u[:, 0] = y*(1-y)
        exact_p = 2*(1-x)
        errors.append([np.linalg.norm(r['velocity']-exact_u)/np.linalg.norm(exact_u),
                       np.linalg.norm(r['pressure']-exact_p)/np.linalg.norm(exact_p)])
        assert r['interface_traction_defect_max'] < 1e-10
        assert r['energy_balance_error'] < 1e-10
        np.testing.assert_allclose(r['total_traction_integrated'][m.neighbor < 0].sum(0), 0, atol=1e-10)
    assert np.all(np.diff(np.asarray(errors)[:, 0]) < 0), errors
    assert errors[-1][1] < errors[-2][1], errors
    assert max(errors[-1]) < .03, errors


def test_simple_and_simplec_match_coupled_for_nonconstant_pressure():
    m, solver = pressure_channel(3)
    reference = solver.solve(algorithm='coupled')
    for algorithm in ('simple', 'simplec'):
        _, solver = pressure_channel(3)
        r = solver.solve(algorithm=algorithm, tolerance=1e-10)
        assert r['numerically_converged']
        for key in ('velocity', 'face_velocity', 'pressure', 'flux'):
            np.testing.assert_allclose(r[key], reference[key], atol=2e-8)


def test_closed_flow_gauge_and_pressure_work():
    m = box(2); force = np.column_stack((m.cell_centers[:, 1], -m.cell_centers[:, 0], np.zeros(len(m.cells))))
    answers = []
    for gauge in (0., 4.):
        solver = NativeHybridStokes3D(m, m.neighbor < 0, body_force=force, gauge_value=gauge)
        r = solver.solve(algorithm='simplec', tolerance=1e-10)
        assert r['numerically_converged']
        assert abs(r['pressure_work']) < 1e-12
        assert r['energy_form'] > 0
        assert r['energy_balance_error'] < 1e-9
        answers.append(r)
    np.testing.assert_allclose(answers[0]['velocity'], answers[1]['velocity'], atol=1e-9)
    np.testing.assert_allclose(answers[1]['pressure']-answers[0]['pressure'], 4., atol=1e-9)


def test_rejected_candidate_preserves_hybrid_state(monkeypatch):
    import tensorfvm.native_hybrid_stokes3d as module
    _, solver = pressure_channel(2)
    state, pressure = solver.state.copy(), solver.p.copy()
    original = module.splu
    class BadFactor:
        def __init__(self, matrix): self.factor = original(matrix); self.n = matrix.shape[0]
        def solve(self, rhs):
            return np.full(self.n, np.nan) if self.n == solver.N else self.factor.solve(rhs)
    monkeypatch.setattr(module, 'splu', BadFactor)
    with pytest.raises(RuntimeError, match='accepted state preserved'):
        solver.solve()
    np.testing.assert_array_equal(state, solver.state)
    np.testing.assert_array_equal(pressure, solver.p)
    assert not solver.history


def test_hybrid_boundary_validation():
    m = box(2)
    with pytest.raises(ValueError, match='boundary faces'):
        NativeHybridStokes3D(m, np.ones(len(m.owner), bool))
    with pytest.raises(ValueError, match='unbalanced'):
        NativeHybridStokes3D(m, m.neighbor < 0, m.face_centers)
