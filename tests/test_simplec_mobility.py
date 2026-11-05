"""Verification guards for the collocated SIMPLEC mobility (Van Doormaal & Raithby, 1984).

Why this file exists
--------------------
``tensorfvm.solver.SimpleSolver._predict`` is the only place where the SIMPLEC
scheme physically differs from SIMPLE::

    d_SIMPLE  = A / (aP/alpha)
    d_SIMPLEC = A / max(aP/alpha - sum(aN), simplec_floor_fraction*aP)

``src/tensorfvm/verification/simplec_channel.py`` audits the *end-to-end*
consequence of that change (developed-channel gates, iteration count, the
alpha_p = 1 breakdown of SIMPLE), but nothing checked the **definition itself**
against the 1984 paper, nothing pinned the denominator floor (that audit records
``limited_dofs = 0`` so the floor branch never executed), and no test in ``tests/``
referenced the module at all, so a regression in the mobility would only surface
as a slow drift in a report.

Each test below is anchored on something independent of the implementation:

1. ``test_simple_mobility_is_face_area_over_relaxed_diagonal`` -- SIMPLE returns
   ``A/(aP/alpha)`` recomputed from the momentum stencil ``_momentum`` assembles.
2. ``test_simplec_mobility_is_the_1984_denominator`` -- SIMPLEC returns
   ``A/max(aP/alpha - sum(aN), 0.1 aP)``, the Van Doormaal & Raithby definition,
   for both momentum components at alpha_u = 0.7 and 0.8.
3. ``test_diffusion_dominated_ratio_is_one_over_one_minus_alpha`` -- on a stencil
   whose neighbour sum equals the diagonal (vanishing convection) the scheme
   ratio must be the closed form ``1/(1-alpha)``; this is a property of the
   paper, not of this code.
4. ``test_floor_limiter_engages_and_stays_finite`` -- at alpha_u = 1 the
   denominator degenerates; the floor must bind, stay finite and positive, and
   return exactly ``10 d_SIMPLE`` there (closed form for alpha_u = 1).
5. ``test_simplec_and_simple_reach_the_same_discrete_fixed_point`` -- the scheme
   changes the iteration path, not the discretisation: converged fields agree to
   solver tolerance and both satisfy the analytic Poiseuille gates (< 3%).
6. ``test_simplec_sustains_pressure_relaxation_one_where_simple_fails`` -- the
   capability claim of the audit is machine-checked with the SIMPLE control.

Reference values are analytic (Poiseuille channel) or self-derived from the
assembled stencil; nothing is copied from a report.
"""

import numpy as np
import pytest
import torch

from tensorfvm.solver import SimpleSolver, SolverConfig
from tensorfvm.verification.cases import channel

torch.set_num_threads(1)

INLET_VELOCITY = 5e-4
FLOOR_FRACTION = SimpleSolver.simplec_floor_fraction
assert FLOOR_FRACTION == .1, "the declared safeguard is documented as 0.1 aP"
GATE = .03  # project-wide engineering acceptance gate on the analytic profile


def _solver(algorithm="simplec", velocity_relaxation=.7,
            pressure_relaxation=.3, inlet_velocity=INLET_VELOCITY):
    config = SolverConfig(nx=36, ny=12, length=.12, height=.02,
                          inlet_velocity=inlet_velocity, density=1000.,
                          reynolds=1., cylinder_radius=None, tolerance=1e-6,
                          max_iterations=2,
                          pressure_velocity_algorithm=algorithm,
                          velocity_relaxation=velocity_relaxation,
                          pressure_relaxation=pressure_relaxation)
    solver = SimpleSolver(config)
    solver._boundaries()
    return solver


def _mobility_from_stencil(solver, component):
    """Independent re-derivation of the mobility from the assembled stencil."""
    diagonal, coeffs, _, active, _ = solver._momentum(component)
    alpha = solver.config.velocity_relaxation
    area = solver.dy if component == "u" else solver.dx
    relaxed = diagonal / alpha
    simple = area / relaxed
    if solver.config.pressure_velocity_algorithm != "simplec":
        return torch.where(active, simple, torch.zeros_like(simple))
    floor = FLOOR_FRACTION * diagonal
    denominator = torch.maximum(relaxed - sum(coeffs), floor)
    return torch.where(active, area / denominator, torch.zeros_like(simple))


def _active_values(a, b, active):
    return (a[active].double().numpy(), b[active].double().numpy())


@pytest.mark.parametrize("component", ["u", "v"])
def test_simple_mobility_is_face_area_over_relaxed_diagonal(component):
    solver = _solver(algorithm="simple")
    expected = _mobility_from_stencil(solver, component)
    actual = solver._predict(component)
    got, want = _active_values(actual, expected, solver._momentum(component)[3])
    assert got.size and np.all(np.isfinite(got))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=0.)
    assert float(actual[~solver._momentum(component)[3]].abs().max()) == 0.
    assert solver.simplec_limited_dofs == 0


@pytest.mark.parametrize("velocity_relaxation", [.7, .8])
@pytest.mark.parametrize("component", ["u", "v"])
def test_simplec_mobility_is_the_1984_denominator(velocity_relaxation, component):
    solver = _solver(velocity_relaxation=velocity_relaxation)
    expected = _mobility_from_stencil(solver, component)
    actual = solver._predict(component)
    got, want = _active_values(actual, expected, solver._momentum(component)[3])
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=0.)
    # The scheme must differ from SIMPLE, otherwise the audit is measuring nothing.
    simple = _mobility_from_stencil(
        _solver(algorithm="simple", velocity_relaxation=velocity_relaxation),
        component)
    simple_values = simple[solver._momentum(component)[3]].double().numpy()
    assert np.median(got) > np.median(simple_values)*(1. + 1e-3)


@pytest.mark.parametrize("velocity_relaxation", [.7, .8])
def test_diffusion_dominated_ratio_is_one_over_one_minus_alpha(velocity_relaxation):
    """aP = sum(aN) for pure diffusion => d_SIMPLEC/d_SIMPLE = 1/(1-alpha) exactly.

    A machine-zero inlet velocity is rejected by the configuration, so 1e-12 m/s
    is used: convection then contributes O(1e-12) to the stencil and the identity
    holds to within the printed tolerance.
    """
    solver = _solver(velocity_relaxation=velocity_relaxation,
                     inlet_velocity=1e-12)
    actual = solver._predict("u")
    simple = _mobility_from_stencil(
        _solver(algorithm="simple", velocity_relaxation=velocity_relaxation,
                inlet_velocity=1e-12), "u")
    active = solver._momentum("u")[3]
    interior = active.clone()
    interior[:, 0] = False
    interior[:, -1] = False  # wall-normal faces carry the boundary closure
    ratio = float((actual[interior]/simple[interior]).median())
    assert solver.simplec_limited_dofs == 0, "floor must not bind here"
    assert ratio == pytest.approx(1./(1. - velocity_relaxation), rel=5e-3)


def test_floor_limiter_engages_and_stays_finite():
    """alpha_u = 1 collapses the denominator; the floor is the only thing left."""
    solver = _solver(velocity_relaxation=1.)
    reference = _solver(algorithm="simple", velocity_relaxation=1.)
    # Both mobilities are derived from the *same* pre-predict state: `_predict`
    # advances the velocity field, so a recomputation afterwards would read a
    # stencil whose upwind coefficients have already changed.
    expected = _mobility_from_stencil(solver, "u")
    simple = _mobility_from_stencil(reference, "u")
    active = solver._momentum("u")[3]
    actual = solver._predict("u")
    assert solver.simplec_limited_dofs > 0
    assert solver.simplec_limited_dofs < int(active.sum()), "limiter must not own every dof"
    got, want = _active_values(actual, expected, active)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=0.)
    assert np.all(np.isfinite(got)) and np.all(got > 0.)
    # At alpha_u = 1 the floored denominator is 0.1 aP, i.e. d = 10 d_SIMPLE.
    ratio = float((actual[active]/simple[active]).double().median())
    assert ratio == pytest.approx(1./FLOOR_FRACTION, rel=1e-9)

def _metrics(case):
    return {row["name"]: float(row["value"]) for row in case["metrics"]}


@pytest.fixture(scope="module")
def equilibrium():
    """Converged SIMPLE and SIMPLEC solutions of the developed channel case."""
    simple = channel(12, 2000, algorithm="simple",
                     velocity_relaxation=.7, pressure_relaxation=.3)
    simplec = channel(12, 2000, algorithm="simplec",
                      velocity_relaxation=.7, pressure_relaxation=.3)
    return simple, simplec


def test_simplec_and_simple_reach_the_same_discrete_fixed_point(equilibrium):
    simple, simplec = equilibrium
    scale = simple["config"]["inlet_velocity"]
    for name in ("u_m_s", "v_m_s"):
        a = np.asarray(simple["fields"][name], dtype=float)
        b = np.asarray(simplec["fields"][name], dtype=float)
        assert float(np.max(np.abs(a - b)))/scale < 1e-6, name
    for case in (simple, simplec):
        assert case["converged"] and case["passed"]
        metrics = _metrics(case)
        # Analytic Poiseuille anchors of the developed channel case.
        assert metrics["velocity_l2"] < GATE
        assert metrics["pressure_gradient"] < GATE
        assert metrics["scaled_divergence"] < 1e-6


def test_simplec_sustains_pressure_relaxation_one_where_simple_fails():
    simplec = channel(12, 2000, algorithm="simplec",
                      velocity_relaxation=.8, pressure_relaxation=1.)
    assert simplec["converged"] and simplec["passed"]
    assert _metrics(simplec)["velocity_l2"] < GATE
    with pytest.raises(Exception) as failure:
        channel(12, 2000, algorithm="simple",
                velocity_relaxation=.7, pressure_relaxation=1.)
    assert "positive" in str(failure.value).lower()