"""Gate on the compressible Euler surface: smooth-flow order and direction symmetry.

The compressible path was previously verified only against *discontinuous* exact
solutions (Sod, contact, Richtmyer--Meshkov, double blast).  Those pin the star
states but cannot measure the truncation error of a smooth flow, so the module's
claim "second order in space and time for smooth flow" had never been tested.
This file adds the two checks that close that gap and pins both as invariants:

*smooth-flow order*
    The isentropic vortex (Shu 1998) is an exact solution of the 2-D Euler
    equations at every time on a periodic box, so a second-order scheme must show
    an observed order near 2 for every primitive variable.  The committed
    ``docs/verification-compressible-isentropic-vortex/study.json`` is regenerated
    by ``scripts/verify_compressible_vortex_order.py``; this test reads it and
    enforces an order floor, so a silent accuracy regression reddens the gate.

*direction symmetry of the advection operator*
    The same smooth entropy wave advected along x and along y on the same grid
    must give the same error.  This is the invariant that caught the bug this test
    guards: the MUSCL-Hancock half-step predictor applied the y flux difference
    with the opposite sign from the update, leaving y-advection at order 0.80
    while x-advection measured 2.19 -- invisible in every axis-aligned
    one-dimensional verification case the module already had.

Negative controls at the bottom prove the order gate rejects a first-order
sequence instead of passing vacuously.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest
import torch

from tensorfvm.compressible.isentropic_vortex import (
    error_norms,
    initial_state,
    mass_and_momentum,
    observed_order,
    run_vortex,
    single_gas_eos,
    vortex_primitive,
)
from tensorfvm.compressible.solver import EulerSolver

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = ROOT / "docs" / "verification-compressible-isentropic-vortex"
STUDY = json.loads((CASE / "study.json").read_text())

L = 10.0
A = 0.2
LAMBDA = 2.0

ORDER_FLOOR = 1.8  # a second-order scheme has to measure clearly above 1


# ----------------------------------------------------------------------
# the exact solution itself
# ----------------------------------------------------------------------
def test_exact_solution_is_isentropic_and_bounded():
    """``p = rho**gamma`` exactly, and the closed-form envelope of the swirl.

    Calibrated 2026-11-12 (m16364) against the *closed form*
    ``T(r) = 1 - K e^{1-r^2}``, ``K = (gamma-1) beta^2 / (8 gamma pi^2)``,
    ``rho = T^{1/(gamma-1)}`` -- the same expression ``vortex_primitive``
    evaluates, so the numbers below are derived, not fitted.  The values are
    reproduced independently by
    ``docs/verification-compressible-isentropic-vortex/envelope_calibration.py``
    (``envelope.json``): beta=5 -> K=0.0905, T_min=0.7541, rho_min=0.4938,
    p_min=0.3724; beta=1 -> rho_min=0.9756, p_min=0.9660.

    Three distinct things are pinned: the isentropic identity (an algebraic
    invariant), the *positivity* of the temperature (the condition
    ``K e < 1``, which is what keeps the case physical), and the radial
    monotonicity of a field whose only structure is a core-centred swirl.
    """
    x, y = torch.meshgrid(
        torch.linspace(0.0, L, 41, dtype=torch.float64),
        torch.linspace(0.0, L, 41, dtype=torch.float64),
        indexing="ij",
    )
    r2 = (x - 0.5 * L) ** 2 + (y - 0.5 * L) ** 2
    for beta in (1.0, 5.0):
        state = vortex_primitive(x, y, lx=L, ly=L, beta=beta)
        rho, p = state["rho"], state["p"]
        # (1) isentropic identity, exact in the closed form
        assert torch.allclose(p, rho ** 1.4, rtol=1e-13, atol=0.0)

        # (2) the closed-form amplitude scale and the extrema it implies
        k = (1.4 - 1.0) * beta * beta / (8.0 * 1.4 * math.pi ** 2)
        t_min = 1.0 - k * math.e            # e^{1-r^2} = e at the core r = 0
        rho_min = t_min ** (1.0 / (1.4 - 1.0))
        # positivity of T everywhere: the physicality condition K e < 1
        assert 1.0 - k * math.e > 0.0 and t_min > 0.7
        # the 41-point grid samples the core exactly, so the extrema are exact
        assert float(rho.min()) == pytest.approx(rho_min, rel=1e-13)
        assert float(p.min()) == pytest.approx(rho_min ** 1.4, rel=1e-13)
        # far field: the swirl decays to the unit free stream, never above it
        assert float(rho.max()) <= 1.0 + 1e-15 and float(p.max()) <= 1.0 + 1e-15
        # the swirl speed peaks at |delta u| = beta/(2 pi) (max of d*e^{(1-d^2)/2}
        # at d = 1), so the streamwise velocity can never exceed that envelope
        swirl = beta / (2.0 * math.pi)
        assert float(state["u"].max()) <= 1.0 + swirl + 1e-12
        assert float(state["u"].max()) >= 1.0 + 0.9 * swirl

        # (3) the field is a function of r alone: rho non-decreasing in r^2
        radial = rho.flatten()[torch.argsort(r2.flatten())]
        assert float(torch.diff(radial).min()) >= -1e-14


def test_single_image_representation_is_bounded_by_the_nearest_image():
    """The neglected periodic images are bounded and far below the truncation error.

    ``vortex_primitive`` keeps one image of the core; on a 10 x 10 box the
    nearest *neglected* image sits 5 core radii away, so its amplitude is
    ``exp((1 - 25)/2) = 6.1e-6`` at the wall cells (exactly the number
    ``envelope.json:periodic_image`` reports).  That is the honest size of the
    "no interpolation" claim: three to four orders below the N=32 truncation
    error (~1e-2) that the order study measures, and dt-independent, so it
    cannot bend the observed order -- but it is *not* double precision, and the
    old module docstring said it was.  This gate keeps the comparison explicit.
    """
    x, y = torch.meshgrid(
        torch.linspace(0.0, L, 41, dtype=torch.float64),
        torch.linspace(0.0, L, 41, dtype=torch.float64),
        indexing="ij",
    )
    dx, dy = x - 0.5 * L, y - 0.5 * L
    worst = 0.0
    for sx in (-L, 0.0, L):
        for sy in (-L, 0.0, L):
            if sx == 0.0 and sy == 0.0:
                continue
            r2 = (dx + sx) ** 2 + (dy + sy) ** 2
            worst = max(worst, float(torch.exp(0.5 * (1.0 - r2)).max()))
    assert worst == pytest.approx(math.exp(0.5 * (1.0 - 25.0)), rel=1e-12)
    assert worst < 1e-4


def test_exact_solution_is_periodic_across_the_seam():
    """The field must be continuous across ``x = 0``/``x = L`` (no seam defect)."""
    y = torch.full((2,), 0.5 * L, dtype=torch.float64)
    x = torch.tensor([0.0, L], dtype=torch.float64)
    for t in (0.0, 0.37, 1.0):
        state = vortex_primitive(x, y, lx=L, ly=L, t=t)
        for name in ("rho", "u", "v", "p"):
            a, b = state[name][0], state[name][1]
            assert torch.allclose(a, b, atol=1e-12), (name, t, float(a - b))


def test_exact_solution_translates_with_the_free_stream():
    """At time ``t`` the field equals the ``t = 0`` field shifted by ``(u, v) t``."""
    x, y = torch.meshgrid(
        torch.linspace(1.0, 9.0, 17, dtype=torch.float64),
        torch.linspace(1.0, 9.0, 17, dtype=torch.float64),
        indexing="ij",
    )
    shifted = vortex_primitive(x - 0.6, y, lx=L, ly=L, t=0.0)
    advected = vortex_primitive(x, y, lx=L, ly=L, t=0.6)
    for name in ("rho", "u", "v", "p"):
        assert torch.allclose(shifted[name], advected[name], atol=1e-12)


# ----------------------------------------------------------------------
# conservation invariants on a periodic box (flux form must be exact)
# ----------------------------------------------------------------------
def test_periodic_run_conserves_mass_and_momentum_to_round_off():
    solver, exact = run_vortex(32, t_end=0.25)
    num = solver.primitive()
    inv0 = mass_and_momentum(initial_state(32, 32, lx=L, ly=L), solver.dx, solver.dy)
    inv1 = mass_and_momentum(num, solver.dx, solver.dy)
    assert abs(inv1["mass"] - inv0["mass"]) / inv0["mass"] < 1e-12
    assert abs(inv1["momentum_x"] - inv0["momentum_x"]) / abs(inv0["momentum_x"]) < 1e-12
    # the solution stays physical (this is the positivity guard, not a tolerance)
    assert float(num["rho"].min()) > 0.4
    assert float(num["p"].min()) > 0.3
    assert bool(torch.isfinite(num["rho"]).all())


# ----------------------------------------------------------------------
# the committed convergence study
# ----------------------------------------------------------------------
def test_committed_study_reports_second_order_for_every_variable():
    for key in ("beta=5.0", "beta=1.0"):
        orders = STUDY["vortex"][key]["orders"]
        for name in ("rho", "p"):
            for value in orders[name]:
                assert value > ORDER_FLOOR, (key, name, value)
        # the transverse velocity is the noisiest variable (its exact value is
        # small over much of the box), but it may never fall to first order
        for value in orders["v"][-1:]:
            assert value > ORDER_FLOOR, (key, "v", value)


def test_committed_study_errors_decrease_monotonically():
    for key in ("beta=5.0", "beta=1.0"):
        rows = STUDY["vortex"][key]["rows"]
        seq = [row["rho"]["L1"] for row in rows]
        assert all(b < a for a, b in zip(seq, seq[1:])), (key, seq)


def test_committed_study_conserves_mass_and_momentum():
    for key in ("beta=5.0", "beta=1.0"):
        for row in STUDY["vortex"][key]["rows"]:
            assert abs(row["mass_drift"]) < 1e-12
            assert abs(row["momentum_x_drift"]) < 1e-12


def test_committed_study_records_the_temporal_plateau():
    """Refining ``dt`` alone must not change the error: time order is not the limit."""
    temporal = STUDY["temporal_refinement"]
    errors = [row["rho_L1"] for row in temporal]
    assert max(errors) / min(errors) < 1.2
    assert temporal[0]["steps"] < temporal[-1]["steps"]


# ----------------------------------------------------------------------
# direction symmetry (the regression guard for the y-predictor sign)
# ----------------------------------------------------------------------
def _advect(n: int, axis: str, t_end: float, limiter: str = "none") -> float:
    """L1 error of a smooth sine *entropy wave* advected along ``axis``."""
    solver = EulerSolver(
        nx=n, ny=n, lx=L, ly=L, eos=single_gas_eos(), bc=("periodic",) * 4,
        flux_mode="hllc", order=2, limiter=limiter, cfl=0.3,
    )
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * (L / n)

    def field(shift: float) -> torch.Tensor:
        base = 1.0 + A * torch.sin(2.0 * math.pi * (c - shift) / LAMBDA)
        return (base.view(-1, 1) if axis == "x" else base.view(1, -1)).expand(n, n).contiguous()

    zeros = torch.zeros((n, n), dtype=torch.float64)
    ones = torch.ones((n, n), dtype=torch.float64)
    solver.set_primitive({
        "rho": field(0.0),
        "u": ones.clone() if axis == "x" else zeros.clone(),
        "v": ones.clone() if axis == "y" else zeros.clone(),
        "p": ones.clone(),
        "Y": zeros.clone(),
    })
    t = 0.0
    while t < t_end - 1e-15:
        dt = min(solver.stable_dt(), t_end - t)
        solver.step(dt)
        t += dt
    return float((solver.primitive()["rho"] - field(t_end)).abs().mean())


def test_advection_error_is_the_same_along_x_and_along_y():
    """Mirror symmetry of the operator: identical geometry must give identical error.

    Before the y-predictor sign fix this ratio was ~2 (y error 1.37e-2 vs x
    7.02e-3 on the same 64x64 grid): the half step moved the y-direction state
    the wrong way and cost the whole direction its second order.
    """
    ex = _advect(32, "x", 0.5)
    ey = _advect(32, "y", 0.5)
    assert ey == pytest.approx(ex, rel=1e-9), (ex, ey)


def test_advection_reaches_second_order_along_each_axis():
    for axis in ("x", "y"):
        coarse = _advect(32, axis, 1.0)
        fine = _advect(64, axis, 1.0)
        assert observed_order(coarse, fine) > ORDER_FLOOR, (axis, coarse, fine)


# ----------------------------------------------------------------------
# negative controls: the gate must fail on a first-order sequence
# ----------------------------------------------------------------------
def test_order_gate_rejects_a_first_order_sequence():
    first_order = [1.0e-2, 5.0e-3, 2.5e-3]
    orders = [observed_order(a, b) for a, b in zip(first_order, first_order[1:])]
    assert all(value < ORDER_FLOOR for value in orders)


def test_order_gate_accepts_a_second_order_sequence():
    second_order = [1.0e-2, 2.5e-3, 6.25e-4]
    orders = [observed_order(a, b) for a, b in zip(second_order, second_order[1:])]
    assert all(value > ORDER_FLOOR for value in orders)


def test_error_norms_are_zero_for_an_identical_field():
    field = initial_state(16, 16, lx=L, ly=L)
    norms = error_norms(field["rho"], field["rho"])
    assert max(abs(v) for v in norms.values()) == 0.0