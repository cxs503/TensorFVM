"""Sod shock tube (Sod 1978) — API-level gate of the compressible Euler solver.

Anchor: REPRO-FVM-SOD1978
Reference: Sod, G. A. (1978). "A Survey of Several Finite Difference Methods
for Systems of Nonlinear Hyperbolic Conservation Laws." J. Comput. Phys.
27:1-31. doi:10.1016/0021-9991(78)90023-2

What this file gates
--------------------
``tests/test_repro_fvm_sod1978.py`` grades the anchor at its own output time
``t = 0.14`` (four wave positions, star state, order on smooth data).  This file
gates the *same solver surface* at the time of the paper's own figure,
``t = 0.2``, and is the file the acceptance criterion was originally written
against::

    L1 relative error of (rho, u, p) vs the exact Riemann solution < 3 %
    at nx = 400, second-order MUSCL-Hancock + HLLC, t_end = 0.2

Thresholds are the ledger criterion (3 %) plus round-off-scale budgets; they are
declared here and not tuned after the numbers were seen.  No tabulated reference
number is transcribed into an assertion: the reference profile comes from the
package's own exact Riemann solver, whose star state is separately cross-checked
against the published values (``test_exact_reference_matches_published_star_state``).

Repair history (why this file was red on main)
----------------------------------------------
``EulerSolver`` migrated and this file was not: ``step`` became a method
(``solver.step > 0`` raised ``TypeError``), the ``rho/u/p`` attributes were
replaced by ``primitive()`` (``AttributeError``), ``ExactRiemann.solve`` now
takes an iteration count rather than ``(x, t)`` (``TypeError: only integer
scalar arrays ...``), and the first-order run used the retired
``_limiter_name = 'first_order'`` hook.  The criteria below are unchanged; only
the calls were migrated, so the anchor's 3 % criterion is machine-checked again
instead of silently absent from the suite signal.
"""

import math

import numpy as np
import pytest

from tensorfvm.compressible import cases

REF_NX = 400
T_END = 0.2
TOL_REL = 0.03  # acceptance criterion, unchanged from the ledger
BUDGET_TOL = 1e-12  # round-off scale of the accounted budget
PUBLISHED_STAR = {"p_star": 0.30313, "u_star": 0.92745}


@pytest.fixture(scope="module")
def runs():
    """Sod at the paper time, donor cell and MUSCL-Hancock, one grid."""
    out = {}
    for order in (1, 2):
        solver, meta = cases.sod(nx=REF_NX, order=order)
        initial = solver.integrals()
        t0 = solver.integrals()
        solver.run(T_END, max_steps=5000)
        out[order] = {"solver": solver, "meta": meta, "initial": initial,
                      "initial_repeat": t0}
    return out


def _fields(solver):
    prim = solver.primitive()
    return {k: prim[k][:, 0].double().numpy() for k in ("rho", "u", "p")}


def _exact(meta):
    xi = (meta["xc"] - meta["x0"]) / T_END
    rho, u, p = meta["exact"].sample(xi)
    return {"rho": rho.double().numpy(), "u": u.double().numpy(),
            "p": p.double().numpy()}


def _l1_rel(num, ref):
    return float(np.mean(np.abs(num - ref)) / np.mean(np.abs(ref)))


# ----------------------------------------------------------------------
# reference
# ----------------------------------------------------------------------
def test_exact_reference_matches_published_star_state(runs):
    """The grading reference must itself agree with the published star state.

    Every L1 number below is measured against this reference, so a drift here
    would invalidate them silently.
    """
    meta = runs[2]["meta"]
    exact = meta["exact"]
    exact.solve()
    p_star = float(exact.pstar)
    u_star = float(exact.ustar)
    assert p_star == pytest.approx(PUBLISHED_STAR["p_star"], rel=5e-3)
    assert u_star == pytest.approx(PUBLISHED_STAR["u_star"], rel=5e-3)


# ----------------------------------------------------------------------
# run health
# ----------------------------------------------------------------------
def test_sod_runs_to_the_paper_time(runs):
    """Both orders complete, reach t = 0.2, and stay finite."""
    for order, run in runs.items():
        solver = run["solver"]
        assert solver.nsteps > 0, f"order {order}: solver did not advance"
        assert solver.time == pytest.approx(T_END, rel=1e-12), order
        fields = _fields(solver)
        for name, values in fields.items():
            assert np.all(np.isfinite(values)), f"order {order}: {name} not finite"
        # rho and p stay positive (u is exactly 0 in the undisturbed left state)
        for name in ("rho", "p"):
            assert np.all(fields[name] > 0.0), f"order {order}: {name} non-positive"


# ----------------------------------------------------------------------
# acceptance criterion: L1 < 3 % at nx = 400
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["rho", "u", "p"])
def test_l1_error_within_criterion(runs, name):
    """Criterion: L1 relative error of rho, u, p below 3 % (order 2, nx = 400)."""
    run = runs[2]
    err = _l1_rel(_fields(run["solver"])[name], _exact(run["meta"])[name])
    assert err < TOL_REL, f"{name}: L1 relative error {err:.4%} exceeds 3 %"


def test_second_order_beats_donor_cell(runs):
    """MUSCL-Hancock must be materially more accurate than the donor cell.

    The measured ratio at this grid is ~0.18; the assertion only demands a
    factor 2 so it is a scheme-property gate, not a fit to one number.
    """
    for name in ("rho", "u", "p"):
        exact = _exact(runs[2]["meta"])[name]
        err1 = _l1_rel(_fields(runs[1]["solver"])[name], exact)
        err2 = _l1_rel(_fields(runs[2]["solver"])[name], exact)
        assert err2 < err1, f"{name}: order 2 ({err2:.4e}) not better than 1 ({err1:.4e})"
        assert err2 < 0.5 * err1, (
            f"{name}: order-2/order-1 error ratio {err2 / err1:.3f} is not "
            "convincing for a second-order scheme")


# ----------------------------------------------------------------------
# accounted budget
# ----------------------------------------------------------------------
@pytest.mark.parametrize("order", [1, 2])
def test_accounted_budget_closes_to_round_off(runs, order):
    """``U(t) + outflow(t) - U(0)`` closes for mass, energy and momentum.

    Momentum is the interesting one: the tube starts at rest, so the initial
    momentum integral is *exactly* zero and the relative drift is undefined
    (NaN by construction, see ``budget_drift``).  Reading that NaN as a
    conservation failure would be wrong -- the absolute residual closes at
    round-off, which is what this test asserts.
    """
    run = runs[order]
    solver, initial = run["solver"], run["initial"]
    relative = solver.budget_drift(initial)
    absolute = solver.budget_drift(initial, absolute=True)
    for key in ("mass", "mass_mat1", "energy"):
        assert abs(relative[key]) < BUDGET_TOL, f"order {order}: {key} {relative[key]:.3e}"
    for key in ("mass", "momentum_x", "momentum_y", "energy"):
        assert abs(absolute[key]) < BUDGET_TOL, f"order {order}: {key} {absolute[key]:.3e}"
    # the zero-reference case is NaN relatively, finite absolutely
    assert math.isnan(relative["momentum_x"])
    assert math.isfinite(absolute["momentum_x"])


def test_initial_state_is_the_declared_diaphragm(runs):
    """Guard the initial data: swapping the states would still look plausible."""
    solver = runs[2]["solver"]
    fields = _fields(solver)
    left = fields["rho"][: REF_NX // 2]
    right = fields["rho"][REF_NX // 2:]
    # after t = 0.2 the plateaus are still (1.0, 0.125) away from the waves
    assert left[: REF_NX // 8].mean() == pytest.approx(1.0, rel=1e-6)
    assert right[-REF_NX // 8:].mean() == pytest.approx(0.125, rel=1e-6)