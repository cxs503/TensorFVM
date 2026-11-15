"""Tests for the wall-shear / power-budget audit of CASE-FVM-CAVITY-RE1000.

The analytic cases pin the operators (they must be *exact* where the algebra says
so); the artefact cases pin the published numbers of
``docs/CASE-FVM-CAVITY-RE1000/WALL-SHEAR-BUDGET.md``.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorfvm.cavity_power_budget import (
    budget_spectral,
    budget_staggered,
    staggered_cell_velocity,
    staggered_gradients,
)
from tensorfvm.cavity_wall_shear import (
    drag_decomposition,
    lid_drag,
    lid_wall_shear,
    observed_order,
)

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "CASE-FVM-CAVITY-RE1000"
FIELDS = DOCS / "runs" / "fields.npz"
AUDIT = DOCS / "runs" / "wall_shear_budget.json"


# --------------------------------------------------------------------------- #
# analytic operator tests (no solver involved)
# --------------------------------------------------------------------------- #
def test_cell_velocity_averages_faces():
    n = 4
    u = np.arange((n + 1) * n, dtype=float).reshape(n + 1, n)
    v = np.arange(n * (n + 1), dtype=float).reshape(n, n + 1)
    uc, vc = staggered_cell_velocity(u, v)
    assert uc.shape == (n, n) and vc.shape == (n, n)
    assert np.allclose(uc, 0.5 * (u[:-1] + u[1:]))
    assert np.allclose(vc, 0.5 * (v[:, :-1] + v[:, 1:]))


def test_gradients_exact_for_linear_field_in_interior():
    n, h = 16, 1.0 / 16
    y = (np.arange(n) + 0.5) * h
    u = np.tile(y, (n + 1, 1))
    v = np.zeros((n, n + 1))
    g = staggered_gradients(u, v, h, lid_velocity=1.0)
    assert np.allclose(g["du_dy"], 1.0, atol=1e-12)
    assert np.allclose(g["dv_dx"], 0.0, atol=1e-12)
    assert np.allclose(g["du_dx"][1:-1, :], 0.0, atol=1e-12)


def test_total_vorticity_identity_linear_field():
    """u = y, v = 0 => int omega dA = -U L exactly on the staggered grid."""
    for n in (8, 16, 32):
        h = 1.0 / n
        y = (np.arange(n) + 0.5) * h
        u = np.tile(y, (n + 1, 1))
        v = np.zeros((n, n + 1))
        out = budget_staggered(u, v, h, nu=1e-3)
        assert out["total_vorticity"] == pytest.approx(-1.0, abs=1e-12)
        assert out["total_vorticity_rel"] == pytest.approx(1.0, abs=1e-12)


def test_shear_estimators_exact_on_linear_and_quadratic_profiles():
    n, h, U = 8, 1.0 / 8, 1.0
    delta = (n - 0.5 - np.arange(n)) * h                 # distance below the lid

    # u = U - a delta  ->  du/dy = +a for both estimators
    a = 3.0
    u = np.tile(U - a * delta, (n + 1, 1))
    for order in (1, 2):
        x, s = lid_wall_shear(u, h, order=order)
        assert np.allclose(s, a, atol=1e-12), order

    # u = U - a delta^2  ->  du/dy = 0 at the wall: order 2 exact, order 1 in error O(h)
    u = np.tile(U - a * delta ** 2, (n + 1, 1))
    _, s1 = lid_wall_shear(u, h, order=1)
    _, s2 = lid_wall_shear(u, h, order=2)
    assert np.allclose(s1, a * h / 2.0, atol=1e-12)
    assert np.allclose(s2, 0.0, atol=1e-12)
    assert lid_drag(u, h, nu=1.0, order=2) == pytest.approx(0.0, abs=1e-12)


def test_drag_decomposition_sums_to_total():
    n, h = 32, 1.0 / 32
    delta = (np.arange(n) + 0.5) * h
    u = np.tile(1.0 - 0.1 * np.sqrt(delta), (n + 1, 1))
    d = drag_decomposition(u, h, nu=1e-3, order=1)
    assert d["left_strip"] + d["centre"] + d["right_strip"] == pytest.approx(d["total"])
    assert 0.0 < d["corner_share"] < 1.0


def test_spectral_budget_operator_on_analytic_streamfunction():
    """psi = sin(pi x) sin(pi y): int omega dA = 8 and int|grad u|^2 = pi^4 exactly."""
    from tensorfvm.cavity_spectral import SpectralCavity

    order = 24
    solver = SpectralCavity(order)
    X = np.asarray(solver.X)
    psi = np.sin(np.pi * X)[:, None] * np.sin(np.pi * X)[None, :]
    omega = 2.0 * np.pi ** 2 * psi          # omega = -lap(psi)
    out = budget_spectral(psi, omega, nu=1e-3, order=order)
    assert out["total_vorticity"] == pytest.approx(8.0, rel=1e-9)
    assert out["dissipation"] == pytest.approx(1e-3 * np.pi ** 4, rel=1e-9)


def test_observed_order_sign_aware():
    assert observed_order(0.004, 0.001) == pytest.approx(2.0)
    assert observed_order(0.001, 0.004) == pytest.approx(-2.0)
    assert np.isnan(observed_order(0.001, -0.001))


# --------------------------------------------------------------------------- #
# artefact tests (stored FVM fields + published audit numbers)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fields():
    if not FIELDS.exists():
        pytest.skip("stored cavity fields not available")
    return dict(np.load(FIELDS))


@pytest.fixture(scope="module")
def audit():
    if not AUDIT.exists():
        pytest.skip("wall-shear audit artefact not available")
    return json.loads(AUDIT.read_text())


def test_fvm_drag_ladder_and_power_identity(fields, audit):
    rows = []
    for n in (32, 64, 128):
        h = 1.0 / n
        u, v = fields[f"u_{n}"], fields[f"v_{n}"]
        b = budget_staggered(u, v, h, nu=1e-3)
        d1 = drag_decomposition(u, h, 1e-3, order=1)
        d2 = drag_decomposition(u, h, 1e-3, order=2)
        rows.append((n, b, d1, d2))

    # the published table must match an independent recomputation
    for (n, b, d1, d2), ref in zip(rows, audit["fvm"]):
        assert ref["n"] == n
        assert d1["total"] == pytest.approx(ref["drag_1"], rel=1e-12)
        assert d2["total"] == pytest.approx(ref["drag_2"], rel=1e-12)
        assert b["dissipation"] == pytest.approx(ref["dissipation"], rel=1e-12)
        assert b["total_vorticity"] == pytest.approx(-1.0, abs=1e-12)

    # order-2 estimator is closer to the converged drag than the solver's order-1 one
    spec_total = audit["spectral"]["drag_total"]
    err1 = abs(rows[-1][2]["total"] - spec_total)
    err2 = abs(rows[-1][3]["total"] - spec_total)
    assert err2 < err1

    # power identity residual decreases under refinement
    res = [r[1]["power_residual_rel"] for r in rows]
    assert res[0] > res[1] > res[2] > 0.0


def test_central_drag_agrees_between_two_discretisations(fields, audit):
    """The measured, defensible claim: central 80 % lid drag agrees within 3 %."""
    u, h = fields["u_128"], 1.0 / 128
    centre = drag_decomposition(u, h, 1e-3, order=1)["centre"]
    spec = audit["spectral"]["drag_centre"]
    rel = 100.0 * (centre - spec) / spec
    assert abs(rel) < 3.0, rel
    assert abs(rel) == pytest.approx(abs(audit["cross_check"]["centre_fvm_n128_vs_spectral"]),
                                     abs=1e-6)


def test_total_drag_is_not_qualified(audit):
    """Honest negative result: the whole-lid drag misses the 3 % bar."""
    assert abs(audit["cross_check"]["total_fvm_rich_vs_spectral"]) > 3.0
    assert audit["spectral"]["corner_share"] > 0.5      # corner strips dominate
    assert audit["orders"]["drag_order1"] < 1.0         # first-order wall shear


def test_report_artifacts_present():
    md = DOCS / "WALL-SHEAR-BUDGET.md"
    assert md.exists()
    text = md.read_text()
    assert "是否 <3%" in text and "honest FAIL" in text
    for fig in ("fig_ws_drag_convergence.png", "fig_ws_power_residual.png",
                "fig_ws_fields_shear.png"):
        assert (DOCS / "figures" / fig).exists(), fig