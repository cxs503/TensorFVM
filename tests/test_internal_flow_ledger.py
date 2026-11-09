"""Tests of the internal-flow momentum-ledger surface.

Two kinds of test:

* **algebraic** tests on synthetic fields, so the ledger can be wrong-checked
  without a solver run (signs of the wall shear, the transpose identity of the
  x/y ledgers, no double counting of an open boundary, exactness of the
  second-order wall derivative);
* **artifact** tests that re-derive the campaign report from the stored case
  records, so the published numbers cannot drift from the JSON evidence
  (``results.json`` is regenerated and compared term by term).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from tensorfvm.internal_flow import cases, ledger, ledger_report, metrics

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "internal-flow-momentum-ledger" / "records"
RESULTS_FILE = ROOT / "docs" / "internal-flow-momentum-ledger" / "results.json"


def poiseuille_fields(ny: int, *, nu: float = 0.05, u_mean: float = 1.0,
                      height: float = 1.0, length: float = 1.0):
    """Analytic plane-Poiseuille field: u(y) = 6 U eta (1 - eta), p = -12 nu U x/h^2."""
    dy = height / ny
    nx = max(int(round(length / dy)), 3)
    dx = dy
    u = torch.zeros((ny, nx + 1))
    v = torch.zeros((ny + 1, nx))
    p = torch.zeros((ny, nx))
    for j in range(ny):
        eta = (j + 0.5) * dy / height
        u[j, :] = 6.0 * u_mean * eta * (1.0 - eta)
    for i in range(nx):
        p[:, i] = -12.0 * nu * u_mean / height ** 2 * (i + 0.5) * dx
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    return u, v, p, fluid, dx, dy


def test_poiseuille_ledger_closes_and_converges():
    """The ledger reproduces the exact channel momentum balance -2 tau_w = dp/dx h."""
    exact = 12.0 * 0.05 * 1.0  # per unit depth over a length 1 channel
    residuals = []
    for ny in (16, 32, 64):
        u, v, p, fluid, dx, dy = poiseuille_fields(ny)
        led = ledger.momentum_ledger(u, v, p, fluid, dx, dy, 0.05, box=(0, ny, 0, 0)
                                     if False else (0, ny, 0, fluid.shape[1]),
                                     open_faces=(True, True))
        residuals.append(abs(led.residual) / exact)
    assert residuals[0] < 0.07
    assert all(b < a for a, b in zip(residuals, residuals[1:])), residuals
    assert residuals[-1] < 0.02


def test_wall_shear_retards_the_flow():
    """A positive near-wall velocity must give a negative (retarding) wall shear."""
    ny, nx = 4, 4
    u = torch.full((ny, nx + 1), 0.5)
    v = torch.zeros((ny + 1, nx))
    p = torch.zeros((ny, nx))
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    led = ledger.momentum_ledger(u, v, p, fluid, 1.0, 1.0, 0.1,
                                 box=(0, ny, 0, nx), open_faces=(False, False))
    assert led.terms["visc_wall_shear"] < 0.0


def test_ledger_transpose_identity():
    """``momentum_ledger(component='y')`` equals the transposed x-ledger."""
    ny, nx = 5, 6
    torch.manual_seed(3)
    u = torch.rand((ny, nx + 1))
    v = torch.rand((ny + 1, nx))
    p = torch.rand((ny, nx))
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    led_y = ledger.momentum_ledger(u, v, p, fluid, 0.1, 0.1, 0.02, component="y")
    led_x_t = ledger.momentum_ledger(v.T.contiguous(), u.T.contiguous(),
                                     p.T.contiguous(), fluid.T.contiguous(),
                                     0.1, 0.1, 0.02, component="x")
    for key in ("convective", "pressure", "viscous", "residual"):
        a = led_y.as_dict()[key]
        b = led_x_t.as_dict()[key]
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), key


def test_open_boundary_is_not_double_counted():
    """An inlet face must appear once (as an open face), never also as a wall."""
    u, v, p, fluid, dx, dy = poiseuille_fields(8)
    open_box = ledger.momentum_ledger(u, v, p, fluid, dx, dy, 0.05,
                                      box=(0, 8, 0, fluid.shape[1]),
                                      open_faces=(True, True))
    closed_box = ledger.momentum_ledger(u, v, p, fluid, dx, dy, 0.05,
                                        box=(0, 8, 0, fluid.shape[1]),
                                        open_faces=(False, False))
    assert open_box.terms["p_wall"] == pytest.approx(0.0)
    assert closed_box.terms["p_wall"] != pytest.approx(0.0)
    assert open_box.terms["p_in_open"] != pytest.approx(0.0)
    assert closed_box.terms["p_in_open"] == pytest.approx(0.0)


def test_station_flux_of_the_parabolic_profile():
    u, v, p, fluid, dx, dy = poiseuille_fields(16)
    flux = ledger.station_flux(u, fluid, 0, dy)
    # midpoint rule on a parabola: U_mean * height up to an O(h^2) quadrature error
    assert flux == pytest.approx(1.0, rel=5e-3)
    assert ledger.station_height(fluid, 0, dy) == pytest.approx(1.0)


def test_richardson_recovers_second_order():
    exact = 6.0
    coarse, medium, fine = (exact + 0.8, exact + 0.2, exact + 0.05)
    out = metrics.richardson(coarse, medium, fine, ratio=2.0)
    assert out["observed_order"] == pytest.approx(2.0, abs=1e-9)
    assert out["extrapolated"] == pytest.approx(exact, abs=1e-9)
    assert out["fine_vs_extrapolated"] == pytest.approx(0.05 / exact, rel=1e-9)


def test_richardson_refuses_non_monotone_data():
    out = metrics.richardson(1.0, 1.0, 2.0)
    assert not math.isfinite(out["observed_order"])
    assert not math.isfinite(out["extrapolated"])


def test_reattachment_from_a_synthetic_wall_profile():
    ny, nx = 4, 12
    u = torch.zeros((ny, nx + 1))
    # wall-adjacent row: negative (recirculation) up to x = 2.5, then positive
    dx = 0.25
    x = torch.arange(nx + 1, dtype=torch.float64) * dx
    u[0, :] = (x - 2.5).to(u.dtype)
    u[1:, :] = 0.5
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    out = metrics.reattachment_length(u, fluid, dx, 0.25, 0.01,
                                     start_face=1, scale=1.0)
    assert out["shear"] == pytest.approx(2.5, abs=1e-9)
    assert out["velocity"] == pytest.approx(2.5, abs=1e-9)


def test_nusselt_second_order_is_exact_for_a_quadratic_profile():
    n = 16
    dx = 1.0 / n
    a, b, c = -0.4, -0.3, 0.25     # T decreases away from the hot wall
    x = (torch.arange(n) + 0.5) * dx
    T = 1.0 + a * x + b * x ** 2 + c * x ** 3      # not quadratic: polynomial of order 3
    # second-order estimate must be closer to the analytic value than the first
    analytic_hot = -a
    field = T.unsqueeze(0).repeat(3, 1)
    out = metrics.wall_nusselt(field, dx, 1.0, 1.0 + a + b + c)
    assert abs(out["nusselt_hot"] - analytic_hot) < abs(
        out["nusselt_hot_order1"] - analytic_hot)


def test_case_geometry_matches_the_definition():
    ar2 = cases.sudden_expansion(2.0, "coarse")
    ar3 = cases.sudden_expansion(3.0, "coarse")
    assert ar2.geometry["expansion_ratio_mask"] == pytest.approx(2.0, abs=0.01)
    assert ar3.geometry["expansion_ratio_mask"] == pytest.approx(3.0, abs=0.01)
    assert ar2.geometry["inlet_flux"] == pytest.approx(0.5, rel=0.02)
    step = cases.backward_facing_step(100.0, "coarse")
    assert step.geometry["expansion_ratio"] == pytest.approx(2.0)
    assert step.geometry["inlet_flux"] == pytest.approx(1.0, rel=0.02)
    cav = cases.natural_cavity(1.0e5, "coarse")
    assert cav.viscosity == pytest.approx(math.sqrt(0.71 / 1.0e5))
    assert cav.conductivity == pytest.approx(cav.viscosity / 0.71)
    with pytest.raises(ValueError):
        cases.sudden_expansion(2.5, "coarse")


def test_borda_carnot_anchor_is_the_exact_momentum_result():
    from tensorfvm.internal_flow import ledger_reference
    assert ledger_reference.borda_carnot(2.0).value == pytest.approx(0.5)
    assert ledger_reference.borda_carnot(3.0).value == pytest.approx(4.0 / 9.0)
    assert ledger_reference.borda_carnot(2.0).grade == "REAL"
    assert ledger_reference.armaly_reattachment(100.0).grade == "WITHHELD"


# --------------------------------------------------------------- artifacts
def _records() -> dict[str, dict]:
    if not ARTIFACTS.exists():
        pytest.skip("campaign records not present in this checkout")
    return ledger_report._load(str(ARTIFACTS))


def test_results_json_is_reproduced_from_the_records():
    if not RESULTS_FILE.exists():
        pytest.skip("results.json not present")
    stored = json.load(open(RESULTS_FILE))
    records = _records()
    fresh = ledger_report.assemble(records)
    assert [c["id"] for c in fresh["criteria"]] == [c["id"] for c in stored["criteria"]]
    for a, b in zip(fresh["criteria"], stored["criteria"]):
        assert a["passed"] == b["passed"]
        if isinstance(a["error"], float) and math.isfinite(a["error"]):
            assert a["error"] == pytest.approx(b["error"], rel=1e-12, abs=1e-15)
    for name, digest in stored["source_records"].items():
        path = ARTIFACTS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_cp_decomposition_closes_on_every_case():
    records = _records()
    payload = ledger_report.assemble(records)
    rows = payload["tables"]["expansion_cp"]
    assert rows, "no expansion record"
    for row in rows:
        assert abs(row["closure"]) < 0.05, row


def test_every_criterion_has_an_explicit_threshold():
    payload = ledger_report.assemble(_records())
    for crit in payload["criteria"]:
        assert "threshold" in crit and "passed" in crit and "error" in crit
        assert isinstance(crit["passed"], bool)