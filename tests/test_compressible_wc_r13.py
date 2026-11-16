"""Regression gates for the Round-13 closure of anchor XVAL-FVM-M18737-ROUND2.

The Round-13 verdict is produced by
``docs/XVAL-FVM-M18737-ROUND2/r13_certified_rung.py`` under the criteria frozen in
``PREDECLARATION-r13.md``.  These tests pin the *logic*, not the numbers, so a future
round can re-run the driver on new rungs without silently changing what a verdict means:

* the CFL admission control must reject a pair whose L1 exceeds the frozen threshold;
* the honest-ceiling table must add ``eps_ref`` to every graded rung (triangle inequality)
  instead of reporting the point estimate as an error;
* a reference pair spread over two instruments (CFL 0.4 and 0.8) must be flagged
  ``licensed=False`` so it can never be quoted as a certified bar;
* the independent re-integration and the shared metric must agree on the stored rung
  (a metric artefact would have to survive both implementations).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
DRIVER = HERE / "docs" / "XVAL-FVM-M18737-ROUND2" / "r13_certified_rung.py"


@pytest.fixture(scope="module")
def r13():
    spec = importlib.util.spec_from_file_location("wc_r13", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def verdict(r13):
    return r13.compute()


def test_cfl_admission_threshold_is_frozen(r13):
    """The admission bar is the pre-registered 0.2 %, not a value chosen after the fact."""
    assert r13.CFL_ADMISSION == 0.2
    assert r13.BAR_CERTIFIED == 1.0
    assert r13.BAR_THREE == 3.0
    assert r13.BAR_LEDGER == 15.0


def test_independent_metric_agrees_with_shared_metric(r13, verdict):
    """Shared metric and the independent re-integration must agree on the stored rungs."""
    diff = verdict["independent_check_max_abs_diff_percent"]
    # the two implementations differ only by where the metric is sampled (native
    # abscissae + trapezoid vs fixed stencil + rectangle): they must agree to well
    # under the metric's own change between the two finest rungs (0.893 -> 0.321
    # once the nx=12800 rung landed in R14; 1.776 -> 0.893 before that).
    ks = sorted(int(k) for k in verdict["ladder"])
    delta = (verdict["ladder"][str(ks[-2])]["l1_percent"]
             - verdict["ladder"][str(ks[-1])]["l1_percent"])
    assert diff < 0.15, diff
    assert diff < 0.2 * delta, (diff, delta)
    for row in verdict["ladder"].values():
        assert row["l1_percent"] > 0.0
        assert row["l1_percent"] < r13.BAR_LEDGER


def test_genuineness_gate_reads_real_grid_geometry(verdict):
    """Every rung the verdict used must pass E0 (dx, cells, positivity, drift)."""
    measured = {k: g for k, g in verdict["E0_genuineness"].items() if g["metrics_available"]}
    assert measured, "no rung carries a metrics sidecar"
    for gate in measured.values():
        assert gate["dx_rel_error"] < 1e-12, gate
        assert gate["positive"] and gate["all_finite"], gate
        assert gate["checks"]["E0c_mass_drift"], gate
    for gate in verdict["E0_genuineness"].values():
        if not gate["metrics_available"]:
            assert "E0c_mass_drift" not in gate["checks"], gate
    assert verdict["gates_all_pass"] is True


def test_honest_ceiling_adds_the_reference_bar(r13, verdict):
    """Upper bound = L1 + eps_ref per convention; never the point estimate alone."""
    e4 = verdict["E4_honest_ceiling"]
    if e4.get("status") != "measured":
        pytest.skip("reference-side pair not available in this checkout")
    eps = (e4["bars"]["B0"]["eps_percent"] if isinstance(e4["bars"]["B0"], dict)
           else e4["bars"]["B0"])
    for row in e4["rows"]:
        b0 = row["upper_bound_percent"]["B0"]
        assert b0 == pytest.approx(row["l1_percent"] + eps, rel=0, abs=2e-3)
        assert b0 > row["l1_percent"], "a bar that does not raise the bound is a bug"


def test_mixed_instrument_reference_is_not_licensed(verdict):
    """Carrying two CFLs across the reference pair must be a probe, not a certified bar."""
    e4 = verdict["E4_honest_ceiling"]
    if e4.get("status") != "measured":
        pytest.skip("reference-side pair not available in this checkout")
    cfls = {r["cfl"] for r in e4["reference_pair"]}
    assert e4["licensed"] is (len(cfls) == 1)
    if not e4["licensed"]:
        assert "探针" in e4["license_note"]


def test_status_is_honest_about_the_missing_rung(verdict):
    """not_measured while the nx=12800 graded rung is absent -- never a silent pass."""
    ladder_ks = {int(k) for k in verdict["ladder"]}
    if 12800 not in ladder_ks:
        assert verdict["status"] == "not_measured"
        assert 12800 in verdict["missing_rungs"]
        assert "不裁决" in verdict["adjudication"]
    else:
        assert verdict["status"] == "measured"


def test_cfl_scaling_probe_is_quoted_with_its_own_numbers(r13):
    """The mixed-instrument licence argument leans on the stored CFL probe; keep it bound."""
    p = HERE / "docs" / "XVAL-FVM-M18737-ROUND2" / "cfl_scaling_local.json"
    d = json.loads(p.read_text())
    worst = max(r["d_l1_percent"] for r in d["rows"])
    assert worst < 0.2, "the CFL probe itself must sit below the admission bar"
    # ... and well below the reference-side pair difference it is used to bound
    pair = d["bars_from_stored_ladder"]["pair_l1_percent"]
    assert worst < 0.5 * pair
