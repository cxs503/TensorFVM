"""Machine gate for the m20098 board audit of the m20000 cavity bundle.

The audit numbers live in ``artifacts/CASE-CAVITY-RE1000-FVM/m20098_audit.json``
together with a text copy of every centreline profile used, so this gate can
re-derive each headline number **without** the binary ``.pt`` fields (which are
not tracked in this repo).  If the JSON is hand-edited, the profiles no longer
reproduce it and the gate fails; if the reference transcription drifts, the
paper's published extrema no longer match and the gate fails.

Run::

    PYTHONPATH=src python3 -m pytest tests/test_board_audit_m20098.py -q
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# the committed copy lives under docs/ (artifacts/ is outside this worktree's
# sparse-checkout definition and therefore not committable); the working copy
# under artifacts/ is accepted as a fallback
AUDIT_JSON = next((p for p in (
    ROOT / "docs" / "CASE-CAVITY-RE1000-FVM" / "m20098_audit.json",
    ROOT / "artifacts" / "CASE-CAVITY-RE1000-FVM" / "m20098_audit.json") if p.exists()),
    ROOT / "docs" / "CASE-CAVITY-RE1000-FVM" / "m20098_audit.json")
SCRIPT = ROOT / "scripts" / "audit_cavity_re1000_m20098.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("audit_m20098", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    if not AUDIT_JSON.exists():
        pytest.skip(f"missing {AUDIT_JSON}")
    return json.loads(AUDIT_JSON.read_text())


def test_reference_self_check_matches_the_paper():
    """The Ghia tables must reproduce the published extrema before scoring."""
    mod = _audit_module()
    ref = mod.reference_self_check()
    assert ref["u_min"] == pytest.approx(-0.38289, abs=1e-9)
    assert ref["u_min_at"] == pytest.approx(0.1719, abs=1e-9)
    assert ref["v_max"] == pytest.approx(0.37095, abs=1e-9)
    assert ref["v_max_at"] == pytest.approx(0.1563, abs=1e-9)


def test_headline_metrics_are_reproducible_from_the_committed_profiles(audit):
    """Every stored metric is re-derived from the committed centreline samples."""
    mod = _audit_module()
    rows = {(r["n"], r["rule"]): r for r in audit["rows"]}
    for field in audit["fields"]:
        n = field["n"]
        ys = field["u_centreline"]["y"]
        fu = field["u_centreline"]["u"]
        xs = field["v_centreline"]["x"]
        fv = field["v_centreline"]["v"]
        for rule in ("linear", "quadratic"):
            row = rows[(n, rule)]
            # interior points only: the wall rows (y=1 lid, y=0 floor, x=0/1 side
            # walls) carry their exact Dirichlet value by definition of the case,
            # so they are never extrapolated into the pointwise norm
            du = [abs(mod.INTERP[rule](ys, fu, y) - ref) for y, ref in mod.GHIA_U
                  if 0.0 < y < 1.0]
            dv = [abs(mod.INTERP[rule](xs, fv, x) - ref) for x, ref in mod.GHIA_V
                  if 0.0 < x < 1.0]
            assert max(du) == pytest.approx(row["pointwise_max_abs_du"], rel=1e-9)
            assert max(dv) == pytest.approx(row["pointwise_max_abs_dv"], rel=1e-9)
            # extrema from a dense sweep of the same interpolant
            gy = [k / 2000.0 for k in range(1, 2000)]
            gx = [k / 2000.0 for k in range(1, 2000)]
            u_min = min(mod.INTERP[rule](ys, fu, y) for y in gy)
            v_max = max(mod.INTERP[rule](xs, fv, x) for x in gx)
            assert u_min == pytest.approx(row["u_min"], rel=1e-9)
            assert v_max == pytest.approx(row["v_max"], rel=1e-9)


def test_ledger_verdict_per_resolution(audit):
    """Ledger row of tensor_benchmark_cases, applied to the audited fields."""
    verdicts = audit["verdict_per_resolution"]
    assert verdicts["32"]["pass_3pct"] is False, "the under-resolved control row must fail"
    assert "u_min_rel<0.03" in verdicts["32"]["failing"]
    for n in ("64", "128"):
        assert verdicts[n]["pass_3pct"] is True, f"n={n} must satisfy the 3% band"
        assert verdicts[n]["pass_strict"] is True
    r128 = [r for r in audit["rows"] if r["n"] == 128 and r["rule"] == "quadratic"][0]
    assert r128["u_min_rel_err"] < 0.01 and r128["v_max_rel_err"] < 0.01
    assert r128["pointwise_max_abs_du"] < 0.01
    # the vortex criterion (|d|/L <= 0.01 @ n>=128) must hold for both psi definitions
    assert r128["vortex_centre_psi_poisson"]["distance"] <= 0.01
    assert r128["vortex_centre_psi_integrated"]["distance"] <= 0.01


def test_field_invariants(audit):
    """The audited fields are incompressible, wall-tight and flux-free."""
    for field in audit["fields"]:
        assert field["max_abs_divergence"] < 1e-12, field["n"]
        b = field["boundary"]
        assert b["side_wall_max_abs_u"] < 1e-12
        assert b["top_max_abs_v"] < 1e-12 and b["bottom_max_abs_v"] < 1e-12
        assert b["max_abs_column_mean_u"] < 1e-12  # closed cavity: zero net flux
        psi = field["psi_extrema"]
        assert -0.13 < psi["min"] < -0.10 and 0.0 <= psi["max"] < 0.01


def test_m20000_gate_lid_assertion_is_unreachable(audit):
    """Documents a defect of the audited bundle's own gate.

    ``tests/test_case_cavity_re1000_m20000.py::test_division_free_and_lid_driven``
    asserts ``uc[:, -1].mean() > 0.9`` (mean u in the cell row below the lid).
    That quantity is O(h) short of the lid speed in the shear layer, so the
    assertion cannot hold at any of the three grids (0.54 / 0.73 / 0.85 here) --
    the gate fails for a test-criterion reason, not a solver reason.
    """
    means = {f["n"]: f["boundary"]["top_cell_row_mean_u"] for f in audit["fields"]}
    assert all(v < 0.9 for v in means.values()), means
    assert means[32] < means[64] < means[128], means
    assert means[128] > 0.8, "the lid is genuinely driving the flow"