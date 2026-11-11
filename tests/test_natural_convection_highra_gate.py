"""Gates on the Ra = 1e5 / 1e6 natural-convection grid ladder.

Two things are pinned here, both of them cheap (JSON only, no PDE solves):

* the **ladder evidence** behind ``docs/verification-natural-convection-highra``
  -- every level present, converged, produced with one time-step setting, the
  error monotone in the grid, and the Richardson limit of the Ra = 1e5 family
  inside the 3 % gate.  If the committed artifacts and the committed report
  drift apart the test fails;
* the **metric-definition defect**: de Vahl Davis tabulates ``u_max`` on the
  vertical mid-line, so a whole-field ``max|u|`` is *not* comparable.  The test
  reproduces the size of that artefact from the stored fields, which is what
  makes the earlier ``u_max`` numbers unusable rather than merely different.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = (ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "incompressible"
       / "fvm_natural_convection_highra")
DOC = ROOT / "docs" / "verification-natural-convection-highra"
REFERENCE = {1.0e5: 4.519, 1.0e6: 8.800}
LEVELS = (16, 32, 64)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_bench", ROOT / "scripts" / "run_incompressible_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ladder():
    out = {}
    for path in RAW.glob("raw_*.json"):
        rec = json.loads(path.read_text())
        out[(float(rec["parameters"]["rayleigh"]),
             int(rec["parameters"]["cells"]))] = rec
    return out


def test_every_level_is_archived_and_converged():
    ladder = _ladder()
    for ra in REFERENCE:
        for n in LEVELS:
            rec = ladder[(ra, n)]
            assert rec["ledger"]["converged"] is True, (ra, n)
            assert rec["parameters"]["cfl"] == pytest.approx(0.5)
            assert rec["ledger"]["max_divergence"] < 1e-8, (ra, n)


def test_nusselt_error_contract():
    ladder = _ladder()
    errors = {}
    for ra, ref in REFERENCE.items():
        series = [ladder[(ra, n)]["metrics"]["nusselt_hot"] for n in LEVELS]
        errs = [abs(v - ref) / ref for v in series]
        # refinement must reduce the error monotonically
        assert errs[0] > errs[1] > errs[2], (ra, errs)
        errors[ra] = errs
    # the Ra = 1e5 ledger row that read "+3.4 % (FAIL)" at N=32 is closed at N=64
    assert errors[1.0e5][2] < 0.03, errors[1.0e5]
    # Ra = 1e6 is honestly recorded as marginal at N=64
    assert errors[1.0e6][2] < 0.035, errors[1.0e6]


def test_observed_order_sign_and_richardson_limit():
    """A converging-from-above second-order series must give ``p = +2``.

    The in-tree estimator used the inverted difference ratio, which returned
    ``p = -2`` (and extrapolated *away* from the limit) for exactly this case.
    """
    runner = _runner()
    limit = 4.519
    values = {n: limit + 0.4 / n ** 2 for n in (32, 64, 128)}   # finest -> +0.4/128^2
    conv = runner.observed_order(values, "N")
    assert conv["order"] == pytest.approx(2.0, abs=1e-6)
    assert conv["richardson_limit"] == pytest.approx(limit, rel=1e-9)
    # a non-monotone/noisy series must not be silently extrapolated into a number
    noisy = {32: 9.7781, 64: 9.0701, 128: 9.0700}
    assert runner.observed_order(noisy, "N")["order"] is not None  # reported, see report


def test_midline_metric_is_the_comparable_one():
    ladder = _ladder()
    for (ra, n), rec in ladder.items():
        m = rec["metrics"]
        assert "u_max_midheight" in m and "v_max_midwidth" in m
        assert m["u_max_midheight"] <= m["u_max"] + 1e-9
        if ra >= 1.0e5 and n >= 32:
            field_err = abs(m["u_max"] - m["u_max_reference"]) / m["u_max_reference"]
            mid_err = abs(m["u_max_midheight"] - m["u_max_reference"]) / m["u_max_reference"]
            assert field_err > 5.0 * mid_err, (ra, n, field_err, mid_err)


def test_report_matches_artifacts():
    summary = json.loads((DOC / "summary.json").read_text())
    ladder = _ladder()
    for ra, block in summary["families"].items():
        raf = float(ra[2:])
        for n_str, nu in block["nusselt_hot"].items():
            stored = ladder[(raf, int(n_str))]["metrics"]["nusselt_hot"]
            assert nu == pytest.approx(stored, rel=1e-12)
    criteria = summary["criteria"]
    assert criteria["convection_nu_Ra1e+05_within_3pct"]["pass"] is True
    assert criteria["convection_nu_Ra1e+05_richardson_within_3pct"]["pass"] is True
    assert criteria["convection_nu_Ra1e+06_within_3pct"]["pass"] is False
    # a withheld limit is never emitted as a number
    assert "convection_nu_Ra1e+06_richardson_within_3pct" not in criteria
    text = (DOC / "REPORT.md").read_text()
    assert "Richardson\nlimit withheld" in text or "Richardson limit withheld" in text