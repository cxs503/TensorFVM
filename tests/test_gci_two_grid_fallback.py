"""Can the two-grid ``Fs = 3`` fall-back band be confidently wrong?

``richardson_gci`` promises that "a broken refinement study can never report a
confident band".  Before this gate the fall-back took its whole width from the
fine/medium pair, ``Fs|(f2-f1)/f1|/(r21^p-1)``, scaled by ``|f_ext|``.  A
non-monotone series has ``f2-f1`` and ``f3-f2`` of opposite sign, and the
physical way to get there is two error terms that cancel between the fine and
the medium grid (*false convergence*): the fine pair then looks converged while
the coarse grid carries the error, so the band collapses precisely when the
discretisation error is largest.

These gates pin the audit in ``scripts/audit_gci_two_grid_fallback.py``:

* the pre-fix replica agrees with the library whenever the series is consistent
  (the change is confined to the fall-back -- no error bar in the repository
  moves for a monotone study),
* a false-convergence case whose legacy band missed the exact value by 90 GCI
  widths is covered after the fix with efficiency ``<= 1``,
* the fall-back band is never narrower than the spread of its own three values,
* coverage over the analytic extremum family is complete on the informative
  subset, and the noisy-oscillation cases of the committed self-verification
  artifact are still inconsistent/flagged.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from tensorfvm.verification.richardson import SAFETY_FALLBACK, richardson_gci

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "audit_gci_two_grid_fallback.py"
ARTIFACT = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorfvm" / "verification"
            / "artifacts" / "richardson-gci-selfverification" / "summary.json")


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_gci_two_grid_fallback", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _consistent_series():
    rows = []
    for exact in (1.0, 10.0):
        for p in (1.0, 2.0, 3.0):
            for spacings in ((1.0, 2.0, 4.0), (1.0, 1.5, 2.25), (1.0, 2.0, 3.0)):
                h = np.asarray(spacings)
                for alpha in (0.0, 0.1, 0.3):
                    rows.append((exact, spacings, tuple(exact + h ** p + alpha * h ** (p + 1.))))
    return rows


def test_fix_is_confined_to_the_fall_back(audit):
    """Consistent series must be untouched: same extrapolation, same band."""
    for exact, spacings, values in _consistent_series():
        study = richardson_gci(*values, spacings, theory_order=2.0)
        legacy = audit.legacy_band(*values, spacings, theory_order=2.0)
        assert not legacy["fallback"], (spacings, values)
        assert study.convergence == "monotone"
        assert study.band == pytest.approx(legacy["band"], rel=0.0, abs=0.0)
        assert study.gci == pytest.approx(legacy["gci"], rel=0.0)


def test_false_convergence_band_was_confidently_wrong_and_is_now_covered(audit):
    """``f(h) = 10 + h^2 - 4h``: the fine pair cancels, the coarse grid is at 10."""
    spacings = (1.0, 2.0, 4.0)
    values = (7.0, 6.0, 10.0)          # exact value is the coarse grid itself
    exact = 10.0
    study = richardson_gci(*values, spacings, theory_order=2.0)
    legacy = audit.legacy_band(*values, spacings, theory_order=2.0)

    assert legacy["fallback"], "this series must trigger the two-grid fall-back"
    assert study.convergence == "inconsistent" and not study.asymptotic
    assert study.safety_factor == SAFETY_FALLBACK

    # Pre-fix: the band sat around 7.33 and did not reach the exact 10.
    assert not (legacy["band"][0] <= exact <= legacy["band"][1])
    legacy_efficiency = abs(legacy["extrapolated"] - exact) / legacy["half"]
    assert legacy_efficiency > 2.5, legacy_efficiency

    # Post-fix: covered, and the band is a bound (Roache efficiency <= 1).
    assert study.band[0] <= exact <= study.band[1], study.band
    half = 0.5 * (study.band[1] - study.band[0])
    assert abs(study.extrapolated - exact) / half <= 1.0


def test_fall_back_band_never_narrower_than_the_study_spread(audit):
    """An inconsistent study cannot claim precision better than its own data."""
    checked = 0
    for row in audit.extremum_rows() + audit.random_rows():
        study = richardson_gci(*row["values"], tuple(row["spacings"]), theory_order=2.0)
        if study.convergence != "inconsistent":
            continue
        checked += 1
        half = 0.5 * (study.band[1] - study.band[0])
        spread = max(abs(value - study.extrapolated) for value in row["values"])
        assert half >= spread * (1.0 - 1e-12), (row["values"], half, spread)
    assert checked >= 300, checked


def test_narrow_monotone_bands_are_flagged_non_asymptotic(audit):
    """Residual limitation: a pre-asymptotic series can look monotone, report an
    inflated observed order (4-5 instead of 2) and a band far narrower than the
    spread of its own values.  The band is not fixed here -- the editable gate is
    ``RefinementStudy.asymptotic``, which must reject every such study."""
    narrow, flagged = 0, 0
    for row in audit.extremum_rows():
        study = richardson_gci(*row["values"], tuple(row["spacings"]), theory_order=2.0)
        if study.convergence == "inconsistent":
            continue
        half = 0.5 * (study.band[1] - study.band[0])
        spread = max(abs(value - study.extrapolated) for value in row["values"])
        if half < spread * (1.0 - 1e-12):
            narrow += 1
            flagged += 0 if study.asymptotic else 1
            assert study.order_used / study.theory_order > 1.5
    assert narrow >= 20, narrow
    assert flagged == narrow, (narrow, flagged)


def test_extremum_family_coverage_is_complete_on_the_informative_subset(audit):
    """Every fall-back case whose exact value lies inside the data span is covered."""
    rows = audit.extremum_rows() + audit.random_rows()
    informative = [r for r in rows if r["fallback"] and r["informative"]]
    assert len(informative) >= 50, len(informative)
    misses = [r for r in informative if not r["covers"]]
    assert not misses, misses[:3]
    assert max(r["efficiency"] for r in informative) <= 1.0
    # The control arm must show that this is not vacuous.
    legacy_misses = [r for r in informative if not r["legacy_covers"]]
    assert len(legacy_misses) / len(informative) > 0.5
    assert max(r["legacy_efficiency"] for r in informative) > 10.0


def test_committed_selfverification_artifact_records_the_widened_band():
    """The two oscillatory cases keep their GCI, gain an explicit band."""
    payload = json.loads(ARTIFACT.read_text())
    rows = [row for row in payload["rows"] if row.get("convergence") == "inconsistent"]
    assert len(rows) >= 2
    for row in rows:
        study = richardson_gci(*row["values"], tuple(row["spacing_m"]),
                               theory_order=row["theory_order"])
        assert study.convergence == "inconsistent"
        assert row["gci"] == pytest.approx(study.gci, rel=0.0)
        assert row["band"] == pytest.approx(study.band, rel=0.0)
        half = 0.5 * (row["band"][1] - row["band"][0])
        assert half >= max(abs(v - study.extrapolated) for v in row["values"]) * (1.0 - 1e-12)