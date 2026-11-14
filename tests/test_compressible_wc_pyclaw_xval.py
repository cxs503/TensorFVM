"""Gate for the third-code (Clawpack/PyClaw) cross-validation of the double blast.

The stored artefact ``docs/woodward-colella-double-blast/pyclaw/results.json``
was produced by ``wc_pyclaw_xval.py`` running Clawpack's Classic wave-propagation
solver (independent authors, independent flux decomposition) on the Woodward &
Colella two-blast datum.  The assertions here bind the claims of
``docs/CASE-WOODWARD-COLELLA-pyclaw-xval.md`` to that artefact; they never
re-run the solver (which is skipped when Clawpack is not installed).

Asserted:

* Clawpack (Roe + entropy fix) at nx = 800/1600 reproduces the digitised
  published reference to < 3 % on the peak and < 3 % on the L1 profile norm —
  i.e. an independent code lands on the same numbers as TensorFVM, which is what
  makes the <10 % ledger verdict a property of the case and not of our solver;
* Clawpack vs TensorFVM (matched grids) is < 3 % in the L1 norm at nx >= 800 —
  an independent bound on the discretisation error of the graded run;
* the L1 error of Clawpack vs the reference decreases from nx = 400 to 1600;
* the closed tube conserves mass to < 2e-3 (Classic's wall-boundary quadrature);
* the Roe-vs-HLLE spread stays in the 2-8 % band recorded in the report, i.e.
  the contact-smearing of a contact-free flux is *larger* than the cross-code
  agreement — the honest width of "flux-model" uncertainty for this case.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("clawpack", reason="Clawpack/PyClaw is the third code under test")

RESULT = (Path(__file__).resolve().parents[1] / "docs"
          / "woodward-colella-double-blast" / "pyclaw" / "results.json")


@pytest.fixture(scope="module")
def data():
    if not RESULT.exists():                      # pragma: no cover
        pytest.skip("pyclaw artefact missing; run wc_pyclaw_xval.py")
    return json.loads(RESULT.read_text())


def _row(data, nx, flux):
    for r in data["rows"]:
        if r["nx"] == nx and r["flux"] == flux:
            return r
    pytest.skip(f"grid {nx}/{flux} not in artefact")


def test_artefact_is_the_expected_case(data):
    assert data["t_end"] == pytest.approx(0.038)
    assert "Classic" in data["third_code"]
    assert data["grids"] == [200, 400, 800, 1600]
    for r in data["rows"]:
        assert r["t"] == pytest.approx(0.038, abs=1e-3)


def test_clawpack_reproduces_reference_at_800(data):
    r = _row(data, 800, "roe")
    assert r["calibrated"]["peak_rel_percent"] < 3.0
    assert r["calibrated"]["l1_percent"] < 3.0
    assert r["calibrated"]["window_mass_percent"] < 1.0
    assert r["calibrated"]["verdict"] == "PASS"


def test_clawpack_reproduces_reference_at_1600(data):
    r = _row(data, 1600, "roe")
    assert r["calibrated"]["l1_percent"] < 3.0
    assert r["calibrated"]["peak_rel_percent"] < 3.0
    assert r["calibrated"]["verdict"] == "PASS"


def test_independent_code_agrees_with_tensorfvm(data):
    for nx in (800, 1600):
        r = _row(data, nx, "roe")
        assert r["vs_tensorfvm"]["l1_percent"] < 3.0, nx
        assert r["vs_tensorfvm"]["peak_x_abs"] < 0.02, nx


def test_l1_vs_reference_decreases_with_refinement(data):
    errs = [_row(data, nx, "roe")["calibrated"]["l1_percent"] for nx in (400, 800, 1600)]
    assert errs[1] < errs[0] and errs[2] < errs[1], errs


def test_closed_tube_conserves_mass(data):
    assert data["conservation"]["max_rel_drift"] < 2e-3
    assert data["conservation"]["initial"] == pytest.approx(1.0)


def test_flux_model_band_is_documented(data):
    """Contact-free HLLE is *wider* than the cross-code difference — recorded,
    not hidden: this is the honest flux-model width for this case."""
    for p in data["flux_pair"]:
        assert 2.0 < p["roe_vs_hlle_l1_percent"] < 8.0, p