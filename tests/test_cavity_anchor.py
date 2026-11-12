"""The de Vahl Davis reference layers and the cavity anchors are consistent.

Covers three things that have all gone wrong in this worktree at least once:

* the reference *variant* registry (a value must be looked up by variant name, and
  an ambiguous quantity must refuse to be guessed);
* the superseded-digit scan (a wrong transcription that survives its repair);
* the ``|psi|_max`` diagnostic of ``NaturalConvectionCase`` -- the m16902 defect,
  where ``argmax`` was taken on the signed streamfunction so the benchmark
  quantity came back as ``0`` and its "location" was a zero of ``psi``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tensorfvm.internal_flow.natural_convection import (
    NaturalConvectionCase,
    NaturalConvectionConfig,
)
from tensorfvm.verification import reference_families as rf

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- the registry
def test_acceptance_family_matches_the_two_source_tables() -> None:
    """The registry must agree with both in-repo reference tables, digit for digit."""
    from tensorfvm.inc_bench.reference import CAVITY_REFERENCES
    from tensorfvm.internal_flow.reference import DE_VAHL_DAVIS_NU, DE_VAHL_DAVIS_VELOCITY

    for rayleigh, ref in CAVITY_REFERENCES.items():
        assert rf.cavity_reference("nusselt", rayleigh).value == pytest.approx(DE_VAHL_DAVIS_NU[rayleigh])
        assert rf.cavity_reference("nusselt", rayleigh).value == pytest.approx(ref.nusselt)
        assert rf.cavity_reference("u_max", rayleigh).value == pytest.approx(DE_VAHL_DAVIS_VELOCITY[rayleigh]["u_max"])
        assert rf.cavity_reference("v_max", rayleigh).value == pytest.approx(DE_VAHL_DAVIS_VELOCITY[rayleigh]["v_max"])
    # the streamfunction digit exists in inc_bench only, and only for the acceptance
    # family; the finite-grid digit (16.32) is registered as a *variant*.
    for rayleigh in (1.0e3, 1.0e4, 1.0e5):
        assert rf.cavity_reference("psi_abs_max", rayleigh).value == pytest.approx(
            CAVITY_REFERENCES[rayleigh].psi_max)


def test_ambiguous_quantity_refuses_to_guess() -> None:
    with pytest.raises(ValueError):
        rf.cavity_reference("psi_abs_max", 1.0e6)
    assert rf.cavity_reference("psi_abs_max", 1.0e6, variant="acceptance").value == pytest.approx(16.750)
    assert rf.cavity_reference("psi_abs_max", 1.0e6, variant="finite_grid").value == pytest.approx(16.320)
    # the spread is what makes the guess dangerous: larger than a 3 % gate
    assert rf.AMBIGUITIES["psi_abs_max"]["spread_percent"] > 3.0


def test_unregistered_combinations_raise() -> None:
    with pytest.raises(KeyError):
        rf.cavity_reference("nusselt", 2.0e4)
    with pytest.raises(KeyError):
        rf.cavity_reference("u_max", 1.0e4, variant="finite_grid")


def test_registry_json_is_complete() -> None:
    payload = rf.registry_json()
    for quantity in ("nusselt", "psi_abs_max", "u_max", "v_max"):
        table = payload["families"][quantity]
        assert len(table) == 4
        assert all(isinstance(v, dict) and "acceptance" in v for v in table.values())
    assert payload["ambiguities"]["psi_abs_max"]["variants"]["acceptance"] == 16.750
    assert payload["ambiguities"]["psi_abs_max"]["variants"]["finite_grid"] == 16.320


# ------------------------------------------------------ superseded digit scan
def test_superseded_scan_finds_planted_digit(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("| Ra=1e4 | u_max = 4.535 |\n", encoding="utf-8")
    (tmp_path / "clean.md").write_text("| Ra=1e4 | u_max = 16.178 |\n", encoding="utf-8")
    hits = rf.superseded_hits(tmp_path)
    assert [h["value"] for h in hits] == [4.535]
    assert hits[0]["quantity"] == "u_max"


def test_superseded_scan_is_exact_not_substring(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("x = 11.3340 and 1.3340\n", encoding="utf-8")
    assert rf.superseded_hits(tmp_path) == []


def test_superseded_digits_survive_only_in_documented_history() -> None:
    """Two stale quotes are still in historical reports; they are logged in D15.

    The gate here is that the scan is *live* (it finds them) and that the set of
    files carrying a superseded digit does not grow silently.
    """
    hits = rf.superseded_hits(ROOT)
    paths = {h["path"] for h in hits}
    allowed = {"docs/three-point-wall-projection/report.md",
               "docs/native-viscous-three-point-closure/nonorthogonal-audit-report.md"}
    assert paths, "the scan found nothing: it is not actually live"
    assert paths <= allowed, paths - allowed


# ------------------------------------------------- the |psi|max diagnostic path
@pytest.fixture(scope="module")
def small_case() -> dict[str, object]:
    case = NaturalConvectionCase(NaturalConvectionConfig(n=16, rayleigh=1.0e3, tolerance=1e-6))
    case.run()
    diag = case.diagnostics()
    return {"case": case, "diag": diag}


def test_psi_abs_max_is_the_benchmark_magnitude(small_case) -> None:
    """Regression for m16902: the signed argmax returned a zero of psi, not the vortex centre."""
    diag = small_case["diag"]
    alpha = small_case["case"].config.diffusivity
    psi = small_case["case"].solver.streamfunction() / alpha
    assert diag["psi_abs_max_over_alpha"] == pytest.approx(float(psi.abs().max()), rel=1e-12)
    assert diag["psi_abs_max_over_alpha"] > 0.5
    # the old (defective) reading was ~0: the signed maximum lives at the corner where psi = 0
    assert float(psi.max()) < 1e-6 * diag["psi_abs_max_over_alpha"]


def test_psi_extremum_is_interior_and_signed_consistently(small_case) -> None:
    diag = small_case["diag"]
    x, y = diag["psi_max_location"]
    assert 0.05 < x < 0.95 and 0.05 < y < 0.95
    assert diag["psi_extremum_signed_over_alpha"] == pytest.approx(-diag["psi_abs_max_over_alpha"], rel=1e-12)


def test_mid_plane_and_global_extrema_agree(small_case) -> None:
    diag = small_case["diag"]
    assert diag["mid_plane_vs_global"]["u"] < 0.05
    assert diag["mid_plane_vs_global"]["v"] < 0.05
    assert min(diag["u_max_over_alpha"], diag["v_max_over_alpha"]) > 0.5


def test_two_streamfunction_quadratures_agree(small_case) -> None:
    """The solver's left-point cumsum and an independent trapezoid rule must agree."""
    from tensorfvm.verification.cavity_anchor import _independent_streamfunction

    case = small_case["case"]
    alpha = case.config.diffusivity
    solver_route = case.solver.streamfunction() / alpha
    independent = _independent_streamfunction(case) / alpha
    gap = float((solver_route - independent).abs().max()) / float(solver_route.abs().max())
    # n = 16 is the coarsest mesh: the two quadratures differ by their truncation
    # term, which is O(h) -- 7.3e-3 at n = 32, larger here.  A normalisation slip
    # would be O(1), so the band still has teeth.
    assert gap < 5e-2, gap


def test_rayleigh_identity_is_preserved(small_case) -> None:
    case = small_case["case"]
    assert case.rayleigh_check() == pytest.approx(1.0e3, rel=1e-10)
    assert math.isfinite(case.rayleigh_check())