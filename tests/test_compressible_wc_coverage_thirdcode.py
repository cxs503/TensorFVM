"""Round-3 gates: an *independent third code* bound on the un-referenced strip.

The m20339 coverage audit argued that the blind strip ``x in [0, 0.4]`` -- the
part of the domain with no digitised literature reference (6.4 % of the mass) --
is bounded by "reference-free evidence".  Both witnesses it used (cross-scheme
WENO5/Rusanov vs MUSCL/HLLC/MC, cross-grid) live *inside* this repository and
share the grid/ghost-cell machinery.

``docs/woodward-colella-double-blast/pyclaw/profiles.npz`` holds Clawpack 5.14.0
Classic (independent authors, Fortran, own limiter, Roe+entropy-fix and HLLE) on
the **whole** domain at the same ``t = 0.038``.  This file gates the third-code
bound and the admissibility criterion that keeps a contaminated witness (HLLE
nx=1600, spurious contact overshoot ``rho_max = 9.04`` vs 6.34 at nx=800) out of
the convergence argument.

Every assertion recomputes from the committed profiles; the committed JSON is
cross-checked but never used as the oracle.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WC = REPO / "docs" / "woodward-colella-double-blast"
COV = WC / "coverage"
PYCLAW = WC / "pyclaw" / "profiles.npz"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cr():
    return _load("wc_coverage_robustness_tc", COV / "wc_coverage_robustness.py")


@pytest.fixture(scope="module")
def tc(cr):
    if not PYCLAW.exists():                      # pragma: no cover - data is committed
        pytest.skip("committed Clawpack profiles absent")
    return cr.third_code_evidence(cr.asy.load_profiles())


# --------------------------------------------------------------- 1. the witness
def test_third_code_witness_spans_the_blind_strip(tc):
    """The whole point: the third code must cover the strip the paper does not."""
    assert tc["available"] is True
    assert tc["domain"]["x_min"] < 0.01 and tc["domain"]["x_max"] > 0.99, tc["domain"]
    assert tc["domain"]["covers_blind_strip"] is True
    # two flux families -> the bound brackets the flux-model error
    assert set(tc["distance"]["1600"]) == {"roe", "hlle"}


def test_matched_grid_comparison_is_truly_matched(tc, cr):
    """No interpolation between codes at the headline grids."""
    profiles = cr.asy.load_profiles()
    third = cr.load_third_code()
    for g, tag in tc["matched_grids"].items():
        assert profiles[tag]["nx"] == int(g), (g, tag, profiles[tag]["nx"])
        assert third[g]["roe"]["nx"] == int(g)


# ----------------------------------------------------- 2. the blind-strip bound
def test_blind_strip_is_bounded_by_an_independent_code(tc):
    """<0.25 % of L1 density distance on the strip -- two orders under the tolerance."""
    v = tc["verdict"]
    assert v["blind_strip_max_percent_matched_nx1600"] < 0.25, v
    assert v["blind_strip_max_percent_clean_witnesses"] < 0.10, v
    assert v["tolerance_percent"] == pytest.approx(10.0)   # the acceptance is unchanged
    assert v["independently_corroborates_reference_free_bound"] is True
    # per-region explicit values (recomputed, hence stable to 1e-6)
    for flux, row in tc["distance"]["1600"].items():
        assert row["blind_left_(0,0.4)"]["percent"] < 0.25, (flux, row)


def test_window_is_where_independent_codes_diverge(tc):
    """The shock-bearing window, not the strip, carries the code disagreement."""
    v = tc["verdict"]
    assert v["window_is_where_independent_codes_diverge"] is True
    clean_window = v["window_max_percent_clean_witnesses"]
    clean_blind = v["blind_strip_max_percent_clean_witnesses"]
    assert clean_window > 5.0 * clean_blind, (clean_window, clean_blind)
    assert tc["distance"]["1600"]["roe"]["window_(0.4,1.0)"]["percent"] == pytest.approx(
        0.7301, abs=5e-3)


def test_cross_code_distance_converges_in_the_blind_strip(tc):
    """4-point sequence at fixed finest in-repo solution: 0.88 -> 0.36 -> 0.14 -> 0.05 %."""
    conv = tc["convergence"]["roe"]["blind_left_(0,0.4)"]
    seq = [conv["sequence"][g] for g in ("200", "400", "800", "1600")]
    assert all(b < a for a, b in zip(seq, seq[1:])), seq
    assert conv["monotone_decreasing"] is True
    assert conv["ratio_finest"] < 0.5, conv          # each doubling at least halves it
    assert seq[0] / seq[-1] > 10.0, seq              # ~16x over three doublings
    # the window converges too, from a much larger level
    w = tc["convergence"]["roe"]["window_(0.4,1.0)"]["sequence"]
    assert w["1600"] > 10.0 * seq[-1]


# -------------------------------------- 3. admissibility of a convergence witness
def test_contaminated_witness_is_rejected(tc):
    """HLLE nx=1600 inflates rho_max by 42 % on one refinement -> inadmissible."""
    q = tc["quality"]
    hlle = q["families"]["hlle"]
    assert hlle["admissible"] is False
    assert hlle["inadmissible_grids"] == [1600]
    flagged = [s for s in hlle["refinement_steps"] if s["exceeds_tolerance"]]
    assert len(flagged) == 1 and flagged[0]["grids"] == [800, 1600]
    assert flagged[0]["growth_percent"] > 40.0
    assert [list(x) for x in q["rejected_witnesses"]] == [["hlle", 1600]]
    assert q["admissible_families"] == ["roe"]


def test_quality_criterion_is_not_vacuous_growth(tc):
    """Ordinary refinement growth (+13 %, +15 %) must pass; only the jump is flagged."""
    for flux in ("roe", "hlle"):
        steps = tc["quality"]["families"][flux]["refinement_steps"]
        ordinary = [s for s in steps if s["grids"] in ([200, 400], [400, 800])]
        assert len(ordinary) == 2, steps
        for s in ordinary:
            assert 0.0 < s["growth_percent"] < 25.0, (flux, s)
            assert s["exceeds_tolerance"] is False, (flux, s)


def test_contaminated_witness_would_inflate_the_window_verdict(tc):
    """Quantify the defect the gate prevents: 5.38 % vs 0.73 % in the window."""
    d = tc["distance"]["1600"]
    contaminated = d["hlle"]["window_(0.4,1.0)"]["percent"]
    clean = d["roe"]["window_(0.4,1.0)"]["percent"]
    assert contaminated > 7.0 * clean, (contaminated, clean)   # 5.38 % vs 0.73 %
    # while in the strip the two fluxes agree to within 0.21 %, i.e. the artefact
    # is confined to the shock/contact region and cannot be blamed on the strip
    assert d["hlle"]["blind_left_(0,0.4)"]["percent"] < 0.25


# ----------------------------------------------------------- 4. the gate's teeth
def test_bound_gate_rejects_a_perturbed_strip(cr):
    """Anti-vacuity: a 2 % density bump in the strip must break the 0.25 % bound."""
    profiles = cr.asy.load_profiles()
    prof = profiles["nx1600_hllc"]
    rho = np.asarray(prof["rho"], float).copy()
    x = np.asarray(prof["x"], float)
    bump = rho * (1.0 + 0.02 * np.exp(-((x - 0.2) / 0.05) ** 2))
    base = cr.region_distance({"nx": prof["nx"], "x": x, "rho": rho},
                              {"nx": prof["nx"], "x": x, "rho": rho}, 0.0, 0.4)
    pert = cr.region_distance({"nx": prof["nx"], "x": x, "rho": bump},
                              {"nx": prof["nx"], "x": x, "rho": rho}, 0.0, 0.4)
    assert base["percent"] == pytest.approx(0.0, abs=1e-12)
    assert pert["percent"] > 0.25, pert["percent"]
    # and a numerically irrelevant perturbation is not pushed over the bound
    tiny = rho * (1.0 + 1e-8)
    t = cr.region_distance({"nx": prof["nx"], "x": x, "rho": tiny},
                           {"nx": prof["nx"], "x": x, "rho": rho}, 0.0, 0.4)
    assert t["percent"] < 1e-5, t["percent"]


def test_window_only_artifacts_are_never_used_as_third_code_input(cr):
    """The third-code block reads only whole-domain data; assert the contract."""
    third = cr.load_third_code()
    assert third, "committed Clawpack profiles must be readable"
    for grid, fams in third.items():
        for flux, row in fams.items():
            assert row["x"].min() < 0.01 and row["x"].max() > 0.99, (grid, flux)


# ------------------------------------------------------- 5. artifact consistency
def test_committed_artifact_matches_recomputation(tc, cr):
    committed = json.loads((COV / "coverage_robustness.json").read_text())
    assert "third_code" in committed, "coverage_robustness.json must carry the round-3 block"
    c = committed["third_code"]
    assert c["available"] is True
    assert c["verdict"]["blind_strip_max_percent_matched_nx1600"] == pytest.approx(
        tc["verdict"]["blind_strip_max_percent_matched_nx1600"], abs=1e-9)
    assert c["verdict"]["window_max_percent_clean_witnesses"] == pytest.approx(
        tc["verdict"]["window_max_percent_clean_witnesses"], abs=1e-9)
    assert c["quality"]["admissible_families"] == tc["quality"]["admissible_families"]
    for g in ("200", "1600"):
        for flux in ("roe", "hlle"):
            for name in ("blind_left_(0,0.4)", "window_(0.4,1.0)", "full_(0,1)"):
                assert c["distance"][g][flux][name]["percent"] == pytest.approx(
                    tc["distance"][g][flux][name]["percent"], abs=1e-9), (g, flux, name)
    # the coverage figure must exist and be non-trivial (it now carries the third-code bars)
    figs = committed["figures"]
    assert "fig_wc_domain_coverage.png" in figs
    assert (COV / "fig_wc_domain_coverage.png").stat().st_size > 20_000