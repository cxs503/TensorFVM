"""Gates for the literature-referenced check of the double blast waves.

The reference is the digitised "Reference" curve of Dumbser et al. (2019)
Fig. 21 — the pure RPJ solution on 3200 cells at T = 0.038 — see
``docs/woodward-colella-double-blast/reference/``.  What is asserted here:

* the digitised table is well formed and its two independent panel
  digitisation passes agree (they are the error budget of the reference),
* the face detector recovers a step it is given,
* the committed nx=800 profile, re-compared from scratch, still meets the ledger
  tolerance on peak density, peak position, the L1 profile norm, **and** the
  window mass integral (the well-posed replacement for the pathological
  ``low_plateau`` metric — see ``CASE-WOODWARD-COLELLA-plateau-diagnosis.md``),
* the shallow low-density plateau is reported honestly as *not* meeting 10%
  (its absolute deviation is ~2% of the peak density), so the case cannot be
  silently promoted to "all metrics pass",
* a fresh solver run is finite and conserves mass to round-off.
"""

import numpy as np
import pytest
import torch

from tensorfvm.compressible.wc_reference import (
    DEFAULT_REFERENCE,
    faces,
    load_reference,
    metrics_from_profiles,
    profile,
)

ARTIFACTS = DEFAULT_REFERENCE.parents[1] / "artifacts"


def test_reference_table_is_well_formed():
    ref = load_reference()
    x, rho = ref["x"], ref["rho"]
    assert np.all(np.diff(x) > 0), "reference grid must be strictly increasing"
    assert 0.39 < x.min() and x.max() < 1.01
    assert 6.0 < rho.max() < 6.6, "published peak density is ~6.3"
    assert 0.10 < rho.min() < 0.30, "published trough density is ~0.16"
    # the two independent digitisation passes (400-cell and 800-cell panels)
    assert ref["panel_spread"] < 1e-3, "panel digitisation disagree"


def test_face_detector_on_analytic_step():
    x = np.linspace(0.0, 1.0, 401)
    rho = np.where(x < 0.6, 1.0, 3.0)
    got = faces(x, rho)
    assert got, "a step must register at least one face"
    assert min(abs(np.array(got) - 0.6)) < 5e-3, got


def test_committed_nx800_profile_against_reference():
    """Re-compare the committed nx=800 profile from scratch.

    Primary verdict uses well-posed metrics only (peak, position, L1, faces,
    window_mass).  The ``low_plateau`` metric is diagnosed as pathological
    (edge-degenerate truncation) and excluded from the primary verdict.
    """
    data = np.load(ARTIFACTS / "wc_blast_profile_nx800.npz")
    ref = load_reference()
    m = metrics_from_profiles(ref["x"], ref["rho"], data["x_got"], data["rho_got"])
    # primary well-posed metrics
    assert m["peak_density"]["rel_error_percent"] < 3.0, m["peak_density"]
    assert m["peak_position"]["abs_error"] < 1e-2, m["peak_position"]
    assert m["l1_relative"]["percent"] < 5.0, m["l1_relative"]
    assert max(m["faces_abs_error"]) < 0.02, m["faces_abs_error"]
    assert m["window_mass"]["rel_error_percent"] < 3.0, m["window_mass"]
    # honest: the low-density plateau does not meet the 10% ledger tolerance
    assert not m["checks"]["low_plateau_lt_10pct"]
    assert m["low_plateau"]["rel_error_percent"] < 25.0
    assert m["low_plateau"]["percent_of_peak_density"] < 2.0
    # primary verdict should PASS (well-posed metrics only)
    assert m["verdict"] == "PASS"
    assert "low_plateau excluded" in m["verdict_note"]


def test_fresh_run_is_finite_and_conservative():
    run = profile(nx=120, t_end=0.004)
    rho = run["rho"]
    assert np.all(np.isfinite(rho)) and rho.min() > 0.0
    mass = torch.as_tensor(rho).mean().item()
    assert abs(mass - 1.0) < 1e-10, "closed tube must conserve mass to round-off"
