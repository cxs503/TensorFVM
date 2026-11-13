"""Gates for the axis-calibration correction of the Woodward-Colella reference.

The committed reference profile of the double blast was digitised from the
tick *label* text boxes of Dumbser et al. (2019) Fig. 21.  For the vertical axis
that method has a constant offset: a digit's bbox centre sits 0.337 pt above the
data line it labels, and a least-squares fit over equally spaced labels absorbs
that offset into the intercept, so the fit residual cannot reveal it.  Every
digitised density is therefore 0.0220 too low — negligible at the peak (rho~6.3)
and dominant in the near-vacuum trough (rho~0.16, the one metric the case could
not close at 13-14 %).

What is asserted here:

* the geometry-derived calibration (plot frame + stroked tick marks, which carry
  no glyph metrics) is internally sound: uniform tick spacing, frame corners on
  the extreme ticks, an *x* fit that lands on the frame to <0.01 pt (i.e. the
  method is unbiased horizontally, so the vertical offset is a glyph effect and
  not a fit failure);
* the label centres are offset from their tick marks by a *constant* (std < 5e-3
  pt), which is what makes the correction a clean additive shift;
* the two panels give the same bias (spread < 1e-3 pt);
* the calibrated table is exactly the committed table plus the bias (the raw
  digitisation is never overwritten);
* re-scoring the stored profiles against the calibrated reference closes the
  near-vacuum plateau at every grid (nx = 200 ... 1600), i.e. the 14 % was the
  instrument, not the solver — and the committed-reference numbers are still
  reproduced, so the honest raw record stays visible;
* a fresh geometry extraction reproduces the committed bias when the PDF is
  available (skipped otherwise: the PDF is not committed, only its sha256).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.compressible import wc_reference as wcr

REFDIR = Path(wcr.DEFAULT_REFERENCE).parent
CALIB_JSON = REFDIR / "axis_calibration.json"
CALIBRATED_CSV = REFDIR / "reference_density_calibrated.csv"
PROFILE_DIR = Path(wcr.DEFAULT_REFERENCE).parents[1] / "artifacts"
LADDER = Path(wcr.DEFAULT_REFERENCE).parents[1] / "ladder" / "data"


@pytest.fixture(scope="module")
def calib():
    return json.loads(CALIB_JSON.read_text())


def test_geometry_calibration_is_sound(calib):
    assert len(calib["panels"]) == 2, "both panels of Fig. 21 must be checked"
    for p in calib["panels"]:
        assert len(p["y_tick_marks"]) == 10, p["y_tick_marks"]
        # a constant spacing was *measured*, not assumed
        assert p["y_tick_spacing_max_deviation_pt"] < 1e-3, p
        # ... and the frame corners are the extreme tick marks
        assert max(p["frame_vs_extreme_tick_pt"].values()) < 0.05, p
        # the same text-fit method is unbiased horizontally (lands on the frame)
        assert abs(p["x_fit_lands_on_frame"]["left_pt"]) < 0.01, p
        assert abs(p["x_fit_lands_on_frame"]["right_pt"]) < 0.01, p
        # => the vertical offset is a glyph-metric effect, not a bad fit
        assert p["y_label_residual_pt"]["std"] < 5e-3, p
        assert 0.30 < abs(p["y_label_residual_pt"]["mean"]) < 0.40, p


def test_bias_is_panel_independent_and_as_recorded(calib):
    assert calib["bias_pt_spread"] < 1e-3, calib["bias_pt_spread"]
    assert calib["bias_density_spread"] < 1e-5, calib["bias_density_spread"]
    assert abs(calib["bias_pt"] - 0.337) < 5e-3, calib["bias_pt"]
    # 0.3370 pt at 15.3070 pt per unit of density
    assert abs(calib["bias_density"] - 0.022017) < 1e-4, calib["bias_density"]
    scale = calib["panels"][0]["y_tick_scale_pt_per_unit"]
    assert abs(calib["bias_pt"] / scale - calib["bias_density"]) < 1e-5


def test_calibrated_table_is_committed_table_plus_bias(calib):
    raw = wcr.load_reference()
    cal = wcr.load_reference(CALIBRATED_CSV)
    assert np.array_equal(raw["x"], cal["x"])
    d = cal["rho"] - raw["rho"]
    assert np.max(np.abs(d - calib["bias_density"])) < 1e-6, "not a pure shift"
    # the correction moves the published peak by well under the tolerance
    assert abs(cal["rho"].max() / raw["rho"].max() - 1) < 5e-3


def _profile(tag: str):
    p = PROFILE_DIR / f"wc_blast_profile_{tag}.npz"
    if p.exists():
        d = np.load(p)
        return d["x_got"], d["rho_got"]
    q = LADDER / f"ladder_profile_{tag}.npz"
    if q.exists():
        d = np.load(q)
        return d["x"], d["rho"]
    pytest.skip(f"profile {tag} not stored")


def test_committed_reference_reproduces_the_raw_honest_numbers():
    """The uncalibrated record still shows the 14 % plateau (no data was edited)."""
    x, rho = _profile("nx800")
    m = wcr.metrics_from_profiles(wcr.load_reference()["x"],
                                  wcr.load_reference()["rho"], x, rho)
    assert 13.0 < m["low_plateau"]["rel_error_percent"] < 15.0, m["low_plateau"]
    assert m["peak_density"]["rel_error_percent"] < 3.0, m["peak_density"]


@pytest.mark.parametrize("tag,plateau_max,l1_max", [
    ("nx400", 1.0, 7.5),
    ("nx800", 1.0, 2.5),
    ("nx1600", 1.0, 2.5),
])
def test_calibrated_reference_closes_the_plateau(tag, plateau_max, l1_max):
    x, rho = _profile(tag)
    cal = wcr.load_reference(CALIBRATED_CSV)
    m = wcr.metrics_from_profiles(cal["x"], cal["rho"], x, rho)
    # the near-vacuum trough, previously 13-14 %, is now sub-percent at *every*
    # grid -- the signature of an additive reference bias, not a discretisation
    # error that would shrink with resolution
    assert m["low_plateau"]["rel_error_percent"] < plateau_max, m["low_plateau"]
    assert m["l1_relative"]["percent"] < l1_max, m["l1_relative"]


def test_calibrated_peak_and_faces_at_finest_grid():
    x, rho = _profile("nx1600")
    cal = wcr.load_reference(CALIBRATED_CSV)
    m = wcr.metrics_from_profiles(cal["x"], cal["rho"], x, rho)
    assert m["peak_density"]["rel_error_percent"] < 3.0, m["peak_density"]
    assert m["peak_position"]["abs_error"] < 1e-2, m["peak_position"]
    assert max(m["faces_abs_error"]) < 0.02, m["faces_abs_error"]
    assert m["verdict"] == "PASS", m["checks"]


def test_optional_reextraction_from_pdf(tmp_path):
    """Re-derive the bias from the paper PDF if it happens to be present."""
    pdf = Path("/tmp/wc.pdf")
    if not pdf.exists():
        pytest.skip("paper PDF not available (not committed); sha256 is recorded")
    import importlib.util
    import fitz  # noqa: F401

    script = REFDIR / "verify_axis_calibration.py"
    spec = importlib.util.spec_from_file_location("vac", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    curves = json.loads((REFDIR / "curves.json").read_text())
    res = mod.analyse(fitz.open(pdf)[curves["page"]], curves["panels"])
    rec = json.loads(CALIB_JSON.read_text())
    assert abs(res["bias_density"] - rec["bias_density"]) < 1e-6, res["bias_density"]
    assert abs(res["bias_pt"] - rec["bias_pt"]) < 1e-3, res["bias_pt"]