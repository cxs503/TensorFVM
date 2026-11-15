"""Round 21: independent field-based metric + the two definitions of ``x_r/H``.

Everything here is decidable without a solver: synthetic fields with prescribed
reversed-flow extents, and (when the round-20 artefacts are present) the real
mean fields of the closure-free sister cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_field_metrics import (
    DEFINITION_ENSEMBLE, DEFINITION_MEAN_FIELD, backflow_extent,
    backflow_extent_rows, definition_spread,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def synthetic_u(x, x_start, x_end, u_outer=1.0, u_rev=-0.3):
    """A plug-like profile: reversed exactly for ``x_start <= x <= x_end``."""
    u = np.full_like(np.asarray(x, dtype=float), u_outer)
    u[(x >= x_start) & (x <= x_end)] = u_rev
    return u


# --------------------------------------------------------------------------- #
# 1. the field metric itself
# --------------------------------------------------------------------------- #
def test_backflow_extent_is_the_prescribed_edge():
    x = np.linspace(0.05, 8.95, 96)
    for start, end in ((1.0, 4.0), (2.0, 7.75), (0.5, 1.5)):
        out = backflow_extent(synthetic_u(x, start, end), x)
        # the last cell inside [start, end], i.e. the largest grid point <= end
        assert out["x_end"] == pytest.approx(x[x <= end + 1e-12].max())
        assert out["contiguous_from_first"] is True


def test_no_backflow_is_none_not_zero():
    x = np.linspace(0.05, 8.95, 96)
    out = backflow_extent(np.ones_like(x), x)
    assert out["x_end"] is None and out["n_cells"] == 0


def test_detached_patch_is_not_merged_into_the_bubble():
    x = np.linspace(0.05, 8.95, 96)
    u = synthetic_u(x, 1.0, 3.0)
    u[(x >= 6.0) & (x <= 6.5)] = -0.2  # detached reversed patch
    out = backflow_extent(u, x)
    assert out["contiguous_from_first"] is False
    assert out["n_runs"] == 2
    assert out["x_end"] == pytest.approx(x[x <= 3.0].max())


def test_periodic_seam_is_reported():
    x = np.linspace(0.05, 8.95, 96)
    u = synthetic_u(x, 0.0, 1.0)
    u[x > 8.0] = -0.2
    out = backflow_extent(u, x)
    assert out["seam_wrapped"] is True


def test_threshold_counts_below_only():
    x = np.linspace(0.05, 8.95, 96)
    u = synthetic_u(x, 1.0, 4.0)
    assert backflow_extent(u, x)["x_end"] is not None
    assert backflow_extent(u, x, threshold=-0.5)["x_end"] is None


def test_rows_helper_indexes_the_right_row():
    x = np.linspace(0.05, 8.95, 48)
    mean_u = np.vstack([synthetic_u(x, 1.0, 3.0), synthetic_u(x, 1.0, 6.0),
                        synthetic_u(x, 1.0, 8.0)])
    rows = backflow_extent_rows(mean_u, x, rows=(0, 2))
    assert rows[0]["x_end"] < rows[2]["x_end"]


def test_validation_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        backflow_extent(np.zeros(4), np.zeros(5))


# --------------------------------------------------------------------------- #
# 2. the two definitions
# --------------------------------------------------------------------------- #
def test_identical_definitions_are_flagged_insensitive():
    out = definition_spread(5.40, 5.40)
    assert out["flag"] == "definition_insensitive"
    assert out["delta_rel"] == pytest.approx(0.0)


def test_two_fold_disagreement_is_flagged_sensitive():
    out = definition_spread(8.08, 3.99)
    assert out["flag"] == "definition_sensitive"
    assert out["delta_rel"] == pytest.approx(abs(3.99 - 8.08) / 8.08)
    # the flag threshold is the case's own 10 % tolerance
    assert definition_spread(5.0, 5.4)["flag"] == "definition_insensitive"
    assert definition_spread(5.0, 5.6)["flag"] == "definition_sensitive"


def test_missing_definition_is_not_measured_not_a_pass():
    assert definition_spread(None, 4.0)["flag"] == "not_measured"
    assert definition_spread(8.0, None)["flag"] == "not_measured"


def test_definition_names_are_distinct_and_descriptive():
    assert DEFINITION_MEAN_FIELD != DEFINITION_ENSEMBLE
    assert "time_mean" in DEFINITION_MEAN_FIELD
    assert "instantaneous" in DEFINITION_ENSEMBLE


# --------------------------------------------------------------------------- #
# 3. the landed artefacts (skipped when round-20 output is absent)
# --------------------------------------------------------------------------- #
ARTEFACTS = sorted(DOCS.glob("round20_laminar_Re*_96x48.json"))


@pytest.mark.parametrize("path", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_mean_field_and_wall_shear_agree_on_the_converged_runs(path):
    """Two data paths, one answer: the bubble edge from ``mean_u`` and from
    ``tau_b`` must agree to a few percent, otherwise the wall-shear
    post-processing is not measuring the recirculation length."""
    payload = json.loads(path.read_text())
    if payload["config"].get("diverged_at_step") is not None:
        pytest.skip("diverged run: no reattachment length exists")
    npz = np.load(path.with_suffix(".npz"))
    x = npz["x_over_h"]
    xr = payload["lower_re"]["x_r_over_H"]
    x_field = backflow_extent(npz["mean_u"][0], x)["x_end"]
    assert xr is not None and x_field is not None
    assert abs(x_field - xr) / xr < 0.10


@pytest.mark.parametrize("path", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_definitions_are_reported_for_every_landed_run(path):
    payload = json.loads(path.read_text())
    series = [v for v in payload["series"]["x_r_over_H"] if v is not None]
    spread = definition_spread(payload["lower_re"]["x_r_over_H"],
                              float(np.median(series)) if series else None)
    assert spread["flag"] in ("definition_sensitive", "definition_insensitive",
                             "not_measured")


def test_diverged_run_is_not_scored():
    path = DOCS / "round20_laminar_Re1400_96x48.json"
    if not path.exists():
        pytest.skip("Re=1400 artefact not fetched")
    payload = json.loads(path.read_text())
    assert payload["config"]["diverged_at_step"] is not None
    assert payload["lower_re"]["x_r_over_H"] is None
    assert payload["lower_re"]["verdict"] == "UNDEFINED"