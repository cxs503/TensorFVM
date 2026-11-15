"""Round 22: stationarity of the *published* periodic-hill metric.

The published ``x_r/H`` is the crossing of the **time-mean** wall shear.  Round 21
found the artefact's ``stationarity`` block carries two disagreeing statements
(per-sample legacy-operator report vs per-block certified-operator report) and
asked (lead #1) to make the *published* metric's stationarity explicit and to
persist the per-block mean shear so it can be re-scored independently.

Everything here is decidable without a full solver: synthetic per-block shear
profiles with prescribed zeros certify the re-scoring operator, and the landed
round-20/21 artefacts (when present) are re-scored from their saved per-block
mean shear / JSON ``x_r_over_H_by_block``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_stationarity import (
    STATIONARY_TOLERANCE,
    block_stationarity,
    published_metric_stationarity,
    reconcile_definitions,
)
from tensorfvm.verification.wall_shear_bubbles import (
    separation_reattachment,
    tanh_shear_profile,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


# --------------------------------------------------------------------------- #
# 1. the re-scoring operator on synthetic per-block shear
# --------------------------------------------------------------------------- #
def test_block_stationarity_rescores_prescribed_zeros():
    """Per-block mean shear with a *prescribed* reattachment must be re-scored
    back to that value by the certified operator -- the operator is certified
    against a known truth, not against the solver it judges."""
    x = np.linspace(0.05, 8.95, 96)
    # four blocks, each a tanh shear profile with reattachment at 5.4 (the
    # reference) -- a stationary run should come back ~5.4 on every block.
    blocks = np.stack([tanh_shear_profile(x, [0.3, 5.4])[0]
                       for _ in range(4)])
    out = block_stationarity(blocks, x)
    assert out["status"] == "stationary"
    assert out["dispersion"]["spread_rel"] <= STATIONARY_TOLERANCE
    for v in out["x_r_over_H_by_block"]:
        assert v == pytest.approx(5.4, abs=0.2)


def test_block_stationarity_reports_drift_as_not_stationary():
    """Blocks whose reattachment drifts by >10 % must be flagged
    ``not_stationary`` -- a moving target is not a single published number."""
    x = np.linspace(0.05, 8.95, 96)
    blocks = np.stack([tanh_shear_profile(x, [0.3, 5.4 + 0.9 * k])[0]
                       for k in range(4)])
    out = block_stationarity(blocks, x)
    assert out["status"] == "not_stationary"
    assert out["dispersion"]["spread_rel"] > STATIONARY_TOLERANCE


def test_block_stationarity_nan_block_is_excluded_not_a_pass():
    x = np.linspace(0.05, 8.95, 96)
    blocks = np.stack([tanh_shear_profile(x, [0.3, 5.4])[0],
                       np.full_like(x, np.nan),
                       tanh_shear_profile(x, [0.3, 5.4])[0]])
    out = block_stationarity(blocks, x)
    assert out["x_r_over_H_by_block"][1] is None
    # two finite blocks: still measurable, still stationary
    assert out["status"] == "stationary"


def test_block_stationarity_requires_shared_streamwise_axis():
    with pytest.raises(ValueError):
        block_stationarity(np.zeros((4, 96)), np.linspace(0, 1, 48))


# --------------------------------------------------------------------------- #
# 2. the landed artefacts: published-metric stationarity
# --------------------------------------------------------------------------- #
ARTEFACTS = sorted(DOCS.glob("round20_laminar_Re*_96x48.json"))


@pytest.mark.parametrize("path", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_published_metric_is_stationary_at_block_level(path):
    """The published ``x_r/H`` (crossing of time-mean wall shear) must be
    stationary at block level -- the per-block certified-operator values in the
    JSON must sit inside the 10 % band, so the time-mean is a single number."""
    payload = json.loads(path.read_text())
    if payload["config"].get("diverged_at_step") is not None:
        pytest.skip("diverged run: no reattachment length exists")
    out = published_metric_stationarity(payload)
    assert out["status"] == "stationary", (
        f"published metric not stationary: {out['dispersion']}")
    assert out["dispersion"]["spread_rel"] <= STATIONARY_TOLERANCE


@pytest.mark.parametrize("path", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_published_metric_stationarity_is_distinct_from_per_sample_report(path):
    """The per-sample ``stationarity_report`` (legacy operator) and the
    per-block certified-operator report must not be conflated: on the closure-free
    branch they disagree (the per-sample report is ``ok=False``), and the
    published-metric stationarity is the per-block one."""
    payload = json.loads(path.read_text())
    if payload["config"].get("diverged_at_step") is not None:
        pytest.skip("diverged run")
    stat = payload["stationarity"]
    per_sample = stat["x_r_over_H"]
    # the per-sample report is on the *instantaneous* reattachment (legacy
    # operator); it is not the published metric's stationarity and is allowed to
    # disagree -- but it must be recorded as such (ok flag present).
    assert "ok" in per_sample
    # the published-metric stationarity is the per-block certified one
    assert published_metric_stationarity(payload)["status"] == "stationary"


# --------------------------------------------------------------------------- #
# 3. the two-definition reconciliation (round-21 lead #3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_mean_field_is_corroborated_by_independent_backflow(path):
    """The mean-field ``x_r/H`` must agree with the ``mean_u<0`` backflow extent
    (an independent data path) within 3 % -- that is what makes it the *measured*
    recirculation length.  The ensemble definition is not required to agree."""
    payload = json.loads(path.read_text())
    if payload["config"].get("diverged_at_step") is not None:
        pytest.skip("diverged run")
    npz = np.load(path.with_suffix(".npz"))
    rec = reconcile_definitions(payload, mean_u=npz["mean_u"],
                                x_over_h=npz["x_over_h"],
                                mean_tau_b=npz["mean_tau_b"])
    assert rec["mean_field_corroborated"] is True, (
        f"mean field {rec['mean_field_x_r_over_H']} vs backflow "
        f"{rec['backflow_extent_row0']} not corroborated")


def test_reconcile_definitions_reports_sensitive_when_definitions_differ():
    """On the closure-free branch the two definitions differ by ~50 %, so the
    reconciliation must flag ``definition_sensitive`` -- quoting one without the
    other is a demonstrable error."""
    path = DOCS / "round20_laminar_Re2800_96x48.json"
    if not path.exists():
        pytest.skip("Re=2800 artefact not present")
    payload = json.loads(path.read_text())
    npz = np.load(path.with_suffix(".npz"))
    rec = reconcile_definitions(payload, mean_u=npz["mean_u"],
                                x_over_h=npz["x_over_h"],
                                mean_tau_b=npz["mean_tau_b"])
    assert rec["definitions"]["flag"] == "definition_sensitive"
    # the mean field is the corroborated one; the ensemble is not
    assert rec["mean_field_corroborated"] is True
    assert rec["ensemble_corroborated"] is False


def test_reconcile_definitions_without_npz_uses_json_derived():
    """Without the ``.npz`` the reconciliation falls back to the JSON's derived
    ``x_r_over_H`` -- it must still run and report a (possibly ``None``)
    corroboration, never raise."""
    path = DOCS / "round20_laminar_Re2800_96x48.json"
    if not path.exists():
        pytest.skip("Re=2800 artefact not present")
    payload = json.loads(path.read_text())
    rec = reconcile_definitions(payload)
    assert "mean_field_x_r_over_H" in rec
    assert "definitions" in rec
