"""Tests for :mod:`tensorfvm.cavity_uncertainty` (CASE-FVM-CAVITY-RE1000 debt).

Two layers:

* estimator tests on synthetic data with a known answer (a power-law sequence
  ``f = f_exact + C h^p`` must return ``p`` and ``f_exact`` exactly, for a uniform
  *and* a non-uniform refinement ratio), plus the failure mode (non-monotonic
  data must be reported as unusable instead of extrapolated);
* artefact tests that recompute the committed uncertainty study from the stored
  face fields — the numbers in ``docs/CASE-FVM-CAVITY-RE1000/UNCERTAINTY.md`` are
  therefore reproducible by a reader who only has the repository.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from tensorfvm.cavity_uncertainty import (
    DEFAULT_SAFETY,
    functionals_from_fields,
    reference_functionals,
    richardson,
    study,
    time_step_spread,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs" / "CASE-FVM-CAVITY-RE1000" / "runs"


def _payload() -> dict:
    return json.loads((RUNS / "uncertainty.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fields() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    data = np.load(RUNS / "fields.npz")
    return {int(key.split("_")[1]): (torch.as_tensor(data[f"u_{key.split('_')[1]}"]).double(),
                                     torch.as_tensor(data[f"v_{key.split('_')[1]}"]).double())
            for key in data.files if key.startswith("u_")}


# --------------------------------------------------------------------------- #
# estimator correctness on synthetic data
# --------------------------------------------------------------------------- #
def test_uniform_refinement_order_and_extrapolation_are_exact() -> None:
    exact, c, p = -0.4000, 1.2e-2, 2.0
    samples = [(n, exact + c * (1.0 / n) ** p) for n in (32, 64, 128)]
    block = richardson(samples, safety=DEFAULT_SAFETY, name="synthetic")
    assert block["observed_order"] == pytest.approx(p, abs=1e-10)
    assert block["extrapolated"] == pytest.approx(exact, abs=1e-10)
    assert block["data_quality"]["usable"] is True
    assert block["gci_fine_relative"] > 0.0


def test_non_uniform_refinement_order_is_recovered() -> None:
    exact, c, p = 0.3770, -2.5e-2, 1.5
    samples = [(n, exact + c * (1.0 / n) ** p) for n in (64, 128, 192)]
    block = richardson(samples, name="synthetic-nonuniform")
    assert block["refinement_ratios"] == pytest.approx([2.0, 192 / 128], rel=1e-12)
    assert block["observed_order"] == pytest.approx(p, abs=1e-8)
    assert block["extrapolated"] == pytest.approx(exact, abs=1e-8)


def test_non_monotonic_sequence_is_not_extrapolated() -> None:
    samples = [(32, -0.30), (64, -0.38), (128, -0.37)]  # overshoot: eps changes sign
    block = richardson(samples, name="synthetic-oscillatory")
    assert block["data_quality"]["monotonic"] is False
    assert block["data_quality"]["usable"] is False
    assert "extrapolated" not in block
    assert "not quoted" in block["data_quality"]["note"]


def test_two_grids_is_an_error() -> None:
    with pytest.raises(ValueError):
        richardson([(64, -0.37), (128, -0.385)])


def test_time_step_spread_separates_identical_from_perturbed_runs() -> None:
    base = {"u_min": -0.3853, "v_max": 0.3739, "psi_min": -0.11937, "kinetic": 0.0447}
    same = time_step_spread({"a": base, "b": dict(base)})
    assert same["max_relative_spread"] == pytest.approx(0.0, abs=1e-15)
    assert same["time_step_independent"] is True

    moved = dict(base, u_min=base["u_min"] * (1.0 + 2e-2))
    changed = time_step_spread({"a": base, "b": moved})
    assert changed["max_relative_spread"] == pytest.approx(1e-2, rel=1e-2)
    assert changed["time_step_independent"] is False


# --------------------------------------------------------------------------- #
# artefacts: the committed study is what the stored fields give
# --------------------------------------------------------------------------- #
def test_functionals_reproduce_the_case_artefacts(fields) -> None:
    """Stored streamfunction minimum / vortex centre must come out of the fields."""
    summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    for case in summary["cases"]:
        got = functionals_from_fields(*fields[case["n"]])
        assert got["psi_min"] == pytest.approx(case["streamfunction"]["psi_min"], abs=1e-10)
        assert got["vortex_x"] == pytest.approx(case["vortex_centre"]["computed"][0], abs=1e-12)
        assert got["vortex_y"] == pytest.approx(case["vortex_centre"]["computed"][1], abs=1e-12)
        # the benchmark cut sits exactly on a face for even n: no interpolation
        n = case["n"]
        assert got["u_min"] == pytest.approx(float(fields[n][0][n // 2, :].min()), abs=1e-12)
        assert got["v_max"] == pytest.approx(float(fields[n][1][:, n // 2].max()), abs=1e-12)


def test_reference_functionals_are_the_shared_transcription() -> None:
    ref = reference_functionals(1000.0)
    assert ref["u_min"] == pytest.approx(-0.38289, abs=5e-6)
    assert ref["v_max"] == pytest.approx(0.37095, abs=5e-6)
    assert (ref["vortex_x"], ref["vortex_y"]) == pytest.approx((0.5313, 0.5625), abs=1e-9)


def test_uncertainty_payload_recomputed_from_fields(fields) -> None:
    stored = _payload()
    recomputed = study({n: fields[n] for n in stored["grids"]})
    for q, block in stored["quantities"].items():
        fresh = recomputed["quantities"][q]
        assert fresh["observed_order"] == pytest.approx(block["observed_order"], abs=1e-10), q
        assert fresh["extrapolated"] == pytest.approx(block["extrapolated"], abs=1e-10), q
        assert fresh["gci_fine_relative"] == pytest.approx(block["gci_fine_relative"], abs=1e-12), q
        assert fresh["data_quality"]["usable"] == block["data_quality"]["usable"], q
        if "reference" in block:
            assert fresh["reference_within_gci_band"] == block["reference_within_gci_band"], q


def test_claimed_numbers_are_the_ones_in_the_report() -> None:
    """The three statements the report makes must still hold for this bundle."""
    payload = _payload()
    u = payload["quantities"]["u_min"]
    v = payload["quantities"]["v_max"]
    # 1. the grid sequence is monotonic and in the asymptotic range -> a GCI exists
    assert u["data_quality"]["usable"] and v["data_quality"]["usable"]
    assert 1.5 <= u["observed_order"] <= 3.0
    assert 1.5 <= v["observed_order"] <= 3.0
    # 2. the finest grid is inside the displayed case-pool band with a wide margin
    assert abs(u["fine_minus_reference_relative"]) < 0.03
    assert abs(v["fine_minus_reference_relative"]) < 0.03
    # 3. ... but the fine-grid agreement is smaller than the discretisation band,
    #    so it must be quoted as "within the GCI", never as an accuracy figure
    assert abs(u["fine_minus_reference_relative"]) <= u["gci_fine_relative"]
    assert abs(v["fine_minus_reference_relative"]) <= v["gci_fine_relative"]


def test_time_step_independence_is_recorded() -> None:
    payload = _payload()
    if "time_step" not in payload:
        pytest.skip("this bundle carries no same-grid/different-Courant runs")
    ts = payload["time_step"]
    assert len(ts["runs"]) >= 2
    assert ts["max_relative_spread"] <= ts["tolerance"]
    assert ts["time_step_independent"] is True