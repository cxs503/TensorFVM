"""Contract tests for the SA flat-plate criteria audit.

The audit (``scripts/audit_flat_plate_sa_criteria.py``) exists because the same
three stored skin-friction coefficients were judged PASS by the shipped 20 %
gate and FAIL by a 3 % criterion written into the refinement report.  These
tests pin the machine-readable consequences: the arithmetic is re-derived
(never hand-typed), both criteria are evaluated, the reference-law dependence
of the verdict is visible, and the grid sequence is honestly reported as
non-asymptotic instead of being extrapolated.
"""
import json
import math
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_flat_plate_sa_criteria.py"

sys.path.insert(0, str(REPO / "src"))

from tensorfvm.flat_plate_reference import (  # noqa: E402
    band_aware_verdict,
    band_membership,
    correlation_values,
    gate_is_informative,
    reference_band,
    window_to_band_ratio,
)

RE = 100_000.0


def run_audit(out_dir):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(REPO), "--out", str(out_dir), "--quiet"],
        check=True,
        capture_output=True,
    )
    return (out_dir / "results.json").read_bytes()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("flat-plate-criteria")
    payload = run_audit(out)
    return json.loads(payload), payload


def test_audit_is_deterministic_and_self_contained(tmp_path, results):
    _, first = results
    second = run_audit(tmp_path / "rerun")
    assert first == second, "audit output must be byte-identical across runs"
    text = first.decode()
    assert str(REPO) not in text, "no machine-specific absolute path may leak"
    payload = json.loads(first)
    assert payload["evidence"]["levels_parsed"] == 3
    assert len(payload["evidence"]["source_sha256"]) == 64
    assert payload["case"]["reynolds"] == RE


def test_stored_errors_are_re_derived_not_trusted(results):
    payload, _ = results
    for level in payload["levels"]:
        assert level["converged"] is True
        assert level["stored_error_matches_recomputed"] is True, level["name"]
        # the stored column is the absolute Schlichting error
        assert abs(
            level["error_percent_vs_schlichting_recomputed"] + level["stored_error_percent"]
        ) < 0.01
        assert level["mean_cf"] > 0.0
    names = [level["name"] for level in payload["levels"]]
    assert names == ["sa-64x56", "sa-128x112", "sa-256x224"]


def test_two_criteria_disagree_on_the_same_numbers(results):
    payload, _ = results
    criteria = payload["criteria"]
    assert criteria["documented_gate"] == 0.20
    assert criteria["refinement_criterion"] == 0.03
    assert criteria["undocumented_factor_between_criteria"] == pytest.approx(20.0 / 3.0)
    assert criteria["verdict_flips_between_criteria"] is True
    for level in payload["levels"]:
        assert level["passes_documented_gate"] is True, level["name"]
        assert level["passes_refinement_criterion"] is False, level["name"]


def test_verdict_depends_on_the_reference_law(results):
    payload, _ = results
    verdicts = payload["per_law_verdicts_at_3pct"]
    assert set(verdicts) == set(correlation_values(RE))
    # White accepts every level, ITTC-57 rejects every level, Prandtl-Schlichting
    # flips between the coarse and the fine grids: the 3 % verdict is a property
    # of the chosen law, not of the solution.
    assert verdicts["white"] == ["PASS", "PASS", "PASS"]
    assert verdicts["ittc57"] == ["FAIL", "FAIL", "FAIL"]
    assert verdicts["prandtl_schlichting"] == ["FAIL", "PASS", "PASS"]
    assert verdicts["schlichting_0.074"] == ["FAIL", "FAIL", "FAIL"]
    for level in payload["levels"]:
        assert level["in_band"] is True
        assert level["band_verdict"] == "in_band"


def test_band_is_wider_than_the_single_law_error(results):
    payload, _ = results
    band = payload["band"]
    assert band["low"] < band["values"]["schlichting_0.074"] < band["high"]
    assert band["spread"] == pytest.approx(0.19098, abs=1e-5)
    # both conventions are reported so that a report cannot quote the smaller
    # ratio as if it were the full gate window
    assert band["full_window_to_band_ratio_at_documented_gate"] == pytest.approx(
        2.0 * 0.20 / band["spread"]
    )
    assert band["half_window_to_band_ratio_at_documented_gate"] == pytest.approx(
        0.20 / band["spread"]
    )
    assert (
        band["full_window_to_band_ratio_at_documented_gate"]
        == pytest.approx(2.0944, abs=1e-3)
    )
    for level in payload["levels"]:
        error = abs(level["error_percent_per_law"]["schlichting_0.074"]) / 100.0
        assert error < band["spread"], level["name"]


def test_grid_sequence_is_not_extrapolated(results):
    payload, _ = results
    convergence = payload["convergence"]
    assert convergence["monotone"] is True
    assert convergence["differences_shrink"] is False
    assert convergence["state"] == "differences_grow_not_asymptotic"
    assert convergence["richardson_usable"] is False
    assert convergence["richardson_limit"] is None
    assert convergence["observed_order"] is None
    assert convergence["delta_ratio"] > 1.0
    assert convergence["delta_ratio"] == pytest.approx(
        abs(convergence["delta_23"]) / abs(convergence["delta_12"])
    )


def test_wall_resolution_is_met_so_it_cannot_explain_the_deficit(results):
    payload, _ = results
    y_plus = [level["first_cell_y_plus"] for level in payload["levels"]]
    assert all(value < 1.0 for value in y_plus), y_plus
    assert y_plus[0] > y_plus[1] > y_plus[2]
    # halving the cell size halves y+ at constant friction velocity
    assert y_plus[0] / y_plus[1] == pytest.approx(2.0, rel=0.02)
    assert y_plus[1] / y_plus[2] == pytest.approx(2.0, rel=0.02)


def test_window_ratio_helper_matches_gate_helper():
    band = reference_band(RE)
    assert window_to_band_ratio(RE, 0.20) == pytest.approx(2.0 * 0.20 / band["spread"])
    assert gate_is_informative(RE, 0.20) is False
    assert window_to_band_ratio(RE, 0.20) > 1.0
    # a gate narrower than the band must be reported as informative on both scales
    assert window_to_band_ratio(RE, 0.05) < 1.0
    assert gate_is_informative(RE, 0.05) is True
    with pytest.raises(ValueError):
        window_to_band_ratio(RE, 0.0)
    with pytest.raises(ValueError):
        window_to_band_ratio(RE, math.nan)


def test_band_aware_verdict_is_consistent_with_band_membership():
    band = reference_band(RE)
    for value in (band["low"], band["mean"], band["high"]):
        verdict = band_aware_verdict(value, RE)
        assert verdict["in_band"] is (band_membership(value, RE) is True)
        assert verdict["verdict"] == "in_band"
        assert set(verdict["error_per_law"]) == set(band["values"])
    outside = band_aware_verdict(band["high"] * 1.02, RE)
    assert outside["in_band"] is False and outside["verdict"] == "outside_band"
    assert outside["offset_in_band_widths"] > 0.0
    assert band_aware_verdict(band["mean"], RE)["error_vs_band_mean"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        band_aware_verdict(-1.0, RE)
    with pytest.raises(ValueError):
        band_aware_verdict(math.inf, RE)

def test_analytic_first_cell_height_matches_the_mesh_class():
    """The closed-form clustering must reproduce the shipped mesh exactly."""
    import importlib.util

    from tensorfvm.body_fitted import FlatPlateMesh
    from tensorfvm.solver import SolverConfig

    spec = importlib.util.spec_from_file_location("audit", SCRIPT)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    config = SolverConfig(
        mesh_type="flat-plate",
        turbulence_model="spalart-allmaras",
        nx=64,
        ny=56,
        length=1.0,
        height=0.2,
        cylinder_radius=None,
        inlet_velocity=1.0,
        reynolds=RE,
        flat_plate_stretching=4.0,
    )
    mesh = FlatPlateMesh(config)
    from_mesh = float(mesh.centers[0, :, 1].min())
    analytic = audit.first_cell_centre_height(56, height=0.2, stretching=4.0)
    assert analytic == pytest.approx(from_mesh, rel=1e-12)
