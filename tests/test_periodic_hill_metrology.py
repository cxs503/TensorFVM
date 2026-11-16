"""Tests for the CASE-FVM-PERIODIC-HILLS metrology audit (round 31).

Every assertion is against something *outside* the function under test: the
published DNS profiles, a closed-form derivation, the solver's own bulk weights,
or an archived field.  The two negative controls are explicit -- a mirrored
station map (shown *not* to be discriminable from wall heights, an honest
negative result) and a uniform field (shown to carry zero bias, so the audit is
not reporting a fixed offset).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillMesh,
    PeriodicHillSolver,
)
from tensorfvm.verification import periodic_hill_metrology as pm
from tensorfvm.verification import periodic_hill_reference as ref

CASE_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs/case-periodic-hills"


# ----------------------------------------------------------------- reference side
def test_flux_is_the_same_at_every_reference_station():
    """Continuity + periodicity: ``Q/U_b`` cannot depend on the station."""
    inv = pm.reference_flux_invariant()
    assert inv["periodic_consistent"] is True
    assert inv["relative_spread"] < 1.0e-3
    # The reference's normalisation constant is *not* the x-averaged band height;
    # it is the geometric height above the crest (to the sampling truncation).
    assert abs(inv["mean_minus_top_minus_crest"]) / inv["top_minus_crest"] < 5.0e-3
    assert 2.02 < inv["mean_q_over_ub"] < 2.04


def test_reference_wall_heights_reproduce_the_code_hill_contour():
    """The files start at the wall they were sampled from -> geometry check."""
    geo = pm.reference_wall_surface()
    assert geo["geometry_matches_reference"] is True
    assert geo["worst_abs_residual"] < 0.01 * pm.HILL_HEIGHT
    # the crest station must sit at y/H = 1 up to the first-sample offset
    crest = next(row for row in geo["rows"] if row["x_over_h"] == pytest.approx(0.05))
    assert crest["sampled_wall_y_over_h"] == pytest.approx(1.0, abs=2.0e-3)
    # periodic pair h(1) == h(8) is reproduced from the data itself
    h1 = next(row for row in geo["rows"] if row["x_over_h"] == 1.0)["sampled_wall_y_over_h"]
    h8 = next(row for row in geo["rows"] if row["x_over_h"] == 8.0)["sampled_wall_y_over_h"]
    assert abs(h1 - h8) < 5.0e-3


def test_wall_heights_cannot_discriminate_a_mirrored_station_map():
    """Honest negative result: the hill is symmetric about L/2, so this control
    carries no information.  Documented so it is not mistaken for a check."""
    mir = pm.mirrored_assignment_residual()
    assert mir["worst_abs_residual_mirrored"] < 0.01 * pm.HILL_HEIGHT
    assert abs(mir["worst_abs_residual_mirrored"] - mir["worst_abs_residual_direct"]) < 0.01


def test_reference_deficit_bounds_the_definition_mismatch():
    deficit = pm.reference_deficit()
    assert 0.05 < deficit["mean_deficit_over_flux"] < 0.08
    implied = pm.reynolds_bias_from_deficit(deficit["mean_deficit_over_flux"])
    assert implied == pytest.approx(deficit["implied_relative_reynolds_bias"])
    assert 0.05 < implied < 0.08


def test_profile_flux_is_orientation_blind_and_rejects_bad_input():
    prof = ref.load_profile(1.0)
    forward = pm.profile_flux(prof)
    reversed_prof = {**prof,
                     "y_over_h": prof["y_over_h"][::-1],
                     "u_over_ub": prof["u_over_ub"][::-1]}
    assert pm.profile_flux(reversed_prof) == pytest.approx(forward, rel=1e-12)
    with pytest.raises(ValueError):
        pm.profile_flux({"y_over_h": np.array([0.0, 1.0]),
                         "u_over_ub": np.array([0.0, np.nan])})


# --------------------------------------------------------------------- code side
def test_zero_deficit_field_has_zero_bias():
    """Negative control on the audit itself: no valley flux, no mislabel.

    The two functionals normalise *different* regions, so a uniform field is not
    the control (it has a valley deficit like any field does); the control is a
    field that carries no flux below the crest level.
    """
    config = PeriodicHillConfig(nx=36, ny=48)
    mesh = PeriodicHillMesh(config)
    field = np.where(mesh.yc >= pm.HILL_HEIGHT, 1.7, 0.0)
    out = pm.functionals(field, config)
    # residual is the cell that straddles the crest level (partial volume), not a
    # definitional offset -- it shrinks with the mesh and is bounded here.
    assert abs(out["relative_reynolds_bias"]) < 0.035
    assert out["valley_deficit_over_flux"] < 0.03
    assert out["valley_deficit_over_flux"] == pytest.approx(0.0, abs=1e-12)
    assert out["relative_reynolds_bias"] == pytest.approx(0.0, abs=1e-12)


def test_crest_band_weights_reproduce_the_solver_functional():
    """``u_b_code`` must be the functional the solver imposes, not a lookalike."""
    config = PeriodicHillConfig(nx=36, ny=12)
    mesh = PeriodicHillMesh(config)
    field = 1.0 + 0.3 * np.cos(2.0 * np.pi * mesh.yc / config.channel_top_m)
    ours = pm.functionals(field, config)
    solver = PeriodicHillSolver(config=config)
    weights, weight_sum = solver._bulk_weights()
    assert np.allclose(ours["weights"], weights, rtol=0, atol=0)
    assert weight_sum == pytest.approx(float((weights * mesh.volume).sum()), rel=1e-14)
    assert ours["u_b_code"] == pytest.approx(solver.bulk_velocity(field), rel=1e-12)


def test_discrete_bias_matches_the_closed_form():
    config = PeriodicHillConfig(nx=36, ny=12)
    mesh = PeriodicHillMesh(config)
    # a valley jet: fast below the crest level, slower above it
    field = np.where(mesh.yc < pm.HILL_HEIGHT, 1.0, 0.6)
    out = pm.functionals(field, config)
    assert out["valley_deficit_over_flux"] > 0.0
    assert out["relative_reynolds_bias"] == pytest.approx(
        pm.reynolds_bias_from_deficit(out["valley_deficit_over_flux"]), rel=1e-12)
    assert pm.audited_reynolds(
        out["relative_reynolds_bias"], config.reynolds
    ) == pytest.approx(out["reynolds_in_reference_convention"], rel=1e-12)


def test_closed_form_limits_and_validation():
    assert pm.reynolds_bias_from_deficit(0.0) == pytest.approx(
        (pm.BAND_HI - pm.BAND_LO) / (pm.CHANNEL_TOP - pm.HILL_HEIGHT) - 1.0)
    neg = pm.reynolds_bias_from_deficit(-0.1)
    assert neg < 0.0  # a surplus of near-wall forward flux raises the imposed Re
    for bad in (1.0, 1.5, np.inf):
        with pytest.raises(ValueError):
            pm.reynolds_bias_from_deficit(bad)
    with pytest.raises(ValueError):
        pm.audited_reynolds(-1.0, 2800.0)


def test_field_shape_and_finiteness_are_enforced():
    config = PeriodicHillConfig(nx=18, ny=8)
    good = np.ones(PeriodicHillMesh(config).volume.shape)
    with pytest.raises(ValueError):
        pm.functionals(good[:-1], config)
    bad = good.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        pm.functionals(bad, config)


# ------------------------------------------------------------------ archived run
def _archived(tag: str):
    field = np.load(CASE_DIR / f"{tag}.npz")
    payload = json.loads((CASE_DIR / f"{tag}.json").read_text())["config"]
    config = PeriodicHillConfig(
        nx=int(payload["nx"]), ny=int(payload["ny"]),
        time_step_s=float(payload["time_step_s"]), reynolds=float(payload["reynolds"]))
    return field, config


def test_archived_headline_run_bias_is_about_one_percent():
    """The published Re = 2800 laminar run is the reference's Re to ~1 %."""
    field, config = _archived("round20_laminar_Re2800_96x48")
    out = pm.functionals(field["mean_u"], config)
    assert 0.005 < out["relative_reynolds_bias"] < 0.015
    assert out["reynolds_in_reference_convention"] == pytest.approx(
        2800.0 * (1.0 + out["relative_reynolds_bias"]), rel=1e-12)


def test_bias_is_solution_dependent_across_the_closure_ladder():
    """Bias is a property of the *field*, not of the label: closures differ."""
    biases = {}
    for tag in ("round20_laminar_Re2800_96x48", "round9_96x48",
                "round11_sst_r18_96x48"):
        field, config = _archived(tag)
        biases[tag] = pm.functionals(field["mean_u"], config)["relative_reynolds_bias"]
    ordered = [biases["round20_laminar_Re2800_96x48"],
               biases["round11_sst_r18_96x48"], biases["round9_96x48"]]
    assert ordered == sorted(ordered)
    assert max(ordered) - min(ordered) > 0.02


def test_station_map_is_confirmed_by_the_flow_not_by_the_geometry():
    field, config = _archived("round20_laminar_Re2800_96x48")
    score = pm.station_assignment_score(field["mean_u"], config)
    assert score["direct_wins"] is True
    assert score["direct"]["correlation"] > 0.0 > score["mirrored"]["correlation"]


def test_verdict_refuses_to_blame_metrology_for_the_reattachment_error():
    """The bias is ~1 % on the headline run while x_r/H is 53 % off."""
    field, config = _archived("round20_laminar_Re2800_96x48")
    out = pm.functionals(field["mean_u"], config)
    table = [{"tag": "round20_laminar_Re2800_96x48", "bulk": out,
              "x_r_error": 0.528}]
    verdict = pm.verdict(table)
    assert verdict["bias_is_solution_dependent"] is False
    assert verdict["bias_can_explain_x_r_error"] is False
    assert verdict["status"] == "metrology_defect_confirmed_bounded"
    assert pm.verdict([])["status"] == "no_usable_fields"