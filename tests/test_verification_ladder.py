"""Verification points for the six-level ladder (:mod:`tensorfvm.verification.ladder`).

Two kinds of checks live here:

* **analytic** -- the reference solutions and the discrete operators are tested
  against each other directly (exact profiles, exact orders, exact panel flows),
  so a regression in the ladder itself fails even with no solver artefacts;
* **audit contract** -- the verdict machinery must report PASS only when a
  measurement exists and satisfies the threshold, must report WITHHELD (never
  PASS) for the two levels that cannot be measured, and must keep the campaign's
  thresholds intact.
"""
import json
import math
import pathlib

import numpy as np
import pytest

from tensorfvm.verification import ladder as L

REPO = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = pathlib.Path(L.ARTIFACTS)


@pytest.fixture(scope="module")
def report():
    """One full ladder run shared by the report-level checks (it costs seconds)."""
    return L.audit()


# --------------------------------------------------------------------------- #
# Level 1a -- exact profile and discretisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("peclet", [0.1, 1.0, 10.0, 100.0])
def test_exact_profile_hits_its_boundary_values(peclet):
    """The analytic profile satisfies T(0) = 0 and T(1) = 1 for every Peclet number."""
    values = L.convection_diffusion_exact(np.array([0.0, 1.0]), peclet)
    assert values[0] == pytest.approx(0.0, abs=1e-14)
    assert values[1] == pytest.approx(1.0, rel=1e-12)


def test_zero_peclet_limit_is_linear():
    """``Pe -> 0`` degenerates to pure conduction with the linear profile."""
    x = np.linspace(0.0, 1.0, 11)
    assert np.allclose(L.convection_diffusion_exact(x, 0.0), x)


@pytest.mark.parametrize("peclet", [1.0, 10.0])
def test_central_scheme_is_second_order(peclet):
    """Doubling the grid must reduce the L2 error at an observed order of two."""
    errors = [L.convection_diffusion(n, peclet=peclet, scheme="central")["relative_l2"]
              for n in (64, 128, 256)]
    order = L.observed_order(errors[0], errors[1], ratio=2.0)
    order_fine = L.observed_order(errors[1], errors[2], ratio=2.0)
    assert 1.9 <= order <= 2.1
    assert 1.9 <= order_fine <= 2.1


def test_upwind_scheme_is_first_order():
    """First-order upwind must show an observed order near one, not near two."""
    errors = [L.convection_diffusion(n, peclet=1.0, scheme="upwind")["relative_l2"]
              for n in (64, 128, 256)]
    order = L.observed_order(errors[1], errors[2], ratio=2.0)
    assert 0.85 <= order <= 1.15


def test_power_law_scheme_beats_upwind_in_the_layer():
    """At ``Pe = 10`` the power-law scheme is second order where upwind is first."""
    power = [L.convection_diffusion(n, peclet=10.0, scheme="power_law")["relative_l2"]
             for n in (64, 128)]
    upwind = [L.convection_diffusion(n, peclet=10.0, scheme="upwind")["relative_l2"]
              for n in (64, 128)]
    assert power[1] < upwind[1]
    assert L.observed_order(power[0], power[1], ratio=2.0) > 1.8


@pytest.mark.parametrize("scheme", ["central", "upwind", "power_law"])
def test_discrete_profile_stays_bounded(scheme):
    """No scheme may overshoot the Dirichlet data on a layer-resolved grid."""
    run = L.convection_diffusion(512, peclet=1.0, scheme=scheme)
    assert run["min"] > -1e-12
    assert run["max"] < 1.0 + 1e-12
    assert run["monotone"]


def test_resolution_rule_tracks_the_boundary_layer():
    """The C1 grid rule is monotone in Peclet and never below its floor."""
    cells = [L.c1_cells(peclet) for peclet in (0.1, 1.0, 10.0, 100.0)]
    assert cells == sorted(cells)
    assert cells[0] == L.C1_MIN_CELLS
    assert L.c1_cells(100.0) == 512 * 100


def test_c1_verdicts_pass_and_carry_their_numbers():
    """C1a/C1b pass on the stated rule and report the measured error."""
    verdicts = {verdict.cid: verdict for verdict in L.level1a_verdicts()}
    assert verdicts["C1a"].verdict == "PASS"
    assert verdicts["C1b"].verdict == "PASS"
    assert verdicts["C1a"].measured < L.CRITERIA["C1a"].threshold
    assert verdicts["C1b"].measured < L.CRITERIA["C1b"].threshold
    assert verdicts["C1a"].detail["resolution_table"]["pe_100"]["l2_central"] < 1e-3


# --------------------------------------------------------------------------- #
# Level 3 -- grid convergence index
# --------------------------------------------------------------------------- #
def test_gci_recovers_an_exact_second_order_sequence():
    """A synthetic exact second-order sequence must return p = 2 and round GCI."""
    exact = 0.4
    fine, medium, coarse = (exact + 1.0 / (n ** 2) for n in (64, 32, 16))
    gci = L.gci_three_grid(fine, medium, coarse, ratio=2.0, safety=1.25)
    assert gci["observed_order"] == pytest.approx(2.0, abs=1e-12)
    assert gci["monotone"] is True
    assert gci["oscillatory"] is False
    # Fs |d21/f1| / (r^p - 1) with the numbers just used
    expected = 1.25 * abs((medium - fine) / fine) / (4.0 - 1.0)
    assert gci["gci_fine"] == pytest.approx(expected, rel=1e-12)


def test_gci_flags_oscillation():
    """A sign-changing increment is an oscillation, not convergence."""
    gci = L.gci_three_grid(1.0, 0.9, 0.95, ratio=2.0)
    assert gci["oscillatory"] is True
    assert gci["monotone"] is False


def test_gci_rejects_a_single_zero_increment():
    with pytest.raises(ValueError):
        L.gci_three_grid(1.0, 1.0, 0.9, ratio=2.0)


def test_gci_constant_sequence_is_exactly_converged():
    gci = L.gci_three_grid(1.0, 1.0, 1.0, ratio=2.0)
    assert gci["converged_exactly"] is True
    assert gci["gci_fine"] == 0.0


@pytest.mark.parametrize("scheme,cells,formal", [
    ("upwind", (32, 64, 128), 1.0),
    ("central", (16, 32, 64), 2.0),
])
def test_gci_study_recovers_the_formal_order(scheme, cells, formal):
    """Both schemes must recover their formal order and keep the exact value in band."""
    study = L.gci_study(scheme=scheme, cells=cells)
    assert study["observed_order"] == pytest.approx(formal, abs=0.2)
    assert study["monotone"] is True
    assert study["band_contains_exact"] is True
    assert study["safety_factor"] == 1.25


def test_level3_verdict_passes_with_both_schemes_reported():
    verdict = L.level3_verdict()
    assert verdict.verdict == "PASS"
    assert verdict.measured < L.CRITERIA["C5"].threshold
    assert set(verdict.detail["upwind"]) >= {"observed_order", "gci_fine"}
    assert set(verdict.detail["central"]) >= {"observed_order", "gci_fine"}


# --------------------------------------------------------------------------- #
# Level 4 -- panel method and integral boundary layer
# --------------------------------------------------------------------------- #
def test_panel_ring_orientation_is_counter_clockwise():
    """Outward normals must point away from the body (the influence algebra assumes it)."""
    nodes = L.naca0012_panels(80)
    solution = L._solve_panels(nodes, alpha_deg=0.0, kutta=False)
    outward = np.sum((solution["mid"] - np.array([0.5, 0.0])) * solution["normal"],
                     axis=1)
    assert np.all(outward > 0.0)


def test_cylinder_cp_reproduces_the_exact_potential_flow():
    """The panel kernel must return ``Cp = 1 - 4 sin^2`` to round-off on a cylinder."""
    check = L.cylinder_panel_check(n_panels=240)
    assert check["relative_l2"] < 1e-10
    assert check["max_abs"] < 1e-10
    assert abs(check["pressure_drag"]) < 1e-10          # d'Alembert


def test_flat_plate_lift_matches_two_pi_sine():
    """The lifting panel formulation must reproduce the exact flat-plate lift."""
    check = L.flat_plate_panel_check()
    assert check["worst_deviation"] < 1e-3
    for case in check["cases"].values():
        assert case["cl_circulation"] == pytest.approx(case["cl_exact"], rel=1e-3)


def test_naca0012_geometry_is_a_closed_twelve_percent_section():
    nodes = L.naca0012_panels(160)
    chord = nodes[:, 0].max() - nodes[:, 0].min()
    assert chord == pytest.approx(1.0, abs=1e-12)
    assert 2.0 * np.abs(nodes[:, 1]).max() == pytest.approx(0.12, rel=0.02)
    assert np.allclose(nodes[0], nodes[-1], atol=1e-12)   # closed ring at the TE


def test_naca0012_is_symmetric_at_zero_incidence():
    """A symmetric section at alpha = 0 must give zero lift and mirror Cp."""
    panel = L.panel_method_naca0012(n_panels=160)
    case = panel["cases"]["alpha_0"]
    assert abs(case["cl_circulation"]) < 1e-12
    solution = L._solve_panels(L.naca0012_panels(160), alpha_deg=0.0, kutta=True)
    upper = solution["cp"][: solution["cp"].size // 2]
    lower = solution["cp"][::-1][: solution["cp"].size // 2]
    assert np.allclose(upper, lower, atol=1e-6)
    assert case["suction_peak_cp"] == pytest.approx(-0.413, abs=0.02)


def test_naca0012_lift_slope_gap_is_documented_not_hidden():
    """The known lift-slope gap is pinned: a silent fix must update the report."""
    panel = L.panel_method_naca0012(n_panels=160)
    dev = max(case["relative_deviation"] for case in panel["cases"].values())
    assert 0.05 < dev < 0.15
    verdict = L.level4_inviscid_verdict()[0]
    assert "above thin-aerofoil theory" in verdict.note


def test_thwaites_reproduces_blasius_on_a_flat_plate():
    """Thwaites' integral method must land within 3 % of the Blasius solution."""
    verdict = L.level4_viscous_verdict()
    assert verdict.verdict == "PASS"
    assert verdict.measured < L.CRITERIA["C7"].threshold
    errors = verdict.detail["relative_errors"]
    assert errors["theta"] < 0.03
    assert errors["delta_star"] < 0.03
    assert errors["cf"] < 0.03


@pytest.mark.parametrize("reynolds", [1e5, 1e6, 1e7])
def test_thwaites_coefficients_are_reynolds_independent(reynolds):
    """The similarity coefficients must not depend on the unit Reynolds number."""
    run = L.thwaites_flat_plate(reynolds=reynolds)
    assert run["theta_coefficient"] == pytest.approx(math.sqrt(0.45), rel=1e-12)
    assert run["cf_coefficient"] == pytest.approx(
        2 * L.THWAITES_LAMBDA0["l"] / math.sqrt(0.45), rel=1e-12)


# --------------------------------------------------------------------------- #
# Audit contract -- stored-artefact levels
# --------------------------------------------------------------------------- #
def _cavity_payload(reynolds, cells, u_error, v_error, *, converged=True):
    return {
        "reynolds": float(reynolds), "n": int(cells), "steps": 1000,
        "converged": converged, "residual": 1e-9, "max_divergence": 1e-14,
        "wall_seconds": 1.0,
        "u": {"max_abs_error": u_error, "rms_error": u_error,
              "max_abs_error_consistent": u_error,
              "suspect_points": {}},
        "v": {"max_abs_error": v_error, "rms_error": v_error,
              "max_abs_error_consistent": v_error,
              "suspect_points": {}},
        "vortex_centre": {"computed": [0.5, 0.5]},
    }


def test_cavity_verdict_is_withheld_without_a_re1000_run(tmp_path):
    """A missing Re = 1000 run must not be replaced by a lower Reynolds number."""
    path = tmp_path / "lid_cavity_Re400_n64.json"
    path.write_text(json.dumps(_cavity_payload(400, 64, 0.005, 0.005)))
    verdict = L.lid_cavity_verdict(artifact_dir=tmp_path)[0]
    assert verdict.verdict == "WITHHELD"
    assert "Re = 1000" in verdict.note


def test_cavity_verdict_passes_on_a_good_re1000_run(tmp_path):
    path = tmp_path / "lid_cavity_Re1000_n64.json"
    path.write_text(json.dumps(_cavity_payload(1000, 64, 0.008, 0.011)))
    verdict = L.lid_cavity_verdict(artifact_dir=tmp_path)[0]
    assert verdict.verdict == "PASS"
    assert verdict.measured == pytest.approx(0.011)


def test_cavity_verdict_fails_above_two_percent(tmp_path):
    path = tmp_path / "lid_cavity_Re1000_n64.json"
    path.write_text(json.dumps(_cavity_payload(1000, 64, 0.05, 0.02)))
    verdict = L.lid_cavity_verdict(artifact_dir=tmp_path)[0]
    assert verdict.verdict == "FAIL"
    assert verdict.measured == pytest.approx(0.05)


def test_cavity_verdict_rejects_an_unconverged_run(tmp_path):
    path = tmp_path / "lid_cavity_Re1000_n64.json"
    path.write_text(json.dumps(_cavity_payload(1000, 64, 0.001, 0.001,
                                               converged=False)))
    verdict = L.lid_cavity_verdict(artifact_dir=tmp_path)[0]
    assert verdict.verdict == "FAIL"
    assert "did not converge" in verdict.note


def test_cavity_runs_reproduce_the_committed_ladder():
    """The same solver scored by a different script must agree to extraction noise."""
    verdict = L.lid_cavity_verdict()[0]
    corroboration = verdict.detail["independent_reproduction"]
    assert corroboration["available"] is True
    assert corroboration["matched_rows"]
    assert corroboration["max_abs_delta_u"] < 1e-5
    assert corroboration["max_abs_delta_v"] < 1e-5
    # the committed ladder never finished this row; the ladder here does not claim it
    assert corroboration["rows_missing_from_committed_ladder"] == ["Re1000_n128"]


def test_natural_convection_matches_de_vahl_davis():
    """C3 reads the stored Ra = 1e4 run and re-derives the deviation from the table."""
    verdict = L.natural_convection_verdict()[0]
    assert verdict.verdict == "PASS"
    assert verdict.measured < L.CRITERIA["C3"].threshold
    reference = L.LITERATURE["de_vahl_davis_ra1e4_nu"][0]
    nusselt = verdict.detail["runs"][
        max(verdict.detail["runs"], key=lambda key: verdict.detail["runs"][key]["cells"])
    ]["nusselt"]
    assert abs(nusselt - reference) / reference == pytest.approx(verdict.measured,
                                                                 rel=1e-12)


def test_natural_convection_verdict_is_withheld_for_an_unknown_rayleigh(tmp_path):
    verdict = L.natural_convection_verdict(artifact_dir=tmp_path, rayleigh=1e5)[0]
    assert verdict.verdict == "WITHHELD"


def test_tgv_conservation_artefact_carries_both_statements():
    """The energy audit must report the inviscid drift and the viscous integral."""
    payload = L.tgv_conservation(cache=ARTIFACTS / "tgv_conservation.json")
    assert payload["records"]
    for record in payload["records"].values():
        assert record["inviscid_energy_drift"] < 1e-6
        assert record["inviscid_convective_power_max"] < 1e-10
        assert record["viscous_integral_energy_error"] > 0.0
        assert record["viscous_final_energy_error"] > 0.0


def test_mass_residual_verdict_uses_the_steady_divergence():
    verdicts = {verdict.cid: verdict for verdict in L.conservation_verdict()}
    mass = verdicts["C4a"]
    assert mass.measured is not None
    assert mass.measured < L.CRITERIA["C4a"].threshold
    assert verdicts["C4b"].verdict == "PASS"


def test_withheld_levels_are_never_passed():
    """Levels 5 and 6 must stay WITHHELD with an explicit reason."""
    withheld = [L.turbulence_verdict()[0], L.suboff_verdict()[0]]
    assert [item.cid for item in withheld] == ["C8", "C9"]
    for item in withheld:
        assert item.verdict == "WITHHELD"
        assert item.measured is None
        assert len(item.note) > 40


# --------------------------------------------------------------------------- #
# Audit contract -- table, self-audit, report
# --------------------------------------------------------------------------- #
def test_thresholds_match_the_campaign_table():
    """The acceptance thresholds are the campaign's; no local relaxation."""
    expected = {"C1a": 1e-6, "C1b": 1e-3, "C2": 0.02, "C3": 0.03, "C4a": 1e-10,
                "C4b": 1e-6, "C4c": 1e-6, "C5": 0.2, "C6": 0.03, "C7": 0.03,
                "C8": 0.05, "C9": 0.10, "C10": 0.0}
    assert {cid: L.CRITERIA[cid].threshold for cid in expected} == expected


def test_criterion_judgement_handles_missing_values():
    assert L.CRITERIA["C1a"].satisfied(None) is None
    assert L.CRITERIA["C1a"].satisfied(float("nan")) is None
    assert L.CRITERIA["C1a"].satisfied(1e-9) is True
    assert L.CRITERIA["C1a"].satisfied(1e-3) is False


def test_self_audit_rejects_a_pass_without_a_measurement():
    verdicts = {cid: L._judge(L.CRITERIA[cid], 0.5 * L.CRITERIA[cid].threshold)
                for cid in L.CRITERIA if cid != "C10"}
    good = L._self_audit(verdicts)
    assert good.verdict == "PASS"
    verdicts["C1a"].measured = None
    broken = L._self_audit(verdicts)
    assert broken.verdict == "FAIL"
    assert broken.measured >= 1


def test_self_audit_flags_a_missing_criterion():
    verdicts = {cid: L._judge(L.CRITERIA[cid], 0.0) for cid in ("C1a", "C1b")}
    audit = L._self_audit(verdicts)
    assert audit.verdict == "FAIL"
    assert any("missing criteria" in problem for problem in audit.detail["problems"])


def test_full_audit_is_json_serialisable_and_complete(report):
    text = json.dumps(report)
    assert json.loads(text)["summary"]["criteria_count"] == len(L.CRITERIA)
    assert set(report["criteria"]) == set(L.CRITERIA)
    counts = report["summary"]["counts"]
    assert counts["PASS"] + counts["FAIL"] + counts["WITHHELD"] == len(L.CRITERIA)
    for cid, item in report["criteria"].items():
        assert item["verdict"] in ("PASS", "FAIL", "WITHHELD")
        if item["verdict"] == "WITHHELD":
            assert item["measured"] is None


def test_markdown_lists_every_criterion(report):
    markdown = L.render_markdown(report)
    for cid in L.CRITERIA:
        assert f"| {cid} |" in markdown
    assert "TensorFVM verification ladder" in markdown


def test_cli_writes_both_outputs(tmp_path, monkeypatch, report):
    monkeypatch.setattr(L, "audit", lambda **kwargs: report)
    assert L.main(["--out", str(tmp_path), "--quiet"]) in (0, 1)
    payload = json.loads((tmp_path / "ladder_audit.json").read_text())
    assert payload["summary"]["criteria_count"] == len(L.CRITERIA)
    assert (tmp_path / "ladder_audit.md").read_text().startswith(
        "# TensorFVM verification ladder")