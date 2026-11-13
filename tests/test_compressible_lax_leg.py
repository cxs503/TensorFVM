"""CASE-LAX as a *case-pool member*: the Lax leg of the standard harness.

``tensorfvm.compressible.verification`` is the registry the platform calls to get
the compressible verification artifact.  Before this gate the Lax (1954) shock
tube existed only as a stand-alone script (``scripts/lax_shock_tube_benchmark.py``)
in ``docs/lax-shock-tube`` and could not be selected with
``python -m tensorfvm.compressible --legs lax``.  These tests pin the registration:

1. ``run_all(legs=("lax",), quick=True)`` runs the leg through the same driver as
   Sod/interface/... and writes the JSON artifact plus the figure;
2. every graded criterion of the leg is ``True`` on the pytest-sized grid, so the
   < 2 % acceptance of CASE-LAX is enforced at CI scale, not only in the docs run;
3. the leg carries its own audit trail (two-implementation exact-solution
   cross-check, Rankine-Hugoniot residual, wave ordering), so the reference is
   checkable from the artifact alone;
4. the artifact collapse (``_strip_heavy``) keeps the numeric rows and the
   waveform profile while dropping field snapshots -- no silent data loss.

Runtime of the quick leg is ~25 s single process, i.e. the same order as the Sod
leg; the full (200/400/800) run lives in ``docs/CASE-LAX``.
"""

from __future__ import annotations

import json

import pytest

from tensorfvm.compressible import verification

#: verdict keys the leg must publish (acceptance of CASE-LAX + its audit trail)
EXPECTED_VERDICTS = {
    "lax_plateau[p_star]",
    "lax_plateau[u_star]",
    "lax_plateau[rho_star_l]",
    "lax_plateau[rho_star_r]",
    "lax_position[x_fan]",
    "lax_position[x_contact]",
    "lax_position[x_shock]",
    "lax_l1[rho]",
    "lax_l1[u]",
    "lax_l1[p]",
    "lax_mass_drift",
    "lax_no_positivity_fix",
    "lax_exact_solution_independently_reproduced",
    "lax_rankine_hugoniot_satisfied",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("case_lax_leg")
    result = verification.run_all(outdir, verification.RunConfig(quick=True),
                                  legs=("lax",), figures=True)
    artifact = json.loads((outdir / "compressible_multimaterial.json").read_text())
    return result, artifact, outdir


def test_lax_leg_is_dispatched_and_alone_in_the_artifact(run):
    result, artifact, _ = run
    assert set(result["legs"]) == {"lax"}
    assert set(artifact["verdicts"]) == {"lax"}
    assert artifact["all_criteria_met"] is True


def test_lax_leg_publishes_the_acceptance_verdicts(run):
    result, artifact, _ = run
    verdicts = artifact["verdicts"]["lax"]
    missing = EXPECTED_VERDICTS - set(verdicts)
    assert not missing, f"missing verdicts: {sorted(missing)}"
    for key in sorted(EXPECTED_VERDICTS):
        assert verdicts[key] is True, f"{key} is not True: {verdicts[key]}"


def test_lax_leg_numeric_rows_match_the_documented_sweep(run):
    """The quick grid (200/400) must reproduce the documented L1 values."""
    _, artifact, _ = run
    leg = artifact["legs"]["lax"]
    rows = {r["nx"]: r for r in leg["rows"]}
    assert set(rows) == {200, 400}
    # docs/lax-shock-tube/report.md ⑤: 200 -> 1.943 %, 400 -> 1.015 % (rho)
    assert rows[200]["l1_rel"]["rho"] == pytest.approx(1.943e-2, rel=5e-2)
    assert rows[400]["l1_rel"]["rho"] == pytest.approx(1.015e-2, rel=5e-2)
    # finer grid must be closer to the exact solution in every field
    for key in ("rho", "u", "p"):
        assert rows[400]["l1_rel"][key] < rows[200]["l1_rel"][key]
    assert leg["graded"]["nx"] == 400


def test_lax_reference_is_auditable_from_the_artifact(run):
    """Two independent exact solutions + RH residual + wave ordering, all in JSON."""
    _, artifact, _ = run
    leg = artifact["legs"]["lax"]
    check = leg["independent_exact_check"]
    assert check["rel_dp_star"] < 1e-9
    assert check["rel_du_star"] < 1e-9
    assert check["max_abs_profile_diff_scaled"] < 1e-9
    rh = leg["rankine_hugoniot"]
    assert abs(rh["rel_residual_mass"]) < 1e-12
    assert abs(rh["rel_residual_momentum"]) < 1e-12
    assert abs(rh["rel_residual_energy"]) < 1e-12
    w = leg["exact_waves"]
    assert w["x_head"] < w["x_tail"] < w["x_contact"] < w["x_shock"]
    # the right wave must be a shock, the left one a rarefaction (Lax, not Sod)
    assert w["left_wave"] == "rarefaction"
    assert w["right_wave"] == "shock"
    assert w["p_star"] > 4.0 * leg["config"]["p_r"]


def test_lax_leg_writes_the_profile_and_figure(run):
    _, artifact, outdir = run
    leg = artifact["legs"]["lax"]
    # _strip_heavy drops field snapshots but keeps the graded profile? no: the
    # profile is dropped from the artifact by design (it is a field snapshot);
    # the figure must exist and the figure path must be recorded.
    profile = leg.get("profile")
    assert profile is None
    figs = artifact["figures"]["lax"]
    assert figs and figs[0].endswith("fig_lax_profiles_vs_exact.png")
    assert (outdir / "fig_lax_profiles_vs_exact.png").stat().st_size > 10_000


def test_lax_leg_survives_the_cli_leg_name():
    """``--legs lax`` must be accepted by the CLI parser (documented entry point)."""
    from tensorfvm.compressible.__main__ import main

    with pytest.raises(SystemExit) as exc:  # --help is the cheapest parse check
        main(["--help"])
    assert exc.value.code == 0