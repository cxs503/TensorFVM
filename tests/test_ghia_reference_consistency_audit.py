"""The Ghia reference-consistency audit must keep its own claims checkable.

``src/tensorfvm/verification/ghia_reference_audit.py`` classifies every published
ordinate of the three Ghia tables as consistent, under-resolved or a reference
defect, and cross-checks that verdict against
``tensorfvm.lid_cavity_reference.DISPUTES``.  A classification that no longer
reproduces the committed scoring, that silently widens the register, or that cannot
see an injected defect, would make the whole exercise circular -- so each of those
is pinned here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil

import pytest

from tensorfvm.lid_cavity import ghia_acceptance
from tensorfvm.lid_cavity_reference import DISPUTES, disputes_by_case, table
from tensorfvm.verification import ghia_reference_audit as gra

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(REPO, "docs", "verification-lid-cavity-ladder", "summary.json")
SCRIPT = os.path.join(REPO, "scripts", "audit_ghia_reference_consistency.py")
DOCS = os.path.join(REPO, "docs", "verification-ghia-reference-consistency")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def bundle():
    if not os.path.exists(BUNDLE):
        pytest.skip("ladder bundle not checked out")
    return gra.load_bundle(BUNDLE)


@pytest.fixture(scope="module")
def report(bundle):
    return gra.audit(BUNDLE)


# --------------------------------------------------------------------------- #
# the audit reproduces the committed evidence
# --------------------------------------------------------------------------- #
def test_missing_bundle_exits_with_a_readable_reason():
    with pytest.raises(SystemExit) as error:
        gra.load_bundle(os.path.join(REPO, "docs", "does-not-exist", "summary.json"))
    assert "not found" in str(error.value)


def test_committed_profile_errors_follow_from_the_stored_profiles(bundle, report):
    checks = gra.profile_error_reproduction(bundle)
    assert len(checks) == 2 * len(bundle["cases"])
    assert all(item["agrees"] for item in checks)
    assert report["profile_errors_reproduced"] is True


def test_grids_present_and_the_documented_gap(report):
    meta = report["meta"]
    assert meta["grids_present"] == {"100": [32, 64, 128],
                                     "400": [32, 64, 128],
                                     "1000": [64]}
    # the bundle advertises the gap itself; the audit must not paper over it
    assert meta["grids_missing"] == [{"reynolds": 1000.0, "n": 32},
                                     {"reynolds": 1000.0, "n": 128}]
    assert meta["bundle_is_partial"] is True


def test_every_published_ordinate_is_classified_once(report):
    expected = sum(len(table(re, which))
                   for re in (100.0, 400.0, 1000.0) for which in ("u", "v"))
    assert report["verdict"]["published_points"] == expected
    assert sum(report["verdict"]["counts"].values()) == expected


# --------------------------------------------------------------------------- #
# exactly one ordinate is a defect, and it is the registered one
# --------------------------------------------------------------------------- #
def test_exactly_one_published_ordinate_is_a_reference_defect(report):
    defects = report["verdict"]["reference_defects"]
    assert [(d["reynolds"], d["which"], d["x"]) for d in defects] == [(400.0, "v", 0.9063)]
    counts = report["verdict"]["counts"]
    assert counts.get("consistent", 0) == 101
    assert counts.get("under_resolved", 0) == 0
    assert counts.get("unverified_single_grid", 0) == 0


def test_the_defect_residual_is_saturated_while_its_neighbours_converge(report):
    rows = {r["x"]: r for r in
            report["pointwise"]["reynolds"]["400.0"]["which"]["v"]["rows"]}
    bad = rows[0.9063]
    assert bad["ratio_previous_over_finest"] < 1.5
    assert bad["residual_over_uncertainty"] > 10.0
    assert bad["convergence"] == "saturated"
    assert bad["verdict"] == "reference_defect"
    # clean second order: the increments shrink by ~4x
    assert bad["observed_order"] == pytest.approx(2.0, abs=0.2)
    assert bad["residual"]["128"] == pytest.approx(0.14931, abs=2e-4)
    others = [r["residual"]["128"] for x, r in rows.items() if x != 0.9063]
    assert max(others) < 0.005


def test_the_register_and_the_detection_agree_in_both_directions(report):
    register = report["register"]
    assert register["agreement"] is True
    assert register["missing_from_register"] == []
    assert register["missing_from_detection"] == []
    stripped = [{k: v for k, v in item.items() if k != "published"}
                for item in register["declared"]]
    assert stripped == register["detected"] == [
        {"reynolds": 400.0, "which": "v", "x": 0.9063}]
    # the register keeps the published number untouched
    assert DISPUTES[(400.0, "v", 0.9063)]["published"] == -0.23827


def test_a_new_defect_would_break_the_register_report(bundle):
    poisoned = gra.inject_defect(bundle, 100.0, "u", 0.5000, 0.05)
    verdict = gra._verdict_block(gra.pointwise(poisoned))
    register = gra.register_crosscheck(verdict)
    assert register["agreement"] is False
    assert {"reynolds": 100.0, "which": "u", "x": 0.5} in register["missing_from_register"]


def test_the_shape_control_is_reported_but_carries_no_verdict(report):
    shape = report["shape_control"]
    assert shape["unexplained"] == []
    assert "400/v/0.9063" in {item["key"] for item in shape["flagged"]}
    # the audit says so explicitly instead of letting the shape test look decisive
    assert shape["used_for_verdict"] is False
    assert "near-extremum" in shape["why_not"]


# --------------------------------------------------------------------------- #
# the gate is re-derived, not trusted
# --------------------------------------------------------------------------- #
def test_gate_verdicts_are_reproduced_and_exclusions_are_named(report):
    assert report["gate_verdicts_reproduced"] is True
    by_case = {(item["reynolds"], item["n"]): item for item in report["acceptance"]}
    for case in ((100.0, 32), (100.0, 64), (100.0, 128), (1000.0, 64)):
        assert by_case[case]["strict_pass"] is True
        assert by_case[case]["excluded_points"] == []
    red = by_case[(400.0, 128)]
    assert red["strict_pass"] is False
    assert red["failing_criteria"] == ["v_max_abs_error"]
    assert red["lenient_pass"] is True
    assert len(red["excluded_points"]) == 1
    assert red["excluded_points"][0]["x"] == 0.9063
    assert red["excluded_points"][0]["published"] == -0.23827


def test_register_aware_gate_cannot_hide_a_second_bad_point(report):
    """The exclusion is keyed by ordinate: a second failure still fails."""
    record = gra.case_record(gra.load_bundle(BUNDLE), 400.0, 128)
    result = {"reynolds": 400.0, "u": record["u"], "v": record["v"],
              "vortex_centre": record["vortex_centre"]}
    broken = json.loads(json.dumps(result))
    broken["u"]["profiles"]["delta"][0] += 0.05
    checks = ghia_acceptance(broken, allow_registered_disputes=True)
    assert checks["criteria"]["u_max_abs_error"]["pass"] is False
    assert checks["pass"] is False
    assert len(checks["registered_disputes"]) == 1


def test_register_grouping_matches_between_gate_and_audit():
    grouped = disputes_by_case()
    assert set(grouped) == {(re, which) for re, which, _ in DISPUTES}
    for (re, which), entries in grouped.items():
        for x, record in entries.items():
            assert record is DISPUTES[(re, which, x)]


# --------------------------------------------------------------------------- #
# the audit has teeth
# --------------------------------------------------------------------------- #
def test_negative_control_sees_the_injected_corruption_only(report):
    control = report["negative_control"]
    assert control["detected"] is True
    assert control["no_other_new_defects"] is True
    assert control["new_defects"] == [{"reynolds": 100.0, "which": "u", "x": 0.5}]


def test_a_shrinking_residual_is_called_under_resolved_not_a_defect(bundle):
    """A resolution artefact must be distinguishable from a table defect."""
    poisoned = json.loads(json.dumps(bundle))
    for record in poisoned["cases"]:
        if record["reynolds"] != 100.0:
            continue
        profile = record["v"]["profiles"]
        coords = [round(c, 4) for c in profile["coord"]]
        index = coords.index(0.5)
        # error proportional to h^2: 4x smaller on each 2:1 refinement
        magnitude = {32: 0.8, 64: 0.2, 128: 0.05}[record["n"]]
        profile["computed"][index] += magnitude
        profile["delta"][index] = profile["computed"][index] - profile["reference"][index]
    verdict = gra._verdict_block(gra.pointwise(poisoned))
    rows = {(r["reynolds"], r["which"], r["x"]): r
            for r in gra._flatten(gra.pointwise(poisoned))}
    row = rows[(100.0, "v", 0.5)]
    assert row["verdict"] == "under_resolved"
    assert row["ratio_previous_over_finest"] > 2.0
    assert (100.0, "v", 0.5) not in {(d["reynolds"], d["which"], d["x"])
                                     for d in verdict["reference_defects"]}


def test_injection_at_a_non_published_ordinate_is_refused(bundle):
    with pytest.raises(KeyError):
        gra.inject_defect(bundle, 100.0, "u", 0.1234, 0.05)


# --------------------------------------------------------------------------- #
# committed artefacts
# --------------------------------------------------------------------------- #
def _script_module():
    spec = importlib.util.spec_from_file_location("ghia_consistency_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_package_re_renders_from_its_own_record():
    """``--check`` re-renders results.json/report.md and the two figures.

    JSON and Markdown are compared exactly; figures are compared by geometry
    because their bytes carry the freetype/Matplotlib build of the rendering
    machine (see the module docstring of the script).  The plotted numbers are
    pinned exactly by ``results.json``, so nothing numeric is lost by that.
    """
    if not os.path.isdir(DOCS):
        pytest.skip("verification package not generated")
    module = _script_module()
    assert module.main(["--check"]) == 0


def _committed_package_copy(target):
    for name in ("results.json", "report.md", "profile.png"):
        shutil.copy(os.path.join(DOCS, name), target / name)
    return target


def test_figure_bytes_follow_the_rasteriser_while_the_data_does_not(tmp_path):
    """Why ``--check`` compares geometry, not bytes.

    A PNG's size moves with the freetype/Matplotlib build that rendered it -- the
    same drift a different runner produces.  Rendering the *same* record under a
    different hinting mode reproduces that: the raster changes, the numbers and
    the figure geometry do not.  Byte equality would therefore have made the
    check a test of the machine, which is what the CI failure was.
    """
    module = _script_module()
    bundle = module.load_bundle(module.DEFAULT_BUNDLE)
    report = module.audit(module.DEFAULT_BUNDLE)
    baseline = module.build(report, bundle, str(tmp_path / "baseline"))
    moved = 0
    for knobs in ({"text.hinting": "no_hinting"}, {"text.hinting": "force_autohint"},
                  {"font.size": 9.0}):
        with module.matplotlib.rc_context(knobs):
            variant = module.build(report, bundle, str(tmp_path / f"v{moved}"))
        assert (module._png_geometry(variant["profile.png"])
                == module._png_geometry(baseline["profile.png"]))
        with open(baseline["results.json"]) as a, open(variant["results.json"]) as b:
            assert a.read() == b.read()
        if os.path.getsize(variant["profile.png"]) != os.path.getsize(baseline["profile.png"]):
            moved += 1
    if not moved:
        pytest.skip("this rasteriser ignores the hinting switches; drift not demonstrable")
    assert moved >= 1


def test_re_render_check_notices_a_figure_of_the_wrong_geometry(tmp_path):
    """The geometry comparison is what keeps ``--check`` portable **and** sharp."""
    if not os.path.isdir(DOCS):
        pytest.skip("verification package not generated")
    module = _script_module()
    _committed_package_copy(tmp_path)
    figure = module.plt.figure(figsize=(1.0, 1.0))
    figure.savefig(tmp_path / "convergence.png", dpi=20)
    module.plt.close(figure)
    assert module.main(["--check", "--out", str(tmp_path)]) == 1


def test_re_render_check_notices_a_hand_edited_number(tmp_path):
    """The note must not be able to drift from ``results.json``."""
    if not os.path.isdir(DOCS):
        pytest.skip("verification package not generated")
    module = _script_module()
    _committed_package_copy(tmp_path)
    text = (tmp_path / "report.md").read_text()
    (tmp_path / "report.md").write_text(text.replace("0.03", "0.05", 1))
    assert module.main(["--check", "--out", str(tmp_path)]) == 1


def test_re_render_check_refuses_a_non_png_artefact(tmp_path):
    if not os.path.isdir(DOCS):
        pytest.skip("verification package not generated")
    module = _script_module()
    _committed_package_copy(tmp_path)
    (tmp_path / "convergence.png").write_bytes(b"not a png at all")
    assert module.main(["--check", "--out", str(tmp_path)]) == 1


def test_committed_results_carry_the_same_verdict_as_a_fresh_audit(report):
    path = os.path.join(DOCS, "results.json")
    if not os.path.exists(path):
        pytest.skip("verification package not generated")
    with open(path) as handle:
        committed = json.load(handle)
    assert committed["verdict"]["counts"] == report["verdict"]["counts"]
    assert committed["register"]["agreement"] is True
    assert committed["meta"]["bundle_md5"] == report["meta"]["bundle_md5"]
    assert committed["ok"] is True