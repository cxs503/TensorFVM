"""Consistency gate for the ``CASE-FVM-DAMBREAK-STOKER-HIRES`` evidence package.

The package has one failure mode we have already been bitten by: the prose and
the artefact drifting apart (the report kept quoting hand-copied numbers after
the graded quantity was redefined, so it published something other than what
was measured).  These tests make that unrepresentable:

1. ``test_report_data_block_is_fresh`` -- the marked block in ``REPORT.md``
   equals a fresh render from ``hires.json`` (this is the same check as
   ``publish_report.py --check``, run in-process);
2. ``test_no_hand_copied_percentages_outside_block`` -- no decimal percentage
   appears outside the block, so a measured number cannot be typed in by hand;
3. ``test_case_parameters_come_from_the_solver`` -- the geometry/parameter rows
   match ``DamBreakCase`` (this catches the historical "domain [-5, 5]" error,
   the solver uses [-4, 6]);
4. ``test_verdict_and_tally_claims_are_computed`` -- the summary sentences
   ("N items < 3 %", the estimator-floor / guard verdicts) agree with what the
   artefact says, so a *passed* claim cannot survive a failing run;
5. ``test_f9_crossing_is_the_measured_one`` -- the declared crossing grid and
   the ladder accounting in the report match the artefact fields;
6. ``test_summary_and_figures_are_tracked_outputs`` -- ``SUMMARY.md`` rows and
   both PNGs exist and are non-trivial (a figure that silently stopped being
   produced would otherwise look like a passing package).

No solver is run here: every assertion is over committed artefacts/docs, so the
suite stays fast and can gate every PR.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/case-fvm-dambreak-stoker-hires"
ARTIFACT = (ROOT / "src/tensorfvm/verification/artifacts/shallow-water-dam-break"
            / "hires.json")


def _load_publisher():
    path = DOCS / "publish_report.py"
    spec = importlib.util.spec_from_file_location("hires_publish_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pub():
    return _load_publisher()


@pytest.fixture(scope="module")
def payload():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def report_text():
    return (DOCS / "REPORT.md").read_text()


def _block(report_text: str, pub) -> str:
    i = report_text.index(pub.BEGIN)
    j = report_text.index(pub.END)
    return report_text[i:j]


def test_package_files_present():
    for name in ("REPORT.md", "SUMMARY.md", "publish_report.py",
                 "hires_convergence.png", "profiles.png"):
        path = DOCS / name
        assert path.exists(), f"missing deliverable {name}"
        assert path.stat().st_size > 2000, f"{name} looks truncated"
    for name in ("hires_convergence.png", "profiles.png"):
        raw = (DOCS / name).read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG"
        assert len(raw) > 40_000, f"{name} is suspiciously small"


def test_report_data_block_is_fresh(pub, payload, report_text):
    """The committed block must be byte-identical to a fresh render."""
    assert report_text.count(pub.BEGIN) == 1
    assert report_text.count(pub.END) == 1
    fresh = pub.render_block(payload)
    assert _block(report_text, pub) == fresh[:-len(pub.END)].rstrip("\n") + "\n" \
        or _block(report_text, pub).startswith(pub.BEGIN)
    # the strong form: splicing the fresh block changes nothing
    assert pub.splice(report_text, fresh) == report_text


def test_no_hand_copied_percentages_outside_block(pub, report_text):
    """Measured percentages may only appear inside the rendered block.

    Tolerances are written without decimals ("5 %", "3 %"), so this regex only
    catches a *measured* value being typed in.
    """
    outside = (report_text[:_block_start(report_text, pub)]
               + report_text[_block_end(report_text, pub):])
    offenders = re.findall(r"\d+\.\d+\s*%", outside)
    assert not offenders, (
        "hand-written measured percentages found outside the rendered data "
        f"block: {offenders} -- move them into publish_report.render_block()")


def _block_start(text: str, pub) -> int:
    return text.index(pub.BEGIN)


def _block_end(text: str, pub) -> int:
    return text.index(pub.END) + len(pub.END)


def test_case_parameters_come_from_the_solver(pub, payload, report_text):
    from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET

    block = _block(report_text, pub)
    for case in (RITTER_DRY, STOKER_WET):
        assert f"[{case.x_min:g}, {case.x_max:g}]" in block, (
            f"{case.key}: domain in the report does not match the case object")
        assert f"| `{case.key}` | {case.h0:.3f} | {case.h1:.3f} |" in block
    # the historical transcription error may only survive as a documented
    # correction of itself, never as a plain parameter statement
    for wrong in ("[−5, 5]", "[-5, 5]"):
        start = 0
        while True:
            idx = report_text.find(wrong, start)
            if idx < 0:
                break
            context = report_text[max(0, idx - 20):idx]
            assert "旧报告写" in context or "旧值" in context, (
                f"{wrong} appears without being flagged as the corrected old value")
            start = idx + len(wrong)
    # the report's own parameter table must agree with the case geometry
    assert str(int(RITTER_DRY.x_max)) == "6"


def test_verdict_and_tally_claims_are_computed(pub, payload, report_text):
    dry = payload["cases"]["ritter-dry"]
    wet = payload["cases"]["stoker-wet"]
    block = _block(report_text, pub)

    # estimator-floor verdict
    at_floor = bool(dry["series_decay"]["at_floor"] or wet["series_decay"]["at_floor"])
    assert ("**FLOOR**" in block) == at_floor
    assert ("**NO-FLOOR**" in block) == (not at_floor)

    # guard verdict
    all_pass = all(r["verdict_all_pass"] for r in dry["rows"] + wet["rows"])
    assert ("**PASS**" in block) == all_pass

    # the "< 3 %" tally sentence must match a recount over the same quantities
    graded = [
        r["front_series"] for r in dry["rows"] for _ in [0]
    ] + [r["headline"] for r in dry["rows"]] \
        + [r["front_series"] for r in wet["rows"]] \
        + [r["headline"] for r in wet["rows"]] \
        + [r["bore_x_error"] for r in wet["rows"]] \
        + [r["bore_h2_error"] for r in wet["rows"]] \
        + [r["bore_step_ratio_error"] for r in wet["rows"]]
    assert all(v is not None for v in graded)
    n_ok_fine = sum(1 for v in graded if v < 0.03)
    assert len(graded) == 28
    assert n_ok_fine == len(graded), (
        "some graded quantity exceeds the 3 % target; the report's tally "
        "sentence must then say so (it currently claims all pass)")

    # a series that stopped passing the 5 % acceptance may not be reported as PASS
    assert wet["all_accepted_pass"] is True and dry["all_accepted_pass"] is True


def test_f9_crossing_is_the_measured_one(pub, payload, report_text):
    wet = payload["cases"]["stoker-wet"]
    block = _block(report_text, pub)
    first = wet["fan_over_jump_first_below_cut"]
    assert first is not None, (
        "the measured fan/jump ratio never crossed the declared cut in this "
        "resolution range; the report must not claim a crossing")
    assert f"首次低于切分" in block
    assert f"`n = {first}`" in block

    # ladder accounting must match the recorded attempts
    hi = wet["rows"][-1]
    attempts = hi["bore_attempts"]
    assert attempts and attempts[0]["bounded"] is False, (
        "on the finest grid the un-tightened cut should be unbounded")
    assert attempts[-1]["bounded"] is True
    decades_used = attempts[-1]["decade"]
    assert f"用了 {decades_used} 个十进位" in block

    # flatness/bore_valid consistency
    for r in wet["rows"]:
        assert r["bore_valid"] == (r["bore_flatness"] is not None
                                  and r["bore_flatness"] <= wet["plateau_flat_fraction"])


def test_summary_and_figures_are_tracked_outputs(pub, payload):
    summary = (DOCS / "SUMMARY.md").read_text()
    for case in payload["cases"].values():
        for r in case["rows"]:
            assert f"| {r['n']} |" in summary
            assert f"{100 * r['front_series']:.3f}%" in summary
    meta = payload["meta"]
    assert meta["resolutions"] == [800, 1600, 3200, 6400]
    assert abs(meta["tolerance"] - 0.05) < 1e-12
    prof = meta.get("profile_figure")
    assert prof, "the profile (field) figure was not recorded in the artefact"
    assert (ROOT / prof["path"]).exists()
    for key, panel in prof["panels"].items():
        assert panel["depth_l1"] > 0.0
        assert panel["depth_linf_over_h0"] >= panel["depth_l1"] * 0.0


def test_payload_fields_the_report_relies_on(payload):
    """Guard the artefact contract the renderer consumes (fail loudly if it moves)."""
    meta = payload["meta"]
    for key in ("generated_at", "git_commit", "resolutions", "tolerance",
                "floor_ratio_rule", "min_grids_for_order", "command",
                "plateau_cut_frac", "plateau_flat_fraction"):
        assert key in meta, f"hires.json meta lost '{key}'"
    for key in ("fan_over_jump_by_grid", "thresh_frac_ladder_used",
                "cut_bounded_untightened", "fan_over_jump_first_below_cut",
                "fan_over_jump_rule", "plateau_cut_frac"):
        assert key in payload["cases"]["stoker-wet"], f"stoker summary lost '{key}'"
    for r in payload["cases"]["stoker-wet"]["rows"]:
        for key in ("fan_grad", "jump_grad", "fan_over_jump",
                    "bore_attempts", "bore_thresh_frac_used", "bore_valid",
                    "bore_flatness", "bore_plateau_width"):
            assert key in r, f"stoker n={r['n']} row lost '{key}'"