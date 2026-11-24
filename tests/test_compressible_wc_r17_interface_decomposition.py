"""Regression gate for the round-17 interface decomposition (CD dominance).

The round-16 defect (`R16-FLUX-DEFECT-CLOSURE.md`): E2 (max|D| h^1 shrinkage)
was falsified at nx=25600 (p_D = 0.6116, frozen band [1.85, 2.15]).
R16's "next hand" was to decompose D(x) by interface and identify which
segment contributes to the slow shrinkage.

This test asserts that the contact discontinuity (CD) region dominates
the total |D(x)| integral, confirming R16's hypothesis.

Criteria F1–F4 are frozen in ``PREDECLARATION-r17.md`` (written before
any number was computed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "docs" / "XVAL-FVM-M18737-ROUND2"
sys.path.insert(0, str(HERE))

import wc_w1_r17_interface_decomposition as r17  # noqa: E402

FROZEN = r17.FROZEN


def test_e0_artifact_gate_passes():
    """E0: 12800/25600 CFL 0.8 rungs must be present and valid."""
    e0 = r17.e0_artifact_gate()
    assert e0["pass"] is True
    assert e0["profile_12800_present"] is True
    assert e0["profile_25600_present"] is True
    assert e0["metrics_12800_present"] is True
    assert e0["metrics_25600_present"] is True
    assert e0["nx_12800"] == 12800
    assert e0["nx_25600"] == 25600
    assert e0["cfl_12800"] == 0.8
    assert e0["cfl_25600"] == 0.8
    assert e0["status_12800"] == "ok"
    assert e0["status_25600"] == "ok"


def test_interfaces_detected():
    """F0: At least 3 interfaces must be detected in the window [0.4, 1.0]."""
    prof_25600 = r17.load(25600, r17.CFL08_FAMILY)
    interfaces = r17.locate_interfaces(prof_25600["rho"], prof_25600["x"])
    assert len(interfaces) >= 3, f"Only {len(interfaces)} interfaces detected"
    # All interfaces should be in the window
    for xi in interfaces:
        assert r17.WINDOW[0] <= xi <= r17.WINDOW[1], f"Interface {xi} outside window"


def test_segment_contributions_sum_to_total():
    """F4: Sum of segment contributions must equal total integral (within 1%)."""
    out = r17.build(verbose=False)
    ratio = out["total_contribution"]["ratio"]
    assert abs(ratio - 1.0) < FROZEN["F4_total_conservation"], \
        f"Segment sum ratio {ratio} not within {FROZEN['F4_total_conservation']} of 1.0"


def test_cd_contribution_passes():
    """F1: CD (region_2) contribution must be >= 30% of total integral."""
    out = r17.build(verbose=False)
    cd_contrib = out["criteria"]["F1"]["value"]
    threshold = FROZEN["F1_CD_contribution_min"] * 100
    assert cd_contrib >= threshold, \
        f"CD contribution {cd_contrib}% < threshold {threshold}%"
    assert out["criteria"]["F1"]["verdict"] == "PASS"


def test_cc_contribution_fails():
    """F2: CC (region_3) contribution must be < 20% (honest FAIL)."""
    out = r17.build(verbose=False)
    cc_contrib = out["criteria"]["F2"]["value"]
    threshold = FROZEN["F2_CC_contribution_min"] * 100
    assert cc_contrib < threshold, \
        f"CC contribution {cc_contrib}% >= threshold {threshold}%"
    assert out["criteria"]["F2"]["verdict"] == "FAIL"


def test_pre_cd_and_post_cc_pass():
    """F3: pre-CD and post-CC contributions must each be < 15%."""
    out = r17.build(verbose=False)
    pre_cd = out["criteria"]["F3"]["pre_cd_value"]
    post_cc = out["criteria"]["F3"]["post_cc_value"]
    threshold = FROZEN["F3_LS_RS_contribution_max"] * 100
    assert pre_cd < threshold, f"pre-CD contribution {pre_cd}% >= threshold {threshold}%"
    assert post_cc < threshold, f"post-CC contribution {post_cc}% >= threshold {threshold}%"
    assert out["criteria"]["F3"]["verdict"] == "PASS"


def test_main_verdict_is_pass():
    """F1 is the承重判据; if it passes, the main verdict is PASS."""
    out = r17.build(verbose=False)
    assert out["verdict"] == "PASS", \
        f"Main verdict should be PASS (F1 passes), got {out['verdict']}"


def test_json_output_is_consistent():
    """The JSON output must match the build() result."""
    json_path = HERE / "r17_interface_decomposition.json"
    if json_path.exists():
        with open(json_path) as f:
            saved = json.load(f)
        out = r17.build(verbose=False)
        # Check key fields match
        assert saved["e0_artifact_gate"]["pass"] is True
        assert saved["criteria"]["F1"]["verdict"] == "PASS"
        assert saved["criteria"]["F4"]["verdict"] == "PASS"
        assert abs(saved["total_contribution"]["ratio"] - 1.0) < 0.01


def test_figure_is_generated():
    """The figure file must exist and be non-empty."""
    fig_path = HERE / "figures" / "fig_r17_interface_decomposition.png"
    assert fig_path.exists(), f"Figure not found: {fig_path}"
    assert fig_path.stat().st_size > 0, "Figure file is empty"
