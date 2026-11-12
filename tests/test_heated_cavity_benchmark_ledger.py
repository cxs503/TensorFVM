#!/usr/bin/env python
"""Regression guard for the m15410 heated-cavity benchmark ledger.

Pure-data test (no solver run, < 1 s): it re-checks the run ledger archived by
the board audit against an **independent** transcription of de Vahl Davis (1983)
Tables 2-3.  Nothing here is copied from the mission's report tables — the paper
values below are the ones printed in the 1983 paper, and the ledger has to meet
them.

Why this file exists: mission m15410 shipped the benchmark run + a driver, but
no test executed on CI consumed those numbers (the only test that asserts the
de Vahl Davis band, ``tests/test_fvm_internal_flow_verified.py::
test_cavity_nusselt_against_de_vahl_davis``, ``pytest.skip``s when
``benchmarks/verified/fvm_natural_convection/summary.json`` is absent — which it
is in a plain checkout).  This test closes that gap for the archived ledger.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "audit-m15410-heated-cavity" / "original-artifacts" / "summary.json"

#: de Vahl Davis, G. (1983), Int. J. Numer. Methods Fluids 3, 249-264, Pr = 0.71.
#: Table 2 (mean Nusselt number on the hot wall) and Table 3 (velocity extrema
#: normalised by alpha/H).
PAPER = {
    "1000": {"nu": 1.118, "u_max": 3.649, "v_max": 3.697},
    "10000": {"nu": 2.243, "u_max": 16.178, "v_max": 19.617},
    "100000": {"nu": 4.519, "u_max": 34.73, "v_max": 68.59},
    "1000000": {"nu": 8.800, "u_max": 64.63, "v_max": 219.36},
}
TOL = 0.03          # code gate used by the internal-flow test suite
PLATFORM_LIMIT = 0.15


@pytest.fixture(scope="module")
def ledger() -> dict:
    if not LEDGER.exists():
        pytest.fail(f"archived benchmark ledger missing: {LEDGER}")
    return json.loads(LEDGER.read_text())


def test_ledger_covers_every_paper_rayleigh_number(ledger: dict) -> None:
    assert set(ledger["cases"]) == set(PAPER)
    for ra, case in ledger["cases"].items():
        assert set(case["meshes"]) == {"32", "64", "128"}, ra
        for block in case["meshes"].values():
            assert block["converged"] is True
            assert block["steps"] > 0
            assert block["max_divergence"] < 1e-10


def test_recorded_references_match_the_paper(ledger: dict) -> None:
    """Guards the reference-transcription bug class fixed inside m15410."""
    for ra, ref in PAPER.items():
        acc = ledger["acceptance"][ra]
        assert acc["nu_reference"] == pytest.approx(ref["nu"], abs=5e-4), ra
        assert acc["u_max_reference"] == pytest.approx(ref["u_max"], rel=1e-4), ra
        assert acc["v_max_reference"] == pytest.approx(ref["v_max"], rel=1e-4), ra


def test_finest_mesh_is_inside_the_three_percent_band(ledger: dict) -> None:
    for ra, ref in PAPER.items():
        acc = ledger["acceptance"][ra]
        errors = {
            "nu": abs(acc["nu_computed"] - ref["nu"]) / ref["nu"],
            "u_max": abs(acc["u_max_computed"] - ref["u_max"]) / ref["u_max"],
            "v_max": abs(acc["v_max_computed"] - ref["v_max"]) / ref["v_max"],
        }
        worst = max(errors.values())
        assert worst < TOL, (ra, errors)
        assert worst < PLATFORM_LIMIT, (ra, errors)


def test_nu_error_shrinks_with_mesh_refinement(ledger: dict) -> None:
    for ra, ref in PAPER.items():
        meshes = sorted(ledger["cases"][ra]["meshes"], key=int)
        errors = [
            abs(ledger["cases"][ra]["meshes"][n]["nu_mean"] - ref["nu"]) / ref["nu"]
            for n in meshes
        ]
        assert errors[-1] < errors[0], (ra, errors)
        assert all(a >= b for a, b in zip(errors, errors[1:])), (ra, errors)


def test_nu_mean_is_the_mean_of_the_wall_profile(ledger: dict) -> None:
    """The headline number must be derived from the field, not typed in."""
    for ra, case in ledger["cases"].items():
        for n, block in case["meshes"].items():
            profile = block["nu_local"]
            assert len(profile) == int(n)
            assert math.isclose(sum(profile) / len(profile), block["nu_mean"],
                                rel_tol=0.0, abs_tol=1e-9), (ra, n)


def test_hot_wall_profile_is_physical(ledger: dict) -> None:
    """Boundary layer on the hot wall: peak just above the bottom corner, then decay."""
    for ra, case in ledger["cases"].items():
        for n, block in case["meshes"].items():
            profile = block["nu_local"]
            peak = max(range(len(profile)), key=profile.__getitem__)
            assert peak <= 0.30 * len(profile), (ra, n, peak)
            tail = profile[peak:]
            assert all(a >= b for a, b in zip(tail, tail[1:])), (ra, n)
            assert profile[-1] < 2.0, (ra, n, profile[-1])