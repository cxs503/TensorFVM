"""Criterion gates for the two-phase CSF capillary chain (m3646 evidence refresh).

The published record ``docs/two-phase-surface-tension-audit/report.md`` reports
``dP = -1.19e22 Pa`` against an exact ``sigma/R = 0.35 Pa``.  These tests pin what
the *current* source actually delivers, at the grids the published record used,
so that neither the (irreproducible) catastrophic numbers nor a silent regression
can be mistaken for the measured state.

Measured values (``scripts/audit_two_phase_capillary_evidence.py``):

| n  | dP [Pa]  | rel err | kappa rel err | Ca       |
|----|----------|---------|---------------|----------|
| 24 | 0.355870 | +1.68 % | +0.32 %       | 1.92e-06 |
| 32 | 0.343787 | -1.78 % | +10.05 %      | 1.07e-05 |
| 48 | 0.284004 | -18.86 %| +30.61 %      | 4.12e-05 |
| 64 | 0.328046 | -6.27 % | +2.40 %       | 4.79e-05 |

The 3 % Young--Laplace bar is therefore **met at n = 24/32 and missed at
n = 48/64**, non-monotonically: the error tracks the alignment of the smeared
circular interface with the grid, not the grid size.  The suite keeps that
verdict visible instead of tuning a tolerance until it passes.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "audit_two_phase_surface_tension.py"
RADIUS, SIGMA, MU, DT, STEPS = 0.2, 0.07, 0.01, 1e-5, 10
EXACT_JUMP = SIGMA / RADIUS

#: `dP` recorded by the stale published artifact at the same grids.
PUBLISHED_JUMP = {24: -1.18714e22, 32: 8.49545e23}


@lru_cache(maxsize=None)
def _module():
    spec = importlib.util.spec_from_file_location("_st_audit_for_tests", AUDIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _laplace(n):
    return _module().laplace_study(n, RADIUS, SIGMA, MU, STEPS, DT)


@lru_cache(maxsize=None)
def _curvature(n):
    return _module().curvature_study(n, RADIUS)


def test_static_droplet_run_is_finite_with_quiet_spurious_currents():
    """A static drop must not diverge: the published 1e20 m/s is not the state."""
    for n in (24, 32):
        run = _laplace(n)
        assert run["finite"], n
        assert run["capillary_number"] < 1e-2, (n, run["capillary_number"])
        assert run["final_divergence"] < 1e-12, (n, run["final_divergence"])


def test_published_catastrophic_jump_is_not_reproducible_from_current_source():
    """Guard against re-citing the stale record as if it were current evidence."""
    for n, published in PUBLISHED_JUMP.items():
        run = _laplace(n)
        ratio = abs(published) / abs(run["pressure_jump_pa"])
        assert ratio > 1e10, (n, ratio)
        assert abs(run["pressure_jump_pa"] - EXACT_JUMP) < 0.25 * EXACT_JUMP


def test_young_laplace_within_three_percent_where_the_interface_is_grid_aligned():
    for n in (24, 32):
        relative = _laplace(n)["relative_error"]
        assert abs(relative) < 0.03, (n, relative)


def test_finer_grids_miss_the_three_percent_bar_non_monotonically():
    """Recorded limitation: the error follows interface/grid alignment, not h."""
    rough = _laplace(48)["relative_error"]
    finer = _laplace(64)["relative_error"]
    assert rough < -0.10, rough
    assert abs(rough) > abs(_laplace(32)["relative_error"])
    assert abs(finer) > 0.03, finer


def test_curvature_of_the_exact_circle_is_second_order_at_the_finest_grid():
    coarse = abs(_curvature(32)["relative_error"])
    finest = abs(_curvature(64)["relative_error"])
    assert finest < 0.03, finest
    assert finest < coarse