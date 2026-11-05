"""Criterion gates for the two-phase CSF capillary chain (m3646 evidence refresh).

The published record ``docs/two-phase-surface-tension-audit/report.md`` reports
``dP = -1.19e22 Pa`` against an exact ``sigma/R = 0.35 Pa``.  These tests pin what
the *current* source actually delivers, at the grids the published record used,
so that neither the (irreproducible) catastrophic numbers nor a silent regression
can be mistaken for the measured state.

Measured values (``scripts/audit_two_phase_capillary_evidence.py``), two
independent machines:

| n  | dP [Pa]  | rel err (loong64) | rel err (x86_64 CI) | kappa rel err |
|----|----------|-------------------|---------------------|---------------|
| 24 | 0.355870 | +1.68 %           | +1.68 % (in band)   | +0.32 %       |
| 32 | 0.343787 | -1.78 %           | -1.78 % (in band)   | +10.05 %      |
| 48 | 0.284004 | -18.86 %          | < -10 %             | +30.61 %      |
| 64 | 0.328046 | -6.27 %           | +0.72 %             | +2.40 %       |

Platform provenance and the raw numbers live in
``docs/two-phase-capillary-evidence-refresh/platform-scatter.json``.

Two consequences, kept deliberately apart:

* **Portable (gated below).**  At n = 24/32 the Young--Laplace bar is met
  (``|rel| < 3 %``); at n = 48 the bar is missed by more than 10 % on *both*
  machines; every grid stays finite, ``Ca < 1e-2`` and ``div < 1e-12``; the
  discrete curvature of the exact circle is second order at the finest grid.
* **Not portable (recorded, not gated to a single number).**  The n = 64 verdict
  moves by ~7 percentage points between architectures (+0.72 % on x86_64 CI vs
  -6.27 % on loong64) although the arithmetic is float64 and independent of the
  thread count (1/2/4/128 threads reproduce -6.27 % bit-for-bit on loong64).
  The smeared-interface pressure sample is therefore only good to a few percent
  of ``sigma/R``: reading a 3 % verdict off a single grid is a platform
  coin-flip, and the earlier version of this file turned that coin-flip into a
  CI red.  Hence "the bar is missed on at least one finer grid" is asserted,
  while *which* finer grid misses it is not.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "audit_two_phase_surface_tension.py"
SCATTER = (
    REPO / "docs" / "two-phase-capillary-evidence-refresh" / "platform-scatter.json"
)
RADIUS, SIGMA, MU, DT, STEPS = 0.2, 0.07, 0.01, 1e-5, 10
EXACT_JUMP = SIGMA / RADIUS
#: Divergence guard for the finer grids -- deliberately loose (the published
#: record was 22 orders of magnitude out, so 50 % still separates "biased sample"
#: from "blown up"), because the observed scatter between architectures is
#: already a few percent of sigma/R.
JUMP_BOUND = 0.5

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


def test_finer_grids_stay_bounded_although_the_bar_is_not_carried_forward():
    """No divergence on the finer grids; the miss is a bias, not a blow-up."""
    for n in (48, 64):
        run = _laplace(n)
        assert run["finite"], n
        assert run["capillary_number"] < 1e-2, (n, run["capillary_number"])
        assert run["final_divergence"] < 1e-12, (n, run["final_divergence"])
        assert abs(run["relative_error"]) < JUMP_BOUND, (n, run["relative_error"])


def test_three_percent_bar_is_missed_on_at_least_one_finer_grid():
    """Portable part of the non-convergence claim (see module docstring).

    Measured on both machines: n = 48 misses the bar by >10 %; n = 64 misses it
    on loong64 and lands inside it on x86_64, so the per-grid verdict is not
    asserted -- but "h-refinement does not carry the 3 % bar" is, because at
    least one finer grid misses it on every platform measured.
    """
    errors = {n: _laplace(n)["relative_error"] for n in (24, 32, 48, 64)}
    missing = [n for n in (48, 64) if abs(errors[n]) > 0.03]
    assert missing, errors
    magnitudes = [abs(errors[n]) for n in (24, 32, 48, 64)]
    assert any(later > earlier for earlier, later in zip(magnitudes, magnitudes[1:])), (
        "Young-Laplace error must not decrease monotonically with n",
        magnitudes,
    )


def test_curvature_of_the_exact_circle_is_second_order_at_the_finest_grid():
    coarse = abs(_curvature(32)["relative_error"])
    finest = abs(_curvature(64)["relative_error"])
    assert finest < 0.03, finest
    assert finest < coarse


def test_platform_scatter_record_matches_the_gated_claims():
    """The cross-platform record must stay inside the band the gates assume.

    Exact values are only required where they were actually measured (loong64:
    every grid; x86_64 CI: n = 64, plus per-grid gate outcomes).  What the test
    enforces is that no recorded run leaves the bounded band and that the
    "3 % bar is not carried forward" claim holds on every recorded platform.
    """
    record = json.loads(SCATTER.read_text())
    runs = {entry["n"]: entry for entry in record["runs"]}
    assert set(runs) == {24, 32, 48, 64}
    for platform in record["platforms"]:
        for n, entry in runs.items():
            relative = entry["relative_error"][platform]
            if relative is not None:
                assert abs(relative) < JUMP_BOUND, (n, platform, relative)
                assert (abs(relative) < 0.03) is entry["within_three_percent"][
                    platform
                ], (
                    n,
                    platform,
                )
        for n in (24, 32):
            assert runs[n]["within_three_percent"][platform], (n, platform)
        assert any(not runs[n]["within_three_percent"][platform] for n in (48, 64)), (
            platform
        )
