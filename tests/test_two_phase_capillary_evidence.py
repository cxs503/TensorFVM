"""Criterion gates for the two-phase CSF capillary chain (m3646 evidence refresh).

The published record ``docs/two-phase-surface-tension-audit/report.md`` reports
``dP = -1.19e22 Pa`` against an exact ``sigma/R = 0.35 Pa`` -- 22 orders of
magnitude away, i.e. an unreadable record.  The first refresh of this evidence
(db97955-era source) measured ``+1.68 / -1.78 / -18.86 / -6.27 %`` at
``n = 24/32/48/64`` and concluded that the 3 % Young--Laplace bar was met only
where the smeared interface happened to line up with the grid.

Merging the current ``main`` brought the ``two_phase_solver`` projection/CSF
changes, and re-running the *same* study (``scripts/audit_two_phase_capillary_evidence.py``,
same parameters) at the *same* grids moves the record again:

| n  | dP [Pa]  | rel err | kappa rel err | Ca       | |u|max [m/s] |
|----|----------|---------|---------------|----------|--------------|
| 24 | 0.356521 | +1.86 % | +0.32 %       | 7.75e-09 | 5.43e-08     |
| 32 | 0.354851 | +1.39 % | +10.05 %      | 8.26e-09 | 5.78e-08     |
| 48 | 0.352524 | +0.72 % | +30.61 %      | 2.24e-08 | 1.57e-07     |
| 64 | 0.351070 | +0.31 % | +2.40 %       | 3.64e-08 | 2.54e-07     |

Two things changed relative to the first refresh: the pressure jump is now
inside 3 % on *every* grid and its error shrinks monotonically under refinement,
and the spurious-current level falls by ~2.5 orders of magnitude (Ca 1.9e-06 ->
7.8e-09 at n = 24).  The discrete-curvature column is bit-identical to the first
refresh (``+0.32/+10.05/+30.61/+2.40 %``), which localises the change: the
geometry/VOF operators did not move, the solver path did.

The residual limitation is therefore narrower than in the first refresh: the
band-mean curvature of the exact circle still misses 3 % at n = 32/48 (a
smeared-interface/grid alignment effect that the refinement does not remove),
even though the pressure jump the CSF chain actually delivers is inside 3 %
everywhere.

Numbers are pinned as *bands* on purpose: a reordered floating-point reduction
must not turn this file red, a real regression must.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "audit_two_phase_surface_tension.py"
CAPILLARY_AUDIT = REPO / "scripts" / "audit_two_phase_capillary_evidence.py"
RADIUS, SIGMA, MU, DT, STEPS = 0.2, 0.07, 0.01, 1e-5, 10
GRIDS = (24, 32, 48, 64)
EXACT_JUMP = SIGMA / RADIUS

#: `dP` recorded by the stale published artifact at the same grids.
PUBLISHED_JUMP = {24: -1.18714e22, 32: 8.49545e23}

#: `dP` relative error measured on the merged source (see module docstring).
MEASURED_RELATIVE_ERROR = {24: 0.01863, 32: 0.01386, 48: 0.00721, 64: 0.00306}


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
    for n in GRIDS:
        run = _laplace(n)
        assert run["finite"], n
        assert run["capillary_number"] < 1e-6, (n, run["capillary_number"])
        assert run["final_divergence"] < 1e-12, (n, run["final_divergence"])


def test_published_catastrophic_jump_is_not_reproducible_from_current_source():
    """Guard against re-citing the stale record as if it were current evidence."""
    for n, published in PUBLISHED_JUMP.items():
        run = _laplace(n)
        ratio = abs(published) / abs(run["pressure_jump_pa"])
        assert ratio > 1e10, (n, ratio)
        assert abs(run["pressure_jump_pa"] - EXACT_JUMP) < 0.25 * EXACT_JUMP


def test_young_laplace_is_within_three_percent_on_every_grid():
    """The 3 % bar is met on all four grids after the solver-path merge."""
    for n in GRIDS:
        relative = _laplace(n)["relative_error"]
        assert abs(relative) < 0.03, (n, relative)
        assert abs(relative - MEASURED_RELATIVE_ERROR[n]) < 0.01, (n, relative)


def test_pressure_jump_error_shrinks_monotonically_under_refinement():
    """+1.86 -> +1.39 -> +0.72 -> +0.31 %: the first refresh's non-monotone
    verdict no longer describes the current source."""
    errors = [abs(_laplace(n)["relative_error"]) for n in GRIDS]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse - 0.002, (GRIDS, errors)


def test_curvature_of_the_exact_circle_keeps_its_n48_alignment_dip():
    """Recorded limitation: the band-mean curvature is *not* refined away at
    n = 48 (+30.6 %) even though the pressure jump there is inside 1 %."""
    coarse = abs(_curvature(32)["relative_error"])
    rough = abs(_curvature(48)["relative_error"])
    finest = abs(_curvature(64)["relative_error"])
    assert rough > 0.10, rough
    assert finest < 0.03, finest
    assert finest < coarse, (coarse, finest)


def test_the_audit_script_pins_this_checkout_over_a_foreign_tensorfvm(tmp_path):
    """The first refresh of this record was produced against another work tree.

    ``python scripts/...`` resolves ``tensorfvm`` from whatever the environment
    offers; an editable install of a *different* TensorFVM checkout on
    ``PYTHONPATH`` then silently wins and the evidence describes a source state
    that is not the one under audit.  The audit script pins ``REPO/src`` first,
    so a decoy package on ``PYTHONPATH`` must never be imported.
    """
    decoy = tmp_path / "decoy"
    (decoy / "tensorfvm").mkdir(parents=True)
    (decoy / "tensorfvm" / "__init__.py").write_text(
        "raise ImportError('evidence measured against a foreign checkout')\n"
    )
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('cap', r'{CAPILLARY_AUDIT}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.load_audit()\n"
        "print(sys.modules['tensorfvm'].__file__)\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(decoy))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert str((REPO / "src").resolve()) in completed.stdout, completed.stdout
