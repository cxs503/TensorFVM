"""Round 16 of ``CASE-FVM-PERIODIC-HILLS``: the sink A/B must be runnable and recordable.

Two defects can silently destroy the evidence of this case, and both are pinned
here:

1. **The non-stationary record could not be written.**  ``averaging_window``
   attached the attempt list to ``attempts[0]``, which *is* the object it
   returns, so ``worst["attempts"][0]["attempts"] -> worst`` was a reference
   cycle.  ``json.dumps`` and ``write_json``'s ``_clean`` both raised
   ``RecursionError`` -- i.e. precisely the honest-FAIL case (no admissible
   window) lost its artefact, while the PASS case wrote fine.  A gate whose
   failure mode is "no file" cannot be distinguished from "no run".
2. **The A/B variant must be measured, not declared.**  Rounds 10--14 recorded
   a ``convection_blend`` knob that was never wired into the closure, so an
   artefact could disagree with itself.  ``scripts/round16_acceptance.py``
   therefore *measures* the in-force ``omega`` sink coefficient from
   ``SSTKOmega.source_terms()``; the tests check that the measurement returns
   1.0 for HEAD (Patankar) and 2.0 for the legacy override, so a run cannot be
   labelled with a variant it did not execute.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_sst import SSTKOmega, SSTKOmegaConfig
from tensorfvm.verification.periodic_hill_round9 import averaging_window

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
          / "round16_acceptance.py")


def _driver():
    """Load ``scripts/round16_acceptance.py`` as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("round16_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drifting_record(n=400, dt=0.02):
    """A record that no contiguous window can call stationary."""
    t = np.arange(n) * dt
    return t, 3.0 + 0.5 * (t - t[0]) / (t[-1] - t[0])


# --------------------------------------------------------------------------- #
# 1. the honest-FAIL record must be serialisable
# --------------------------------------------------------------------------- #
def test_failed_window_record_is_json_serialisable():
    t, v = _drifting_record()
    out = averaging_window(t, v, n_blocks=4)
    assert out["ok"] is False, "the drifting record must not pass"
    assert out["blocks_used"] == 4
    # the previous defect: RecursionError here, so the artefact was never written
    text = json.dumps(out, allow_nan=False)
    assert len(text) > 100
    assert len(out["attempts"]) == 3, "one attempt per admissible block drop"


def test_attempt_diagnostics_are_not_a_reference_cycle():
    t, v = _drifting_record()
    out = averaging_window(t, v, n_blocks=4)
    assert out["attempts"][0] is not out, "returned record must not alias attempt[0]"
    assert all("attempts" not in a for a in out["attempts"]), (
        "nested attempt lists are how the cycle was built")
    # the diagnostic content itself must survive: one row per dropped block
    assert [a["blocks_dropped"] for a in out["attempts"]] == [0, 1, 2]
    assert all(np.isfinite(a["spread_rel"]) for a in out["attempts"])


def test_stationary_window_does_not_attach_attempts():
    t = np.arange(400) * 0.02
    out = averaging_window(t, np.full(400, 3.0), n_blocks=4)
    assert out["ok"] is True
    assert "attempts" not in out


# --------------------------------------------------------------------------- #
# 2. the A/B variant is measured from the model, not declared
# --------------------------------------------------------------------------- #
def _solver():
    solver = PeriodicHillSolver(
        PeriodicHillConfig(nx=8, ny=16, reynolds=2800.0, time_step_s=0.02))
    solver.turbulence = SSTKOmega(solver, SSTKOmegaConfig(max_viscosity_ratio=1.0e4))
    turb = solver.turbulence
    turb.k = np.full((16, 8), 4.0e-3)
    turb.omega = np.full((16, 8), 20.0)
    return solver


def test_head_sink_is_the_patankar_coefficient():
    mod = _driver()
    measured = mod.measure_sink_ratio(_solver())
    assert measured["samples"] > 0
    assert measured["ratio"] == pytest.approx(1.0, rel=1e-12)


def test_legacy_override_is_detected_as_the_newton_coefficient():
    mod = _driver()
    solver = _solver()
    mod.apply_legacy_newton_sink(solver)
    measured = mod.measure_sink_ratio(solver)
    assert measured["ratio"] == pytest.approx(2.0, rel=1e-12), (
        "the rounds-10--15 defect must show up as a factor 2 in the measurement")


def test_override_does_not_change_anything_else():
    """The A/B must differ *only* in the ``omega`` sink coefficient."""
    mod = _driver()
    solver = _solver()
    before = {k: np.array(v, dtype=float) for k, v in solver.turbulence.source_terms().items()}
    mod.apply_legacy_newton_sink(solver)
    after = solver.turbulence.source_terms()
    changed = {k for k in before if not np.allclose(before[k], np.asarray(after[k], dtype=float))}
    assert changed == {"omega_destruction_linear"}, changed