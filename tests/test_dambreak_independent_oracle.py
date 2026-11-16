"""Regression tests for the independent dam-break oracle and instruments.

These tests protect the *verifier* side of ``CASE-FVM-DAMBREAK-STOKER``:
the oracle must stay independent of ``shallow_water.exact`` yet agree with it
to round-off, the instruments must be able to see the errors they claim to
measure (injected shift / plateau scaling), the negative control must be
rejected, and the grading of the production solver must reproduce the case
pool's committed ``depth_l1``.

Runtime is kept low by grading at ``n = 800`` except for the single pool-grid
binding test (``n = 1600``, a few seconds).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from tensorfvm.verification import dambreak_independent_oracle as oracle

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOL_ARTIFACT = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                             "artifacts", "shallow-water-dam-break",
                             "metrics.json")


def test_module_does_not_import_the_shipped_exact_module_at_load():
    """Independence is mechanical: importing this module must not load the oracle.

    ``oracle_crosscheck`` imports ``shallow_water.exact`` lazily on purpose; if
    a future edit hoists that import, the "independent" oracle silently becomes
    the same code and the whole exercise loses its value.  A subprocess import
    is the only way to observe ``sys.modules`` before pytest has loaded the rest
    of the suite.
    """
    code = (
        "import sys, tensorfvm.verification.dambreak_independent_oracle as o;"
        "print('tensorfvm.shallow_water.exact' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=REPO_ROOT,
                         env={**os.environ, "PYTHONPATH": os.path.join(REPO_ROOT, "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_oracle_agrees_with_the_shipped_reference():
    """Two independently written closed forms must agree to round-off."""
    check = oracle.oracle_crosscheck(t_star=2.0, points=2001)
    assert check["oracle_agrees"] is True
    for key, entry in check["cases"].items():
        assert entry["max_depth_diff_scaled"] <= 1e-12, key
        assert entry["max_velocity_diff_scaled"] <= 1e-12, key
        for name, const in entry["constants"].items():
            assert const["relative"] <= 1e-12, (key, name)


def test_jump_conditions_and_rarefaction_invariant_close():
    """The independent middle state really solves the Riemann problem."""
    residuals = oracle.rankine_hugoniot_residuals(1.0, 0.5)
    assert residuals["mass_relative"] <= 1e-12
    assert residuals["momentum_relative"] <= 1e-12
    assert residuals["rarefaction_invariant_defect"] <= 1e-12
    assert residuals["middle_state_scalar_residual"] <= 1e-12


def test_ritter_known_values_at_the_dam():
    """Ritter's classic values: ``h = 4/9 h0`` and ``u = 2/3 c0`` at the dam."""
    h0, g = 1.0, 9.81
    t = 0.5
    h, u = oracle.ritter_reference(np.array([0.0]), t, h0, g)
    assert h[0] == pytest.approx(4.0 / 9.0 * h0, rel=1e-12)
    assert u[0] == pytest.approx(2.0 / 3.0 * math.sqrt(g * h0), rel=1e-12)


def test_ritter_fan_has_constant_riemann_invariant():
    """``u + 2 sqrt(g h)`` is constant across the fan (self-similar identity)."""
    h0, g, t = 1.0, 9.81, 0.4
    c0 = math.sqrt(g * h0)
    xi = np.linspace(-0.9 * c0, 1.9 * c0, 200)
    h, u = oracle.ritter_reference(xi * t, t, h0, g)
    invariant = u + 2.0 * np.sqrt(g * h)
    assert np.max(np.abs(invariant - 2.0 * c0)) <= 1e-12


def test_stoker_regions_and_continuity():
    """The bore sits exactly at ``s t`` and the states either side are ``h2``/``h1``."""
    h0, h1, g, t = 1.0, 0.5, 9.81, 0.6
    sol = oracle.stoker_constants_independent(h0, h1, g)
    x_shock = sol["shock_speed"] * t
    eps = 1e-6
    h, u = oracle.stoker_reference(np.array([x_shock - eps, x_shock + eps]),
                                   t, h0, h1, g)
    assert h[0] == pytest.approx(sol["h2"], rel=1e-6)
    assert u[0] == pytest.approx(sol["u2"], rel=1e-6)
    assert h[1] == pytest.approx(h1, rel=1e-12)
    assert u[1] == 0.0


@pytest.mark.parametrize("key", ["ritter-dry", "stoker-wet"])
def test_negative_control_is_rejected(key):
    """A wrong reference (the other case's solution) must fail the 5 % gate."""
    control = oracle.negative_control(key, n=400)
    assert control["rejected"] is True
    assert control["depth_l1_vs_wrong_reference"] > 0.05


def test_instruments_recover_injected_perturbations():
    """Calibration: a 5 % bore shift and a 1 % plateau scaling must be recovered."""
    profile = oracle.run_profile("stoker-wet", n=800)
    shifts = oracle.detection_power_shift(profile, (-0.1, 0.1))
    for row in shifts:
        assert row["recovery_relative_error"] <= 0.05, row
    scales = oracle.detection_power_step(profile, (0.99, 1.01))
    for row in scales:
        assert row["recovery_relative_error"] <= 0.05, row


def test_wiggle_detection_threshold_is_quantified():
    """The monotonicity gate's blind spot must be finite and reported."""
    profile = oracle.run_profile("stoker-wet", n=400)
    rows = oracle.wiggle_detection_threshold(profile, cells=(4, 8),
                                             prominences=(1e-4,))
    assert len(rows) == 2
    for row in rows:
        assert row["amplitude_over_h0"] >= 1e-5
        # a longer wiggle is harder to see at a fixed per-cell prominence
    assert rows[1]["amplitude_over_h0"] > rows[0]["amplitude_over_h0"]


@pytest.mark.parametrize("key", ["ritter-dry", "stoker-wet"])
def test_grading_passes_at_common_grid(key):
    """The anchor's acceptance (5 %) holds under the independent instruments."""
    graded = oracle.grade_case(key, n=800)
    assert graded["pass"] is True
    assert graded["worst_graded"] <= 0.05
    assert graded["monotone"] is True
    assert graded["volume_drift"] <= 1e-9


def test_pool_grid_reproduces_the_committed_depth_l1():
    """At the pool grid the independent L1 must match the shipped artifact.

    The instrument is independent, the *definition* is the same, so the numbers
    must coincide: this is the binding between this verifier and the published
    case pool, and it fails loudly if either side drifts.
    """
    if not os.path.exists(POOL_ARTIFACT):        # pragma: no cover
        pytest.skip("pool artifact not present in this checkout")
    with open(POOL_ARTIFACT, encoding="utf-8") as handle:
        artifact = json.load(handle)
    graded = oracle.grade_case("stoker-wet", n=1600)
    committed = artifact["cases"]["stoker-wet"]["metrics"]
    assert graded["depth_l1"] == pytest.approx(committed["depth_l1"], rel=1e-6)
    assert graded["shock_position"]["relative_error"] == pytest.approx(
        committed["shock_error"], abs=2e-6)
    assert graded["step_ratio_error"] <= 5.0 * committed["step_ratio_error"]