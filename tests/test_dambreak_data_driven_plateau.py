"""Data-driven (F4) wet-bed plateau/bore cross-check for the dam-break case.

Why this file exists
--------------------
The audit (``docs/case-fvm-dambreak-stoker/AUDIT.md``, finding F4) recorded that
the wet-bed ``h2``/``step_ratio``/``shock_position`` metrics in
:func:`tensorfvm.shallow_water.dam_break._wet_metrics` are **oracle-guided**:
they place the measurement window at the *analytic* ``x_fan_end``/``x_shock``
and set the crossing level from the *analytic* ``h2``.  Consequence: they only
measure the plateau *amplitude* and are insensitive to a misplaced bore -- a
solver that puts the plateau in the wrong place but at the right amplitude
still reads a machine-precision ``h2`` (``step_ratio_error = 5.76e-7`` at
n=1600 is itself the "near-tautology" signature).

This file locks the fix: :func:`data_driven_plateau` locates the bore and the
plateau from the **numerical profile alone** (no analytic position or level),
so the ``h2``/``step_ratio``/``bore_x`` reading is a genuine
position-and-amplitude check.  The cross-check must agree with the closed form
within the case's own 5 % tolerance on the accepted grid, and the ``bore_x``
must be a *position* measurement (it tracks a displaced bore -- the property
the oracle-guided ``h2`` lacks).

Thresholds are the case's own (``TOLERANCE = 5 %``); ``thresh_frac = 1e-2`` is
declared in the module (a fraction of the max gradient -- the plateau is ~1e-4
of the bore gradient while the fan is ~5e-2, so ``1e-2`` separates them and is
grid-independent because both scale with ``1/dx``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    STOKER_WET,
    TOLERANCE,
    data_driven_plateau,
    run_case,
)
from tensorfvm.shallow_water.exact import stoker_constant_state


@pytest.fixture(scope="module")
def runs():
    return {
        ("wet", 400): run_case(STOKER_WET, n=400),
        ("wet", 800): run_case(STOKER_WET, n=800),
    }


def test_detector_takes_only_the_numerical_profile(runs):
    """The detector must not consume any analytic position or level.

    Its only inputs are the cell centres and depths; if it ever needs the
    closed form, the signature (and this test) must change.
    """
    r = runs[("wet", 800)]
    x = np.asarray(r["profile"]["x"])
    h = np.asarray(r["profile"]["h_numeric"])
    dd = data_driven_plateau(x, h)
    assert dd["n_plateau"] > 100
    assert dd["bore_x"] > 0.0 and dd["h2"] > 0.0 and dd["h1"] > 0.0


def test_data_driven_crosscheck_agrees_with_closed_form(runs):
    """The independent reading sits inside the case's 5 % tolerance."""
    for n in (400, 800):
        dd = runs[("wet", n)]["metrics"]["dd"]
        for key in ("bore_x_error", "h2_error", "h1_error", "step_ratio_error"):
            assert dd[key] < TOLERANCE, (n, key, dd[key])
        # the plateau is genuinely flat and wide (a real constant state)
        assert dd["n_plateau"] > 50


def test_data_driven_bore_x_is_a_position_measurement(runs):
    """Negative control: ``bore_x`` tracks a displaced bore.

    The oracle-guided ``h2`` measures the plateau *amplitude* at the analytic
    window, so it is blind to where the bore actually sits.  The data-driven
    ``bore_x`` is the cell of maximal ``|dh/dx|`` -- a *position* reading.  If
    the whole profile is translated by ``k`` cells, ``bore_x`` must move by
    ``k * dx`` (it is not pinned to an analytic location), while ``h2`` (the
    plateau amplitude) stays put because the profile is merely translated.
    """
    r = runs[("wet", 800)]
    x = np.asarray(r["profile"]["x"])
    h = np.asarray(r["profile"]["h_numeric"])
    dx = x[1] - x[0]
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    dd0 = data_driven_plateau(x, h)
    # translate the profile by k cells (same amplitude, bore in a new place)
    k = 40
    h_shifted = np.concatenate([h[k:], h[-k:]])
    dd = data_driven_plateau(x, h_shifted)
    # the bore must move by ~k*dx (a position reading), not stay pinned
    assert abs(abs(dd["bore_x"] - dd0["bore_x"]) - k * dx) < 2.0 * dx
    # while the plateau amplitude is unchanged by a pure translation
    assert abs(dd["h2"] - sol["h2"]) / sol["h2"] < 1e-5


def test_detector_guards_degenerate_input():
    x = np.linspace(0.0, 1.0, 11)
    h = np.zeros_like(x)
    dd = data_driven_plateau(x, h)
    assert np.isnan(dd["bore_x"]) and np.isnan(dd["h2"]) and dd["n_plateau"] == 0
