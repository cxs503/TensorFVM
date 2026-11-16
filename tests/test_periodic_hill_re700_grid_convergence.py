"""Round 31: Re=700 (laminar, 2-D-admissible) reattachment for CASE-FVM-PERIODIC-HILLS.

The Re=700 sister case is the one regime where a two-dimensional solution is
physically admissible (the flow is laminar, so there is no spanwise instability
for a 2-D mean to miss).  The round-20 96x48 artefact therefore carries the
cleanest possible test of the *solver* (as opposed to the model-form error that
dominates Re=2800).  These tests pin what that artefact actually shows, so a
future change cannot silently move the number.

The headline, pinned here: the 2-D mean-flow reattachment is ``x_r/H = 8.08``,
**+54 %** against the DNS reference ``5.24``, and the certified operator flags
it ``out_of_window`` because the reattachment (8.078) lies *beyond the flat
bottom* (which ends at ``x/H = 7.071``) -- the 2-D bubble is so long it
reattaches in the next hill's approach region.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_lower_re import reattachment_reference
from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

OUTDIR = (Path(__file__).resolve().parents[1]
          / "docs" / "case-periodic-hills")
NPZ = OUTDIR / "round20_laminar_Re700_96x48.npz"
JSON = OUTDIR / "round20_laminar_Re700_96x48.json"
WINDOW = (1.9, 7.1)
FLAT_BOTTOM_END = 7.071  # x/H where the hill surface returns to y=0


def _certified_x_r() -> dict:
    with np.load(NPZ) as npz:
        tau = np.asarray(npz["mean_tau_b"], dtype=float)
        x = np.asarray(npz["x_over_h"], dtype=float)
    return separation_reattachment(tau, x, window=WINDOW)


def test_artefact_present():
    assert NPZ.is_file(), f"missing {NPZ}"
    assert JSON.is_file(), f"missing {JSON}"


def test_reattachment_is_long_and_out_of_window():
    scored = _certified_x_r()
    x_r = scored["x_r_over_H"]
    assert np.isfinite(x_r)
    # the 2-D bubble reattaches beyond the flat bottom -> out of the certified
    # window, i.e. the bubble is not a compact recirculation on the bed
    assert scored["status"] == "out_of_window"
    assert x_r > FLAT_BOTTOM_END, (
        f"reattachment {x_r:.3f} is not beyond the flat bottom "
        f"({FLAT_BOTTOM_END})")


def test_error_vs_dns_reference_is_honest_fail():
    scored = _certified_x_r()
    x_r = scored["x_r_over_H"]
    ref = reattachment_reference(700.0)
    assert ref is not None and ref["x_r_over_H"] == 5.24
    err = abs(x_r - ref["x_r_over_H"]) / ref["x_r_over_H"]
    # honest: the 2-D laminar solution is far outside the 10 % band
    assert err > 0.10
    # and the number is stable to the stored artefact (not a fluke of the run)
    payload = json.loads(JSON.read_text())
    payload_x_r = payload["time_averaged"]["x_r_over_H"]
    assert abs(x_r - payload_x_r) < 1.0e-6


def test_stationarity_of_the_96x48_run():
    payload = json.loads(JSON.read_text())
    stat = payload["stationarity"]["x_r_over_H"]
    # the instantaneous reattachment is stationary (the mean is trustworthy)
    assert stat["ok"] is True
    assert stat["spread_rel"] < 0.05


def test_lower_re_verdict_recorded():
    payload = json.loads(JSON.read_text())
    lower = payload["lower_re"]
    assert lower["reynolds"] == 700.0
    assert lower["reference_x_r_over_H"] == 5.24
    assert lower["verdict"] in ("FAIL", "UNDEFINED")
    # the gate is hard-wired to Re=2800 and must be flagged as such
    assert payload["gate_reference_reynolds"] == 2800.0
