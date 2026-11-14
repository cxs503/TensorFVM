"""Round 11: closure-parametric time-averaged periodic-hill gate (unit tests).

These tests do NOT assert the physics verdict (that is a cluster run and lands in
``docs/case-periodic-hills/ROUND11.md``).  They assert the things that would
make such a verdict untrustworthy if they were wrong:

* the closure selector really selects a closure and rejects typos;
* the round-11 loop is *the same loop* as the round-9 driver (same numbers on the
  same input), so an SST-vs-SA comparison is not confounded by the driver;
* the gate can only pass when all three conditions hold, and an *undefined*
  ``x_r/H`` fails rather than being skipped;
* the round-11 gate re-derives the verdict that round 9 *recorded* for SA at
  96x48 from round 9's own artefact (no retro-fitting of the SA result);
* a diverged run still writes an analysable artefact (honest FAIL, not a gap).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.periodic_hill_sst import SSTKOmega
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras
from tensorfvm.verification.periodic_hill_round11 import (
    CLOSURES,
    build_solver,
    closure_comparison,
    gate_summary,
    run_time_averaged_case,
    strict_finite,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
ROUND9_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "round9_long_time_average.py"


# --------------------------------------------------------------------------- #
# closure selection
# --------------------------------------------------------------------------- #
def test_closures_tuple_and_selection():
    assert CLOSURES == ("sa", "sst")
    assert isinstance(build_solver("sst", nx=16, ny=8).turbulence, SSTKOmega)
    assert isinstance(build_solver("sa", nx=16, ny=8).turbulence, SpalartAllmaras)
    # case/whitespace tolerant, but a typo must never fall back to a default
    assert isinstance(build_solver(" SST ", nx=16, ny=8).turbulence, SSTKOmega)
    with pytest.raises(ValueError, match="unknown closure"):
        build_solver("k-epsilon", nx=16, ny=8)


def test_build_solver_records_the_requested_time_step():
    solver = build_solver("sst", nx=16, ny=8, dt=0.01)
    assert solver.config.time_step_s == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# the loop is round 9's loop
# --------------------------------------------------------------------------- #
def _round9_module():
    spec = importlib.util.spec_from_file_location("round9_driver", ROUND9_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round11_loop_reproduces_round9_driver_numbers(tmp_path, monkeypatch):
    """Same closure, same mesh, same steps -> identical metrics.

    ``round9_96x48.json`` is the SA baseline the SST result is compared against;
    if the round-11 loop disagreed with the round-9 loop on identical input, the
    comparison would be measuring the driver, not the closure.
    """
    mod = _round9_module()
    monkeypatch.setattr(mod, "OUTDIR", tmp_path)

    common = dict(nx=16, ny=8, steps=60, sample_every=10, dt=0.02)
    r9 = mod.run_case("tiny", burn_in=20, **common, verbose=False)
    r11 = run_time_averaged_case(
        "tiny", "sa", solver=build_solver("sa", nx=16, ny=8, dt=0.02),
        burn_in_steps=20, **common)

    assert r11["samples"] == r9["samples"]
    for key in ("x_s_over_H", "x_r_over_H"):
        a, b = r11["time_averaged"][key], r9["time_averaged"][key]
        # both "undefined" is agreement (an instantaneous crossing can be absent
        # on a 16x8 mesh); otherwise the numbers must match to round-off
        assert (a is None) == (b is None), key
        if a is None or b is None:
            continue
        assert math.isnan(a) == math.isnan(b), key
        if not math.isnan(b):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-12), key
    np.testing.assert_allclose(r11["horizontal_profile_y_H_0.5"]["u_over_Ub"],
                               r9["horizontal_profile_y_H_0.5"]["u_over_Ub"],
                               rtol=1e-12, atol=1e-12)
    assert r11["profile_comparison"]["n_stations"] == r9["profile_comparison"]["n_stations"]
    assert r11["profile_comparison"]["n_pass"] == r9["profile_comparison"]["n_pass"]
    assert r11["stationarity"]["reattachment_coverage"] == pytest.approx(
        r9["stationarity"]["reattachment_coverage"])


# --------------------------------------------------------------------------- #
# gate semantics
# --------------------------------------------------------------------------- #
def _payload(x_r, station_pass=7, station_total=7, spread=0.01, cover=1.0,
             nu_t=10.0, closure="sst"):
    return {
        "closure": {"name": closure},
        "config": {"nx": 96, "ny": 48, "end_time": 400.0},
        "samples": 601,
        "time_averaged": {"x_r_over_H": x_r, "max_nu_t_ratio": nu_t},
        "profile_comparison": {"n_stations": station_total, "n_pass": station_pass},
        "stationarity": {"x_r_over_H_block_spread_rel": spread,
                         "reattachment_coverage": cover},
    }


def test_gate_passes_only_on_the_reference_value():
    summary = gate_summary(_payload(5.34))
    assert summary["pass"] is True
    assert summary["x_r_over_H_error"] == pytest.approx(0.0)
    assert summary["reasons"] == []


def test_gate_rejects_undefined_reattachment_with_a_reason():
    summary = gate_summary(_payload(None, cover=0.0))
    assert summary["pass"] is False
    assert summary["x_r_over_H"] is None
    assert any("undefined" in r for r in summary["reasons"])
    assert any("0.0%" in r for r in summary["reasons"])


def test_gate_rejects_each_condition_independently():
    # x_r/H outside the 10 % band (5.34 * 1.10 = 5.874)
    assert gate_summary(_payload(6.5))["pass"] is False
    # one profile station out
    assert gate_summary(_payload(5.34, station_pass=6))["pass"] is False
    # non-stationary window
    assert gate_summary(_payload(5.34, spread=0.5))["pass"] is False
    # boundary: the band is inclusive at the 10 % edge (5.874), exclusive beyond
    assert gate_summary(_payload(5.874))["pass"] is True
    assert gate_summary(_payload(5.874 + 1e-6))["pass"] is False
    assert gate_summary(_payload(5.34 * 0.90))["pass"] is True


def test_gate_reasons_name_the_failing_condition():
    summary = gate_summary(_payload(8.0, station_pass=2, spread=None))
    text = " | ".join(summary["reasons"])
    assert "x_r/H = 8.000" in text
    assert "2/7 stations" in text
    assert "not stationary" in text


# --------------------------------------------------------------------------- #
# cross-closure table
# --------------------------------------------------------------------------- #
def test_closure_comparison_is_schema_compatible_with_round9_artefacts():
    table = closure_comparison({"sa": _payload(None, 0, 7, 0.0, 0.0),
                                "sst": _payload(5.4, 7, 7, 0.02, 1.0, 8.0)})
    rows = {r["closure"]: r for r in table["rows"]}
    assert table["n_pass"] == 1
    assert table["reference"]["x_r_over_H"] == pytest.approx(5.34)
    assert rows["sa"]["gate_pass"] is False
    assert rows["sst"]["gate_pass"] is True
    assert rows["sa"]["profile_stations_pass"] == "0/7"
    assert rows["sst"]["mesh"] == [96, 48]
    json.dumps(table, allow_nan=False)


def test_round11_gate_reproduces_the_recorded_round9_verdict():
    """The round-9 SA artefact must be judged the same way it was recorded.

    This pins that round 11 did not relax (or tighten) the acceptance
    convention while adding a closure: it re-derives round 9's own recorded gate.
    """
    path = DOCS / "round9_96x48.json"
    if not path.exists():
        pytest.skip("round-9 SA artefact not in the tree")
    recorded = json.loads(path.read_text())
    rederived = gate_summary(recorded)
    assert rederived["pass"] == bool(recorded["gate"]["pass"]) is False
    assert rederived["x_r_over_H"] is None
    assert rederived["profile_stations_pass"] == recorded["gate"]["y_H_0.5_profile"]["stations_pass"]
    assert rederived["stationary"] == bool(recorded["gate"]["stationary_window"]) is False
    assert rederived["reattachment_coverage"] == pytest.approx(
        recorded["gate"]["reattachment_coverage"])


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def test_strict_finite_removes_nan_and_inf():
    payload = {"a": float("nan"), "b": [float("inf"), -float("inf"), 1.0],
               "c": np.float64("nan"), "d": np.int64(3), "e": {"f": float("nan")}}
    cleaned = strict_finite(payload)
    assert cleaned["a"] is None and cleaned["e"]["f"] is None
    assert cleaned["b"] == [None, None, 1.0]
    assert cleaned["d"] == 3
    json.dumps(cleaned, allow_nan=False)


def test_artefact_is_written_with_strict_json(tmp_path):
    payload = run_time_averaged_case(
        "tiny", "sst", nx=16, ny=8, steps=40, sample_every=10, burn_in_steps=10,
        out_dir=tmp_path, save_fields=True)
    path = tmp_path / "round11_sst_tiny.json"
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    reloaded = json.loads(text)
    assert reloaded["closure"]["name"] == "sst"
    assert reloaded["closure"]["class"] == "SSTKOmega"
    assert reloaded["samples"] == payload["samples"] > 0
    assert reloaded["gate"]["pass"] is False  # 40 steps is not a converged flow
    assert (tmp_path / "round11_sst_tiny.npz").exists()


def test_diverged_run_still_yields_an_analysable_artefact(tmp_path):
    """A blow-up is an honest FAIL with numbers, not a missing shard."""

    class Exploding:
        def __init__(self, solver, fail_at=25):
            self.solver = solver
            self.fail_at = fail_at
            self.steps = 0

        def update(self, dt):
            self.steps += 1
            if self.steps >= self.fail_at:
                raise FloatingPointError("boom")
            return np.zeros_like(self.solver.mesh.xc)

        def viscosity(self):
            return np.zeros_like(self.solver.mesh.xc)

    solver = build_solver("sst", nx=16, ny=8)
    solver.turbulence = Exploding(solver)
    payload = run_time_averaged_case(
        "boom", "sst", solver=solver, nx=16, ny=8, steps=100, sample_every=10,
        burn_in_steps=10, out_dir=tmp_path)
    assert payload["config"]["diverged_at_step"] == 25
    assert payload["samples"] > 0
    assert math.isfinite(payload["time_averaged"]["body_force"])
    written = json.loads((tmp_path / "round11_sst_boom.json").read_text())
    assert written["config"]["diverged_at_step"] == 25
    assert written["gate"]["pass"] is False