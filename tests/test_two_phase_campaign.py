"""Tests for the IN99-N01 three-benchmark two-phase campaign.

The campaign itself (three benchmark legs) takes tens of minutes, so it is run
by ``python -m tensorfvm.verification.two_phase_campaign``; these tests pin the
pieces that decide the verdict:

* the C2 metric -- including the cases that *must* fail it (a filled notch, a
  uniform ``alpha = 0.5`` field that conserves mass exactly and still has no
  opening at all),
* the donor-cell baseline (conservative but smeared),
* the Grace anchor algebra,
* the C4 gate (including a synthetic 20 % shift that must be rejected),
* a real but very short dam-break integration (volume ledger + front origin),
* the leg -> summary -> report chain.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tensorfvm.verification import two_phase_campaign as tpc
from tensorfvm.verification.dam_break import mm_surge_reference
from tensorfvm.verification.vof_rotation import rotation_velocity, zalesak_disk


# --------------------------------------------------------------------------- #
# C2 metric: slot opening
# --------------------------------------------------------------------------- #
def test_slot_opening_exact_field_is_recovered_exactly():
    n = 100
    exact = zalesak_disk(n)
    rec = tpc.slot_opening_width(exact, n)
    # The exact field has alpha = 0 for |x-0.5| <= 0.05: the alpha = 0.5 crossings
    # sit exactly on the slot walls, 0.10 apart.
    assert rec["exact_width"] == pytest.approx(0.10)
    assert rec["measured_width"] == pytest.approx(0.10, abs=1e-12)
    assert rec["recovery"] == pytest.approx(1.0, abs=1e-12)
    assert rec["cells_across_opening"] == pytest.approx(10.0)


def test_slot_metrics_exact_field_area_and_chord_agree():
    n = 100
    m = tpc.slot_metrics(zalesak_disk(n), n)
    assert m["area_recovery"] == pytest.approx(1.0, abs=1e-12)
    assert m["chord_recovery_at_2_3"] == pytest.approx(1.0, abs=1e-12)
    assert m["chord_recovery_depth_mean"] == pytest.approx(1.0, abs=1e-12)


def test_filled_notch_fails_the_opening_metric():
    """A notch filled with liquid has no opening: recovery must collapse."""
    n = 100
    filled = zalesak_disk(n).clone()
    dx = 1.0 / n
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    x, y = torch.meshgrid(centres, centres, indexing="xy")
    slot = ((x - 0.5).abs() <= 0.05) & (y >= 0.75)
    filled[slot] = 1.0
    m = tpc.slot_metrics(filled, n)
    assert m["area_recovery"] < 1e-12
    assert m["chord_recovery_at_2_3"] == 0.0
    assert m["area_recovery"] < tpc.C2_OPENING_TOL


def test_uniform_half_field_conserves_mass_but_has_no_opening():
    """The C2/C1 separation: alpha = 0.5 everywhere is exactly conservative.

    Such a field passes any mass criterion and represents a completely smeared
    interface -- which is precisely why the campaign does not let C1 stand alone.
    """
    n = 100
    exact = zalesak_disk(n)
    dx = 1.0 / n
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    x, y = torch.meshgrid(centres, centres, indexing="xy")
    slot = ((x - 0.5).abs() <= 0.05) & (y >= 0.75)
    disk = exact > 0.5
    # Half-fill the notch, then take exactly that volume back out of the disk
    # interior: the field is mass-matched with the exact disk to 1e-12 while its
    # notch is half-full (i.e. the interface is smeared, not sharp).
    haze = exact.clone()
    haze[slot] = 0.5
    delta = 0.5 * float(slot.sum()) / float(disk.sum())
    haze[disk] = (exact[disk] - delta).clamp_min(0.0)
    assert float((haze.sum() - exact.sum()).abs()) / float(exact.sum()) < 1e-6
    m = tpc.slot_metrics(haze, n)
    assert m["chord_recovery_at_2_3"] == 0.0
    assert m["area_recovery"] == pytest.approx(0.5, abs=1e-12)
    assert m["area_recovery"] < tpc.C2_OPENING_TOL


# --------------------------------------------------------------------------- #
# Donor-cell baseline
# --------------------------------------------------------------------------- #
def test_donor_cell_step_conserves_volume_but_smears():
    n = 32
    alpha0 = zalesak_disk(n)
    u, v, dx = rotation_velocity(n)
    dt = 0.25 * dx / float(u.abs().max())
    a = alpha0.clone()
    for k in range(40):
        a = tpc.donor_cell_step(a, u, v, dt, dx, dx)
    assert float(a.min()) >= -1e-12 and float(a.max()) <= 1.0 + 1e-12
    drift = abs(float(a.sum()) - float(alpha0.sum())) / float(alpha0.sum())
    assert drift < 1e-12                     # flux form: conservative
    assert float(a.max()) < 0.95             # ... and still smeared


def test_run_zalesak_plic_beats_donor_on_shape():
    """Small real run: both schemes conserve, only PLIC keeps the slot."""
    z = tpc.run_zalesak(n=32, cfl=0.4)
    plic, donor = z["schemes"]["plic"], z["schemes"]["donor_cell"]
    assert abs(plic["mass_relative_drift"]) <= tpc.C1_MASS_TOL
    assert abs(donor["mass_relative_drift"]) <= tpc.C1_MASS_TOL
    assert plic["peak_retention"] > donor["peak_retention"]
    assert plic["slot"]["area_recovery"] > donor["slot"]["area_recovery"]
    assert plic["l1"] < donor["l1"]
    # The geometry leg and the rotation leg must share one dx.
    assert z["geometry"]["domain"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Bubble: anchor algebra and the C3/C5 gates
# --------------------------------------------------------------------------- #
def test_grace_anchor_is_the_declared_formula():
    for eo in (1.23, 10.0, 40.0):
        assert tpc.grace_anchor(eo) == pytest.approx(math.sqrt(2.14 / eo + 0.505),
                                                     rel=1e-15)


def test_grace_anchor_decreases_with_eotvos():
    assert tpc.grace_anchor(1.0) > tpc.grace_anchor(10.0) > tpc.grace_anchor(40.0)


def _fake_bubble_run(eotvos: float, u_t: float, cells: int = 16,
                     width: float = 4.0) -> dict:
    anchor = tpc.grace_anchor(eotvos)
    return {
        "case": {"name": f"fake-{eotvos}", "eotvos": eotvos, "morton": 1e-3,
                 "domain_width": width, "domain_height": 8.0, "end_time": 2.5},
        "grid": {"cells_per_diameter": cells, "nx": 4 * cells, "ny": 8 * cells},
        "terminal_velocity": u_t,
        "grace_anchor": anchor,
        "relative_error": (u_t - anchor) / anchor,
        "pass_c3": abs((u_t - anchor) / anchor) <= tpc.C3_VELOCITY_TOL,
        "2d_reference": None,
        "history": [],
    }


def _fake_zalesak(area_recovery: float, chord: float) -> dict:
    return {
        "cells": 100,
        "dx": 0.01, "steps": 100, "dt": 1e-3, "courant_max": 0.25,
        "geometry": {"domain": 1.0, "discrete_area": 0.03, "analytic_area": 0.03},
        "schemes": {"plic": {
            "mass_relative_drift": 0.0,
            "worst_step_relative_drift": 0.0,
            "slot": {"area_recovery": area_recovery,
                     "chord_recovery_at_2_3": chord,
                     "chord_recovery_depth_mean": chord,
                     "chord_recovery_range": [chord, chord],
                     "slot_fill": 1.0 - area_recovery},
        }},
    }


def test_criteria_pass_when_numbers_pass():
    z = _fake_zalesak(1.0, 1.0)
    bubble = [_fake_bubble_run(eo, 1.0 * tpc.grace_anchor(eo)) for eo in (1.23, 10.0, 40.0)]
    conv = tpc.convergence_metric([0.60, 0.61])
    crit = tpc.evaluate(z, bubble, conv, {"points": [], "pass": True,
                                          "worst_relative_error": 0.05})
    assert crit["C1_zalesak_mass"]["pass"]
    assert crit["C2_zalesak_slot_opening"]["pass"]
    assert crit["C3_bubble_terminal_velocity"]["pass"]
    assert crit["C5_grid_convergence"]["pass"]


def test_criteria_fail_when_bubble_is_off_anchor():
    z = _fake_zalesak(1.0, 1.0)
    bubble = [_fake_bubble_run(eo, 0.5 * tpc.grace_anchor(eo)) for eo in (1.23, 10.0, 40.0)]
    crit = tpc.evaluate(z, bubble, tpc.convergence_metric([0.6, 0.8]), None)
    assert not crit["C3_bubble_terminal_velocity"]["pass"]
    assert not crit["C5_grid_convergence"]["pass"]
    assert all(abs(l["relative_error"] - (-0.5)) < 1e-12
               for l in crit["C3_bubble_terminal_velocity"]["levels"])
    assert not crit["all_pass"]


def test_criteria_report_c2_chord_next_to_area():
    z = _fake_zalesak(area_recovery=0.99, chord=0.90)
    crit = tpc.evaluate(z, [], None, None)
    c2 = crit["C2_zalesak_slot_opening"]
    assert c2["pass"]                                   # area reading
    assert c2["chord_recovery_at_2_3"] == pytest.approx(0.90)
    assert not c2["chord_recovery_at_2_3_pass"]          # single chord misses it


def test_convergence_metric_gate():
    assert tpc.convergence_metric([0.60, 0.62])["pass"]
    assert not tpc.convergence_metric([0.50, 0.62])["pass"]
    assert tpc.convergence_metric([0.60])["pass"] is False   # one grid is not a trend


# --------------------------------------------------------------------------- #
# Dam break: reference curve, gate, and a short real integration
# --------------------------------------------------------------------------- #
def test_martin_moyce_reference_anchors():
    assert mm_surge_reference(0.0) == pytest.approx(0.0)
    assert mm_surge_reference(1.0) == pytest.approx(1.40)
    assert mm_surge_reference(2.0) == pytest.approx(2.45)


def test_dam_gate_passes_exact_curve_and_rejects_a_shift():
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    exact = [{"t_star": t, "front_star": mm_surge_reference(t)} for t in times]
    good = tpc.evaluate_dam_break({"history": exact})
    assert good["pass"]
    shifted = [{"t_star": t, "front_star": 1.2 * mm_surge_reference(t)} for t in times]
    bad = tpc.evaluate_dam_break({"history": shifted})
    assert not bad["pass"]
    assert bad["worst_relative_error"] == pytest.approx(0.20, rel=1e-6)


def test_dam_gate_reports_unreached_times_as_fail():
    history = [{"t_star": 0.0, "front_star": 0.0}, {"t_star": 0.5, "front_star": 0.75}]
    out = tpc.evaluate_dam_break({"history": history}, times_star=(1.0, 2.0, 3.0))
    assert not out["pass"]
    assert out["points_reached"] == "0/3"


def test_short_dam_break_run_conserves_volume_and_starts_at_the_gate():
    """Real integration (small grid, < 1 s of wall clock) of the dam-break leg."""
    # The campaign grid (11 cells across the column) for two or three steps: the
    # advancing tongue is only ~1 cell deep, so a coarser grid would measure the
    # *film* rather than the front.
    cfg = tpc.DamBreakRun(name="tiny", nx=100, ny=30, max_t_star=0.03)
    run = tpc.run_dam_break_case(cfg, progress=False)
    assert run["steps"] > 0
    assert run["history"], "the run produced no history"
    first = run["history"][0]
    # Martin & Moyce measure the front from the gate: x*(t=0) = 0, not 1.
    assert abs(first["front_star"]) <= 0.3
    assert first["front_star"] > -0.5
    assert abs(run["final_volume_drift"]) < 1e-10
    assert run["worst_step_relative_drift"] < 1e-10
    assert run["grid"]["cells_across_column"] > 2


def test_dam_break_rejects_a_bad_pressure_form():
    with pytest.raises(ValueError):
        tpc.DamBreakRun(pressure_form="nonsense")
    with pytest.raises(ValueError):

        tpc.run_dam_break_case(tpc.DamBreakRun(name="bad", pressure_form="nonsense"))


# --------------------------------------------------------------------------- #
# Leg -> summary -> report chain
# --------------------------------------------------------------------------- #
def test_json_safe_replaces_non_finite_floats():
    out = tpc._json_safe({"a": float("nan"), "b": [float("inf")], "c": torch.ones(2)})
    assert out["a"] is None and out["b"] == [None] and out["c"] == [1.0, 1.0]


def test_legs_roundtrip_and_compose(tmp_path):
    out = tmp_path / "campaign"
    z = _fake_zalesak(1.0, 1.0)
    z["_leg"] = "zalesak"
    z["geometry"] = {"domain": 1.0, "discrete_area": 0.03, "analytic_area": 0.03}
    tpc.save_leg(out, "zalesak", z,
                 {"exact": zalesak_disk(100), "plic": zalesak_disk(100),
                  "donor_cell": zalesak_disk(100) * 0.5})
    specs = [(0, 1.23, 12), (0, 1.23, 16), (1, 10.0, 16), (2, 40.0, 16)]
    for k, (idx, eo, cells) in enumerate(specs):
        run = _fake_bubble_run(eo, tpc.grace_anchor(eo), cells=cells, width=4.0)
        run.update({"_leg": "bubble", "_leg_name": f"b{k}", "_level_index": idx,
                    "derived": {"sigma": 1 / eo, "mu_liquid": 0.03, "mu_gas": 0.0005},
                    "conservation": {"total_relative_drift": 1e-14,
                                     "worst_step_relative_drift": 1e-15},
                    "cost": {"steps": 10, "elapsed_seconds": 1.0,
                             "cg_iterations_last": 5, "cg_converged_last": True},
                    "shape": {"aspect_ratio": 1.2, "circularity": 0.9},
                    "regime": {"regime": "ellipsoidal"},
                    "dimensionless": {"reynolds": 20.0},
                    "snapshots": []})
        tpc.save_leg(out, f"b{k}", run, {"final_alpha": zalesak_disk(16)})
    dam = {"_leg": "dam", "config": {"name": "mm-plic"}, "grid": {"nx": 10, "ny": 5,
                                                                "dx": 0.1, "dy": 0.1,
                                                                "cells_across_column": 3},
           "steps": 5, "elapsed_seconds": 1.0, "worst_step_relative_drift": 1e-12,
           "final_volume_drift": 1e-12,
           "history": [{"t_star": t, "front_star": mm_surge_reference(t),
                        "height_star": 1.0} for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]}
    tpc.save_leg(out, "dam_mm-plic", dam)

    summary = tpc.compose(out, skip_figures=True)
    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "C1_zalesak_mass" in text and "PASS" in text
    assert summary["criteria"]["C4_dam_break_front"]["pass"]
    assert summary["criteria"]["C5_grid_convergence"]["pass"]
    assert len(summary["figures"]) == 0
    # every criterion carries its numbers, not just a flag
    for key in ("C1_zalesak_mass", "C2_zalesak_slot_opening",
                "C3_bubble_terminal_velocity", "C4_dam_break_front",
                "C5_grid_convergence"):
        assert key in summary["criteria"]
        assert "criterion" in summary["criteria"][key]
    json.loads((out / "summary.json").read_text())


def test_report_states_both_slot_readings():
    """The report must expose the chord reading next to the area reading."""
    out_summary = {"zalesak": {"cells": 100, "dx": 0.01, "steps": 10, "dt": 1e-3,
                               "courant_max": 0.25,
                               "geometry": {"discrete_area": 0.03,
                                            "analytic_area": 0.03},
                               "schemes": {"plic": {
                                   "mass_relative_drift": 0.0,
                                   "worst_step_relative_drift": 0.0,
                                   "peak_retention": 1.0, "l1": 1e-3, "linf": 1.0,
                                   "distorted_fraction": 0.01,
                                   "slot": {"area_recovery": 0.98,
                                            "chord_recovery_at_2_3": 0.92,
                                            "chord_recovery_depth_mean": 0.99,
                                            "chord_recovery_range": [0.92, 1.11],
                                            "slot_fill": 0.02}}}},
                    "bubble": [], "dam_break": [], "figures": [],
                    "criteria": tpc.evaluate(_fake_zalesak(0.98, 0.92), [], None, None)}
    text = tpc.build_report(out_summary, __import__("pathlib").Path("/tmp/x"))
    assert "单弦口径" in text and "面积口径" in text