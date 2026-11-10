"""Tests for the interface-tracking verification surface (three classic cases).

The suite pins the *measurement instruments* (analytic area, sub-cell contour
area, transition width, vortex velocity) and the small-grid versions of the three
benchmarks.  The production-size ladders run in
:mod:`tensorfvm.verification.vof_interface_tracking`; here the grids are chosen so
the whole file stays in the tens-of-seconds range.

Two tests are deliberately about *geometry bookkeeping* rather than accuracy:

* ``test_classic_slotted_disk_matches_analytic_area`` — the geometry used here is
  the published Zalesak shape (notch from the centre upwards), area ``0.05597``;
* ``test_legacy_zalesak_geometry_is_a_full_strip`` — the helper used by the
  earlier rotation verification cuts the **whole** vertical strip (area
  ``0.0413``).  It is a valid transport test but a different shape, so the
  deviation is pinned here instead of being silently "fixed" (which would move
  the numbers of an already-published verification case).
"""

import math

import pytest
import torch

from tensorfvm.verification.vof_interface_tracking import (
    CRITERIA,
    classic_slotted_disk,
    contour_area,
    diagonal_advection,
    diagonal_step_field,
    discrete_divergence,
    field_metrics,
    full_strip_disk_area,
    max_transition_width,
    slotted_disk_area,
    verdicts,
    vortex_reversal,
    vortex_time_symmetry_study,
    vortex_velocity,
    zalesak_rotation,
)
from tensorfvm.verification.vof_rotation import zalesak_disk

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# analytic geometry
# ---------------------------------------------------------------------------
def test_slotted_disk_area_matches_quadrature():
    """Closed form vs numeric integration of ``πr² − ∫ sqrt(r² − x²) dx``."""
    r, h = 0.15, 0.05
    steps = 200001
    xs = torch.linspace(-h, h, steps, dtype=torch.float64)
    integrand = torch.sqrt((r * r - xs * xs).clamp_min(0.0))
    trapz = getattr(torch, "trapezoid", torch.trapz)
    numeric = math.pi * r * r - float(trapz(integrand, xs))
    assert abs(numeric - slotted_disk_area()) < 1e-8


def test_slotted_disk_area_rejects_bad_slot():
    with pytest.raises(ValueError):
        slotted_disk_area(radius=0.1, slot_half_width=0.2)


def test_classic_slotted_disk_matches_analytic_area():
    """Rasterised geometry converges to the analytic area (O(Δx) but < 1 %)."""
    analytic = slotted_disk_area()
    for n in (128, 256):
        a = classic_slotted_disk(n)
        volume = float(a.sum()) / n ** 2
        assert abs(volume - analytic) / analytic < 0.02
        assert float(a.min()) == 0.0 and float(a.max()) == 1.0
    fine = classic_slotted_disk(512)
    assert abs(float(fine.sum()) / 512 ** 2 - analytic) / analytic < 0.005


def test_classic_slotted_disk_slot_is_open_upwards_only():
    """The notch must remove liquid above the centre and *keep* it below."""
    n = 128
    dx = 1.0 / n
    a = classic_slotted_disk(n)
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    xx, yy = torch.meshgrid(c, c, indexing="xy")
    disk = ((xx - 0.5) ** 2 + (yy - 0.75) ** 2) <= 0.15 ** 2
    strip = (xx - 0.5).abs() <= 0.05
    upper = disk & strip & (yy >= 0.75)
    lower = disk & strip & (yy < 0.75)
    assert bool(upper.any()) and bool(lower.any())
    assert float(a[upper].sum()) == 0.0
    assert float(a[lower].sum()) > 0.0


def test_legacy_zalesak_geometry_is_a_full_strip():
    """Pin the historical helper: it removes the strip *and* the lower lobe."""
    legacy = float(zalesak_disk(512).sum()) / 512 ** 2
    classic = float(classic_slotted_disk(512).sum()) / 512 ** 2
    analytic_full_strip = full_strip_disk_area()
    assert abs(legacy - analytic_full_strip) / analytic_full_strip < 0.01, legacy
    assert (classic - legacy) / classic > 0.2
    assert full_strip_disk_area() < slotted_disk_area()
    with pytest.raises(ValueError):
        full_strip_disk_area(radius=0.05, slot_half_width=0.1)


# ---------------------------------------------------------------------------
# sub-cell contour area
# ---------------------------------------------------------------------------
def test_contour_area_of_a_binary_disk_is_its_rasterised_area():
    n = 96
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    for radius in (0.15, 0.25):
        disk = (((xx - 0.5) ** 2 + (yy - 0.5) ** 2) <= radius ** 2).to(torch.float64)
        area = contour_area(disk, dx=1.0 / n)
        assert abs(area - float(disk.sum()) / n ** 2) < 5e-3
        assert abs(area - math.pi * radius ** 2) / (math.pi * radius ** 2) < 0.02


def test_contour_area_quadrature_is_converged():
    """Quadrature sensitivity must stay an order of magnitude below the criterion."""
    n = 64
    a = classic_slotted_disk(n)
    smeared = 0.5 * (a + torch.roll(a, 1, dims=1))
    coarse = contour_area(smeared, dx=1.0 / n, subcells=8)
    fine = contour_area(smeared, dx=1.0 / n, subcells=16)
    finer = contour_area(smeared, dx=1.0 / n, subcells=32)
    assert abs(coarse - fine) / fine < 5e-3     # < 0.5 %
    assert abs(fine - finer) / finer < 2e-3     # < 0.2 %
    assert abs(fine - finer) < abs(coarse - fine)  # monotone convergence


def test_contour_area_degenerate_fields():
    ones = torch.ones((8, 8), dtype=torch.float64)
    zeros = torch.zeros((8, 8), dtype=torch.float64)
    assert contour_area(ones, dx=0.125) == pytest.approx(1.0)
    assert contour_area(zeros, dx=0.125) == 0.0
    # a uniform 0.6 field is entirely above the 0.5 level and entirely below 0.7
    assert contour_area(ones * 0.6, dx=0.125) == pytest.approx(1.0)
    assert contour_area(ones * 0.6, level=0.7, dx=0.125) == 0.0


def test_contour_area_periodic_and_clamped_wrap_differ():
    """The boundary convention must be explicit, not accidental."""
    a = torch.full((8, 8), 0.4, dtype=torch.float64)
    a[:, -1] = 0.9
    periodic = contour_area(a, dx=0.125, periodic=True)
    clamped = contour_area(a, dx=0.125, periodic=False)
    assert periodic != clamped
    assert periodic > clamped  # wrap spreads liquid across the seam


def test_contour_area_rejects_bad_input():
    with pytest.raises(ValueError):
        contour_area(torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        contour_area(torch.zeros((4, 4), dtype=torch.float64), subcells=0)


def test_max_transition_width_of_a_binary_field_is_zero():
    """No cell is *partially* filled in a binary field, so the run length is 0."""
    a = classic_slotted_disk(64)
    assert max_transition_width(a) == 0
    ramp = torch.zeros((1, 8), dtype=torch.float64)
    ramp[0, 2:6] = torch.tensor([0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
    assert max_transition_width(ramp) == 4


def test_field_metrics_requires_exact_for_errors():
    a = classic_slotted_disk(32)
    m = field_metrics(a, dx=1.0 / 32)
    assert "l1" not in m and "peak_retention" not in m
    m = field_metrics(a, a, dx=1.0 / 32)
    assert m["l1"] == 0.0 and m["peak_retention"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# vortex velocity field
# ---------------------------------------------------------------------------
def test_vortex_velocity_is_bounded_and_staggered():
    n = 48
    u, v, dx = vortex_velocity(n, phase=1.0)
    assert u.shape == (n, n + 1) and v.shape == (n + 1, n)
    assert float(u.abs().max()) <= 1.0 + 1e-12
    assert float(v.abs().max()) <= 1.0 + 1e-12
    assert dx == pytest.approx(1.0 / n)
    # sign reversal is exactly the opposite field
    u2, v2, _ = vortex_velocity(n, phase=-1.0)
    assert torch.allclose(u2, -u) and torch.allclose(v2, -v)


def test_vortex_velocity_divergence_is_small_but_not_zero():
    """The discrete field is *not* exactly solenoidal — report it, do not hide it."""
    for n in (32, 64):
        u, v, dx = vortex_velocity(n)
        div = discrete_divergence(u, v, dx)
        assert 0.0 < div < 1e-10


def test_diagonal_step_field_is_the_half_space_intersection():
    n = 64
    a = diagonal_step_field(n)
    assert float(a.min()) == 0.0 and float(a.max()) == 1.0
    # the band area must match the analytic triangle/trapezoid area
    exact = 0.75 ** 2 / 2.0
    assert abs(float(a.sum()) / n ** 2 - exact) < 0.03


# ---------------------------------------------------------------------------
# the three benchmark cases (small grids)
# ---------------------------------------------------------------------------
def test_zalesak_rotation_bounded_conservative_and_sharp():
    n = 32
    row = zalesak_rotation(n, revolutions=1, scheme="plic")
    assert row["cells"] == n and row["steps"] > 1
    assert row["mass_drift"] < CRITERIA["step_mass_residual_max"]
    assert row["max_step_residual"] < CRITERIA["step_mass_residual_max"]
    assert row["area_drift"] < CRITERIA["zalesak_area_drift_max"]
    assert row["metrics"]["minimum"] >= 0.0
    assert row["metrics"]["maximum"] <= 1.0 + 1e-12
    assert row["metrics"]["maximum"] > 0.5  # the disk is still there


def test_zalesak_rotation_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        zalesak_rotation(32, revolutions=0.5)
    with pytest.raises(ValueError):
        zalesak_rotation(32, revolutions=0)
    with pytest.raises(ValueError):
        zalesak_rotation(32, scheme="elvira")


def test_zalesak_donor_cell_is_the_worse_baseline():
    """The geometric scheme must beat first-order donor cell on shape error."""
    plic = zalesak_rotation(32, revolutions=1, scheme="plic")
    donor = zalesak_rotation(32, revolutions=1, scheme="donor_cell")
    assert plic["metrics"]["l1"] < donor["metrics"]["l1"]
    assert plic["metrics"]["maximum"] > donor["metrics"]["maximum"]


def test_vortex_reversal_conserves_mass_and_beats_donor_cell():
    plic = vortex_reversal(32, half_period=0.25, scheme="plic")
    donor = vortex_reversal(32, half_period=0.25, scheme="donor_cell")
    assert plic["mass_drift"] < CRITERIA["vortex_mass_drift_max"]
    assert plic["max_step_residual"] < CRITERIA["step_mass_residual_max"]
    assert plic["metrics"]["peak_retention"] > donor["metrics"]["peak_retention"]
    assert plic["metrics"]["relative_l1"] < donor["metrics"]["relative_l1"]
    assert plic["metrics"]["max_transition_width"] <= donor["metrics"]["max_transition_width"]
    assert plic["velocity"]["max_discrete_divergence"] < 1e-10


def test_vortex_reversal_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        vortex_reversal(32, scheme="plic_fct")
    with pytest.raises(ValueError):
        vortex_reversal(16, half_period=0.125, reversal="adiabatic")


def test_hard_reversal_still_conserves_mass():
    """The reversal mode changes accuracy, never conservation."""
    smooth = vortex_reversal(16, half_period=0.125, reversal="smooth")
    hard = vortex_reversal(16, half_period=0.125, reversal="hard")
    for row in (smooth, hard):
        assert row["mass_drift"] < CRITERIA["vortex_mass_drift_max"]
        assert row["max_step_residual"] < CRITERIA["step_mass_residual_max"]
    # the discontinuous reversal is the less accurate of the two
    assert hard["metrics"]["relative_l1"] > smooth["metrics"]["relative_l1"]


def test_time_symmetry_study_reports_both_modes():
    rows = vortex_time_symmetry_study(n=16, cfls=(0.4,), reversals=("smooth", "hard"))
    assert {(r["reversal"], r["cfl"]) for r in rows} == {("smooth", 0.4), ("hard", 0.4)}
    assert all(r["mass_drift"] < CRITERIA["vortex_mass_drift_max"] for r in rows)


def test_diagonal_advection_is_mass_conservative_and_first_order_or_better():
    coarse = diagonal_advection(32, scheme="plic")
    fine = diagonal_advection(64, scheme="plic")
    assert coarse["max_step_residual"] < CRITERIA["step_mass_residual_max"]
    assert fine["max_step_residual"] < CRITERIA["step_mass_residual_max"]
    assert fine["l1"] < coarse["l1"]
    order = math.log(coarse["l1"] / fine["l1"]) / math.log(2.0)
    assert order >= CRITERIA["diagonal_order_min"], order
    donor = diagonal_advection(64, scheme="donor_cell")
    assert fine["l1"] < donor["l1"]
    with pytest.raises(ValueError):
        diagonal_advection(16, scheme="maccormack")


# ---------------------------------------------------------------------------
# verdict logic (negative evidence: it must be able to fail)
# ---------------------------------------------------------------------------
def _fake_row(scheme: str, cells: int, area_drift: float, mass_drift: float,
              residual: float) -> dict:
    return {
        "scheme": scheme,
        "cells": cells,
        "area_drift": area_drift,
        "mass_drift": mass_drift,
        "max_step_residual": residual,
    }


def _fake_diagonal(order: float) -> dict:
    return {"plic_order_last": order, "plic_order_fit": order}


def test_verdicts_pass_on_good_numbers():
    zalesak = [_fake_row("plic", 32, 0.01, 1e-15, 1e-15),
               _fake_row("plic", 64, 0.005, 1e-15, 1e-15)]
    vortex = [_fake_row("plic", 32, 0.0, 1e-15, 1e-15),
              _fake_row("plic", 64, 0.0, 1e-15, 1e-15)]
    v = verdicts(zalesak, vortex, _fake_diagonal(1.1))
    assert v["all_pass"] is True
    assert all(c["pass"] for c in v["checks"].values())


def test_verdicts_fail_on_each_criterion():
    bad_area = [_fake_row("plic", 64, 0.05, 1e-15, 1e-15)]
    good = [_fake_row("plic", 64, 0.0, 1e-15, 1e-15)]
    v = verdicts(bad_area, good, _fake_diagonal(1.1))
    assert v["checks"]["zalesak_area_drift"]["pass"] is False and not v["all_pass"]

    bad_mass = [_fake_row("plic", 64, 0.0, 1e-2, 1e-15)]
    v = verdicts(good, bad_mass, _fake_diagonal(1.1))
    assert v["checks"]["vortex_mass_drift"]["pass"] is False and not v["all_pass"]

    bad_residual = [_fake_row("plic", 64, 0.0, 0.0, 1e-6)]
    v = verdicts(bad_residual, good, _fake_diagonal(1.1))
    assert v["checks"]["step_mass_residual"]["pass"] is False and not v["all_pass"]

    v = verdicts(good, good, _fake_diagonal(0.5))
    assert v["checks"]["diagonal_l1_order"]["pass"] is False and not v["all_pass"]
    v = verdicts(good, good, _fake_diagonal(None))
    assert v["checks"]["diagonal_l1_order"]["pass"] is False


def test_verdicts_ignores_the_donor_baseline_rows():
    """Criteria are judged on the scheme under test, not on the reference."""
    plic = [_fake_row("plic", 64, 0.001, 1e-15, 1e-15)]
    donor = [_fake_row("donor_cell", 64, 0.9, 1e-1, 1e-1)]
    v = verdicts(plic + donor, plic, _fake_diagonal(1.2))
    assert v["all_pass"] is True


# ---------------------------------------------------------------------------
# driver smoke test
# ---------------------------------------------------------------------------
def test_run_writes_json_and_figures(tmp_path):
    from tensorfvm.verification.vof_interface_tracking import run

    report = run(tmp_path, resolutions=(16,), make_figures=True)
    payload = tmp_path / "vof_interface_tracking.json"
    assert payload.exists()
    import json
    saved = json.loads(payload.read_text())
    assert saved["verdict"]["criteria"] == CRITERIA
    assert {r["scheme"] for r in saved["zalesak"]} == {"plic", "donor_cell"}
    assert set(report["figures"]) == {
        "zalesak_rotation.png", "vortex_reversal.png",
        "diagonal_convergence.png", "zalesak_area_drift.png",
    }
    for name in report["figures"]:
        assert (tmp_path / name).stat().st_size > 1000
    # no resolution pair ⇒ no order ⇒ that criterion honestly fails
    assert report["verdict"]["checks"]["diagonal_l1_order"]["pass"] is False


def test_main_prints_a_verdict_and_returns_zero_or_one(tmp_path, capsys):
    from tensorfvm.verification.vof_interface_tracking import main

    code = main(["--resolutions", "16", "--no-figures", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert "all_pass:" in out
    assert code in (0, 1)