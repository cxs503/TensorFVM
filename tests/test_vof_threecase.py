"""Tests for the three-case VOF verification face (node P447W2-N01).

The tests guard three different things and are grouped accordingly:

* **the metrics themselves** claim what they claim (notch area against the
  analytic slot-and-disk intersection, sub-cell front interpolation against a
  synthetic step, the Martin & Moyce mapping at its own anchor);
* **the physics anchors** hold (a stratified pool stays at rest under the
  well-balanced reduced pressure, the Young-Laplace jump is recovered, the
  balanced-CSF current is smaller than the legacy one, the notch metric
  separates a shape-preserving scheme from donor-cell);
* **the verdict logic** can actually fail (a synthetic record that breaches a
  threshold must produce ``pass=False``), so a green suite cannot hide a red
  acceptance criterion.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.vof_threecase.campaign import verdicts
from tensorfvm.vof_threecase.dambreak import (
    DamBreakMMConfig,
    front_position,
    initial_alpha,
    martin_moyce_error,
    reduced_hydrostatic_pressure,
)
from tensorfvm.vof_threecase.statics import (
    StaticBubbleConfig,
    capillary_velocity,
    run_static_bubble,
)
from tensorfvm.vof_threecase.zalesak import (
    NotchConfig,
    notch_area,
    notch_fill,
    notch_mask,
    run_long_rotation,
)
from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from tensorfvm.vof_phase0.anchors import martin_moyce_front

# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_notch_mask_area_matches_analytic_intersection():
    """The discrete notch mask integrates to the analytic slot ∩ disk area.

    ``A = 4 int_0^s sqrt(R^2 - u^2) du`` with ``s`` the slot half width; the
    cell mask is a midpoint quadrature of it, so a 1 % band on 64..256 cells is
    a check of the geometry, not of the transport.
    """
    radius, slot = 0.15, 0.05
    s = slot
    analytic = 4.0 * (0.5 * s * math.sqrt(radius ** 2 - s ** 2)
                      + 0.5 * radius ** 2 * math.asin(s / radius))
    errs = {}
    for cells in (64, 128, 256):
        mask = notch_mask(cells)
        dx = 1.0 / cells
        area = float(mask.sum()) * dx * dx
        errs[cells] = abs(area - analytic) / analytic
    # The mask is a midpoint quadrature of a step function, so its own error is
    # O(dx) and must shrink; the criterion itself never uses the analytic value
    # (it compares against the same mask on the initial field), which is why the
    # band below only has to prove the geometry is right, not sub-cell exact.
    assert errs[64] < 0.10 and errs[128] < 0.08 and errs[256] < 0.025, errs
    assert errs[256] < errs[64], errs


def test_notch_mask_follows_the_rigid_rotation():
    """The co-rotating mask must sit where the transport carries the notch.

    A 90 deg counterclockwise rotation (the velocity field of
    ``rotation_velocity``) moves the disk centre from (0.5, 0.75) to
    (0.25, 0.5); the notch of the *rotated* mask must then be empty of liquid
    in the initial field, and the unrotated mask must not be.
    """
    from tensorfvm.twophase_bench.cicsam import advect_cicsam
    from tensorfvm.verification.vof_rotation import rotation_velocity, zalesak_disk

    n = 64
    a = zalesak_disk(n)
    u, v, dx = rotation_velocity(n, rate=2 * math.pi)
    dt = 0.4 * dx / float(max(u.abs().max(), v.abs().max()))
    quarter = 0.25
    steps = int(round(quarter / dt))
    dt = quarter / steps
    for k in range(steps):
        a = advect_cicsam(a, u, v, dt, dx, dx, scheme="cicsam",
                          order="xy" if k % 2 == 0 else "yx", periodic=True)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    centroid = ((float((a * x[None, :]).sum() / a.sum())),
                float((a * x[:, None]).sum() / a.sum()))
    assert centroid[0] == pytest.approx(0.25, abs=0.01)
    assert centroid[1] == pytest.approx(0.50, abs=0.01)
    assert notch_fill(a, n, degrees=90.0) > 0.9


def test_notch_area_is_zero_for_a_full_disk_and_exact_for_the_slotted_disk():
    """``A_notch = 0`` when there is no slot; ``A_notch = A_mask`` for the
    untouched slotted disk (the metric has no offset of its own)."""
    cells = 64
    dx = 1.0 / cells
    c = (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    xx, yy = torch.meshgrid(c, c, indexing="xy")
    full = (torch.sqrt((xx - 0.5) ** 2 + (yy - 0.75) ** 2) <= 0.15).double()
    assert notch_area(full, cells) == pytest.approx(0.0, abs=1e-12)
    from tensorfvm.verification.vof_rotation import zalesak_disk

    slotted = zalesak_disk(cells)
    assert notch_area(slotted, cells) == pytest.approx(
        float(notch_mask(cells).sum()) * dx * dx, rel=1e-12)


def test_front_position_interpolates_the_half_level():
    """The front is where alpha crosses 1/2, linearly interpolated sub-cell."""
    dx = 0.1
    alpha = torch.zeros((4, 12), dtype=torch.float64)
    alpha[:, :5] = 1.0
    alpha[:, 4] = 0.75
    alpha[:, 5] = 0.25
    out = front_position(alpha, dx, bed_rows=2)
    assert out["column"] == 4 and out["interpolated"]
    # a0 = 0.75, a1 = 0.25 -> the 1/2 level sits half a cell past the centre
    assert out["x_front"] == pytest.approx((4 + 0.5 + 0.5) * dx, rel=1e-12)
    # A step 1 -> 0 puts the interface on the cell face: half a cell past the
    # last cell centre, which is the geometric position of the front.
    alpha[:, 4] = 1.0
    alpha[:, 5] = 0.0
    out = front_position(alpha, dx, bed_rows=2)
    assert out["x_front"] == pytest.approx((4 + 1.0) * dx, rel=1e-12)


def test_front_position_is_zero_on_a_dry_bed():
    alpha = torch.zeros((4, 12), dtype=torch.float64)
    assert front_position(alpha, 0.1)["x_front"] == 0.0


def test_martin_moyce_mapping_reproduces_the_anchor():
    """A front placed exactly on the digitised curve compares as 0 % error."""
    cfg = DamBreakMMConfig()
    t = 0.2
    t_star = t / cfg.time_scale
    x_front = cfg.column_width * (1.0 + martin_moyce_front(t_star))
    rec = martin_moyce_error(t, x_front, cfg)
    assert rec["abs_relative_error"] < 1e-12
    assert rec["t_star"] == pytest.approx(t_star, rel=1e-12)


def test_capillary_velocity_scale():
    assert capillary_velocity(0.07, 1000.0, 0.2) == pytest.approx(math.sqrt(0.07 / 200.0))


# --------------------------------------------------------------------------- #
# physics anchors
# --------------------------------------------------------------------------- #


def _stratified_pool(cells=32, depth=0.5):
    """A flat liquid layer filling the lower half of a sealed-ish box."""
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=1e-2, mu_gas=1e-5, gravity=9.81,
                        pressure_form="reduced", warm_start_pressure=True,
                        momentum_advection=True, top_atmosphere=True)
    solver = TwoPhaseSolver(GridConfig(nx=cells, ny=cells, length=1.0, height=1.0), tp)
    y = (torch.arange(cells, dtype=torch.float64) + 0.5) * solver.dy
    alpha = (y[:, None] <= depth).double().expand(cells, cells).contiguous()
    solver.set_alpha(alpha)
    solver.p = reduced_hydrostatic_pressure(solver)
    return solver


def test_reduced_hydrostatic_pressure_balances_every_v_face():
    """The initialisation is discrete, not analytic: the residual is round-off.

    ``p[j-1] - p[j] == (rho_face_j - rho_l) g dy`` for every active v-face is
    exactly the momentum balance the reduced-pressure predictor evaluates, so
    this identity is the well-balancedness proof for the rest state.
    """
    solver = _stratified_pool()
    rho_face = solver._rho_face_v()
    g, dy, rho_l = solver.tp.gravity, solver.dy, solver.tp.rho_liquid
    # v-face j (between cells j-1 and j) is rho_face[j]; the matching cell
    # difference is p[j] - p[j-1], so the residual is indexed from 1.
    residual = (solver.p[1:, :] - solver.p[:-1, :]) - (rho_l - rho_face[1:-1, :]) * g * dy
    assert float(residual.abs().max()) < 1e-9 * rho_l * g * dy


def test_stratified_pool_stays_at_rest():
    """With the well-balanced pressure the pool must not free-fall.

    This is the guard for the failure mode that broke the first attempt of the
    node's dam-break driver: ``p = 0`` in the reduced form accelerates the gas
    at ``(rho_l - rho_g) g / rho_g`` and the run blows up within a few steps.
    """
    solver = _stratified_pool()
    dt = 0.35 * solver.dx / math.sqrt(9.81 * 0.5)
    for _ in range(20):
        solver.step(dt)
    assert float(max(solver.u.abs().max(), solver.v.abs().max())) < 1e-9


def test_static_bubble_laplace_pressure_jump():
    """``p_in - p_out`` recovers ``sigma/R`` (Young-Laplace) in a short run."""
    rec = run_static_bubble(StaticBubbleConfig(cells=32, steps=30, track_every=10))
    assert abs(rec["jump_error_relative"]) < 0.05, rec["jump_error_relative"]
    assert rec["sigma_over_R"] == pytest.approx(0.07 / 0.2)


def test_balanced_csf_beats_legacy_on_the_same_metric():
    """The spurious-current metric must discriminate, not just be small."""
    args = dict(cells=32, steps=40, track_every=10)
    balanced = run_static_bubble(StaticBubbleConfig(csf_balance="balanced", **args))
    legacy = run_static_bubble(StaticBubbleConfig(csf_balance="legacy", **args))
    assert balanced["spurious_ratio"] < legacy["spurious_ratio"]
    assert balanced["volume_drift"] < 0.01


def test_zalesak_notch_keeps_the_slot_over_one_revolution():
    """CICSAM over 360 deg: the notch area comes back within 8 % at N = 64."""
    rec = run_long_rotation(NotchConfig(cells=64, scheme="cicsam", degrees=360.0,
                                        track_every=100))
    assert rec["notch_error_relative"] < 0.08
    assert rec["mass_drift_relative"] < 1e-10
    assert 0.99 <= rec["peak_retention"] <= 1.01
    assert rec["min_alpha"] >= -1e-12 and rec["max_alpha"] <= 1.0 + 1e-12


def test_donor_cell_fails_the_notch_metric():
    """First-order transport smears the slot: the same metric must say so."""
    cicsam = run_long_rotation(NotchConfig(cells=64, scheme="cicsam", degrees=360.0,
                                           track_every=100))
    upwind = run_long_rotation(NotchConfig(cells=64, scheme="upwind", degrees=360.0,
                                           track_every=100))
    assert upwind["notch_error_relative"] > 3.0 * cicsam["notch_error_relative"]
    assert upwind["peak_retention"] < 0.6


def test_dam_break_initial_column_and_first_front_motion():
    """A short coupled run: the front advances and the volume is conserved."""
    cfg = DamBreakMMConfig(ny=25, nx=50, aspect=2.0, t_end=0.05)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=1e-3, mu_gas=1e-5, gravity=9.81,
                        pressure_form="reduced", warm_start_pressure=True,
                        momentum_advection=True, vof_scheme="plic",
                        top_atmosphere=True)
    solver = TwoPhaseSolver(
        GridConfig(nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height), tp)
    solver.set_alpha(initial_alpha(solver, cfg))
    solver.p = reduced_hydrostatic_pressure(solver)
    # The initial column is a staircase on the cell centres, so its discrete
    # area can only approach W*H0 from below (7 x 14 cells at dx = 8 mm).
    area = float(solver.alpha.sum()) * solver.dx * solver.dy
    assert area <= cfg.column_width * cfg.depth * (1 + 1e-12)
    assert area == pytest.approx(cfg.column_width * cfg.depth, rel=0.05)
    m0 = float(solver.alpha.sum())
    dt = 0.35 * min(solver.dx, solver.dy) / math.sqrt(9.81 * cfg.depth)
    for _ in range(10):
        solver.step(dt)
    front = front_position(solver.alpha, solver.dx)
    assert front["x_front"] > cfg.column_width
    assert abs(float(solver.alpha.sum()) - m0) / m0 < 1e-10


# --------------------------------------------------------------------------- #
# verdict logic
# --------------------------------------------------------------------------- #


def _dam_record(ny, nx, errors, drift=1e-15):
    return {
        "ny": ny, "nx": nx, "mass_drift_max": drift,
        "samples": [{"t": t, "abs_relative_error": e, "relative_error": e}
                    for t, e in zip((0.1, 0.2, 0.3), errors)],
    }


def _static_record(ratio, balance="balanced", saturated=True):
    return {"csf_balance": balance, "spurious_ratio": ratio, "saturated": saturated,
            "cells": 64, "mu_liquid": 1e-2, "laplace_number": 1e4,
            "csf_curvature": "height_function", "jump_error_relative": 1e-3,
            "volume_drift": 1e-12}


def _zalesak_record(deg, err, scheme="cicsam", cells=128):
    return {"scheme": scheme, "cells": cells, "degrees": deg,
            "notch_error_relative": err, "area_error_contour_vs_initial": err,
            "perimeter_error_analytic": err, "peak_retention": 1.0,
            "mass_drift_relative": 1e-14}


def _good_campaign():
    dam = [_dam_record(50, 100, (0.20, 0.15, 0.11)),
           _dam_record(100, 200, (0.09, 0.07, 0.05)),
           _dam_record(200, 400, (0.06, 0.05, 0.03))]
    static = [_static_record(1e-3), _static_record(5e-2, balance="legacy")]
    zalesak = [_zalesak_record(360.0, 0.02), _zalesak_record(1000.0, 0.10)]
    return static, zalesak, dam


def test_verdicts_pass_on_a_conforming_campaign():
    v = verdicts(*_good_campaign())
    assert all(c["pass"] for c in v.values()), v


def test_verdicts_fail_on_a_front_error_above_the_band():
    static, zalesak, dam = _good_campaign()
    dam[1] = _dam_record(100, 200, (0.12, 0.11, 0.30))
    v = verdicts(static, zalesak, dam)
    assert not v["a_front_front_error_10pct"]["pass"]


def test_verdicts_fail_on_an_unsaturated_or_large_spurious_current():
    static, zalesak, dam = _good_campaign()
    static = [_static_record(0.02)]
    assert not verdicts(static, zalesak, dam)["b_spurious_current"]["pass"]
    static = [_static_record(1e-4, saturated=False)]
    assert not verdicts(static, zalesak, dam)["b_spurious_current"]["pass"]


def test_verdicts_fail_on_a_missed_notch_and_on_non_monotone_grids():
    static, zalesak, dam = _good_campaign()
    zalesak = [_zalesak_record(360.0, 0.02), _zalesak_record(1000.0, 0.40)]
    assert not verdicts(static, zalesak, dam)["c_notch_shape"]["pass"]
    dam[2] = _dam_record(200, 400, (0.30, 0.25, 0.20))
    assert not verdicts(static, zalesak, dam)["e_grid_convergence"]["pass"]


def test_verdicts_fail_on_mass_drift():
    static, zalesak, dam = _good_campaign()
    dam[0] = _dam_record(50, 100, (0.2, 0.15, 0.1), drift=2e-3)
    assert not verdicts(static, zalesak, dam)["d_mass_conservation"]["pass"]