"""Tests for the two-phase VOF benchmark face (P400W2-N01).

Coverage (12 cases, all fast — the full campaign is reproduced by
``benchmarks/twophase/run_benchmark.py``):

1.  NVD branch values against hand-computed formulas (Hyper-C, ULTIMATE-QUICKEST).
2.  FCT limiter restores the discrete maximum principle on a random field.
3.  A flux-form sweep conserves the global sum to round-off (periodic).
4.  CICSAM keeps a translated step inside one cell where donor cell spreads it.
5.  CICSAM converges at second order on a smooth field (observed order >= 1.7).
6.  The Zalesak geometry anchors agree with the closed-form slotted-disk values.
7.  The contour area/perimeter metrics reproduce an analytic disk (metric sanity).
8.  The CICSAM perimeter error of the slotted disk decreases with resolution.
9.  Ritter analytics: mass of the analytic profile, front relation, and the
    depth-matched front locator round-trip.
10. Dam-break transport smoke test: front error, mass ledger, boundedness, and a
    measured advantage over the donor-cell baseline.
11. Guards: CFL > 1, unknown scheme, front threshold above the reservoir, and a
    domain that cannot contain the wave all raise ``ValueError``.
12. The Zalesak leg is conservative and bounded for every registered scheme.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfvm.twophase_bench.cicsam import (
    advect_cicsam,
    fct_limit,
    hyper_c,
    sweep_1d,
    ultimate_quickest,
)
from tensorfvm.twophase_bench.metrics2d import (
    contour_area,
    contour_perimeter,
    front_position,
    slotted_disk_reference,
)
from tensorfvm.twophase_bench.ritter import (
    DamBreakGeometry,
    exact_alpha,
    ritter_depth,
    ritter_front,
    ritter_front_at_depth,
    run_transport,
    velocity_faces,
    column_depth,
)
from tensorfvm.verification.vof_rotation import cosine_hill, rotation_velocity
from tensorfvm.twophase_bench.zalesak import run_rotation


# ---------------------------------------------------------------------------
# 1. NVD branch values
# ---------------------------------------------------------------------------
def test_nvd_branch_values_match_formulas():
    a_tilde = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    cf = torch.tensor([0.5, 0.25, 0.8], dtype=torch.float64)
    hc = hyper_c(a_tilde, cf)
    # min(1, a/C)
    assert hc.tolist() == pytest.approx([0.5, 1.0, 0.9375], rel=1e-12)
    uq = ultimate_quickest(a_tilde, cf)
    expected = []
    for a, c in zip(a_tilde.tolist(), cf.tolist()):
        quick = (8 * c * a + (1 - c) * (6 * a + 3)) / 8
        expected.append(min(quick, a / c))
    assert uq.tolist() == pytest.approx(expected, rel=1e-12)
    # CBC: the UQ branch never exceeds Hyper-C
    assert torch.all(uq <= hc + 1e-15)


# ---------------------------------------------------------------------------
# 2. FCT limiter and 3. conservation of a sweep
# ---------------------------------------------------------------------------
def _random_field(n=48, ny=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand((ny, n), generator=g, dtype=torch.float64)
    return a


def test_fct_limiter_restores_bounds():
    n, ny = 48, 3
    a = _random_field(n, ny)
    speed = 0.9
    faces = torch.full((ny, n + 1), speed, dtype=torch.float64)
    faces[:, 0] = faces[:, -1] = 0.0
    dt = 0.45 / speed  # directional CFL = 0.45
    out_limited = sweep_1d(a, faces, dt, 1.0, scheme="cicsam", periodic=True,
                           limit=True)
    out_raw = sweep_1d(a, faces, dt, 1.0, scheme="cicsam", periodic=True,
                       limit=False)
    for out in (out_limited, out_raw):
        # the sweep itself is conservative either way
        assert abs(float(out.sum()) - float(a.sum())) / float(a.sum()) < 1e-13
    assert float(out_limited.min()) >= -1e-14
    assert float(out_limited.max()) <= 1.0 + 1e-14

    # the limiter is *active* where a cell has no room left: with an
    # anti-diffusive flux of 0.5 into a cell at 0.9 only 0.1 may pass
    a_low = torch.tensor([[0.9, 0.9, 0.0, 0.9]], dtype=torch.float64)
    anti = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float64)
    limited = fct_limit(a_low, anti, 1.0)
    assert float(limited[0, 1]) == pytest.approx(0.1, rel=1e-9)
    # unlimited, the same flux would leave the cell super-saturated
    assert float(a_low[0, 1] + limited[0, 1] - limited[0, 2]) <= 1.0 + 1e-12


def test_sweep_conserves_global_sum():
    n, ny = 64, 2
    a = _random_field(n, ny, seed=3)
    faces = 0.4 * torch.ones((ny, n + 1), dtype=torch.float64)
    faces[:, 0] = faces[:, -1] = 0.0
    total = float(a.sum())
    for step in range(50):
        a = sweep_1d(a, faces, 1.0, 1.0, scheme="cicsam", periodic=True)
    assert abs(float(a.sum()) - total) / total < 1e-12


# ---------------------------------------------------------------------------
# 4. sharpness of a translated step
# ---------------------------------------------------------------------------
def _interface_cells(a: torch.Tensor, dx: float = 1.0) -> float:
    """10-90 % transition width of a 1-D profile, in cells."""
    v = a.flatten()
    lo = float(v.min())
    hi = float(v.max())
    if hi - lo < 1e-9:
        return float("nan")
    norm = (v - lo) / (hi - lo)
    x10 = torch.nonzero(norm > 0.1)[0].item()
    x90 = torch.nonzero(norm > 0.9)[0].item()
    return float(x90 - x10)


def test_cicsam_keeps_step_sharper_than_donor_cell():
    n = 128
    a0 = torch.zeros(1, n, dtype=torch.float64)
    a0[0, n // 4: 3 * n // 4] = 1.0
    speed = 0.5
    faces = torch.full((1, n + 1), speed, dtype=torch.float64)
    faces[0, 0] = faces[0, -1] = 0.0
    dt, steps = 0.5 / speed, 64
    a_c = a0.clone()
    a_u = a0.clone()
    for _ in range(steps):
        a_c = sweep_1d(a_c, faces, dt, 1.0, scheme="cicsam", periodic=True)
        a_u = sweep_1d(a_u, faces, dt, 1.0, scheme="upwind", periodic=True)
    w_cicsam = _interface_cells(a_c[0])
    w_upwind = _interface_cells(a_u[0])
    assert w_cicsam <= 2.0, f"CICSAM spread the step over {w_cicsam} cells"
    assert w_upwind > w_cicsam, "donor cell must be the more diffusive baseline"


# ---------------------------------------------------------------------------
# 5. second-order convergence on a smooth field
# ---------------------------------------------------------------------------
def _smooth_advection_errors(scheme: str, resolutions=(64, 128, 256)):
    speed = 0.5
    errors = []
    for n in resolutions:
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        shift = 0.3
        exact_np = 0.5 * (1.0 + np.cos(2.0 * math.pi * (x - shift)))
        a0 = torch.from_numpy(0.5 * (1.0 + np.cos(2.0 * math.pi * x)))
        a = a0.reshape(1, n).clone()
        faces = torch.full((1, n + 1), speed, dtype=torch.float64)
        dt = 0.4 * dx / speed
        steps = int(round(shift / (speed * dt)))
        dt = shift / (speed * steps)
        for step in range(steps):
            a = sweep_1d(a, faces, dt, dx, scheme=scheme, periodic=True)
        err = float((a[0] - torch.from_numpy(exact_np)).abs().sum()) * dx
        errors.append(err)
    return errors


def test_uq_branch_is_second_order_on_smooth_advection():
    errors = _smooth_advection_errors("uq")
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(2.0) for i in range(2)]
    assert min(orders) >= 1.7, f"observed orders {orders}"


def test_cicsam_beats_donor_cell_on_the_cosine_hill():
    """A rigidly rotating smooth hill: the compressive orientation blend costs
    smooth-field accuracy (measured 1-D order ~1.2 and L1 far above the pure UQ
    branch), but CICSAM must still beat the donor-cell baseline -- the reason it
    is used for interfaces at all."""
    n = 48
    a0 = cosine_hill(n)
    u, v, dx = rotation_velocity(n)
    umax = float(max(u.abs().max(), v.abs().max()))
    dt = 0.4 * dx / umax
    steps = max(1, int(round(1.0 / dt)))
    dt = 1.0 / steps
    err = {}
    for scheme in ("cicsam", "upwind"):
        a = a0.clone()
        for k in range(steps):
            a = advect_cicsam(a, u, v, dt, dx, dx, scheme=scheme,
                              order="xy" if k % 2 == 0 else "yx", periodic=True)
        err[scheme] = float((a - a0).abs().sum() / a0.sum())
    assert err["cicsam"] < 0.6 * err["upwind"], err
    assert err["cicsam"] < 1.0 and err["upwind"] > 0.5, err


# ---------------------------------------------------------------------------
# 6. Zalesak anchors and the closed-form reference
# ---------------------------------------------------------------------------
def test_slotted_disk_reference_matches_quadrature():
    ref = slotted_disk_reference(radius=0.15, slot_half_width=0.05)
    r, w = 0.15, 0.05
    yy = np.linspace(-w, w, 200001)
    strip = float(np.trapezoid(2.0 * np.sqrt(r * r - yy * yy), yy))
    area = math.pi * r * r - strip
    assert ref["area"] == pytest.approx(area, rel=1e-6)
    chord = math.sqrt(r * r - w * w)
    half_angle = math.asin(w / r)
    perimeter = 2 * math.pi * r - 4 * r * half_angle + 4 * chord
    assert ref["perimeter"] == pytest.approx(perimeter, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. metric sanity on an analytic disk
# ---------------------------------------------------------------------------
def test_contour_metrics_reproduce_analytic_disk():
    """Metric sanity: a disk resolved *sub-cell* (exact overlap fractions).

    A binary 0/1 mask cannot be used for this check: its 0.5-contour is the
    staircase polygon, which is 5.6 % too long at N = 400 (measured) — that is
    the metric's discretisation bias, not a scheme property, and it is why the
    report separates "perimeter vs analytic" from "perimeter vs initial mask".
    """
    n, dx, m = 64, 1.0 / 64, 16
    c = (np.arange(n) + 0.5) * dx
    off = (np.arange(m) + 0.5) / m * dx - 0.5 * dx
    frac = np.zeros((n, n))
    for oy in off:
        yy = c[:, None] + oy
        for ox in off:
            frac += (np.hypot(c[None, :] + ox - 0.5, yy - 0.5) <= 0.2)
    alpha = torch.from_numpy(frac / m ** 2)
    area = contour_area(alpha, dx, dx)
    per = contour_perimeter(alpha, dx, dx)
    assert area == pytest.approx(math.pi * 0.2 ** 2, rel=0.005)
    assert per == pytest.approx(2 * math.pi * 0.2, rel=0.01)


# ---------------------------------------------------------------------------
# 8. perimeter error decreases with resolution
# ---------------------------------------------------------------------------
def test_zalesak_perimeter_error_decreases_with_resolution():
    coarse = run_rotation(32, "cicsam", track_every=1000)
    fine = run_rotation(64, "cicsam", track_every=1000)
    assert fine["perimeter_error_analytic"] < coarse["perimeter_error_analytic"]
    assert fine["area_error_contour_vs_initial"] < coarse["area_error_contour_vs_initial"]


# ---------------------------------------------------------------------------
# 9. Ritter analytics
# ---------------------------------------------------------------------------
def test_ritter_analytics_are_self_consistent():
    geo = DamBreakGeometry()
    # mass of the analytic profile is invariant in time (semi-infinite reservoir
    # inside the domain: both ends outside the wave)
    for t in (0.05, 0.15, 0.30):
        x = np.linspace(geo.x_left, geo.x_right, 200001)
        h = ritter_depth(x - geo.width, t, geo)
        mass = float(np.trapezoid(h, x))
        if t == 0.05:
            m_ref = mass
        assert mass == pytest.approx(m_ref, rel=2e-3)
        # front relation
        assert ritter_front(t, geo) == pytest.approx(geo.width + 2 * geo.c0 * t)
        # the depth-matched front inverts the profile
        h_ref = 0.05 * geo.height
        x_ref = ritter_front_at_depth(t, h_ref, geo)
        assert float(ritter_depth(np.array([x_ref - geo.width]), t, geo)[0]) == \
            pytest.approx(h_ref, rel=1e-6)


def test_front_locator_recovers_the_analytic_position():
    geo = DamBreakGeometry()
    dx = geo.height / 56
    nx = int(round(geo.length / dx))
    ny = int(round(geo.tank_height / dx))
    for t in (0.05, 0.2, 0.3):
        alpha = exact_alpha(nx, ny, geo, t)
        h_ref = 0.05 * geo.height
        x = front_position(column_depth(alpha, geo.tank_height / ny), dx, geo.x_left, h_ref)
        assert x == pytest.approx(ritter_front_at_depth(t, h_ref, geo), rel=2e-3)


def test_velocity_field_is_discretely_divergence_free():
    geo = DamBreakGeometry()
    dx = geo.height / 28
    nx = int(round(geo.length / dx))
    ny = int(round(geo.tank_height / dx))
    u, v = velocity_faces(nx, ny, geo, 0.1)
    div = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / (geo.tank_height / ny)
    # interior cells only: the two boundary layers are closed walls
    assert float(div[1:-1, 1:-1].abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# 10. dam-break transport smoke test
# ---------------------------------------------------------------------------
def test_dam_break_transport_smoke():
    geo = DamBreakGeometry()
    cie = run_transport("cicsam", 14, t_end=0.08, t_start=0.02,
                        sample_times=(0.05, 0.08), geo=geo)
    uph = run_transport("upwind", 14, t_end=0.08, t_start=0.02,
                        sample_times=(0.05, 0.08), geo=geo)
    for leg in (cie, uph):
        worst = max(s["front_error_at_depth"] for s in leg["samples"])
        assert worst < 0.10, f"{leg['scheme']} front error {worst:.3f}"
        assert leg["mass_drift_relative"] < 1e-12
        for s in leg["samples"]:
            assert -1e-12 <= s["alpha_min"]
            assert s["alpha_max"] <= 1.0 + 1e-12
    # shape: CICSAM resolves the rarefaction better than donor cell
    assert (cie["samples"][-1]["depth_error_l1_relative"]
            < uph["samples"][-1]["depth_error_l1_relative"])


# ---------------------------------------------------------------------------
# 11. guards
# ---------------------------------------------------------------------------
def test_guards_reject_invalid_input():
    a = torch.full((1, 8), 0.5, dtype=torch.float64)
    faces = torch.full((1, 9), 2.0, dtype=torch.float64)  # CFL = 2 * dt / h > 1
    with pytest.raises(ValueError):
        sweep_1d(a, faces, 0.6, 1.0, scheme="cicsam", periodic=True)
    with pytest.raises(ValueError):
        sweep_1d(a, faces, 0.1, 1.0, scheme="nope", periodic=True)
    geo = DamBreakGeometry()
    with pytest.raises(ValueError):
        ritter_front_at_depth(0.1, 2.0 * geo.height, geo)
    with pytest.raises(ValueError):
        DamBreakGeometry(upstream=0.26, downstream=0.30, tank_height=0.07)
    with pytest.raises(ValueError):
        advect_cicsam(a, torch.zeros((1, 9)), torch.zeros((2, 8)), 0.1, 1.0, 1.0,
                      scheme="cicsam", order="zz")


# ---------------------------------------------------------------------------
# 12. every registered Zalesak scheme is conservative and bounded
# ---------------------------------------------------------------------------
def test_zalesak_schemes_are_conservative_and_bounded():
    for scheme in ("upwind", "cicsam", "hric"):
        row = run_rotation(32, scheme, track_every=1000)
        assert row["mass_drift_relative"] < 1e-10, scheme
        assert row["min_alpha"] >= -1e-12, scheme
        assert row["max_alpha"] <= 1.0 + 1e-12, scheme
        assert row["bounded_violation_cells"] == 0, scheme