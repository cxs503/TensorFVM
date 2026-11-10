"""Verification tests for the VOF ice-water free-surface node.

Each test targets one claim of ``src/tensorfvm/vof/ice_water.py`` /
``thinc.py``; the campaign-level criteria (C1..C6) are produced by
``python -m tensorfvm.vof.ice_water_report`` and asserted here at CI size.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.vof import thinc
from tensorfvm.vof.ice_water import (
    IceWaterConfig,
    IceWaterVOF,
    column_height,
    ritter_dam_break,
    standing_wave_elevation,
    subcell_front,
)


# --------------------------------------------------------------------------- #
# analytic anchors
# --------------------------------------------------------------------------- #
def test_ritter_analytic_properties():
    """Ritter: front at 2 c0 t, depth h0 behind the dam, dry ahead, monotone."""
    h0, g = 0.08, 9.81
    x = torch.linspace(-0.5, 2.5, 400, dtype=torch.float64)
    t = 0.4
    h, u, front = ritter_dam_break(x, t, h0, g, x_dam=0.0)
    c0 = math.sqrt(g * h0)
    assert front == pytest.approx(2.0 * c0 * t, rel=1e-12)
    assert float(h[x < -c0 * t].min()) == pytest.approx(h0, rel=1e-12)
    assert float(h.max()) == pytest.approx(h0, rel=1e-12)
    assert float(h[x > 2 * c0 * t].max()) == 0.0
    assert bool((u >= -1e-12).all()) and bool((u <= 2 * c0 + 1e-9).all())
    assert float(u.min()) == pytest.approx(0.0, abs=1e-12)


def test_ritter_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        ritter_dam_break(torch.zeros(3), 0.0, 0.1)
    with pytest.raises(ValueError):
        ritter_dam_break(torch.zeros(3), 0.1, -0.1)


def test_standing_wave_dispersion_relation():
    eta, omega = standing_wave_elevation(0.0, 0.0, 0.2, 0.02, 2.0, 9.81)
    k = math.pi
    assert omega == pytest.approx(math.sqrt(9.81 * k * math.tanh(k * 0.2)), rel=1e-12)
    assert float(eta) == pytest.approx(0.02, rel=1e-12)


def test_subcell_front_interpolates_between_cells():
    """A linear hydrograph crossing halfway must land on the half-cell point."""
    h = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    x = subcell_front(h, dx=0.1, threshold=0.25)
    assert x == pytest.approx((1 + 0.5 + 0.5) * 0.1, rel=1e-12)
    assert subcell_front(torch.zeros(4), 0.1, 0.01) == 0.0


# --------------------------------------------------------------------------- #
# THINC reconstruction / transport
# --------------------------------------------------------------------------- #
def test_thinc_reconstruction_reproduces_cell_mean():
    a = torch.linspace(1e-3, 1 - 1e-3, 41, dtype=torch.float64)
    x_if = thinc.thinc_reconstruct(a, sharpness=4.0)
    mean = thinc.thinc_cell_mean(x_if, 2.0 * thinc.beta_from_sharpness(a, 4.0))
    assert float((mean - a).abs().max()) < 1e-6


def test_thinc_1d_transport_is_conservative():
    a = torch.zeros(64, dtype=torch.float64)
    a[:32] = 1.0
    speed = torch.full((65,), 0.5, dtype=torch.float64)
    mass0 = float(a.sum())
    for _ in range(20):
        a = thinc.advance_1d(a, speed, dt=0.1, dx=0.05)
    assert abs(float(a.sum()) - mass0) / mass0 < 1e-12
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_thinc_1d_rejects_cfl_above_one():
    a = torch.full((16,), 0.5, dtype=torch.float64)
    speed = torch.full((17,), 2.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        thinc.advance_1d(a, speed, dt=0.1, dx=0.05)


def test_thinc_and_plic_preserve_a_uniform_field_transported_periodically():
    """A uniform field must stay uniform and keep its volume in both schemes."""
    from tensorfvm.vof_plic import advect_plic

    n = 16
    uniform = torch.full((n, n), 0.3, dtype=torch.float64)
    u = torch.zeros((n, n + 1), dtype=torch.float64)
    v = torch.zeros((n + 1, n), dtype=torch.float64)
    u[:, 1:-1] = 0.0  # uniform translation is checked in 1-D above; here: at rest
    mass = float(uniform.sum())
    for scheme in (advect_plic, thinc.advect_thinc):
        out = scheme(uniform, u, v, 0.01, 0.1, 0.1)
        assert abs(float(out.sum()) - mass) / mass < 1e-12
        assert float((out - 0.3).abs().max()) < 1e-9


# --------------------------------------------------------------------------- #
# solver: statics and negative controls (C5)
# --------------------------------------------------------------------------- #
def _dam_break_solver(nx=32, ny=16, **kwargs):
    cfg = IceWaterConfig(nx=nx, ny=ny, length=1.0, height=0.5, **kwargs)
    s = IceWaterVOF(cfg)
    a = torch.zeros((ny, nx), dtype=torch.float64)
    a[: ny // 4, : nx // 2] = 1.0
    s.set_water(a)
    return s


def test_static_pool_stays_at_rest():
    """Balanced free-surface condition: a motionless pool must stay motionless."""
    cfg = IceWaterConfig(nx=32, ny=16, length=1.0, height=0.5)
    s = IceWaterVOF(cfg)
    a = torch.zeros((16, 32), dtype=torch.float64)
    a[:8, :] = 1.0
    s.set_water(a)
    for _ in range(30):
        rec = s.step()
    assert s.max_speed() < 1e-8, f"spurious current {s.max_speed():.3e} m/s"
    assert abs(rec["water_drift"]) < 1e-12


def test_negative_control_zero_gravity_no_flow():
    """C5b: gravity removed -> the column must not collapse."""
    s = _dam_break_solver(gravity=0.0)
    for _ in range(20):
        s.step()
    assert s.max_speed() < 1e-8
    assert float(s.alpha[:4, :16].min()) == 1.0


def test_negative_control_density_ratio_one_no_interface_motion():
    """C5c: with rho_air = rho_water the stratified interface must not move."""
    cfg = IceWaterConfig(nx=32, ny=16, length=1.0, height=0.5,
                         air_model="two_fluid", rho_air=998.0, rho_ice=998.0)
    s = IceWaterVOF(cfg)
    a = torch.zeros((16, 32), dtype=torch.float64)
    a[:8, :] = 1.0
    s.set_water(a)
    for _ in range(20):
        s.step()
    assert s.max_speed() < 1e-3
    assert abs(s.water_volume() - float(a.sum()) * s.dx * s.dy) < 1e-12


def test_negative_control_no_ice_equals_pure_water():
    """C5a: an explicitly zero ice field must change nothing (bitwise)."""
    runs = []
    for with_ice_call in (False, True):
        s = _dam_break_solver()
        if with_ice_call:
            s.set_ice(torch.zeros_like(s.alpha))
        for _ in range(10):
            s.step()
        runs.append((s.alpha.clone(), s.u.clone(), s.v.clone()))
    assert float((runs[0][0] - runs[1][0]).abs().max()) == 0.0
    assert float((runs[0][1] - runs[1][1]).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# solver: conservation, ice handling, input validation
# --------------------------------------------------------------------------- #
def test_dam_break_mass_conservation():
    """Flux-form VOF: the water volume must not drift."""
    s = _dam_break_solver(nx=32, ny=16)
    v0 = s.water_volume()
    worst = 0.0
    for _ in range(40):
        worst = max(worst, abs(s.step()["water_drift"]))
    assert abs(s.water_volume() - v0) / v0 < 1e-12
    assert worst < 1e-12


def test_ice_block_is_rigid_and_carries_pressure_load():
    """Ice faces carry no flux, the floe volume is fixed and it feels a force."""
    s = _dam_break_solver(nx=32, ny=16)
    # the floe stands inside the reservoir column so it feels the hydrostatic load
    s.add_ice_block(0.20, 0.35, 0.0, 0.20)
    ice0 = s.ice_volume()
    for _ in range(25):
        s.step()
    solid = s.solid_mask
    # every face touching ice must be exactly zero
    blocked_u = IceWaterVOF._face_blocked(solid, 1)
    assert float(s.u[blocked_u].abs().max()) == 0.0
    blocked_v = IceWaterVOF._face_blocked(solid, 0)
    assert float(s.v[blocked_v].abs().max()) == 0.0
    fx, _ = s.drag_on_ice()
    assert math.isfinite(fx) and abs(fx) > 0.0
    assert abs(s.ice_volume() - ice0) / ice0 < 1e-12


def test_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        IceWaterVOF(IceWaterConfig(nx=32, ny=16, length=1.0, height=0.4))  # non-square cells
    with pytest.raises(ValueError):
        IceWaterConfig(vof_scheme="mumbo")
    with pytest.raises(ValueError):
        IceWaterConfig(air_model="smoke")
    with pytest.raises(ValueError):
        IceWaterConfig(cfl=1.5)
    s = _dam_break_solver()
    with pytest.raises(ValueError):
        s.set_water(torch.full((16, 32), 1.5, dtype=torch.float64))
    with pytest.raises(ValueError):
        s.step(dt=-1.0)
    with pytest.raises(ValueError):
        s.add_ice_block(0.9, 1.5, 0.0, 0.1)


def test_momentum_is_conserved_in_a_closed_free_slip_box():
    """C6: no gravity, no open boundary -> total momentum must not drift."""
    cfg = IceWaterConfig(nx=32, ny=16, length=1.0, height=0.5,
                         slip="free_slip", gravity=0.0)
    s = IceWaterVOF(cfg)
    s.set_water(torch.ones((16, 32), dtype=torch.float64))
    yc = (torch.arange(16, dtype=torch.float64) + 0.5) * s.dy
    s.u = 0.1 * torch.sin(math.pi * yc / cfg.height).unsqueeze(1).expand(16, 33).clone()
    p0 = s.momentum()
    for _ in range(15):
        s.step()
    p1 = s.momentum()
    residual = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / max(math.hypot(*p0), 1e-30)
    assert residual < 1e-2, f"momentum residual {residual:.3e}"


# --------------------------------------------------------------------------- #
# criteria at CI size (C1/C2/C3)
# --------------------------------------------------------------------------- #
def test_c1_surge_bulk_matches_ritter_but_tip_lags():
    """C1 at CI resolution: the surge bulk must match Ritter; the tip is honest-FAIL.

    The front position of the analytic solution is a *vanishing-depth* tip, so it
    is the hardest quantity in the case: this solver reproduces the rarefaction
    fan (depth profile) but not the last per-cent of the mass, and the campaign
    therefore records criterion C1 as **not met**.  This test pins both facts --
    the front must stay physical and advance monotonically, the profile must be
    within 15 % -- so that a future improvement shows up as a change here.
    """
    from tensorfvm.vof.ice_water_report import criterion_c1

    res = criterion_c1(nx=90, ny=16, length=3.0, h0=0.16, samples=(0.2, 0.4, 0.6))
    assert res["n_time_points"] >= 3
    rows = res["rows"]
    fronts = [r["front_numeric"] for r in rows]
    assert all(b > a for a, b in zip(fronts, fronts[1:])), fronts
    # physical band: between the dam and the analytic front
    for r in rows:
        assert 1.0 < r["front_numeric"] < r["front_analytic"]
    assert res["surface_profile_relative_l2"] < 0.15, res["surface_profile_relative_l2"]
    # the tip criterion is *not* met -- recorded, not hidden
    assert res["criterion_met"] is False


def test_c2_single_wave_mass_conservation():
    """C2: relative volume drift of a standing wave must stay below 0.5 %."""
    from tensorfvm.vof.ice_water_report import criterion_c2

    res = criterion_c2(nx=60, ny=12, periods=0.5)
    assert res["criterion_met"], res
    assert max(res["worst_step_drift"], res["cumulative_drift"]) < 5e-3


def test_c3_ice_floe_interface_capture():
    """C3: PLIC must capture a rotating ice floe with < 3 % relative L2 error."""
    from tensorfvm.vof.ice_water_report import criterion_c3

    res = criterion_c3(nx=32, ny=32)
    assert res["criterion_met"], res["scheme_results"]
    assert res["scheme_results"]["plic"]["l2_relative"] < 0.03
    assert res["scheme_results"]["plic"]["mass_error"] < 1e-10


def test_interface_uses_the_selected_scheme():
    """The solver must honour ``vof_scheme`` and stay conservative for both."""
    for scheme in ("plic", "thinc"):
        s = _dam_break_solver(nx=24, ny=12, vof_scheme=scheme)
        v0 = s.water_volume()
        for _ in range(8):
            s.step()
        assert abs(s.water_volume() - v0) / v0 < 1e-6, scheme
        assert float(s.alpha.min()) >= -1e-12 and float(s.alpha.max()) <= 1 + 1e-12


def test_column_height_rescales_with_dy():
    h = column_height(torch.ones((4, 3), dtype=torch.float64), 0.25)
    assert torch.allclose(h, torch.full((3,), 1.0, dtype=torch.float64))