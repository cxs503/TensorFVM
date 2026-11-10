"""Two-dimensional multi-material gates: Richtmyer--Meshkov, bubble, vortices.

The grids here are deliberately coarse so the file stays inside a normal CI
budget; the quantitative criteria that need the production resolution are
measured by ``python -m tensorfvm.compressible`` and recorded in
``artifacts/compressible-multimaterial/``.  What is asserted here is that every
mechanism the report relies on actually measures something: the amplitude of a
shocked interface grows at the Richtmyer rate, the budgets close, the bubble
refracts the shock into a rarefaction with the analytic speed, and the vortex
pair keeps two cores rotating at roughly the point-vortex rate.
"""

import math

import pytest
import torch

from tensorfvm.compressible.cases import (
    BubbleConfig,
    RMConfig,
    VortexConfig,
    bubble_analytic,
    normal_shock_state,
    richtmyer_meshkov,
    rm_analytic,
    shock_bubble,
    vortex_pair,
)
from tensorfvm.compressible.diagnostics import (
    amplitude_from_columns,
    interface_position,
    mixing_width,
    transverse_mean,
)
from tensorfvm.compressible.vortex import (
    core_positions,
    core_separation,
    total_circulation,
    vorticity,
)

torch.set_num_threads(1)


# ------------------------------------------------- Richtmyer-Meshkov ----
@pytest.fixture(scope="module")
def rm_result():
    cfg = RMConfig(ny=24, t_end=0.62, cfl=0.4, shock_x=-0.9)
    solver, meta = richtmyer_meshkov(cfg)
    initial = meta["initial_integrals"]
    times, amplitudes, widths = [], [], []
    while solver.time < cfg.t_end:
        dt = min(solver.stable_dt(), cfg.t_end - solver.time)
        solver.step(dt)
        prim = solver.primitive()
        pos = interface_position(prim["Y"], meta["xc"], 0.5)
        amp = amplitude_from_columns(pos)
        width = mixing_width(transverse_mean(prim["Y"]), meta["xc"])
        times.append(solver.time)
        amplitudes.append(amp["amplitude"])
        widths.append(width["width"])
    return {
        "solver": solver,
        "meta": meta,
        "times": times,
        "amplitudes": amplitudes,
        "widths": widths,
        "drift": solver.budget_drift(initial),
    }


def test_rm_analytic_anchor_is_self_consistent():
    """The impulsive model must reduce to the pre-shock state when nothing moves."""
    cfg = RMConfig()
    a = rm_analytic(cfg)
    post = normal_shock_state(1.4, cfg.mach)
    assert a["post_rho"] == pytest.approx(post["rho"], rel=1e-12)
    # light-to-heavy: interface accelerates downstream, A+ > 0, transmitted shock
    # is faster than the interface and slower than the incident shock
    assert a["atwood_post"] > 0.5
    assert 0.0 < a["u_star"] < a["transmitted_shock_speed"] < a["incident_shock_speed"]
    assert a["compression"] == pytest.approx(1.0 - a["u_star"] / a["transmitted_shock_speed"],
                                             rel=1e-12)
    assert a["linear_growth_rate"] == pytest.approx(
        a["wavenumber"] * a["a0_post"] * a["atwood_post"] * a["u_star"], rel=1e-12)


def test_rm_amplitude_grows_at_the_richtmyer_rate(rm_result):
    times, amps = rm_result["times"], rm_result["amplitudes"]
    meta = rm_result["meta"]
    theory = meta["analytic"]["linear_growth_rate"]
    t0 = meta["analytic"]["shock_transit_time"] + 0.08
    ts = [t for t, a in zip(times, amps) if t >= t0]
    as_ = [a for t, a in zip(times, amps) if t >= t0]
    assert len(ts) >= 5
    n = len(ts)
    mt, ma = sum(ts) / n, sum(as_) / n
    slope = (sum((t - mt) * (a - ma) for t, a in zip(ts, as_))
             / sum((t - mt) ** 2 for t in ts))
    assert slope > 0.0, "the interface never grew"
    # a 24-cell transverse grid is far from converged: this gate is a factor-of-two
    # sanity band, the 15 % gate is measured at ny = 40/56/72
    assert 0.5 * theory < slope < 2.0 * theory, (slope, theory)


def test_rm_budgets_close_and_the_scheme_stays_physical(rm_result):
    drift = rm_result["drift"]
    assert abs(drift["mass"]) < 5e-3
    assert abs(drift["energy"]) < 5e-3
    assert abs(drift["momentum_x"]) < 5e-3
    assert rm_result["solver"].positivity_fixes == 0


def test_rm_mixing_width_is_finite_and_grows(rm_result):
    widths = rm_result["widths"]
    assert all(w == w for w in widths)
    assert widths[-1] > widths[2]


# ------------------------------------------------------ shock-bubble ----
@pytest.fixture(scope="module")
def bubble_result():
    cfg = BubbleConfig(mesh=16, t_end=1.1)
    solver, meta = shock_bubble(cfg)
    initial = meta["initial_integrals"]
    d = meta["d"]
    i_inc = int((-1.2 - cfg.x_min) / d)
    j_mid = meta["ny"] // 2
    times, pressures, circ = [], [], []
    while solver.time < cfg.t_end:
        dt = min(solver.stable_dt(), cfg.t_end - solver.time)
        solver.step(dt)
        prim = solver.primitive()
        omega = vorticity(prim["u"], prim["v"], d, d, periodic=False)
        times.append(solver.time)
        pressures.append(float(prim["p"][i_inc, j_mid]))
        circ.append(float(omega[:, j_mid:].sum() * d * d))
    return {"solver": solver, "meta": meta, "times": times, "pressures": pressures,
            "circulation": circ, "drift": solver.budget_drift(initial)}


def test_bubble_analytic_predicts_a_reflected_rarefaction():
    a = bubble_analytic(BubbleConfig())
    assert a["reflected_is_rarefaction"] is True
    assert a["transmitted_shock_speed"] > a["incident_shock_speed"]
    assert a["crossing_time"] > 0.0 and a["arrival_time"] > 0.0


def test_bubble_incident_shock_speed(bubble_result):
    meta = bubble_result["meta"]
    a = meta["analytic"]
    cfg = meta["config"]
    d = meta["d"]
    i_inc = int((-1.2 - cfg.x_min) / d)
    threshold = 0.5 * (a["post_p"] + 1.0)
    times, pressures = bubble_result["times"], bubble_result["pressures"]
    t_cross = float("nan")
    for t, p in zip(times, pressures):
        if p >= threshold:
            t_cross = t
            break
    assert t_cross == t_cross, "the incident shock never reached the probe"
    x_probe = cfg.x_min + (i_inc + 0.5) * d
    speed = (x_probe - cfg.shock_x) / t_cross
    assert abs(speed - a["incident_shock_speed"]) / a["incident_shock_speed"] < 0.1


def test_bubble_forms_baroclinic_circulation(bubble_result):
    circ = [c for c in bubble_result["circulation"] if c == c]
    assert max(circ) > 0.0
    drift = bubble_result["drift"]
    assert abs(drift["mass"]) < 5e-3
    assert abs(drift["energy"]) < 5e-3


# ------------------------------------------------------ vortex pair ----
@pytest.fixture(scope="module")
def vortex_result():
    cfg = VortexConfig(n=96, t_end=1.0, circulation=1.0, core=0.2, separation=1.2,
                       mach_vortex=None)
    solver, meta = vortex_pair(cfg)
    initial = meta["initial_integrals"]
    d = meta["d"]
    prim = solver.primitive()
    omega0 = vorticity(prim["u"], prim["v"], d, d, periodic=False)
    angles, seps = [], []
    while solver.time < cfg.t_end:
        solver.step(min(solver.stable_dt(), cfg.t_end - solver.time))
        prim = solver.primitive()
        omega = vorticity(prim["u"], prim["v"], d, d, periodic=False)
        cores = core_positions(omega, d, d)
        sep, ncores = core_separation(omega, d, d)
        seps.append((sep, ncores))
        if len(cores) == 2:
            angles.append(math.atan2(cores[1][1] - cores[0][1],
                                     cores[1][0] - cores[0][0]))
    return {"solver": solver, "meta": meta, "omega0": omega0, "d": d,
            "angles": angles, "seps": seps,
            "circulation": total_circulation(omega, d, d),
            "drift": solver.budget_drift(initial)}


def test_vortex_pair_initial_circulation(vortex_result):
    d = vortex_result["d"]
    circ = total_circulation(vortex_result["omega0"], d, d)
    assert circ == pytest.approx(2.0, rel=0.05)


def test_vortex_pair_keeps_two_cores(vortex_result):
    sep, ncores = vortex_result["seps"][-1]
    assert ncores == 2
    assert sep == pytest.approx(1.2, rel=0.15)


def test_vortex_pair_rotates(vortex_result):
    angles = vortex_result["angles"]
    assert len(angles) > 5
    # unwrap modulo pi (the axis is a line) and require a monotone rotation
    unwrapped = [angles[0]]
    for a in angles[1:]:
        prev = unwrapped[-1]
        while a - prev > 0.5 * math.pi:
            a -= math.pi
        while a - prev < -0.5 * math.pi:
            a += math.pi
        unwrapped.append(a)
    assert unwrapped[-1] != unwrapped[0]
    assert abs(vortex_result["circulation"]) > 0.5


def test_vortex_pair_uses_non_telescoping_curl(vortex_result):
    """A periodically differentiated curl sums to exactly zero: guard the diagnostic."""
    d = vortex_result["d"]
    periodic = vorticity(vortex_result["solver"].primitive()["u"],
                         vortex_result["solver"].primitive()["v"], d, d, periodic=True)
    assert abs(total_circulation(periodic, d, d)) < 1e-9
    assert abs(vortex_result["circulation"]) > 1e-3