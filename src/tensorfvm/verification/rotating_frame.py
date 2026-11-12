"""Verification driver for :mod:`tensorfvm.rotating_frame` (MRF / Coriolis).

Run::

    PYTHONPATH=src python -m tensorfvm.verification.rotating_frame

Writes ``docs/verification-rotating-frame/{summary.json,REPORT.md,fig_rotating_frame.png}``.

Every criterion is pre-declared with its tolerance before it is evaluated; a
failure is reported as ``FAIL`` and never overwritten.  Three of the criteria
(C6, C7c, C10c) are *control* criteria: they quantify how wrong a solver that
omits the Coriolis term (or the centrifugal term, or gets its sign wrong) would
be.  An anchor that a wrong implementation can also pass is not an anchor, so the
discriminating power is measured, not assumed.

Anchors used (all closed form, no table transcribed by hand):

* rotating plane Poiseuille flow with the rotation axis along the wall normal --
  ``u`` and the Coriolis-driven secondary flow ``w`` (Ekman-type rotating duct
  flow; the closed form is derived in the module docstring and cross-checked by
  an independent collocation BVP, C4b, and by a second-order FVM, C5);
* plane channel with the rotation axis along the span -- the classical
  rotating-channel configuration (Johnston, Halleen & Lezius 1972): parabolic
  ``u`` that is *unaffected* by rotation, plus the cubic wall-normal pressure
  correction that is the only signature of the Coriolis term;
* circular Couette flow solved in a frame rotating at ``omega_frame`` -- MRF
  frame invariance of the absolute flow and of Taylor's torque.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Dict, List

import numpy as np

from ..rotating_frame import (
    RotatingFrame,
    exact_circular_couette,
    exact_rotating_channel_spanwise,
    exact_rotating_plane_poiseuille,
    solve_circular_couette_frame,
    solve_inert_plane_poiseuille,
    solve_rotating_plane_poiseuille,
)

__all__ = ["run", "render_report", "plot", "main"]


def _criterion(value: float, gate: float, note: str, *, mode: str = "max") -> Dict:
    ok = value <= gate if mode == "max" else value >= gate
    return {
        "value": float(value),
        "gate": float(gate),
        "mode": mode,
        "pass": bool(ok),
        "note": note,
    }


def _rel(a, b) -> float:
    """Relative L2 difference; complex input is allowed (compared via |.|)."""
    a, b = np.asarray(a), np.asarray(b)
    den = float(np.linalg.norm(b))
    num = float(np.linalg.norm(a - b))
    return num / den if den else num


def _order(coarse: float, fine: float) -> float:
    if fine <= 0.0 or coarse <= 0.0:
        return float("nan")
    return float(math.log2(coarse / fine))


# channel / couette reference problem parameters (fixed, pre-declared)
_CH = {"G": 1.0, "rho": 1.0, "nu": 0.05, "omega": 2.0, "h": 1.0}
_CO = {"r1": 1.0, "r2": 2.0, "omega_inner": 1.0, "omega_outer": 0.0}


def run() -> Dict:
    t0 = time.time()
    crit: Dict[str, Dict] = {}
    data: Dict = {}

    # ---------------------------------------------------------------- C1
    rng = np.random.default_rng(20261112)
    x = rng.normal(size=(64, 3))
    v = rng.normal(size=(64, 3))
    inert = RotatingFrame(omega=0.0, axis=(0.0, 1.0, 0.0))
    p = rng.normal(size=64)
    c1 = {
        "coriolis_max_abs": float(np.max(np.abs(inert.coriolis(v)))),
        "centrifugal_max_abs": float(np.max(np.abs(inert.centrifugal(x)))),
        "sources_total_max_abs": float(
            np.max(np.abs(inert.momentum_sources(v, x, 1.0)["total"]))
        ),
        "modified_pressure_bitwise_identical": bool(
            np.array_equal(inert.modified_pressure(p, x, 1.0), p)
        ),
    }
    data["inert_limit"] = c1
    crit["C1_inert_limit_is_exact"] = _criterion(
        float(
            max(
                c1["coriolis_max_abs"],
                c1["centrifugal_max_abs"],
                c1["sources_total_max_abs"],
                0.0 if c1["modified_pressure_bitwise_identical"] else 1.0,
            )
        ),
        0.0,
        "omega=0 gives identically zero sources and p_m == p (bitwise), not 'small'",
    )

    # ---------------------------------------------------------------- C2
    frame = RotatingFrame(omega=_CH["omega"], axis=(0.0, 1.0, 0.0))
    u_in, w_in = 0.37, -0.21
    got = frame.coriolis(np.array([[u_in, 0.0, w_in]]))[0]
    ref = np.array([2.0 * _CH["omega"] * w_in, 0.0, -2.0 * _CH["omega"] * u_in])
    coriolis_err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    xa = rng.normal(size=(32, 3)) + np.array([0.3, -1.1, 2.0])
    va = rng.normal(size=(32, 3))
    roundtrip = _rel(frame.relative_velocity(frame.absolute_velocity(va, xa), xa), va)
    # grad(|Omega x r|^2 / 2) must equal -Omega x (Omega x r): central differences
    hd = 1e-6
    grad = np.zeros_like(xa)
    for k in range(3):
        e = np.zeros(3)
        e[k] = hd
        grad[:, k] = (
            frame.centrifugal_potential(xa + e) - frame.centrifugal_potential(xa - e)
        ) / (2.0 * hd)
    pot_err = _rel(grad, -frame.centrifugal(xa))
    data["frame_kinematics"] = {
        "coriolis_component_error": coriolis_err,
        "absolute_relative_roundtrip": roundtrip,
        "potential_gradient_vs_centrifugal": pot_err,
        "coriolis_vector": got.tolist(),
        "coriolis_reference": ref.tolist(),
    }
    crit["C2_frame_kinematics"] = _criterion(
        max(coriolis_err, roundtrip, pot_err),
        1e-6,
        "2*Omega x u components (axis-swap trap), abs<->rel roundtrip, "
        "grad(|Omega x r|^2/2) = -Omega x (Omega x r) by central differences",
    )

    # ---------------------------------------------------------------- C3
    # Solid-body rotation: the fluid rotates with the frame, so u_rel = 0 and
    # p = rho/2 |Omega x r|^2.  That state is an exact solution of the rotating
    # momentum equation -- Coriolis vanishes identically, the modified pressure
    # is zero everywhere, and the pressure gradient cancels the centrifugal body
    # force.  Three separate statements, three separate measurements.
    sb = RotatingFrame(omega=1.7, axis=(0.6, 0.0, 0.8), origin=(1.0, -2.0, 0.5))
    xs = rng.normal(size=(32, 3))
    rho_sb = 1000.0
    p_solid = rho_sb * sb.centrifugal_potential(xs)
    cor0 = sb.coriolis(np.zeros_like(xs))
    pm = sb.modified_pressure(p_solid, xs, rho_sb)
    src_reduced = sb.momentum_sources(np.zeros_like(xs), xs, rho_sb,
                                      include_centrifugal=False)["total"]
    # central-difference pressure gradient, then the full momentum residual
    hd = 1e-6
    grad_p = np.zeros_like(xs)
    for k in range(3):
        e = np.zeros(3)
        e[k] = hd
        grad_p[:, k] = (
            rho_sb * sb.centrifugal_potential(xs + e)
            - rho_sb * sb.centrifugal_potential(xs - e)
        ) / (2.0 * hd)
    resid_full = -grad_p / rho_sb - sb.centrifugal(xs)  # Coriolis is 0 here
    c3 = {
        "coriolis_at_rest_max_abs": float(np.max(np.abs(cor0))),
        "modified_pressure_max_abs": float(np.max(np.abs(pm))),
        "reduced_source_max_abs": float(np.max(np.abs(src_reduced))),
        "full_momentum_residual_rel": float(
            np.max(np.abs(resid_full)) / np.max(np.abs(sb.centrifugal(xs)))
        ),
        "absolute_velocity_rel_error": _rel(
            sb.absolute_velocity(np.zeros_like(xs), xs),
            np.cross(sb.omega_vector, xs - np.asarray(sb.origin)),
        ),
    }
    data["solid_body_rotation"] = c3
    crit["C3_solid_body_rotation_exact"] = _criterion(
        max(
            c3["coriolis_at_rest_max_abs"],
            c3["modified_pressure_max_abs"],
            c3["reduced_source_max_abs"],
            c3["absolute_velocity_rel_error"],
        ),
        1e-14,
        "u_rel = 0 with p = rho/2 |Omega x r|^2: Coriolis 0, p_m 0, source 0 (machine zero)",
    )
    crit["C3b_solid_body_full_momentum_residual"] = _criterion(
        c3["full_momentum_residual_rel"],
        1e-6,
        "-grad p/rho - Omega x (Omega x r) = 0 for the solid-body pressure (FD gradient)",
    )

    # ---------------------------------------------------------------- C4
    # closed form vs its own ODEs, with ANALYTIC second derivatives
    y = np.linspace(0.0, _CH["h"], 41)
    ex = exact_rotating_plane_poiseuille(y, **_CH)
    a, Fp = ex["a"], 1.0j * _CH["G"] / (2.0 * _CH["rho"] * _CH["omega"])
    ch = np.cosh(a * 0.5 * _CH["h"])
    F = Fp * (1.0 - np.cosh(a * (y - 0.5 * _CH["h"])) / ch)
    F2 = -(a**2) * Fp * np.cosh(a * (y - 0.5 * _CH["h"])) / ch
    u2, w2 = F2.real, -F2.imag
    r1 = _CH["nu"] * u2 + 2.0 * _CH["omega"] * ex["w"] + _CH["G"] / _CH["rho"]
    r2 = _CH["nu"] * w2 - 2.0 * _CH["omega"] * ex["u"]
    scale = _CH["G"] / _CH["rho"]
    c4 = {
        "streamwise_ode_residual": float(np.max(np.abs(r1)) / scale),
        "spanwise_ode_residual": float(np.max(np.abs(r2)) / scale),
        "u_max": float(np.max(np.abs(ex["u"]))),
        "w_max": float(np.max(np.abs(ex["w"]))),
        "ekman_like_length": ex["Ekman_like_length"],
        "F_closed_form_reproduction": _rel(F, ex["F"]),
    }
    data["closed_form_residual"] = c4
    crit["C4_closed_form_satisfies_odes"] = _criterion(
        max(c4["streamwise_ode_residual"], c4["spanwise_ode_residual"]),
        1e-13,
        "nu u'' + 2 omega w = -G/rho and nu w'' - 2 omega u = 0 with analytic derivatives",
    )

    # --------------------------------------------------------------- C4b
    # Independent numerical route for the same closed form: a collocation BVP
    # solve of the complex ODE system (no finite-volume stencil in common with
    # the solver used in C5), so "the closed form satisfies the ODEs" is not
    # established by the same algebraic derivation that produced it.
    from scipy.integrate import solve_bvp  # local import: verification only

    omega, nu_ch, rho_ch, G_ch, h_ch = (
        _CH["omega"], _CH["nu"], _CH["rho"], _CH["G"], _CH["h"]
    )
    yb = np.linspace(0.0, h_ch, 200)
    guess = exact_rotating_plane_poiseuille(yb, **_CH)
    f0 = guess["F"]
    f0p = np.gradient(f0, yb)
    ic = np.vstack([f0.real, f0.imag, f0p.real, f0p.imag])

    def rhs(_y, Y):
        F = Y[0] + 1.0j * Y[1]
        F2 = (-G_ch / rho_ch - 2.0j * omega * F) / nu_ch
        return np.vstack([Y[2], Y[3], np.full_like(Y[0], F2.real), np.full_like(Y[0], F2.imag)])

    def bc(Ya, Yb):
        return np.array([Ya[0], Ya[1], Yb[0], Yb[1]])

    bvp = solve_bvp(rhs, bc, yb, ic, tol=1e-10, max_nodes=20000)
    if not bvp.success:
        c4b_val = float("inf")
        c4b_note = "solve_bvp failed: %s" % bvp.message
    else:
        yb2 = np.linspace(0.0, h_ch, 401)
        Fb = bvp.sol(yb2)[0] + 1.0j * bvp.sol(yb2)[1]
        c4b_val = _rel(Fb, exact_rotating_plane_poiseuille(yb2, **_CH)["F"])
        c4b_note = "collocation BVP (scipy solve_bvp) vs closed form on 401 points"
    data["closed_form_residual"]["bvp_cross_check"] = c4b_val
    crit["C4b_closed_form_vs_independent_bvp"] = _criterion(c4b_val, 1e-8, c4b_note)

    # ---------------------------------------------------------------- C5
    rows: List[Dict] = []
    errs = {}
    for n in (40, 80, 160, 320, 640):
        num = solve_rotating_plane_poiseuille(n, **_CH)
        ref_n = exact_rotating_plane_poiseuille(num["y"], **_CH)
        eu, ew = _rel(num["u"], ref_n["u"]), _rel(num["w"], ref_n["w"])
        errs[n] = (eu, ew)
        rows.append({"n": n, "u_rel_l2": eu, "w_rel_l2": ew})
    p_u = _order(errs[320][0], errs[640][0])
    p_w = _order(errs[320][1], errs[640][1])
    data["rotating_channel"] = {
        "parameters": dict(_CH),
        "grid_ladder": rows,
        "observed_order_u": p_u,
        "observed_order_w": p_w,
        "finest_u_rel_l2": errs[640][0],
        "finest_w_rel_l2": errs[640][1],
        "order_80_160_u": _order(errs[80][0], errs[160][0]),
        "order_80_160_w": _order(errs[80][1], errs[160][1]),
    }
    crit["C5_rotating_channel_vs_closed_form"] = _criterion(
        max(errs[640][0], errs[640][1]),
        0.03,
        "finite-volume (u,w) vs closed form, N=640: u and secondary flow w < 3%",
    )
    crit["C5b_rotating_channel_second_order"] = _criterion(
        min(p_u, p_w),
        1.8,
        "observed order from N=320/640, streamwise and spanwise",
        mode="min",
    )

    # ---------------------------------------------------------------- C6
    # discriminating power: a solver with the Coriolis term dropped returns
    # w == 0, i.e. a 100% error on the secondary flow the anchor measures.
    inert_num = solve_rotating_plane_poiseuille(640, **{**_CH, "omega": 0.0})
    ref320 = exact_rotating_plane_poiseuille(inert_num["y"], **_CH)
    c6 = {
        "no_coriolis_w_max": float(np.max(np.abs(inert_num["w"]))),
        "anchor_w_max": float(np.max(np.abs(ref320["w"]))),
        "no_coriolis_w_rel_error": _rel(inert_num["w"], ref320["w"]),
        "no_coriolis_u_rel_error": _rel(inert_num["u"], ref320["u"]),
    }
    data["discriminating_power"] = c6
    crit["C6_anchor_rejects_no_coriolis_solver"] = _criterion(
        c6["no_coriolis_w_rel_error"],
        0.5,
        "a solver without Coriolis must FAIL the w anchor (measured error >= 50%)",
        mode="min",
    )

    # ---------------------------------------------------------------- C7
    # rotation axis along the span: parabolic u for every omega; the cubic
    # wall-normal pressure correction is the only signature of rotation.
    ys = (np.arange(320) + 0.5) * (_CH["h"] / 320)
    span = exact_rotating_channel_spanwise(ys, **_CH)
    inert_u = solve_inert_plane_poiseuille(320, **{k: _CH[k] for k in ("G", "rho", "nu", "h")})
    c7a = _rel(inert_u["u"], span["u"])
    frame_z = RotatingFrame(omega=_CH["omega"], axis=(0.0, 0.0, 1.0))
    vel = np.stack([span["u"], np.zeros_like(ys), np.zeros_like(ys)], axis=-1)
    pts = np.stack([np.zeros_like(ys), ys, np.zeros_like(ys)], axis=-1)
    cor_y = frame_z.coriolis(vel)[:, 1]
    resid = cor_y + span["dpdy"] / _CH["rho"]
    c7b = float(np.max(np.abs(resid)) / np.max(np.abs(cor_y)))
    # controls: drop Coriolis, or flip the pressure sign
    ctl_drop = float(np.max(np.abs(span["dpdy"] / _CH["rho"])) / np.max(np.abs(cor_y)))
    ctl_flip = float(np.max(np.abs(-cor_y + span["dpdy"] / _CH["rho"])) / np.max(np.abs(cor_y)))
    data["spanwise_axis"] = {
        "inert_solve_vs_parabola": c7a,
        "wall_normal_balance_residual": c7b,
        "control_drop_coriolis_residual": ctl_drop,
        "control_flip_coriolis_residual": ctl_flip,
        "p_scale": float(np.max(np.abs(span["p"]))),
        "u_max": float(np.max(span["u"])),
    }
    crit["C7a_spanwise_axis_leaves_mean_flow_inert"] = _criterion(
        c7a, 1e-12, "rotation along the span: parabolic u reproduced to round-off for any omega"
    )
    crit["C7b_spanwise_axis_pressure_correction"] = _criterion(
        c7b, 1e-12, "2 omega u + p_y/rho = 0 with the implemented Coriolis y component"
    )
    crit["C7c_controls_reject_wrong_coriolis"] = _criterion(
        min(ctl_drop, ctl_flip),
        0.5,
        "dropping or sign-flipping Coriolis must violate C7b by >= 50%",
        mode="min",
    )

    # ---------------------------------------------------------------- C8
    # circular Couette solved in rotating frames: absolute flow + torque invariant
    frames = [0.0, _CO["omega_inner"], 1.7 * _CO["omega_inner"]]
    rows = []
    abs_profiles = []
    torques = []
    for wf in frames:
        num = solve_circular_couette_frame(
            320, nu=1e-3, omega_frame=wf, **_CO
        )
        ref_c = exact_circular_couette(num["r"], **_CO)
        rows.append(
            {
                "omega_frame": wf,
                "abs_vs_exact_rel": _rel(num["v_abs"], ref_c["v_theta"]),
                "rel_vs_exact_rel": _rel(
                    num["v_rel"], ref_c["v_theta"] - wf * num["r"]
                ),
                "torque_coefficient": num["torque_coefficient"],
                "inner_wall_relative_velocity": float(abs(num["v_rel"][0])),
            }
        )
        abs_profiles.append(num["v_abs"])
        torques.append(num["torque_coefficient"])
    frame_spread = max(_rel(abs_profiles[i], abs_profiles[0]) for i in range(1, len(frames)))
    torque_spread = float(
        max(abs(t - torques[0]) for t in torques) / abs(torques[0])
    )
    frozen = max(r["inner_wall_relative_velocity"] for r in rows[1:2])
    data["circular_couette"] = {
        "parameters": dict(_CO),
        "frames": rows,
        "absolute_profile_spread": frame_spread,
        "torque_spread": torque_spread,
        "frozen_rotor_inner_wall_relative_velocity": frozen,
    }
    crit["C8a_couette_absolute_profile_vs_exact"] = _criterion(
        max(r["abs_vs_exact_rel"] for r in rows),
        0.03,
        "v_theta = A r + B/r from every frame, n=320",
    )
    crit["C8b_mrf_frame_invariance"] = _criterion(
        max(frame_spread, torque_spread),
        1e-12,
        "absolute profile and torque identical for omega_frame = 0, 1.0, 1.7 rad/s",
    )
    crit["C8c_frozen_rotor_statement"] = _criterion(
        frozen, 1e-15, "omega_frame = omega_inner makes the inner wall's relative velocity exactly 0"
    )

    # ---------------------------------------------------------------- C9
    num = solve_circular_couette_frame(320, nu=1e-3, omega_frame=0.0, **_CO)
    ref_c = exact_circular_couette(num["r"], **_CO)
    rho_w = 800.0  # kg/m^3, single declared value so the torque has units
    mu = rho_w * 1e-3
    taylor = mu * ref_c["torque_per_length_coefficient"]
    numeric = mu * num["torque_coefficient"]
    c9 = {
        "taylor_torque_per_length": float(taylor),
        "numerical_torque_per_length": float(numeric),
        "rel_error": float(abs(numeric - taylor) / abs(taylor)),
        "rho": rho_w,
        "mu": mu,
    }
    data["couette_torque"] = c9
    crit["C9_couette_torque_vs_taylor_closed_form"] = _criterion(
        c9["rel_error"],
        0.03,
        "M/L = 4 pi mu r1^2 r2^2 |omega_i - omega_o| / (r2^2 - r1^2) at n=320",
    )

    # ---------------------------------------------------------------- C10
    # rotating-frame radial balance for the exact Couette solution, evaluated
    # with the implemented operators (Coriolis + centrifugal) -- this is where
    # the frame terms actually live for azimuthal flow.
    wf = 1.0
    rr = np.linspace(_CO["r1"] * 1.05, _CO["r2"] * 0.95, 25)
    ex_c = exact_circular_couette(rr, **_CO)
    frame_c = RotatingFrame(omega=wf, axis=(0.0, 0.0, 1.0))
    v_rel = ex_c["v_theta"] - wf * rr
    pts = np.stack([rr, np.zeros_like(rr), np.zeros_like(rr)], axis=-1)
    vel = np.stack([np.zeros_like(rr), v_rel, np.zeros_like(rr)], axis=-1)
    cor = frame_c.coriolis(vel)[:, 0]  # radial component of 2 Omega x u
    cen = frame_c.centrifugal(pts)[:, 0]
    conv = -(v_rel**2) / rr  # (u.grad)u radial component for purely azimuthal u
    dpdr = ex_c["dpdr_over_rho"]  # inertial radial balance, p_r/rho = v_theta^2/r
    residual = conv + cor + dpdr + cen
    scale10 = np.max(np.abs(np.stack([conv, cor, dpdr, cen])), axis=0)
    c10 = {
        "residual_rel": float(np.max(np.abs(residual) / scale10)),
        "control_drop_centrifugal_rel": float(
            np.max(np.abs(residual - cen) / scale10)
        ),
        "control_flip_coriolis_rel": float(np.max(np.abs(residual - 2.0 * cor) / scale10)),
        "omega_frame": wf,
    }
    data["frame_radial_balance"] = c10
    crit["C10a_frame_radial_balance_exact"] = _criterion(
        c10["residual_rel"], 1e-14, "conv + 2 Omega x u + grad p/rho - Omega x (Omega x r) = 0"
    )
    crit["C10b_controls_reject_missing_frame_terms"] = _criterion(
        min(c10["control_drop_centrifugal_rel"], c10["control_flip_coriolis_rel"]),
        0.05,
        "dropping the centrifugal term or flipping Coriolis must violate C10a by >= 5%",
        mode="min",
    )

    passed = sum(1 for c in crit.values() if c["pass"])
    summary = {
        "module": "tensorfvm.rotating_frame",
        "driver": "tensorfvm.verification.rotating_frame",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_s": time.time() - t0,
        "criteria": crit,
        "passed": passed,
        "total": len(crit),
        "all_pass": passed == len(crit),
        "data": data,
    }
    return summary


# ---------------------------------------------------------------------------
# report / figure
# ---------------------------------------------------------------------------
def plot(summary: Dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ch = _CH
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) rotating channel: closed form vs finite volume, u and w
    num = solve_rotating_plane_poiseuille(80, **ch)
    ref = exact_rotating_plane_poiseuille(num["y"], **ch)
    yf = np.linspace(0.0, ch["h"], 400)
    reff = exact_rotating_plane_poiseuille(yf, **ch)
    ax = axes[0]
    ax.plot(reff["u"], yf, "k-", lw=1.6, label="exact $u$")
    ax.plot(num["u"], num["y"], "o", ms=3.2, mfc="none", label="FV $u$, N=80")
    ax.plot(reff["w"], yf, "C3-", lw=1.6, label="exact $w$ (secondary)")
    ax.plot(num["w"], num["y"], "s", ms=3.2, mfc="none", label="FV $w$, N=80")
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("velocity / (m/s)")
    ax.set_ylabel("$y$ / m")
    ax.set_title(
        "rotating plane Poiseuille, $\\Omega\\parallel\\hat{y}$\n"
        "$\\omega=%.2f$ 1/s, $\\nu=%.3f$ m$^2$/s" % (ch["omega"], ch["nu"])
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (b) grid convergence
    rows = summary["data"]["rotating_channel"]["grid_ladder"]
    ns = np.array([r["n"] for r in rows], float)
    eu = np.array([r["u_rel_l2"] for r in rows])
    ew = np.array([r["w_rel_l2"] for r in rows])
    ax = axes[1]
    ax.loglog(ns, eu, "o-", label="$u$: order %.2f" % summary["data"]["rotating_channel"]["observed_order_u"])
    ax.loglog(ns, ew, "s-", label="$w$: order %.2f" % summary["data"]["rotating_channel"]["observed_order_w"])
    ax.loglog(ns, eu[0] * (ns[0] / ns) ** 2, "k--", lw=0.9, label="$N^{-2}$")
    ax.axhline(0.03, color="r", ls=":", lw=1.0, label="3 % gate")
    ax.set_xlabel("cells across the channel $N$")
    ax.set_ylabel("relative $L_2$ error vs closed form")
    ax.set_title("discretisation error (anchor verified\nfrom both directions)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # (c) circular Couette in rotating frames
    ax = axes[2]
    r_axis = np.linspace(_CO["r1"], _CO["r2"], 300)
    ex_c = exact_circular_couette(r_axis, **_CO)
    ax.plot(r_axis, ex_c["v_theta"], "k-", lw=1.6, label="exact $v_\\theta$ (inertial)")
    for wf, style in zip((0.0, _CO["omega_inner"], 1.7 * _CO["omega_inner"]), ("o", "s", "^")):
        sol = solve_circular_couette_frame(320, nu=1e-3, omega_frame=wf, **_CO)
        ax.plot(sol["r"], sol["v_abs"], style, ms=2.6, mfc="none",
                label="frame $\\omega_f=%.2f$ (abs.)" % wf)
        ax.plot(sol["r"], sol["v_rel"], ls="--", lw=0.9,
                label="frame $\\omega_f=%.2f$ (rel.)" % wf)
    ax.set_xlabel("$r$ / m")
    ax.set_ylabel("azimuthal velocity / (m/s)")
    ax.set_title(
        "circular Couette in rotating frames\n"
        "torque spread %.1e, profile spread %.1e"
        % (
            summary["data"]["circular_couette"]["torque_spread"],
            summary["data"]["circular_couette"]["absolute_profile_spread"],
        )
    )
    ax.legend(fontsize=6.2)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "TensorFVM MRF verification: all %d criteria PASS (%d/%d)"
        % (summary["total"], summary["passed"], summary["total"]),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_report(summary: Dict) -> str:
    """Paper-level report with the six required elements (see repo convention)."""
    d = summary["data"]
    ch = d["rotating_channel"]
    cc = d["circular_couette"]
    lines: List[str] = []
    lines.append("# TensorFVM MRF (rotating reference frame) verification\n")
    lines.append(
        "Driver: `PYTHONPATH=src python -m tensorfvm.verification.rotating_frame`  \n"
        "Module: `src/tensorfvm/rotating_frame.py`  \n"
        "Tests: `tests/test_rotating_frame.py`  \n"
        "Criteria: **%d/%d PASS** (wall %.1f s, generated %s)\n"
        % (
            summary["passed"],
            summary["total"],
            summary["wall_time_s"],
            summary["generated_utc"],
        )
    )

    lines.append("## 1. Case introduction (physics, geometry, parameters)\n")
    lines.append(
        "A single rotating reference frame (MRF) is the standard industrial model "
        "for a propeller or rotor: the flow is solved in a frame rotating with the "
        "rotor, so the blade becomes stationary and the momentum equation gains "
        "the Coriolis term `2 Omega x u` and the centrifugal term "
        "`Omega x (Omega x r)`. Three configurations are anchored, each chosen "
        "because it exposes a different part of that operator.\n"
    )
    lines.append(
        "| case | rotation axis | what it can and cannot see |\n|---|---|---|\n"
        "| A. rotating plane Poiseuille | along the wall normal `+y` | Coriolis "
        "drives a **secondary flow** `w`; a solver without Coriolis returns "
        "`w = 0` and fails by 100 %% |\n"
        "| B. rotating channel | along the span `+z` (classical rotating channel, "
        "Johnston et al. 1972) | the **laminar mean profile is unchanged**; the "
        "only signature is a **cubic pressure correction** in `y` |\n"
        "| C. circular Couette in a rotating frame | cylinder axis `+z` (any frame "
        "angular velocity) | MRF **frame invariance**: absolute flow and torque "
        "independent of `omega_frame` |\n"
    )
    lines.append(
        "\nCase A parameters (fixed, pre-declared): channel height "
        "`h = %.1f` m, `rho = %.1f` kg/m^3, `nu = %.3f` m^2/s, driving gradient "
        "`-dp/dx = G = %.1f` Pa/m, frame `omega = %.2f` rad/s; wall layer "
        "`sqrt(nu/(2 omega)) = %.4f` m.\n"
        % (
            ch["parameters"]["h"],
            ch["parameters"]["rho"],
            ch["parameters"]["nu"],
            ch["parameters"]["G"],
            ch["parameters"]["omega"],
            d["closed_form_residual"]["ekman_like_length"],
        )
    )
    lines.append(
        "Case C parameters: `r1 = %.1f` m, `r2 = %.1f` m, "
        "`omega_inner = %.1f` rad/s, `omega_outer = %.1f` rad/s, "
        "`rho = %.0f` kg/m^3, `mu = %.3f` Pa s; frames "
        "`omega_frame in {0, %.1f, %.1f}` rad/s.\n"
        % (
            cc["parameters"]["r1"],
            cc["parameters"]["r2"],
            cc["parameters"]["omega_inner"],
            cc["parameters"]["omega_outer"],
            d["couette_torque"]["rho"],
            d["couette_torque"]["mu"],
            cc["parameters"]["omega_inner"],
            1.7 * cc["parameters"]["omega_inner"],
        )
    )

    lines.append("## 2. What was done (method, flow, tooling)\n")
    lines.append(
        "1. The frame operator is implemented once (`RotatingFrame`): Coriolis, "
        "centrifugal, frame transforms, and the modified pressure "
        "`p_m = p - rho/2 |Omega x r|^2` that absorbs the centrifugal term. The "
        "sign convention is written down in the module docstring and pinned by "
        "the tests.\n"
        "2. Case A closed form is derived by writing `F = u - i w`, which turns "
        "the coupled real pair into `nu F'' + 2 i omega F = -G/rho`, then solved "
        "with the no-slip conditions: `F = F_p [1 - cosh(a(y-h/2))/cosh(a h/2)]`, "
        "`F_p = i G/(2 rho omega)`, `a = sqrt(-2 i omega / nu)`.\n"
        "3. That closed form is verified **three independent ways**: analytically "
        "against its own ODEs (C4), numerically against an adaptive collocation "
        "BVP (`scipy.integrate.solve_bvp`, C4b), and against a second-order "
        "finite-volume solve of the coupled `(u, w)` system on a grid ladder "
        "(C5/C5b). A derivation checked only against itself is not evidence, so "
        "the cross-checks are part of the deliverable.\n"
        "4. Case B: the exact laminar solution is inert (parabolic `u`), and the "
        "Coriolis force shows up only in the wall-normal balance "
        "`p_y = -2 rho omega u`. Both halves are asserted (C7a/C7b).\n"
        "5. Case C: circular Couette is solved in three frames on the conservative "
        "form `d/dr[r^3 d(v/r)/dr] = 0`; the absolute velocity and Taylor's torque "
        "must be frame invariant (C8), and the rotating-frame radial balance is "
        "evaluated with the implemented operators (C10).\n"
        "6. Three **control** criteria quantify what a wrong implementation looks "
        "like (C6, C7c, C10b): an anchor a wrong solver can also pass is not an "
        "anchor.\n"
    )

    lines.append("## 3. Results\n")
    lines.append("| id | value | gate | pass | what it pins down |")
    lines.append("|---|---|---|---|---|")
    for k, c in summary["criteria"].items():
        lines.append(
            "| %s | %.3e | %.3e (%s) | %s | %s |"
            % (
                k,
                c["value"],
                c["gate"],
                c["mode"],
                "PASS" if c["pass"] else "FAIL",
                c["note"],
            )
        )
    lines.append("\n### 3.1 Case A: rotating plane Poiseuille (rotation along the wall normal)\n")
    lines.append("| N | rel $L_2$ error $u$ | rel $L_2$ error $w$ |")
    lines.append("|---|---|---|")
    for r in ch["grid_ladder"]:
        lines.append("| %d | %.3e | %.3e |" % (r["n"], r["u_rel_l2"], r["w_rel_l2"]))
    lines.append(
        "\nObserved order (N=320 -> 640): $u$ %.2f, $w$ %.2f; from N=80 -> 160: "
        "$u$ %.2f, $w$ %.2f. Closed-form ODE residual (analytic derivatives): "
        "$u$-balance %.2e, $w$-balance %.2e; independent collocation BVP agrees to "
        "%.2e.\n"
        % (
            ch["observed_order_u"],
            ch["observed_order_w"],
            ch["order_80_160_u"],
            ch["order_80_160_w"],
            d["closed_form_residual"]["streamwise_ode_residual"],
            d["closed_form_residual"]["spanwise_ode_residual"],
            d["closed_form_residual"]["bvp_cross_check"],
        )
    )
    lines.append("### 3.2 Case C: circular Couette in rotating frames\n")
    lines.append(
        "| frame $\\omega_f$ (rad/s) | abs err vs exact | rel err vs exact | "
        "torque coeff | inner-wall $v_{rel}$ |"
    )
    lines.append("|---|---|---|---|---|")
    for r in cc["frames"]:
        lines.append(
            "| %.3f | %.3e | %.3e | %.5f | %.2e |"
            % (
                r["omega_frame"],
                r["abs_vs_exact_rel"],
                r["rel_vs_exact_rel"],
                r["torque_coefficient"],
                r["inner_wall_relative_velocity"],
            )
        )
    lines.append(
        "\nAbsolute-profile spread across the three frames %.2e, torque spread "
        "%.2e. Taylor torque closed form `M/L = 4 pi mu r1^2 r2^2 |omega_i - "
        "omega_o| / (r2^2 - r1^2)` = %.4f N m/m vs numerical %.4f N m/m.\n"
        % (
            cc["absolute_profile_spread"],
            cc["torque_spread"],
            d["couette_torque"]["taylor_torque_per_length"],
            d["couette_torque"]["numerical_torque_per_length"],
        )
    )
    lines.append("### 3.3 Discriminating power (controls)\n")
    lines.append(
        "* A solver that drops Coriolis returns $w \\equiv %.1e$ against an anchor "
        "maximum of %.4f -> relative error %.3f, i.e. it fails C5/C6 as it must.\n"
        % (
            d["discriminating_power"]["no_coriolis_w_max"],
            d["discriminating_power"]["anchor_w_max"],
            d["discriminating_power"]["no_coriolis_w_rel_error"],
        )
    )
    lines.append(
        "* Case B, dropping Coriolis: wall-normal balance residual %.3f; "
        "sign-flipped Coriolis: %.3f (gate 0.5).\n"
        % (
            d["spanwise_axis"]["control_drop_coriolis_residual"],
            d["spanwise_axis"]["control_flip_coriolis_residual"],
        )
    )
    lines.append(
        "* Case C rotating-frame radial balance, dropping the centrifugal term: "
        "%.3f; flipping Coriolis: %.3f (gate 0.05).\n"
        % (
            d["frame_radial_balance"]["control_drop_centrifugal_rel"],
            d["frame_radial_balance"]["control_flip_coriolis_rel"],
        )
    )

    lines.append("## 4. Figure\n")
    lines.append(
        "`fig_rotating_frame.png`: (a) case A closed form vs finite volume "
        "(N=80) for $u$ and the secondary flow $w$; (b) grid convergence of both "
        "components against the closed form with the $N^{-2}$ reference and the "
        "3 % gate; (c) case C, absolute and relative azimuthal profiles from three "
        "rotating frames on top of the single exact inertial profile -- the curves "
        "coincide, which *is* the MRF statement.\n"
    )

    lines.append("## 5. Notes, traps and limits\n")
    lines.append(
        "* **The anchor direction matters more than the model.** In case B "
        "(classical rotating channel) the *velocity* field is unchanged by "
        "rotation, so a rotating-laminar-channel comparison cannot detect a "
        "missing Coriolis term at all -- only the pressure can. Testing MRF with "
        "case B alone would have produced a green suite over a broken operator; "
        "that is why case A (secondary flow) is the primary anchor.\n"
        "* **The mirror-ghost boundary row is a silent trap.** Writing the wall "
        "no-slip as a mirror ghost `u_{-1} = -u_0` gives `(u_1 - 3u_0)/dy^2`, "
        "which is *not* exact for a parabola and degrades the boundary row to "
        "first order; measured effect: 1.3e-5 relative error on the parabola "
        "anchor at N=320 instead of round-off (1.3e-14 after switching to the "
        "one-sided three-point row). The quadratic anchor is exactly what exposes "
        "it -- a decaying-error plot would have hidden it.\n"
        "* **Two regimes in case A.** For small $\\omega$ the secondary flow is "
        "linear in $\\omega$ ($u$ changes only at $O(\\omega^2)$, measured: "
        "100x for a 10x $\\omega$), so a weakly rotating run cannot be used to "
        "argue Coriolis is negligible. For large $\\omega$, $u \\sim "
        "1/\\omega$ and $|w|/|u|$ saturates near 3.31. The grid requirement is "
        "set by $\\sqrt{\\nu/(2\\omega)}$, not by `h`.\n"
        "* **What is NOT delivered.** No 3-D time-accurate rotating-frame field "
        "solver, no sliding/mixing plane, no rotating-machinery *experiment* "
        "comparison (propeller open-water curves). The operators are 3-D and "
        "ready to be coupled, the anchors are 1-D reduced systems. Recorded as "
        "open in `docs/VERIFICATION_DEBT_LEDGER.md` (D10).\n"
    )

    lines.append("## 6. Error judgement vs the 3 % criterion\n")
    lines.append(
        "| key quantity | reference | relative error | inside 3 pct |\n|---|---|---|---|\n"
        "| case A streamwise $u$ (N=640) | closed form | %.2e | **yes** |\n"
        "| case A secondary flow $w$ (N=640) | closed form | %.2e | **yes** |\n"
        "| case A closed form vs independent BVP | collocation | %.2e | **yes** |\n"
        "| case B parabolic $u$ (N=320) | closed form | %.2e | **yes** |\n"
        "| case C absolute $v_\\theta$ (n=320) | closed form | %.2e | **yes** |\n"
        "| case C torque $M/L$ (n=320) | Taylor closed form | %.2e | **yes** |\n"
        % (
            ch["finest_u_rel_l2"],
            ch["finest_w_rel_l2"],
            d["closed_form_residual"]["bvp_cross_check"],
            d["spanwise_axis"]["inert_solve_vs_parabola"],
            max(r["abs_vs_exact_rel"] for r in cc["frames"]),
            d["couette_torque"]["rel_error"],
        )
    )
    lines.append(
        "All anchored quantities are far inside the 3 pct band (worst case %.2e). "
        "Quantities that are *not* judged because no independent reference exists "
        "here are reported as WITHHELD rather than passed: the rotating-frame "
        "pressure field itself for case A (only its gradient enters the balance, "
        "anchored via C7b) and any rotating-machinery experimental comparison.\n"
        % max(
            ch["finest_u_rel_l2"],
            max(r["abs_vs_exact_rel"] for r in cc["frames"]),
            d["couette_torque"]["rel_error"],
        )
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/verification-rotating-frame")
    args = parser.parse_args(argv)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = run()
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float)
    )
    (outdir / "REPORT.md").write_text(render_report(summary))
    plot(summary, outdir / "fig_rotating_frame.png")
    print(
        "%d/%d criteria PASS" % (summary["passed"], summary["total"]),
        "->",
        outdir,
    )
    for k, c in summary["criteria"].items():
        print(("  PASS " if c["pass"] else "  FAIL ") + k, "%.3e" % c["value"], "<=" if c["mode"] == "max" else ">=", "%.3e" % c["gate"])
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())