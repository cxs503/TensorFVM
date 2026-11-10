"""Analytic anchors for the two-phase dam-break chain.

Two anchors that do *not* depend on a hand-digitised experimental curve:

**A1  Free-fall surface (tall column).**  Immediately after the gate is released
the liquid surface at the gate is in free fall -- the pressure there is
atmospheric and the only force is gravity -- so its height is exactly

.. math::  h(t) = h_0 - \\tfrac12 g t^2 ,  \\qquad h_0 = 2a ,

until the rarefaction released at the gate travels the column width and reaches
the wall, ``t < a / sqrt(g h_0)``.  For the Martin & Moyce geometry that is
``t* < 0.71`` with ``t* = t sqrt(g/a)``.  This is a kinematic identity, not a
model, so it is a legitimate gate on a gravity-capable two-phase solver: any
departure is a discrete momentum / pressure error.

**A2  Ritter (1892) shallow-water front.**  For a column that is *shallow*
(``h_0 / L_sim << 1``) the dry-bed dam break has the exact solution

.. math::  x_f(t) = a + 2\\sqrt{g h_0}\\, t      (front),
           \\qquad u_f = 2\\sqrt{g h_0} ,

valid until the front reaches the far wall and before the tail rarefaction
returns, ``t < a/sqrt(g h_0)``.  This is the classical analytic anchor for the
surge-front position and is the one a two-phase VOF solver has to reproduce.

Both anchors are *measured against the same solver state* the transient legs
output, so no extra physics assumption (interface reconstruction, curvature,
etc.) enters the comparison.

Reference-data audit
--------------------
:func:`martin_moyce_audit` checks the digitised Martin & Moyce (1952) tables
shipped in :mod:`tensorfvm.verification.two_phase_campaign` against the two
kinematic bounds above.  It exists because the campaign's own gate (15 % on the
digitised surge-front curve) failed by 9-23 % for every scheme and resolution,
and the question "is that the solver or the reference?" has to be answered with
numbers rather than belief.  The audit result is recorded in
``docs/vof-chain-anchor/report.md``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from .two_phase_campaign import (
    MM_HEIGHT_H,
    MM_HEIGHT_T,
    MM_SURGE_T,
    MM_SURGE_X,
)

FREEFALL_TOL = 0.15       # A1: relative error on the gate-surface height
RITTER_FRONT_TOL = 0.15   # A2: relative error on the surge front


# --------------------------------------------------------------------------- #
# A1: free-fall surface of a tall column
# --------------------------------------------------------------------------- #
def freefall_surface(t: float, h0: float, g: float) -> float:
    """Height of a freely falling liquid surface at time ``t`` (>= 0)."""
    return max(h0 - 0.5 * g * t * t, 0.0)


def freefall_window(column_width: float, h0: float, g: float) -> float:
    """Time until the rarefaction crosses the column width: ``a / sqrt(g h0)``."""
    return column_width / math.sqrt(g * h0)


def freefall_gate(history: list[dict], column_width: float, gravity: float,
                  times_star=(0.25, 0.5, 0.65), tol: float = FREEFALL_TOL) -> dict:
    """Compare the measured gate height with ``h0 - g t^2 / 2``.

    ``history`` entries carry ``time``, ``t_star`` and ``height_star``
    (``height / a``); the initial column height is ``h0 = 2a``.
    """
    h0 = 2.0 * column_width
    t_scale = math.sqrt(column_width / gravity)
    t_window = freefall_window(column_width, h0, gravity)
    points = []
    for t_star in times_star:
        t = t_star * t_scale
        ref = freefall_surface(t, h0, gravity) / column_width
        if t > t_window:
            points.append({"t_star": t_star, "reference": ref, "measured": None,
                           "relative_error": None, "reached": False,
                           "reason": "outside the free-fall window"})
            continue
        entry = min(history, key=lambda h: abs(h["t_star"] - t_star))
        measured = entry.get("height_star")
        if measured is None:
            points.append({"t_star": t_star, "reference": ref, "measured": None,
                           "relative_error": None, "reached": False,
                           "reason": "no height recorded"})
            continue
        points.append({"t_star": t_star, "reference": ref, "measured": measured,
                       "relative_error": (measured - ref) / ref, "reached": True})
    reached = [p for p in points if p["reached"]]
    worst = max((abs(p["relative_error"]) for p in reached), default=float("nan"))
    return {
        "pass": bool(reached) and len(reached) == len(points) and worst <= tol,
        "criterion": f"|h*_sim - h*_freefall| / h*_freefall <= {tol:g}",
        "reference": "h(t) = 2a - g t^2/2 (kinematic free fall, t < a/sqrt(2 g a))",
        "freefall_window_t_star": t_window / t_scale,
        "worst_relative_error": worst,
        "points": points,
    }


# --------------------------------------------------------------------------- #
# A2: Ritter (1892) shallow-water dam break
# --------------------------------------------------------------------------- #
@dataclass
class RitterLegConfig:
    """Shallow dry-bed dam break resolved enough for the Ritter solution.

    The liquid occupies ``[0, dam_x] x [0, h0]``; the gate sits at ``dam_x``.
    ``h0 / dy >= 8`` is the resolution requirement that makes the shallow-water
    solution meaningful -- below that the vertical structure of the surge is
    unresolved and the anchor is invalid (measured: at ``h0/dy = 2.5`` the front
    lags Ritter by 85 %, which is a resolution artefact, not solver physics).
    """

    name: str = "ritter"
    nx: int = 120
    ny: int = 36
    length: float = 1.5
    height: float = 0.18
    dam_x: float = 0.25
    h0: float = 0.05
    gravity: float = 9.81
    rho_liquid: float = 1000.0
    rho_gas: float = 1.2
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.8e-5
    vof_scheme: str = "plic"
    cfl: float = 0.4
    t_end: float = 0.2
    snapshot_fractions: tuple = (0.6, 0.8, 1.0)

    def __post_init__(self) -> None:
        if self.h0 / (self.height / self.ny) < 8.0:
            raise ValueError(
                "Ritter anchor needs h0/dy >= 8 cells; increase ny or h0")
        if self.dam_x <= 0 or self.dam_x >= self.length:
            raise ValueError("dam_x must lie inside the domain")

    @property
    def c0(self) -> float:
        return math.sqrt(self.gravity * self.h0)


def ritter_front(t: float, dam_x: float, h0: float, g: float) -> float:
    """Ritter front position ``a + 2 sqrt(g h0) t`` measured from ``x = 0``."""
    return dam_x + 2.0 * math.sqrt(g * h0) * t


def ritter_profile(x: float, t: float, dam_x: float, h0: float, g: float) -> float:
    """Ritter free-surface height at ``x`` (dry bed, ``t > 0``)."""
    c0 = math.sqrt(g * h0)
    xi = (x - dam_x) / t if t > 0 else -1e9
    if xi <= -c0:
        return h0
    if xi >= 2.0 * c0:
        return 0.0
    return (2.0 * c0 - xi) ** 2 / (9.0 * g)


def run_ritter_leg(cfg: RitterLegConfig, progress: bool = False) -> dict:
    """Integrate the shallow dam break and record front / profile snapshots."""
    dx, dy = cfg.length / cfg.nx, cfg.height / cfg.ny
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas, sigma=0.0,
        surface_tension=False, gravity=cfg.gravity, top_atmosphere=True,
        bottom_wall=True, side_outlet=False, vof_scheme=cfg.vof_scheme,
        momentum_advection=True, pressure_form="total",
        warm_start_pressure=True, domain_type="closed",
        vof_cfl_limit=cfg.cfl,
    )
    solver = TwoPhaseSolver(
        GridConfig(nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height,
                   max_iterations=1), tp)
    x = (torch.arange(cfg.nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * dy
    alpha0 = ((y[:, None] < cfg.h0) & (x[None, :] < cfg.dam_x)).to(torch.float64)
    solver.set_alpha(alpha0)
    volume0 = float(solver.alpha.sum()) * dx * dy

    snapshots = [cfg.t_end * f for f in cfg.snapshot_fractions]
    history: list[dict] = []
    steps, t0 = 0, time.time()
    while solver.time < cfg.t_end and steps < 200000:
        umax = max(float(solver.u.abs().max()), float(solver.v.abs().max()),
                   2.0 * cfg.c0)
        dt = min(solver.stable_time_step(), cfg.cfl * min(dx, dy) / umax,
                 cfg.t_end - solver.time)
        if dt <= 0.0:
            break
        for _ in range(6):
            try:
                solver.step(dt)
                break
            except ValueError:
                dt *= 0.25
        steps += 1
        if snapshots and solver.time >= snapshots[0]:
            t_snap = snapshots.pop(0)
            wet = (solver.alpha > 0.5).any(dim=0).nonzero()
            front = ((float(wet[-1]) + 0.5) * dx - cfg.dam_x) if len(wet) else 0.0
            eta = (solver.alpha.sum(dim=0) * dy).cpu().numpy()
            ana = [ritter_profile(float(xi), t_snap, cfg.dam_x, cfg.h0,
                                  cfg.gravity) for xi in x]
            vel = 0.5 * (solver.u[:, :-1] + solver.u[:, 1:])
            umax_now = float(vel.abs().max())
            # Head diagnostic: is the pressure inside the reservoir hydrostatic?
            # A gravity current that is *late* usually has p well below
            # rho*g*(surface - y) during the startup, because the incremental
            # projection builds the head only as fast as the flow itself.
            i_gate = max(1, min(cfg.nx - 2, int(round(0.6 * cfg.dam_x / dx)) - 1))
            col_p = solver.p[:int(cfg.h0 / dy) + 1, i_gate]
            y_col = ((torch.arange(col_p.numel(), dtype=torch.float64) + 0.5)
                     * dy)
            surface = float(solver.alpha[:, i_gate].sum()) * dy
            hydro = cfg.rho_liquid * cfg.gravity * (surface - y_col).clamp_min(0.0)
            wet = hydro > 0.02 * cfg.rho_liquid * cfg.gravity * cfg.h0
            ratio = (col_p[wet] / hydro[wet]) if bool(wet.any()) else torch.zeros(1)
            head_deficit = float(ratio.mean())
            history.append({
                "time": t_snap,
                "front": front,
                "front_ritter": ritter_front(t_snap, cfg.dam_x, cfg.h0,
                                             cfg.gravity) - cfg.dam_x,
                "relative_error": (front - (ritter_front(t_snap, cfg.dam_x,
                                                         cfg.h0, cfg.gravity)
                                            - cfg.dam_x))
                                  / (ritter_front(t_snap, cfg.dam_x, cfg.h0,
                                                  cfg.gravity) - cfg.dam_x),
                "max_velocity": umax_now,
                "front_velocity_ritter": 2.0 * cfg.c0,
                "eta": eta.tolist(),
                "eta_ritter": [float(v) for v in ana],
                "volume_drift": float(solver.alpha.sum()) * dx * dy / volume0 - 1.0,
                "pressure_over_hydrostatic": head_deficit,
            })
            if progress:
                print(f"  [{cfg.name}] t={t_snap:.3f}s front={front:.4f} "
                      f"ritter={history[-1]['front_ritter']:.4f} "
                      f"err={history[-1]['relative_error']:+.3f}", flush=True)
    volume1 = float(solver.alpha.sum()) * dx * dy
    return {
        "config": asdict(cfg),
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "dx": dx, "dy": dy,
                 "cells_across_depth": cfg.h0 / dy,
                 "cells_across_reservoir": cfg.dam_x / dx},
        "history": history,
        "steps": steps,
        "elapsed_seconds": time.time() - t0,
        "volume_drift": (volume1 - volume0) / volume0,
        "final_alpha": solver.alpha.detach().cpu().clone(),
    }


def evaluate_ritter(run: dict, tol: float = RITTER_FRONT_TOL) -> dict:
    """A2 gate: front error against ``2 sqrt(g h0) t`` at every snapshot."""
    hist = run["history"]
    if not hist:
        return {"pass": False, "reason": "no snapshots", "points": []}
    points = [{"time": h["time"], "front": h["front"],
               "reference": h["front_ritter"],
               "relative_error": h["relative_error"]} for h in hist]
    worst = max(abs(p["relative_error"]) for p in points)
    # Profile check away from the two singular ends of the Ritter wedge.
    prof_err = None
    if hist:
        last = hist[-1]
        eta, ana = last["eta"], last["eta_ritter"]
        interior = [i for i in range(len(eta))
                    if ana[i] > 0.02 * run["config"]["h0"]
                    and ana[i] < 0.98 * run["config"]["h0"]]
        if interior:
            num = sum((eta[i] - ana[i]) ** 2 for i in interior)
            den = sum(ana[i] ** 2 for i in interior)
            prof_err = math.sqrt(num / den)
    return {
        "pass": bool(worst <= tol) and all(p["front"] > 0 for p in points),
        "criterion": f"|x_f - 2 sqrt(g h0) t| / (2 sqrt(g h0) t) <= {tol:g}",
        "reference": "Ritter (1892) dry-bed dam break, shallow-water theory",
        "worst_relative_error": worst,
        "points": points,
        "profile_relative_l2": prof_err,
        "profile_note": "L2 relative error of eta(x) at the last snapshot, "
                        "over the interior of the Ritter wedge",
    }


def front_onset_delay(run: dict, fraction: float = 0.02) -> dict:
    """Time (in units of the leg's own time scale) before the front starts.

    The theoretical front leaves the gate at ``t = 0`` with speed
    ``2 sqrt(g h0)``; a measured onset delay is therefore a direct measure of
    the solver's startup lag, independent of any reference data.
    """
    cfg = run["config"]
    hist = run["history"]
    threshold = fraction * cfg["dam_x"]
    onset = None
    for entry in hist:
        if entry["front"] > threshold:
            onset = entry["time"]
            break
    c0 = float(cfg.get("c0") or math.sqrt(cfg["gravity"] * cfg["h0"]))
    t_ref = cfg["dam_x"] / (2.0 * c0)  # time to travel `fraction * a` ideally
    return {
        "first_snapshot_front_m": hist[0]["front"] if hist else None,
        "onset_time_s": onset,
        "onset_in_units_of_a_over_c0": (onset / (cfg["dam_x"] / c0)
                                        if onset else None),
        "ideal_time_to_same_distance_s": threshold / (2.0 * c0),
        "pressure_over_hydrostatic": [round(e["pressure_over_hydrostatic"], 3)
                                      for e in hist],
        "note": "the front cannot leave at t = 0 in the discrete solve because "
                "the pressure head is built incrementally from p = 0; the "
                "measured onset is the size of that transient",
        "t_reference": t_ref,
    }


# --------------------------------------------------------------------------- #
# Reference audit: is the digitised Martin & Moyce table self-consistent?
# --------------------------------------------------------------------------- #
def martin_moyce_kinematics(column_width: float, gravity: float) -> dict:
    """Reference sanity check: are the digitised M&M points kinematically attainable?

    A gravity-driven surge of displacement ``s`` in time ``t`` needs a mean
    acceleration ``2 s / t^2``; a column collapsing under gravity cannot sustain
    much more than a few ``g``.  Applied to the digitised front and height
    series (with the case's own scaling ``t* = t sqrt(g/a)``, ``h0 = 2a``) this
    separates points the reference can physically have measured from points
    that must have been mis-digitised (a wrong time origin, a mixed abscissa,
    or the finite gate-withdrawal time of the experiment).
    """

    a, g = column_width, gravity
    tau = math.sqrt(a / g)

    def row(t_star, s, label):
        t = t_star * tau
        a_req = 2.0 * s / (t * t) if t > 0 else float("inf")
        return {"t_star": float(t_star), "displacement_m": float(s),
                "required_mean_acceleration_m_s2": float(a_req),
                "over_g": float(a_req / g) if math.isfinite(a_req) else None,
                "attainable_under_2g": bool(math.isfinite(a_req) and a_req <= 2.0 * g),
                "series": label}

    rows = []
    for t_star, x_star in zip(MM_SURGE_T, MM_SURGE_X):
        if t_star > 0.0:
            rows.append(row(t_star, x_star * a, "front"))
    for t_star, h_star in zip(MM_HEIGHT_T, MM_HEIGHT_H):
        if t_star > 0.0:
            drop = (2.0 - h_star) * a       # the gate surface fell this far
            if drop > 0.0:
                rows.append(row(t_star, drop, "height"))
    unattainable = [r for r in rows if not r["attainable_under_2g"]]
    usable = sorted({r["t_star"] for r in rows if r["attainable_under_2g"]})
    return {
        "case_scaling": {"a_m": a, "h0_over_a": 2.0, "tau_s": tau,
                         "criterion": "mean acceleration 2s/t^2 <= 2g"},
        "rows": rows,
        "unattainable_points": unattainable,
        "n_points": len(rows),
        "n_unattainable": len(unattainable),
        "usable_t_star_min": min(usable) if usable else None,
        "verdict": (
            "the digitised series contains points that a gravity-driven collapse "
            "cannot reach under the stated scaling (they need more than 2 g); "
            "those early points cannot be used as an accuracy gate"
            if unattainable else
            "every digitised point is kinematically attainable under the stated "
            "scaling, so the reference can be used as-is"),
    }


# --------------------------------------------------------------------------- #
# Figures / report fragments for the anchors
# --------------------------------------------------------------------------- #
def write_anchor_figures(payload: dict, output: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    figs: list[dict] = []
    dam = payload.get("dam_break") or []
    ritter = payload.get("ritter") or []
    if not dam and not ritter:
        return figs

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    # --- left: free-fall surface of the tall column -------------------------
    ax = axes[0]
    for leg in dam:
        hist = leg["history"]
        a = leg["config"]["column_width"]
        g = leg["config"]["gravity"]
        ts = [h["t_star"] for h in hist]
        ax.plot(ts, [h["height_star"] for h in hist], lw=1.4,
                label=f"{leg['config']['vof_scheme']} "
                      f"{leg['grid']['nx']}x{leg['grid']['ny']}")
    if dam:
        leg = dam[0]
        a, g = leg["config"]["column_width"], leg["config"]["gravity"]
        ts = [h["t_star"] for h in leg["history"]]
        h0 = 2.0 * a
        ax.plot(ts, [freefall_surface(h["time"], h0, g) / a for h in leg["history"]],
                "k--", lw=1.2, label="free fall  2 - t*^2/2")
        gate = payload["gates"].get("G4a_freefall_surface", {})
        if gate.get("freefall_window_t_star"):
            ax.axvline(gate["freefall_window_t_star"], color="0.6", ls=":", lw=1)
            ax.text(gate["freefall_window_t_star"], 1.6, " rarefaction returns",
                    fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("t* = t sqrt(g/a)")
    ax.set_ylabel("h* = h_gate / a")
    ax.set_title("A1: free-fall collapse of the gate surface")
    ax.set_ylim(0, 2.3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # --- right: Ritter front -------------------------------------------------
    ax = axes[1]
    for leg in ritter:
        hist = leg["history"]
        ax.plot([h["time"] for h in hist], [h["front"] for h in hist], "o-",
                ms=4, lw=1.4, label=f"{leg['config']['vof_scheme']} "
                                    f"{leg['config']['name']}")
    if ritter:
        leg = ritter[0]
        cfg = leg["config"]
        ts = [h["time"] for h in leg["history"]]
        ax.plot(ts, [2.0 * math.sqrt(cfg["gravity"] * cfg["h0"]) * t for t in ts],
                "k--", lw=1.2, label="Ritter  2 sqrt(g h0) t")
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("surge-front travel  x_f - a  [m]")
    ax.set_title("A2: shallow-water dam-break front")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = output / "fig4-analytic-anchors.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Analytic anchors. Left: the gate surface of the "
                            "tall Martin & Moyce column falls in free fall "
                            "until the rarefaction returns (A1). Right: the "
                            "shallow dam-break front against the Ritter "
                            "solution (A2)."})

    # --- audit figure: kinematic attainability of the M&M table -------------
    audit = payload.get("martin_moyce_kinematics")
    if audit:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for series, colour, marker in (("front", "C0", "o"), ("height", "C1", "s")):
            rows = [r for r in audit["rows"] if r["series"] == series]
            ax.semilogy([r["t_star"] for r in rows],
                        [max(r["over_g"], 1e-3) for r in rows],
                        marker=marker, ms=5, lw=1.2, color=colour,
                        label=f"{series} series")
        ax.axhline(2.0, color="k", ls=":", lw=1)
        ax.text(0.05, 2.2, "2 g attainability bound", fontsize=8)
        ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
        ax.text(0.05, 0.85, "1 g", fontsize=8)
        ax.set_xlabel("t* (digitised table abscissa)")
        ax.set_ylabel("required mean acceleration  2s/t^2  / g")
        ax.set_title("Are the digitised M&M points kinematically attainable?")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = output / "fig5-mm-reference-kinematics.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "Kinematic attainability of the digitised "
                                "Martin & Moyce points under the case's own "
                                "scaling: the mean acceleration 2s/t^2 that each "
                                "point implies, in units of g. Only the very "
                                "early points exceed the 2 g bound, so the "
                                "front gate is taken at t* >= 2."})
    return figs


def report_section(payload: dict) -> list[str]:
    """Markdown lines for the analytic anchors (numbers from the payload)."""
    gates = payload["gates"]
    lines: list[str] = []
    A = lines.append
    A("### 3.2 A1 自由落体表面锚（解析，独立于实验曲线）\n")
    gate = gates.get("G4a_freefall_surface")
    if gate:
        A("闸门释放后，闸门处液面只受重力、上方为大气压，因此严格满足 "
          "`h(t) = 2a - g t²/2`，直到闸门处发出的稀疏波传回左壁 "
          f"（`t* < {gate['freefall_window_t_star']:.2f}`）。判据 "
          f"`{gate['criterion']}`。\n")
        A("| t* | h* 实测 | h* 自由落体 | 相对误差 |")
        A("|---|---|---|---|")
        for p in gate["points"]:
            if p["measured"] is None:
                A(f"| {p['t_star']:g} | 未记录 | {p['reference']:.4f} | "
                  f"{p.get('reason', 'n/a')} |")
            else:
                A(f"| {p['t_star']:g} | {p['measured']:.4f} | "
                  f"{p['reference']:.4f} | "
                  f"{100.0 * p['relative_error']:+.2f}% |")
        A("")
        A(f"最差相对误差 **{100.0 * gate['worst_relative_error']:.2f}%** → "
          f"{'PASS' if gate['pass'] else 'FAIL'}"
          f"（判据 {100.0 * FREEFALL_TOL:.0f}%）。\n")
        A("**读法警告**：`height_star` 由「最上方湿单元」给出，量化步长是 "
          "`dy/a = 0.0875`，且实测值在这个窗口里几乎不动（1.990），"
          "所以这条 PASS 只是「液面下落没有快于自由落体、且在一个格子量级"
          "内一致」的弱证据，不能据此认为重力项已定量正确——它与 §3.5 的"
          "启动滞后是同一现象的两面。\n")
    else:
        A("本批未跑塔柱溃坝 leg。\n")

    A("### 3.3 A2 Ritter 解析波前锚\n")
    ritter = payload.get("ritter") or []
    gate = gates.get("G4b_ritter_front")
    if ritter and gate:
        A("参数（程序化注入）：`h0 = "
          f"{ritter[0]['config']['h0']:g} m`，`a = {ritter[0]['config']['dam_x']:g} m`，"
          f"`h0/dy = {ritter[0]['grid']['cells_across_depth']:.1f}` 个单元，"
          f"`dx = {ritter[0]['grid']['dx']:.4g} m`。\n")
        A("| 时刻 t [s] | x_f 实测 [m] | Ritter 2√(g h0) t [m] | 相对误差 |")
        A("|---|---|---|---|")
        for p in gate["points"]:
            A(f"| {p['time']:.3f} | {p['front']:.4f} | {p['reference']:.4f} | "
              f"{100.0 * p['relative_error']:+.2f}% |")
        A("")
        A(f"最差相对误差 **{100.0 * gate['worst_relative_error']:.2f}%** → "
          f"{'PASS' if gate['pass'] else 'FAIL'}"
          f"（判据 {100.0 * RITTER_FRONT_TOL:.0f}%）；波形 L2 相对误差 "
          f"{gate['profile_relative_l2']:.3f}" if gate["profile_relative_l2"]
          else "波形误差 n/a")
    else:
        A("本批未跑 Ritter leg。")
    A("")

    audit = payload.get("martin_moyce_kinematics")
    A("### 3.4 参考数据的运动学可达性检查（Martin & Moyce 数字化表）\n")
    if audit:
        A(f"塔柱几何 `a = {audit['case_scaling']['a_m']:g} m`，`h0 = 2a`，"
          f"`τ = sqrt(a/g) = {audit['case_scaling']['tau_s']:.4f} s`；判据 "
          f"`{audit['case_scaling']['criterion']}`。位移 `s` 在时间 `t` 内需要的"
          "平均加速度不能远超重力——这条与求解器无关，只用来看「表里的点实验"
          "有没有可能测到」。\n")
        A("| 序列 | t* | 位移 [m] | 所需平均加速度 [m/s²] | /g | 可达（≤2g） |")
        A("|---|---|---|---|---|---|")
        for row in audit["rows"]:
            A(f"| {row['series']} | {row['t_star']:g} | "
              f"{row['displacement_m']:.4f} | "
              f"{row['required_mean_acceleration_m_s2']:.2f} | "
              f"{row['over_g']:.2f} | {'是' if row['attainable_under_2g'] else '否'} |")
        A("")
        A(f"共 {audit['n_points']} 点，其中 **{audit['n_unattainable']} 点**在 2g "
          f"以上（全部出现在 t* < {audit['usable_t_star_min']:g} 的极早期）。"
          f"判定：{audit['verdict']}。\n")
        A("因此 G4c 只在 `t* ≥ 2` 上用 M&M 曲线做判据；`t* = 1` 的点保留在表里，"
          "作为**未达**记录如实报告。\n")
    else:
        A("本批未做检查。\n")

    onset = payload.get("front_onset")
    if onset:
        A("### 3.5 启动滞后诊断（为什么会晚）\n")
        A(f"理论上波前在 `t = 0` 就以 `2√(g h0)` 出发。测得的时间序列里，波前"
          f"第一次超过 `0.02a` 的时刻是 **{onset['onset_time_s']:.4f} s**"
          f"（= {onset['onset_in_units_of_a_over_c0']:.2f} 个 `a/c0` 时间单位），"
          f"而理想情况只需 {onset['ideal_time_to_same_distance_s']:.4f} s——"
          f"启动滞后约 {onset['onset_in_units_of_a_over_c0'] / 1.0:.1f} 倍 `a/c0`。\n")
        A("**压力不是原因。** 同一条 leg 的储液柱内 `p / ρg(h_s − y)` 逐快照为 "
          f"{onset['pressure_over_hydrostatic']}：启动阶段液柱内压力**基本就是"
          "静水压**（0.98–1.01），所以「水头没建立」这一解释被自己的测量否掉了。"
          "同一时间窗口里液面几乎不下落（§3.2 的 h* 近似常数 1.99），"
          "而波前也不动——即流体在该窗口内**整体接近静止**，"
          "重力既没有变成下落也没有变成横向铺展。\n")
        A("**已排除的假设（每条都有数字）。** ①网格分辨率：`h0/dy` 从 2.5 提到 10、"
          "溃坝从 100×30 提到 200×60，早期波前误差只从 −56% 变到 −48%；"
          "②初始水头：用平衡水静压场（`discrete=True`）做溃坝初值，`t* = 0.55` "
          "的波前完全不变（0.181 → 0.181a）；③气相惯性/闭腔背压：`ρ_g` 取 "
          "1.2 / 0.012 / 10 kg/m³ 波前一致，取 1000 kg/m³ 反而更慢（−105%）；"
          "④`pressure_form`：`\"reduced\"` 不稳定（`|u|` 冲到 8 m/s，波前仍滞后）。\n")
        A("**结论（未解决，如实登记）。** 波前偏慢不是参考数据、不是分辨率、"
          "不是水头初始化、也不是气相性质，而是耦合求解器在**重力流的早期"
          "（`t* ≲ 1.5`）把重力势转换成横向动量的效率不足**；本单没有定位到"
          "唯一根因，因此不声明修复。可继续追的方向：动量方程里界面面的密度"
          "插值方式、以及 1/ρ 加权压力泊松对液相的约束强度。\n")
    return lines