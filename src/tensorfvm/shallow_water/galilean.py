"""Galilean-invariance verification of the 1-D shallow-water solver.

Ritter (1892) and Stoker (1957) published the dam-break solutions for water
*at rest*, so the benchmark in :mod:`tensorfvm.shallow_water.dam_break` only
ever exercises flows whose undisturbed state is ``u = 0``.  That leaves an
entire error class untested: with ``u0 = 0`` the net mass and momentum flux
through the domain ends is zero and no wave ever reaches a boundary, so a wrong
end condition (closed/wall, no-slip, fixed depth), a spurious momentum source,
or an inconsistent treatment of the non-zero-velocity branch of the HLL flux
are all *invisible* to the existing benchmark -- they look like a 0.02 % front
error either way.

This module closes that gap by grading the solver against the **Galilean
boost** of the same exact solutions.  The shallow-water system

    h_t + (h u)_x = 0,     (h u)_t + (h u^2 + g h^2 / 2)_x = 0

is invariant under the constant boost ``x' = x - u0 t``, ``u' = u - u0``,
``h' = h``; hence the solution whose *initial* state carries a uniform velocity
``u0`` on both sides of the dam is

    h(x, t) = h_rest(x - u0 t, t),
    u(x, t) = u_rest(x - u0 t, t) + u0,

where ``h_rest, u_rest`` are the classical Ritter/Stoker fields of
:mod:`tensorfvm.shallow_water.exact`.  The consequences that are graded here:

* the toe/shock advances at ``u0 + speed_rest`` (the fan head retreats at
  ``u0 - c0``), so a wrong sign or a missing ``u h`` factor in the wave-speed
  estimate shifts a *measurable* position;
* the volume budget is no longer trivial: ``dV/dt = h0 u0``, the inlet flux
  through the upstream end.  A closed end accumulates water and fails it;
* the undisturbed current must be transported **exactly**: ``u = u0``,
  ``h = h0`` in the region the waves have not reached.  A uniform state is an
  exact fixed point of the HLL + MUSCL operator, so any drift there is a
  genuine defect (source-term leakage, boundary closure, limiter inconsistency)
  rather than discretisation error.

Refs: Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947; Stoker (1957) *Water
Waves*, ch. 10; Whitham (1974) *Linear and Nonlinear Waves*, §3.1 (Galilean
invariance of the shallow-water system); Toro (2009) ch. 6-7 (HLL flux used by
:class:`tensorfvm.shallow_water.swe1d.ShallowWater1D`).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .dam_break import FAN_WINDOW, _level_crossing, _local_extrema, fan_front_estimate
from .exact import ritter_state, stoker_constant_state, stoker_state, t_star_to_time
from .swe1d import ShallowWater1D

#: Acceptance tolerance of the anchor (``CASE-FVM-DAMBREAK-STOKER``): 5 %.
TOLERANCE = 0.05

#: End-of-run similarity time of the anchor, ``t* = t sqrt(g h0) / h0``.
T_STAR = 2.0

#: Uniform initial velocities graded.  ``0`` is the control (it must reproduce
#: the classical benchmark), the other two are a downstream current and a
#: counter-flow strong enough that the fan head has moved ``0.94 h0``-worth...
#: see the report; both keep every wave inside ``[-4, 6] m``.
U0_VALUES = (0.0, 1.5, -1.5)

#: Gate on the undisturbed-current drift ``max |u - u0|`` (m/s) in the region
#: the waves have not reached.  A uniform state is an *exact* fixed point of the
#: operator, so the only admissible values are round-off; the gate is set three
#: orders of magnitude above measured round-off (see the report table).
DRIFT_TOLERANCE = 1e-6

#: Gate on the through-flow volume budget ``|V_num - V_exact| / V_exact``.
VOLUME_TOLERANCE = 5e-3


@dataclass(frozen=True)
class GalileanCase:
    """A dam break whose initial state carries a uniform velocity ``u0``."""

    key: str
    kind: str                     # "dry" (h1 = 0) or "wet" (h0 > h1 > 0)
    h0: float
    h1: float
    u0: float
    gravity: float = 9.81
    x_dam: float = 0.0
    x_min: float = -4.0
    x_max: float = 6.0
    t_star_end: float = T_STAR
    reference: str = ""

    # ------------------------------------------------------------ kinematics
    @property
    def t_end(self) -> float:
        """Physical end time of the ``t* = t_star_end`` snapshot."""
        return t_star_to_time(self.t_star_end, self.h0, self.gravity)

    @property
    def c0(self) -> float:
        return math.sqrt(self.gravity * self.h0)

    def head(self, t: float) -> float:
        """Upstream edge of the rarefaction, ``x_dam + (u0 - c0) t``."""
        return self.x_dam + (self.u0 - self.c0) * t

    def front(self, t: float) -> float:
        """Graded wave position.

        Dry bed: the zero-depth toe of the fan, ``x_dam + (u0 + 2 c0) t``.
        Wet bed: the bore, ``x_dam + (U + u0) t`` with ``U`` the *rest-frame*
        Stoker shock speed (the Rankine-Hugoniot conditions are unchanged by a
        constant boost, which the module verifies in
        :func:`boosted_shock_residual`).
        """
        if self.kind == "dry":
            return self.x_dam + (self.u0 + 2.0 * self.c0) * t
        sol = stoker_constant_state(self.h0, self.h1, self.gravity)
        return self.x_dam + (sol["shock_speed"] + self.u0) * t

    def shock_state(self) -> dict:
        """Boosted bore state ``(h2, u2, U)`` (wet bed only)."""
        sol = dict(stoker_constant_state(self.h0, self.h1, self.gravity))
        sol["u2"] = sol["u2"] + self.u0
        sol["shock_speed"] = sol["shock_speed"] + self.u0
        return sol

    # --------------------------------------------------------------- fields
    def exact(self, x, t: float):
        """Exact ``(h, u)`` at ``(x, t)``: the rest-frame field seen at ``x'``."""
        xs = np.asarray(x, dtype=float)
        shift = xs - self.u0 * t
        if t <= 0.0:
            depth = np.where(shift <= self.x_dam, self.h0, self.h1)
            return depth, np.full_like(depth, self.u0)
        if self.kind == "dry":
            depth, vel = ritter_state(shift, t, self.h0, self.gravity, self.x_dam)
        else:
            depth, vel = stoker_state(shift, t, self.h0, self.h1, self.gravity,
                                      self.x_dam)
        return depth, vel + self.u0

    # --------------------------------------------------------------- budget
    def exact_volume(self, t: float, samples: int = 200_001) -> float:
        """Water volume inside the domain at time ``t`` (analytic for dry bed).

        Dry bed, closed form: with ``head = x_dam + (u0 - c0) t`` and
        ``front = x_dam + (u0 + 2 c0) t``,

            V(t) = h0 (head - x_min) + t c0 h0,

        whose derivative ``h0 u0`` is exactly the inlet flux -- the identity the
        solver's volume budget is graded against.  The wet bed has no such
        elementary form (rarefaction + shock), so it is integrated on a fine
        auxiliary grid; :func:`galilean_selfcheck` measures the quadrature error
        against the dry-bed closed form.
        """
        if self.kind == "dry":
            if not (self.x_min < self.head(t) and self.front(t) < self.x_max):
                raise ValueError("waves out of the domain: volume formula invalid")
            return self.h0 * (self.head(t) - self.x_min) + t * self.c0 * self.h0
        xs = np.linspace(self.x_min, self.x_max, samples)
        depth, _ = self.exact(xs, t)
        return float(np.trapezoid(depth, xs))


def boosted_shock_residual(case: GalileanCase, t: float = 1.0) -> dict:
    """Rankine-Hugoniot residual of the *boosted* bore (analytic self-check).

    The jump conditions of the wet-bed bore read, in the frame where the water
    far downstream moves at ``u0``,

        (U + u0)(h2 - h1) = h2 (u2 + u0) - h1 u0        (mass)
        (U + u0)(h2 u2) = (h2 u2^2 + g h2^2/2) - g h1^2/2   (momentum, boosted)

    Both must vanish to round-off for the boosted field, which is what makes the
    boosted shock position ``x_dam + (U + u0) t`` an admissible reference.
    """
    if case.kind != "wet":
        raise ValueError("boosted_shock_residual needs a wet-bed case")
    sol = case.shock_state()
    U, g, u0 = sol["shock_speed"], case.gravity, case.u0
    h1, h2, u2 = case.h1, sol["h2"], sol["u2"]
    # frame moving with the bore: the relative velocities are boost-invariant
    m2 = h2 * (u2 - U)
    m1 = h1 * (u0 - U)
    mass = m2 - m1
    mom = (m2 * (u2 - U) + 0.5 * g * h2 * h2) - (m1 * (u0 - U) + 0.5 * g * h1 * h1)
    scale_m = max(abs(m2), abs(m1), 1e-30)
    scale_p = max(abs(h2 * (u2 - U) ** 2), abs(g * h2 * h2), 1e-30)
    return {"mass_residual": float(mass), "momentum_residual": float(mom),
            "mass_relative": float(abs(mass) / scale_m),
            "momentum_relative": float(abs(mom) / scale_p)}


def galilean_selfcheck(gravity: float = 9.81) -> dict:
    """Analytic identities the reference fields must satisfy (no solver)."""
    g = gravity
    # 1) u0 = 0 must reduce to the classical fields, bit for bit.
    x = np.linspace(-4.0, 6.0, 401)
    t = t_star_to_time(T_STAR, 1.0, g)
    ref_h, ref_u = ritter_state(x, t, 1.0, g)
    ctl = GalileanCase("control", "dry", h0=1.0, h1=0.0, u0=0.0, gravity=g)
    h0s, u0s = ctl.exact(x, t)
    # 2) boost identity: h(x + u0 t, t) == h_rest(x, t), u shifted by u0.
    boosted = GalileanCase("boosted", "dry", h0=1.0, h1=0.0, u0=1.5, gravity=g)
    hb, ub = boosted.exact(x + 1.5 * t, t)
    # 3) dry-bed volume derivative equals the inlet flux h0 u0 (central diff).
    dt = 1e-6
    dV = (boosted.exact_volume(t + dt) - boosted.exact_volume(t - dt)) / (2 * dt)
    # 4) quadrature used for the wet bed reproduces the closed form on the dry
    #    bed (same grid, same routine) -- so wet-bed volumes are not a weak link.
    wet = GalileanCase("wet", "wet", h0=1.0, h1=0.5, u0=1.0, gravity=g)
    xs = np.linspace(ctl.x_min, ctl.x_max, 200_001)
    quad_dry = float(np.trapezoid(ctl.exact(xs, t)[0], xs))
    return {
        "u0_zero_matches_ritter_h": float(np.max(np.abs(h0s - ref_h))),
        "u0_zero_matches_ritter_u": float(np.max(np.abs(u0s - ref_u))),
        "boost_identity_h": float(np.max(np.abs(hb - ref_h))),
        "boost_identity_u": float(np.max(np.abs(ub - (ref_u + 1.5)))),
        "dry_volume_rate": float(dV),
        "dry_volume_rate_expected": float(boosted.h0 * boosted.u0),
        "dry_volume_rate_error": float(abs(dV - boosted.h0 * boosted.u0)
                                       / abs(boosted.h0 * boosted.u0)),
        "quadrature_error_on_dry": float(abs(quad_dry - ctl.exact_volume(t))
                                         / ctl.exact_volume(t)),
        "stoker_boost_residual": boosted_shock_residual(wet, t),
    }


# --------------------------------------------------------------------------- #
# Solver grading
# --------------------------------------------------------------------------- #
def _relative(measured: float, exact: float) -> float:
    return abs(measured - exact) / abs(exact) if exact != 0.0 else abs(measured)


def profile_metrics(case: GalileanCase, solver: ShallowWater1D) -> dict:
    """Grade one snapshot of a boosted dam break against the boosted reference."""
    t = solver.time
    x = solver.x
    h, u = solver.primitive()
    h_ex, u_ex = case.exact(x, t)
    dx = solver.dx
    head = case.head(t)
    front = case.front(t)

    # Far field: the upstream half of the region the rarefaction has not
    # reached.  A uniform state is an exact fixed point of the HLL + MUSCL
    # operator, so the current there must be preserved to round-off; the
    # *near-head* decay is not round-off (it is the smeared rarefaction head,
    # recorded separately as ``head_smear_cells``), so the grading window must
    # exclude it by a *physical* distance, not by a few cells.
    far_mid = case.x_min + 0.5 * (head - case.x_min)
    quiet = (x <= far_mid) & (x >= case.x_min + 3.0 * dx)
    drift = np.abs(u - case.u0)
    quiet_drift_u = float(np.max(drift[quiet])) if np.any(quiet) else float("nan")
    quiet_drift_h = (float(np.max(np.abs(h[quiet] - case.h0)) / case.h0)
                     if np.any(quiet) else float("nan"))
    # how far upstream of the exact head the velocity is still disturbed: the
    # smeared head, reported in cells (a numerical-diffusion diagnostic).
    head_smear_cells = int(np.count_nonzero((x < head) & (drift > DRIFT_TOLERANCE)))

    volume_exact = case.exact_volume(t)
    volume_num = solver.total_volume()
    metrics = {
        "key": case.key,
        "kind": case.kind,
        "n": solver.n,
        "dx": dx,
        "u0": case.u0,
        "t": t,
        "t_star": t * case.c0 / case.h0,
        "t_end": case.t_end,
        "head_exact": head,
        "front_exact": front,
        "front_speed_exact": (front - case.x_dam) / t,
        "depth_l1": float(np.abs(h - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(h - h_ex).max() / case.h0),
        "velocity_l1": float(np.abs(u - u_ex).sum() / max(np.abs(u_ex).sum(), 1e-300)),
        "quiet_cells": int(np.count_nonzero(quiet)),
        "head_smear_cells": head_smear_cells,
        "quiet_drift_u": quiet_drift_u,
        "quiet_drift_h": quiet_drift_h,
        "volume_exact": volume_exact,
        "volume_num": volume_num,
        "volume_error": _relative(volume_num, volume_exact),
        "volume_inlet_flux": case.h0 * case.u0,
        "tv_numeric": float(np.abs(np.diff(h)).sum()),
        "tv_exact": float(case.h0 - case.h1),
        "tv_ratio": float(np.abs(np.diff(h)).sum() / (case.h0 - case.h1)),
        "local_extrema": _local_extrema(h, 1e-4 * case.h0),
        "clipped_cells": int(solver.clipped_cells),
        "dry_flips": int(solver.dry_flips),
        "u_max": float(np.max(np.abs(u))),
    }

    if case.kind == "dry":
        fan = fan_front_estimate(x, h, t, case.gravity, case.h0, FAN_WINDOW)
        measured = fan["front"]
        metrics.update({
            "front_estimator": "fan_front_estimate (boost-invariant shape identity)",
            "front_measured": measured,
            "front_error": _relative(measured, front),
            "fan_cells": fan["cells"],
            "fan_spread": fan["spread"],
            "fan_slope_error": fan["slope_error"],
        })
    else:
        sol = case.shock_state()
        level = 0.5 * (case.h1 + sol["h2"])
        measured = _level_crossing(x, h, level)
        x_fan_end = case.x_dam + (sol["u2"] - sol["c2"]) * t
        lo = x_fan_end + 0.25 * (front - x_fan_end)
        hi = x_fan_end + 0.75 * (front - x_fan_end)
        plateau = (x >= lo) & (x <= hi)
        still = x >= front + 0.5 * (case.x_max - front)
        h2_meas = float(h[plateau].mean()) if np.any(plateau) else float("nan")
        h1_meas = float(h[still].mean()) if np.any(still) else float("nan")
        metrics.update({
            "front_estimator": "mid-level crossing (h1 + h2)/2",
            "front_measured": measured,
            "front_error": _relative(measured, front),
            "h2_exact": sol["h2"],
            "h2_measured": h2_meas,
            "h2_error": _relative(h2_meas, sol["h2"]),
            "h1_measured": h1_meas,
            "stage_ratio_exact": sol["h2"] / case.h1,
            "stage_ratio_measured": h2_meas / h1_meas,
            "stage_ratio_error": _relative(h2_meas / h1_meas, sol["h2"] / case.h1),
            "still_cells": int(np.count_nonzero(still)),
        })
    return metrics


def verdict(metrics: dict) -> dict:
    """Pre-declared gates; ``value`` is the shortfall, so ``pass`` means <= 0.

    ``velocity_l1`` and the dry-cell ``dry_flips`` count are *diagnostics*, not
    gates: a dry-bed tip is a zero-depth cusp whose numerical velocity is set by
    the depth floor, and a wet/dry cell legitimately flips as the front sweeps
    through it.  Both are reported in every row so the reader can see them.
    """
    out = {
        "front<5%": metrics["front_error"] - TOLERANCE,
        "depth_l1<5%": metrics["depth_l1"] - TOLERANCE,
        "monotone(no_extrema)": 0.0 if metrics["local_extrema"] == 0 else float(metrics["local_extrema"]),
        "tv_no_growth<0.5%": max(0.0, metrics["tv_ratio"] - 1.0) - 0.005,
        "far_current<1e-6": metrics["quiet_drift_u"] - DRIFT_TOLERANCE,
        "head_smear<=64cells": float(metrics["head_smear_cells"] - 64),
        "volume_budget<0.5%": metrics["volume_error"] - VOLUME_TOLERANCE,
        "no_negative_depth": float(metrics["clipped_cells"]),
    }
    if metrics["kind"] == "wet":
        out["h2_plateau<5%"] = metrics["h2_error"] - TOLERANCE
        out["stage_ratio<5%"] = metrics["stage_ratio_error"] - TOLERANCE
    out = {k: {"value": float(v), "pass": bool(v <= 0.0)} for k, v in out.items()}
    out["all_pass"] = bool(all(item["pass"] for item in out.values()))
    return out


def run_case(case: GalileanCase, n: int = 1600, cfl: float = 0.4,
             order: int = 2, solver_cls=ShallowWater1D) -> tuple[dict, ShallowWater1D]:
    """Initialise the uniform ``(h0, u0) | (h1, u0)`` state and integrate."""
    solver = solver_cls(n=n, x_min=case.x_min, x_max=case.x_max,
                        gravity=case.gravity, cfl=cfl, order=order)
    h_init, u_init = case.exact(solver.x, 0.0)
    solver.set_state(h_init, u_init)
    solver.advance(case.t_end)
    metrics = profile_metrics(case, solver)
    metrics["verdict"] = verdict(metrics)
    metrics["wall_time_s"] = None
    return metrics, solver


DEFAULT_CASES = (
    GalileanCase("ritter-dry-u0-0.0", "dry", 1.0, 0.0, 0.0,
                 reference="Ritter 1892 (control: reproduces the rest-frame case)"),
    GalileanCase("ritter-dry-u0-1.5", "dry", 1.0, 0.0, 1.5,
                 reference="Ritter 1892 + Galilean boost u0 = +1.5 m/s"),
    # Counter-flow: the fan head retreats at u0 - c0 = -4.63 m/s, so the domain
    # is extended upstream (-6 m instead of -4 m) to keep a *quiet* far field
    # outside the numerical head smear at the coarse resolutions.
    GalileanCase("ritter-dry-u0--1.5", "dry", 1.0, 0.0, -1.5, x_min=-6.0,
                 reference="Ritter 1892 + Galilean boost u0 = -1.5 m/s"),
    GalileanCase("stoker-wet-u0-1.0", "wet", 1.0, 0.5, 1.0,
                 reference="Stoker 1957 ch.10 + Galilean boost u0 = +1.0 m/s"),
)


def convergence_study(cases=DEFAULT_CASES, resolutions=(400, 800, 1600),
                      cfl: float = 0.4, order: int = 2) -> dict:
    """Front-error convergence of each boosted case; a gate needs convergence."""
    rows = []
    for case in cases:
        for n in resolutions:
            t0 = time.perf_counter()
            metrics, _ = run_case(case, n=n, cfl=cfl, order=order)
            metrics["wall_time_s"] = time.perf_counter() - t0
            rows.append({k: metrics[k] for k in (
                "key", "kind", "u0", "n", "dx", "front_exact", "front_measured",
                "front_error", "depth_l1", "velocity_l1", "quiet_drift_u",
                "volume_error", "tv_ratio", "local_extrema", "clipped_cells",
                "dry_flips", "wall_time_s")} | {"all_pass": metrics["verdict"]["all_pass"]})
    summary = {}
    for case in cases:
        sub = [r for r in rows if r["key"] == case.key]
        errs = np.array([r["front_error"] for r in sub])
        ok = errs > 0.0
        order_fit = (float(np.polyfit(np.log(np.array([r["dx"] for r in sub])[ok]),
                                      np.log(errs[ok]), 1)[0]) if np.count_nonzero(ok) >= 2
                     else float("nan"))
        summary[case.key] = {
            "front_error_by_n": {int(r["n"]): float(r["front_error"]) for r in sub},
            "worst_front_error": float(errs.max()),
            "observed_order": order_fit,
            "all_pass": bool(all(r["all_pass"] for r in sub)),
        }
    return {"rows": rows, "summary": summary, "tolerance": TOLERANCE}


class ClosedWall1D(ShallowWater1D):
    """Variant with closed (reflecting) ends, for the boundary-sensitivity probe.

    Only the two boundary faces differ from the production solver: the ghost
    state is the mirror image of the edge cell (``u_ghost = -u_edge``), which
    makes the mass flux through the end exactly zero.  The probe exists to show
    that the boosted case *detects* an end-condition change that the
    rest-frame (``u0 = 0``) benchmark cannot see.
    """

    def _face_states(self):  # noqa: D102 - see class docstring
        hL, uL, hR, uR = super()._face_states()
        hL, uL, hR, uR = hL.copy(), uL.copy(), hR.copy(), uR.copy()
        uL[0] = -uR[0]
        uR[-1] = -uL[-1]
        return hL, uL, hR, uR


def boundary_blindness_demo(n: int = 800) -> dict:
    """Run the same cases with transmissive and closed ends.

    ``u0 = 0``: both end conditions are no-flow for this flow (nothing of the
    solution ever reaches an end), so the readings agree to round-off -- the
    classical benchmark has no discriminating power here.
    ``u0 != 0``: the closed end blocks the inlet, the current decelerates and
    the through-flow budget ``dV/dt = h0 u0`` collapses.
    """
    rows = []
    for case in (DEFAULT_CASES[0], DEFAULT_CASES[1]):
        open_m, _ = run_case(case, n=n, solver_cls=ShallowWater1D)
        wall_m, _ = run_case(case, n=n, solver_cls=ClosedWall1D)
        rows.append({
            "key": case.key, "u0": case.u0, "n": n,
            "open_front_error": open_m["front_error"],
            "wall_front_error": wall_m["front_error"],
            "open_quiet_drift_u": open_m["quiet_drift_u"],
            "wall_quiet_drift_u": wall_m["quiet_drift_u"],
            "open_volume_error": open_m["volume_error"],
            "wall_volume_error": wall_m["volume_error"],
            "open_all_pass": open_m["verdict"]["all_pass"],
            "wall_all_pass": wall_m["verdict"]["all_pass"],
        })
    return {"rows": rows,
            "reading": ("u0=0: the two end conditions are indistinguishable; "
                        "u0!=0: the closed end is caught by the current drift "
                        "and the volume budget")}


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
def figure_galilean(runs: dict, study: dict, demo: dict, path) -> str:
    """Depth/velocity profiles + front-error convergence + BC-blindness bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    colors = {"ritter-dry-u0-0.0": "#4d4d4d", "ritter-dry-u0-1.5": "#1f77b4",
              "ritter-dry-u0--1.5": "#d62728", "stoker-wet-u0-1.0": "#2ca02c"}

    ax = axes[0]
    for key, solver in runs.items():
        c = colors.get(key, "#888888")
        case = next(c for c in DEFAULT_CASES if c.key == key)
        ax.plot(solver.x, solver.h / case.h0, "-", color=c, lw=1.6, label=f"num {key}")
        hx, _ = case.exact(solver.x, solver.time)
        ax.plot(solver.x, hx / case.h0, "--", color=c, lw=1.0, alpha=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h / h0")
    ax.set_title("depth: solver (solid) vs boosted exact (dashed)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(TOLERANCE * 100.0, color="k", ls=":", lw=1.0, label="5 % gate")
    for case in DEFAULT_CASES:
        rows = [r for r in study["rows"] if r["key"] == case.key]
        ax.loglog([r["dx"] for r in rows], [100.0 * r["front_error"] for r in rows],
                  "o--", color=colors[case.key], label=case.key)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("front-position error [%]")
    ax.set_title("front error vs dx")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    labels = [f"{r['key']}\nn={r['n']}" for r in demo["rows"]]
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, [100.0 * r["open_front_error"] for r in demo["rows"]],
           width=0.4, color="#1f77b4", label="open ends (production)")
    ax.bar(idx + 0.2, [100.0 * r["wall_front_error"] for r in demo["rows"]],
           width=0.4, color="#d62728", label="closed ends (defective)")
    ax.axhline(5.0, color="k", ls=":", lw=1.0)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("front-position error [%]")
    ax.set_title("boundary sensitivity: only u0 != 0 sees the wall")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def write_artifacts(outdir, payload: dict) -> dict:
    """Write ``metrics.json`` / ``convergence.json`` / ``manifest.json``."""
    from pathlib import Path
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("metrics", "convergence", "boundary_blindness", "selfcheck"):
        if name in payload:
            p = out / f"{name}.json"
            p.write_text(json.dumps(payload[name], indent=2, ensure_ascii=False) + "\n")
            written[name] = str(p)
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({
        "meta": payload.get("meta", {}),
        "artifacts": {k: Path(v).name for k, v in written.items()},
    }, indent=2, ensure_ascii=False) + "\n")
    written["manifest"] = str(manifest)
    return written