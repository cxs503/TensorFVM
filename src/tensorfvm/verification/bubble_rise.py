"""2-D single-bubble rise benchmark: Clift shape band + terminal-velocity anchor.

Physical setting
----------------
A single gas bubble of diameter ``d`` is released from rest in a quiescent
liquid column and rises under gravity.  Unlike the dam break this problem has no
free parameter left: the flow is fixed by three dimensionless groups,

* Eotvos (Bond) number ``Eo = rho_l g d^2 / sigma``  -- surface tension vs buoyancy,
* Morton number ``M = g mu_l^4 drho / (rho_l^2 sigma^3)`` -- liquid properties only,
* the density and viscosity ratios ``rho_l/rho_g``, ``mu_l/mu_g``,

and those place the bubble in a shape regime documented on the
Clift-Grace-Weber map (Clift, Grace & Weber 1978, *Bubbles, Drops and
Particles*, ch. 7): spherical / ellipsoidal / spherical-cap.

What is measured, and against what
----------------------------------
1. **Terminal rise velocity** of the bubble centroid, compared with the
   Mendelson (1967) wave-analogy correlation in the form used by Fan & Tsuchiya
   (1990) for ellipsoidal bubbles,

       U = sqrt( 2.14 sigma/(rho_l d) + 0.505 g d ).

   Its validity window is 3-D, ellipsoidal, low Morton number, uncontaminated;
   our run is 2-D planar and laterally confined, so the comparison is made with
   a *declared* tolerance (:data:`TERMINAL_VELOCITY_TOLERANCE`) and the signed
   error is reported, not asserted away.
2. **Shape regime band**: aspect ratio (equatorial/polar extent at ``alpha=0.5``)
   and perimeter circularity against the ellipsoidal band, together with the
   regime classification from the measured ``(Eo, M, Re)``.  A band check, not a
   pointwise literature value -- the map is qualitative.
3. **Mass conservation** of the interface transport: worst per-step relative
   volume change over the whole run (``< 1e-6`` required).
4. **Grid convergence**: terminal velocity at three interface resolutions plus
   the Cauchy difference of the two finest.

Nondimensionalisation
---------------------
The solver is unit-agnostic, so the benchmark runs in bubble units:
``d = 1`` (length), ``sqrt(d/g) = 1`` (time, hence ``g = 1``), ``rho_l = 1``.
Then ``sigma = 1/Eo`` and ``mu_l`` follows from the Morton number
(:attr:`BubbleCase.mu_liquid`).  Velocities are in ``sqrt(g d)``; for air/water
at ``d = 3 mm`` that unit is ``sqrt(g d) = 0.172 m/s``.

Reproduce::

    PYTHONPATH=src python -m tensorfvm.verification.bubble_rise \
        --output docs/bubble-rise-2d --levels 12,16,20
    PYTHONPATH=src pytest tests/test_bubble_rise.py -q
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

# --------------------------------------------------------------------------- #
# Declared tolerances and bands -- every number the criteria use lives here.
# --------------------------------------------------------------------------- #

#: Declared tolerance of the terminal-velocity comparison.  Fixed *before*
#: running: 15 % is the platform-wide acceptability threshold for a
#: reference-value comparison, and the anchor is a 3-D correlation applied to a
#: 2-D confined run, so a tighter claim would be over-claiming.
TERMINAL_VELOCITY_TOLERANCE = 0.15

#: Ellipsoidal-regime aspect band (equatorial extent / polar extent).  Clift et
#: al. place the ellipsoidal regime between a nearly spherical shape (E -> 1)
#: and cap onset; reported aspect ratios over that regime span roughly
#: 1.05 ... 2.5.  The case here (Eo = 1.23) sits near the spherical/ellipsoidal
#: transition, so the low-Eo half of the band is used.
ASPECT_BAND = (1.00, 2.00)

#: Perimeter circularity band for one unbroken ellipsoidal bubble: a circle is
#: 1.0, a 2:1 ellipsoid ~0.93, a smeared or fragmented interface far below.
CIRCULARITY_BAND = (0.75, 1.00)

#: Mass-conservation criterion (worst per-step and total relative volume change).
MASS_TOLERANCE = 1e-6

#: Cauchy criterion on the terminal velocity between the two finest grids.
CAUCHY_TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# Case definition and analytic anchors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BubbleCase:
    """One bubble-rise case in bubble units (``d = g = rho_l = 1``)."""

    name: str
    eotvos: float
    morton: float
    density_ratio: float = 10.0
    viscosity_ratio: float = 55.0
    cells_per_diameter: int = 16
    domain_width: float = 4.0
    domain_height: float = 8.0
    start_y: float = 2.0
    end_time: float = 2.5
    csf_smoothing: int = 3
    momentum_advection: bool = True

    def __post_init__(self):
        if self.eotvos <= 0:
            raise ValueError("eotvos must be positive")
        if self.morton <= 0:
            raise ValueError("morton must be positive")
        if self.density_ratio < 1.01:
            raise ValueError("density_ratio must exceed 1 (gas lighter than liquid)")
        if self.viscosity_ratio < 1.0:
            raise ValueError("viscosity_ratio must be >= 1")
        if self.cells_per_diameter < 8:
            raise ValueError(
                "cells_per_diameter < 8 cannot resolve a one-cell PLIC interface")
        if self.domain_height <= self.start_y + 2.0:
            raise ValueError(
                "domain_height must leave at least 2 diameters of rise above start_y")
        if self.end_time <= 0:
            raise ValueError("end_time must be positive")

    @property
    def sigma(self) -> float:
        """Surface tension coefficient, ``sigma = rho_l g d^2/Eo`` (rho_l=g=d=1)."""
        return 1.0 / self.eotvos

    @property
    def mu_liquid(self) -> float:
        """Liquid viscosity implied by the Morton number.

        ``M = g mu_l^4 (rho_l - rho_g)/(rho_l^2 sigma^3)`` implies
        ``mu_l = (M rho_l^2 sigma^3/(g drho))^(1/4)``.
        """
        delta_rho = 1.0 - 1.0 / self.density_ratio
        return (self.morton * self.sigma ** 3 / delta_rho) ** 0.25

    @property
    def mu_gas(self) -> float:
        return self.mu_liquid / self.viscosity_ratio

    def fan_tsuchiya_velocity(self) -> float:
        """Mendelson/Fan-Tsuchiya terminal velocity, in ``sqrt(g d)``."""
        return fan_tsuchiya_velocity(self.eotvos)

    def solve_config(self) -> tuple[GridConfig, TwoPhaseConfig]:
        n = self.cells_per_diameter
        grid = GridConfig(
            nx=int(round(self.domain_width * n)),
            ny=int(round(self.domain_height * n)),
            length=self.domain_width,
            height=self.domain_height,
        )
        tp = TwoPhaseConfig(
            enabled=True,
            rho_liquid=1.0,
            rho_gas=1.0 / self.density_ratio,
            mu_liquid=self.mu_liquid,
            mu_gas=self.mu_gas,
            sigma=self.sigma,
            surface_tension=True,
            gravity=1.0,
            pressure_form="reduced",
            top_atmosphere=True,
            bottom_wall=True,
            side_outlet=False,
            vof_scheme="plic",
            momentum_advection=self.momentum_advection,
            warm_start_pressure=True,
            csf_smoothing=self.csf_smoothing,
        )
        return grid, tp

    def initial_alpha(self, nx: int, ny: int, dx: float, dy: float) -> torch.Tensor:
        """Bubble indicator (1 inside the bubble) for a unit-diameter disk."""
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        cx = 0.5 * self.domain_width
        r2 = (x[None, :] - cx) ** 2 + (y[:, None] - self.start_y) ** 2
        return (r2 <= 0.25).to(torch.float64)


def fan_tsuchiya_velocity(eotvos: float) -> float:
    """Terminal velocity of an ellipsoidal bubble (Mendelson 1967, Fan 1990).

    Dimensional form ``U = sqrt(2.14 sigma/(rho_l d) + 0.505 g d)``; in bubble
    units (``sigma = 1/Eo``, ``rho_l = g = d = 1``) this is
    ``U = sqrt(2.14/Eo + 0.505)``.
    """
    if eotvos <= 0:
        raise ValueError("eotvos must be positive")
    return math.sqrt(2.14 / eotvos + 0.505)


def hadamard_rybczynski_velocity(
        eotvos: float, morton: float, density_ratio: float,
        viscosity_ratio: float) -> float:
    """Creeping-flow (``Re -> 0``) rise velocity of a spherical bubble.

    ``U_HR = (2/3)(drho g a^2/mu_l)(mu_l + mu_g)/(2 mu_l + 3 mu_g)`` with
    ``a = d/2``; in bubble units ``drho = 1 - 1/rho_r`` and ``mu_l`` comes from
    the Morton number.  This is the exact answer in the bottom-left corner of the
    Clift map (``Re << 1, Eo << 1``) and is **not** applicable to the ellipsoidal
    benchmark case -- the report says so instead of quoting it as an anchor there.
    """
    case = BubbleCase(name="_anchor", eotvos=eotvos, morton=morton,
                      density_ratio=density_ratio, viscosity_ratio=viscosity_ratio)
    delta_rho = 1.0 - 1.0 / density_ratio
    mu_l, mu_g = case.mu_liquid, case.mu_gas
    a = 0.5
    return ((2.0 / 3.0) * delta_rho * a * a / mu_l
            * (mu_l + mu_g) / (2.0 * mu_l + 3.0 * mu_g))


#: Approximate Clift-Grace-Weber regime windows, tested in order.  These are
#: *windows* quoted from the map's description, not fitted boundaries: the
#: benchmark uses them to state which regime a measured point falls in, and
#: every criterion that depends on them says so.
CLIFT_WINDOWS = (
    ("spherical", 0.4, 1.0, "Eo < 0.4 and Re < 1: surface tension dominates, sphere"),
    ("ellipsoidal", 40.0, 1.0e3, "0.4 < Eo < 40, Re <~ 1e3: oblate ellipsoid, no cap"),
    ("spherical-cap", math.inf, math.inf, "Eo > 40: cap with a flattened rear"),
)


def clift_regime(eotvos: float, morton: float, reynolds: float) -> dict:
    """Classify a measured state on the Clift-Grace-Weber map (approximate).

    Returns the window the point falls in *plus* the raw numbers, so a reader can
    disagree with the classification without losing the data.  ``morton`` is
    echoed but not used by the windows -- a documented limitation (M only shifts
    the ellipsoidal/cap boundary).
    """
    for name, eo_max, re_max, comment in CLIFT_WINDOWS:
        if eotvos < eo_max and reynolds < re_max:
            return {"regime": name, "eotvos": eotvos, "morton": morton,
                    "reynolds": reynolds, "window": comment}
    return {"regime": "spherical-cap", "eotvos": eotvos, "morton": morton,
            "reynolds": reynolds, "window": "Eo > 40"}


# --------------------------------------------------------------------------- #
# Field metrics
# --------------------------------------------------------------------------- #


def bubble_metrics(alpha: torch.Tensor, dx: float, dy: float) -> dict:
    """Geometric metrics of the bubble from its indicator field ``alpha``.

    ``alpha`` is the *bubble* fraction (1 inside the gas).  All lengths are in
    bubble units.

    * ``area`` -- ``sum(alpha) dx dy``;
    * ``d_equivalent = sqrt(4 A/pi)``;
    * ``centroid`` -- alpha-weighted first moment;
    * ``width``/``height`` -- extents of the ``alpha > 0.5`` set;
    * ``aspect_ratio = width/height`` (>1 = oblate, the ellipsoidal signature);
    * ``circularity = pi d_equivalent / perimeter``, the perimeter being the
      length of the ``alpha = 0.5`` crossings counted on cell faces (standard
      interface-length estimator: a circle gives 1, a smeared or fragmented
      interface much less).
    """
    if alpha.ndim != 2:
        raise ValueError("alpha must be 2-D")
    ny, nx = alpha.shape
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    total = float(alpha.sum())
    if total <= 0:
        return {"area": 0.0, "d_equivalent": 0.0, "centroid": [float("nan")] * 2,
                "width": 0.0, "height": 0.0, "aspect_ratio": float("nan"),
                "circularity": 0.0, "perimeter": float("nan")}
    area = total * dx * dy
    xc = float((alpha * x[None, :]).sum() / total)
    yc = float((alpha * y[:, None]).sum() / total)
    mask = alpha > 0.5
    width = float(mask.any(dim=0).sum()) * dx
    height = float(mask.any(dim=1).sum()) * dy
    inside = mask.to(torch.float64)
    crossings_x = (inside[:, 1:] - inside[:, :-1]).abs().sum()
    crossings_y = (inside[1:, :] - inside[:-1, :]).abs().sum()
    perimeter = float(crossings_x) * dy + float(crossings_y) * dx
    d_eq = math.sqrt(4.0 * area / math.pi)
    circularity = math.pi * d_eq / perimeter if perimeter > 0 else 0.0
    return {
        "area": area,
        "d_equivalent": d_eq,
        "centroid": [xc, yc],
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height > 0 else float("nan"),
        "circularity": circularity,
        "perimeter": perimeter,
    }


def second_moment_aspect(alpha: torch.Tensor, dx: float, dy: float) -> float:
    """Oblateness from the alpha-weighted covariance, ``sqrt(var_x/var_y)``.

    The ratio of the semi-axes of the equivalent homogeneous ellipse.  Reported
    next to the extent-based aspect ratio so a reader can see whether the shape
    conclusion depends on the estimator.
    """
    ny, nx = alpha.shape
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    total = float(alpha.sum())
    if total <= 0:
        return float("nan")
    xc = float((alpha * x[None, :]).sum() / total)
    yc = float((alpha * y[:, None]).sum() / total)
    vx = float((alpha * (x[None, :] - xc) ** 2).sum() / total)
    vy = float((alpha * (y[:, None] - yc) ** 2).sum() / total)
    if vy <= 0:
        return float("nan")
    return math.sqrt(vx / vy)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #


def run_case(case: BubbleCase, snapshot_fracs: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0),
             progress: bool = True, max_steps: int | None = None) -> dict:
    """Integrate one case and return its metrics, history and criteria inputs.

    The step is the solver's own explicit stable step
    (:meth:`TwoPhaseSolver.stable_time_step`), recomputed every step, so the
    CFL / capillary / viscous limits hold by construction and the PLIC transport
    never sees a directional Courant number above 1.
    """
    grid, tp = case.solve_config()
    solver = TwoPhaseSolver(grid, tp)
    bubble0 = case.initial_alpha(grid.nx, grid.ny, solver.dx, solver.dy)
    # Geometry validation *before* physics: the discrete disk must reproduce the
    # analytic area to well under a percent, otherwise the reported equivalent
    # diameter is not the one the dimensionless groups were built with.
    area0 = float(bubble0.sum()) * solver.dx * solver.dy
    geometry_error = abs(area0 - math.pi / 4.0) / (math.pi / 4.0)
    solver.set_alpha(1.0 - bubble0)            # the solver stores liquid fraction
    dx, dy = solver.dx, solver.dy

    def bubble() -> torch.Tensor:
        return 1.0 - solver.alpha

    volume0 = float(bubble().sum()) * dx * dy
    history: list[dict] = []
    snapshots: list[dict] = []
    worst_step_drift = 0.0
    worst_repair = 0.0
    volume_prev = volume0
    next_snapshot = 1
    step = 0
    t0 = time.time()
    while solver.time < case.end_time:
        dt = solver.stable_time_step()
        if solver.time + dt > case.end_time:
            dt = case.end_time - solver.time
        solver.step(dt)
        step += 1
        vol = float(bubble().sum()) * dx * dy
        worst_step_drift = max(worst_step_drift, abs(vol - volume_prev) / volume0)
        volume_prev = vol
        repairs = solver.plic_diagnostics.get("sweeps") or []
        if repairs:
            worst_repair = max(worst_repair, max(repairs))
        m = bubble_metrics(bubble(), dx, dy)
        m["time"] = float(solver.time)
        m["aspect_second_moment"] = second_moment_aspect(bubble(), dx, dy)
        m["umax"] = float(max(solver.u.abs().max(), solver.v.abs().max()))
        m["divergence_max"] = float(solver._divergence().abs().max())
        m["volume_drift"] = (vol - volume0) / volume0
        history.append(m)
        frac = solver.time / case.end_time
        while next_snapshot < len(snapshot_fracs) and frac >= snapshot_fracs[next_snapshot]:
            snapshots.append({"time": float(solver.time),
                              "alpha": bubble().detach().cpu().clone()})
            next_snapshot += 1
        if progress and step % 100 == 0:
            print(f"  [{case.name} n={case.cells_per_diameter}] t={solver.time:.3f} "
                  f"yc={m['centroid'][1]:.3f} U?={m['umax']:.3f} "
                  f"aspect={m['aspect_ratio']:.2f}", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    if not snapshots:
        snapshots.append({"time": float(solver.time),
                          "alpha": bubble().detach().cpu().clone()})
    elapsed = time.time() - t0

    # Rise velocity by central difference on the centroid; the terminal value is
    # the mean over the last quarter of the window (documented convention).
    times = torch.tensor([h["time"] for h in history], dtype=torch.float64)
    yc = torch.tensor([h["centroid"][1] for h in history], dtype=torch.float64)
    velocity = torch.zeros_like(yc)
    if len(times) > 2:
        velocity[1:-1] = (yc[2:] - yc[:-2]) / (times[2:] - times[:-2])
        velocity[0] = (yc[1] - yc[0]) / (times[1] - times[0])
        velocity[-1] = (yc[-1] - yc[-2]) / (times[-1] - times[-2])
    for h, u in zip(history, velocity.tolist()):
        h["rise_velocity"] = u
    tail = times >= 0.75 * case.end_time
    terminal = float(velocity[tail].mean()) if int(tail.sum()) else float(velocity[-1])
    peak = float(velocity.max())

    final = history[-1]
    d_measured = final["d_equivalent"]
    reynolds = 1.0 * terminal * d_measured / case.mu_liquid
    weber = terminal ** 2 * d_measured / case.sigma
    capillary = case.mu_liquid * terminal / case.sigma
    final_volume = float(bubble().sum()) * dx * dy

    return {
        "case": asdict(case),
        "derived": {"sigma": case.sigma, "mu_liquid": case.mu_liquid,
                    "mu_gas": case.mu_gas, "u_ref_sqrt_gd": 1.0},
        "grid": {"nx": grid.nx, "ny": grid.ny, "dx": dx, "dy": dy,
                 "cells_per_diameter": case.cells_per_diameter},
        "geometry_validation": {
            "analytic_area": math.pi / 4.0,
            "initial_area": area0,
            "relative_error": geometry_error,
            "cells_across_bubble": float(case.cells_per_diameter),
        },
        "terminal_velocity": terminal,
        "peak_velocity": peak,
        "terminal_velocity_anchor": case.fan_tsuchiya_velocity(),
        "dimensionless": {
            "eotvos": case.eotvos, "morton": case.morton, "reynolds": reynolds,
            "weber": weber, "capillary": capillary,
            "density_ratio": case.density_ratio,
            "viscosity_ratio": case.viscosity_ratio,
        },
        "regime": clift_regime(case.eotvos, case.morton, reynolds),
        "shape": {
            "aspect_ratio": final["aspect_ratio"],
            "aspect_second_moment": final["aspect_second_moment"],
            "circularity": final["circularity"],
            "d_equivalent": d_measured,
            "centroid": final["centroid"],
        },
        "conservation": {
            "initial_volume": volume0,
            "final_volume": final_volume,
            "total_relative_drift": (final_volume - volume0) / volume0,
            "worst_step_relative_drift": worst_step_drift,
            "worst_bound_repair_fraction": worst_repair,
        },
        "cost": {"steps": step, "final_time": float(solver.time),
                 "elapsed_seconds": elapsed,
                 "cg_iterations_last": solver.cg_diagnostics.get("iterations"),
                 "cg_converged_last": solver.cg_diagnostics.get("converged")},
        "history": history,
        "snapshots": snapshots,
    }


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #


def evaluate_criteria(runs: list[dict],
                      tolerance: float = TERMINAL_VELOCITY_TOLERANCE) -> dict:
    """Turn measured runs into the node's acceptance criteria.

    ``runs`` must be ordered coarse -> fine.  Each criterion returns its numbers
    plus a ``pass`` flag.  A failing criterion is reported as failing: the
    benchmark may fail, it is not allowed to hide.
    """
    if not runs:
        raise ValueError("need at least one run")
    finest = runs[-1]
    anchor = finest["terminal_velocity_anchor"]
    error = abs(finest["terminal_velocity"] - anchor) / anchor
    signed = (finest["terminal_velocity"] - anchor) / anchor
    c_terminal = {
        "criterion": f"|U_T - U_anchor|/U_anchor < {tolerance:.0%} "
                     "(Fan & Tsuchiya correlation, declared tolerance)",
        "U_terminal": finest["terminal_velocity"],
        "U_anchor": anchor,
        "relative_error": error,
        "signed_error": signed,
        "declared_tolerance": tolerance,
        "pass": bool(error < tolerance),
    }

    aspect = finest["shape"]["aspect_ratio"]
    circ = finest["shape"]["circularity"]
    shape_ok = (ASPECT_BAND[0] <= aspect <= ASPECT_BAND[1]
                and CIRCULARITY_BAND[0] <= circ <= CIRCULARITY_BAND[1])
    c_shape = {
        "criterion": f"aspect ratio in {ASPECT_BAND} and circularity in "
                     f"{CIRCULARITY_BAND} (ellipsoidal band, qualitative)",
        "aspect_ratio": aspect,
        "aspect_second_moment": finest["shape"]["aspect_second_moment"],
        "circularity": circ,
        "regime": finest["regime"]["regime"],
        "d_equivalent": finest["shape"]["d_equivalent"],
        "pass": bool(shape_ok),
    }

    worst = max(r["conservation"]["worst_step_relative_drift"] for r in runs)
    total = max(abs(r["conservation"]["total_relative_drift"]) for r in runs)
    c_mass = {
        "criterion": f"worst per-step |dV|/V0 < {MASS_TOLERANCE:g} and "
                     f"|total drift| < {MASS_TOLERANCE:g} over the whole run",
        "worst_step_relative_drift": worst,
        "total_relative_drift_finest": finest["conservation"]["total_relative_drift"],
        "worst_total_relative_drift": total,
        "pass": bool(worst < MASS_TOLERANCE and total < MASS_TOLERANCE),
    }

    velocities = [r["terminal_velocity"] for r in runs]
    errors = [abs(v - r["terminal_velocity_anchor"]) / r["terminal_velocity_anchor"]
              for v, r in zip(velocities, runs)]
    monotone = all(errors[i] > errors[i + 1] - 1e-12 for i in range(len(errors) - 1))
    cauchy = (abs(velocities[-1] - velocities[-2]) / abs(velocities[-1])
              if len(velocities) > 1 and velocities[-1] != 0 else float("nan"))
    c_grid = {
        "criterion": "terminal velocity converges: anchor error decreases with "
                     f"resolution and Cauchy(U_N, U_N-1) < {CAUCHY_TOLERANCE:.0%}",
        "cells_per_diameter": [r["grid"]["cells_per_diameter"] for r in runs],
        "terminal_velocity": velocities,
        "anchor_error": errors,
        "errors_monotone_decreasing": bool(monotone),
        "cauchy_finest": cauchy,
        "pass": bool(monotone and cauchy < CAUCHY_TOLERANCE),
    }

    regime = finest["regime"]["regime"]
    c_regime = {
        "criterion": "measured (Eo, M, Re) lies in the ellipsoidal window the "
                     "shape band and the velocity anchor are claimed for",
        "regime": regime,
        "eotvos": finest["dimensionless"]["eotvos"],
        "morton": finest["dimensionless"]["morton"],
        "reynolds": finest["dimensionless"]["reynolds"],
        "pass": bool(regime == "ellipsoidal"),
    }

    geometry = max(r["geometry_validation"]["relative_error"] for r in runs)
    c_geometry = {
        "criterion": "initial bubble reproduces the analytic area pi d^2/4 within 2%",
        "worst_relative_error": geometry,
        "pass": bool(geometry < 0.02),
    }

    criteria = {
        "C1_shape_band": c_shape,
        "C2_mass_conservation": c_mass,
        "C3_grid_convergence": c_grid,
        "C4_terminal_velocity": c_terminal,
        "C5_regime_window": c_regime,
        "C6_geometry_validation": c_geometry,
    }
    criteria["all_pass"] = all(c["pass"] for c in criteria.values())
    return criteria


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def write_figures(result_runs: list[dict], output_dir: Path) -> list[str]:
    """Write the four report figures and return their file names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    finest = result_runs[-1]

    # Fig 1 -- interface snapshots (bubble fraction) on the finest grid
    snaps = finest["snapshots"]
    ncol = len(snaps)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 4.2), squeeze=False)
    nx, ny = finest["grid"]["nx"], finest["grid"]["ny"]
    width = nx * finest["grid"]["dx"]
    height = ny * finest["grid"]["dy"]
    im = None
    for ax, snap in zip(axes[0], snaps):
        im = ax.imshow(snap["alpha"].numpy(), origin="lower",
                       extent=(0, width, 0, height), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"t = {snap['time']:.2f}")
        ax.set_xlabel("x/d")
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("y/d")
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="bubble fraction")
    fig.suptitle(f"Bubble rise: Eo={finest['case']['eotvos']:.2f}, "
                 f"M={finest['case']['morton']:.2e}, "
                 f"n={finest['grid']['cells_per_diameter']}/d")
    fig.savefig(output_dir / "fig1-interface-snapshots.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    names.append("fig1-interface-snapshots.png")

    # Fig 2 -- centroid, rise velocity and shape history with the anchor band
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for run in result_runs:
        h = run["history"]
        t = [q["time"] for q in h]
        label = f"n={run['grid']['cells_per_diameter']}"
        axes[0].plot(t, [q["centroid"][1] for q in h], label=label)
        axes[1].plot(t, [q["rise_velocity"] for q in h], label=label)
        axes[2].plot(t, [q["aspect_ratio"] for q in h], label=label)
    anchor = finest["terminal_velocity_anchor"]
    axes[1].axhline(anchor, color="k", ls="--", lw=1, label="Fan-Tsuchiya anchor")
    axes[1].axhspan(anchor * (1 - TERMINAL_VELOCITY_TOLERANCE),
                    anchor * (1 + TERMINAL_VELOCITY_TOLERANCE),
                    color="k", alpha=0.08,
                    label=f"+/-{TERMINAL_VELOCITY_TOLERANCE:.0%} band")
    axes[2].axhspan(*ASPECT_BAND, color="C2", alpha=0.12, label="ellipsoidal band")
    axes[0].set_xlabel("t sqrt(g/d)")
    axes[0].set_ylabel("centroid y/d")
    axes[0].set_title("Rise")
    axes[1].set_xlabel("t sqrt(g/d)")
    axes[1].set_ylabel("U sqrt(gd)")
    axes[1].set_title("Rise velocity vs anchor")
    axes[2].set_xlabel("t sqrt(g/d)")
    axes[2].set_ylabel("aspect (width/height)")
    axes[2].set_title("Shape")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2-history.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    names.append("fig2-history.png")

    # Fig 3 -- grid convergence of the terminal velocity + volume conservation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ns = [r["grid"]["cells_per_diameter"] for r in result_runs]
    us = [r["terminal_velocity"] for r in result_runs]
    axes[0].plot(ns, us, "o-", label="measured U_T")
    axes[0].axhline(anchor, color="k", ls="--", label="Fan-Tsuchiya anchor")
    axes[0].axhspan(anchor * (1 - TERMINAL_VELOCITY_TOLERANCE),
                    anchor * (1 + TERMINAL_VELOCITY_TOLERANCE),
                    color="k", alpha=0.08)
    axes[0].set_xlabel("cells per diameter")
    axes[0].set_ylabel("U_T sqrt(gd)")
    axes[0].set_title("Grid convergence of the terminal velocity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for run in result_runs:
        h = run["history"]
        axes[1].semilogy([q["time"] for q in h],
                         [max(abs(q["volume_drift"]), 1e-18) for q in h],
                         label=f"n={run['grid']['cells_per_diameter']}")
    axes[1].axhline(MASS_TOLERANCE, color="r", ls="--", label=f"{MASS_TOLERANCE:g} limit")
    axes[1].set_xlabel("t sqrt(g/d)")
    axes[1].set_ylabel("|dV|/V0")
    axes[1].set_title("Volume conservation")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3-convergence-conservation.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    names.append("fig3-convergence-conservation.png")

    # Fig 4 -- position on the Clift map
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    reynolds_axis = [10 ** (-1 + 0.01 * k) for k in range(501)]
    ax.loglog(reynolds_axis, [0.4 for _ in reynolds_axis], "k--", lw=1,
              label="spherical / ellipsoidal (Eo = 0.4)")
    ax.axhline(40.0, color="k", ls=":", lw=1, label="ellipsoidal / cap (Eo = 40)")
    ax.axvline(1.0, color="0.6", ls=":", lw=1)
    for run in result_runs:
        d = run["dimensionless"]
        ax.plot(max(d["reynolds"], 1e-3), d["eotvos"], "o",
                label=f"n={run['grid']['cells_per_diameter']}")
    ax.set_xlabel("Re = rho_l U_T d/mu_l")
    ax.set_ylabel("Eo")
    ax.set_title("Position on the Clift-Grace-Weber map")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4-clift-map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    names.append("fig4-clift-map.png")
    return names


def write_summary(result_runs: list[dict], criteria: dict, figures: list[str],
                  output_dir: Path, elapsed: float | None = None) -> dict:
    """Write ``summary.json`` (numbers only) and return the payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "two-phase-vof-bubble-rise-2d",
        "declared_tolerances": {
            "terminal_velocity": TERMINAL_VELOCITY_TOLERANCE,
            "mass": MASS_TOLERANCE,
            "cauchy": CAUCHY_TOLERANCE,
            "aspect_band": list(ASPECT_BAND),
            "circularity_band": list(CIRCULARITY_BAND),
        },
        "runs": [{k: v for k, v in r.items() if k != "snapshots"} for r in result_runs],
        "criteria": {k: v for k, v in criteria.items() if k != "all_pass"},
        "all_pass": criteria["all_pass"],
        "figures": figures,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2-D bubble rise benchmark")
    parser.add_argument("--output", type=Path, default=Path("docs/bubble-rise-2d"))
    parser.add_argument("--levels", default="12,16,20",
                        help="comma separated cells per bubble diameter (coarse->fine)")
    parser.add_argument("--eotvos", type=float, default=1.23)
    parser.add_argument("--morton", type=float, default=2.63e-11)
    parser.add_argument("--density-ratio", type=float, default=10.0)
    parser.add_argument("--viscosity-ratio", type=float, default=55.0)
    parser.add_argument("--end-time", type=float, default=2.5)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch intra-op threads (0 = library default)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    started = time.time()
    runs = []
    for n in levels:
        case = BubbleCase(name="air-water-3mm", eotvos=args.eotvos,
                          morton=args.morton, density_ratio=args.density_ratio,
                          viscosity_ratio=args.viscosity_ratio,
                          cells_per_diameter=n, end_time=args.end_time)
        print(f"running {case.name} n={n} grid="
              f"{case.solve_config()[0].nx}x{case.solve_config()[0].ny}", flush=True)
        runs.append(run_case(case))
    criteria = evaluate_criteria(runs)
    figures = [] if args.no_figures else write_figures(runs, args.output)
    payload = write_summary(runs, criteria, figures, args.output,
                            elapsed=round(time.time() - started, 1))
    for name, crit in criteria.items():
        if name == "all_pass":
            continue
        print(f"{name}: {'PASS' if crit['pass'] else 'FAIL'} -- {crit['criterion']}")
    print(f"all_pass: {payload['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
