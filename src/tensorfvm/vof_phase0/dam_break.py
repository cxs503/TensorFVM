"""Dam-break driver for the Phase-0 two-phase VOF gate.

The driver couples the Phase-0 algebraic VOF transport
(:mod:`tensorfvm.vof_phase0.advection`) to the variable-density projection
solver of :mod:`tensorfvm.vof_phase0.solver`.

Geometry (Martin & Moyce 1952)
------------------------------
A rectangular column of liquid, half-width ``a`` and height ``h0``, is released
on a rigid bed:

    domain = [0, length] x [0, height]      closed side walls, free-slip bed,
                                            open (atmospheric) top
    liquid = [0, a] x [0, h0]               at t = 0, at rest

``h0 = aspect * a``; the digitised reference curve corresponds to
``aspect = 2``.  The initial pressure field is the column's own hydrostatic head
(``p = rho_l g (h0 - y)`` inside the column, atmospheric outside), so the
initial state is the physical rest state and the surge is driven by the
pressure jump at the dam face rather than by a numerical start-up transient.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .anchors import to_dimensionless_front, to_dimensionless_time
from .solver import Phase0TwoPhaseSolver, cell_centres

__all__ = [
    "DamBreakConfig",
    "run_dam_break",
    "surge_front",
    "column_height",
    "initial_alpha",
]


@dataclass
class DamBreakConfig:
    """Configuration of the Martin & Moyce dam-break case."""

    #: Initial column half-width ``a`` [m] (the Martin & Moyce apparatus value).
    a: float = 0.05715
    #: Initial column height / half-width ratio; the reference curve is 2.
    aspect: float = 2.0
    #: Domain size [m].
    length: float = 0.6
    height: float = 0.2
    #: Grid.
    nx: int = 120
    ny: int = 40
    #: Fluid properties.
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.0e-5
    gravity: float = 9.81
    #: Interface transport: ``"global"`` bounds alpha to [0, 1].
    vof_bounds: str = "global"
    #: Time integration.
    cfl: float = 0.35
    t_star_end: float = 4.0
    max_steps: int = 6000
    #: Record a sample every ``track_every`` steps (1 = every step).
    track_every: int = 1
    #: Initial pressure field: ``"hydrostatic"`` (physical) or ``"zero"``.
    pressure_init: str = "hydrostatic"

    def __post_init__(self) -> None:
        if self.a <= 0.0 or self.aspect <= 0.0:
            raise ValueError("a and aspect must be positive")
        if self.h0 > self.height + 1e-15:
            raise ValueError(
                f"initial column height h0={self.h0:.4g} m exceeds domain "
                f"height {self.height:.4g} m")
        if self.a >= self.length:
            raise ValueError("column half-width must be smaller than the domain")
        if self.nx < 8 or self.ny < 4:
            raise ValueError("grid too coarse for a dam-break simulation")
        if self.rho_liquid <= 0.0 or self.rho_gas <= 0.0:
            raise ValueError("densities must be positive")
        if self.pressure_init not in ("hydrostatic", "zero"):
            raise ValueError(f"unknown pressure_init {self.pressure_init!r}")

    @property
    def h0(self) -> float:
        return self.aspect * self.a

    @property
    def density_ratio(self) -> float:
        return self.rho_liquid / self.rho_gas

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def t_star_scale(self) -> float:
        """Seconds per unit dimensionless time (``sqrt(a/g)``)."""
        return math.sqrt(self.a / self.gravity)

    @property
    def initial_volume(self) -> float:
        return self.a * self.h0


def initial_alpha(cfg: DamBreakConfig) -> torch.Tensor:
    """Unit volume fraction inside the initial column, zero elsewhere."""
    xs = cell_centres(cfg.nx, cfg.dx)
    ys = cell_centres(cfg.ny, cfg.dy)
    inside = (xs[None, :] <= cfg.a) & (ys[:, None] <= cfg.h0)
    return inside.to(torch.float64)


def _depth(alpha: torch.Tensor, dy: float) -> np.ndarray:
    return alpha.detach().cpu().numpy().sum(axis=0) * dy


def surge_front(alpha: torch.Tensor, dx: float, dy: float,
                threshold: float | None = None) -> float:
    """Surge-front position ``x_f`` [m] from the depth profile.

    The front is the outermost column whose liquid depth exceeds ``threshold``
    (default: half a cell -- an alpha of 0.5 in a single bottom cell).  A depth
    criterion is used instead of a single-cell alpha so that a smeared algebraic
    interface is integrated consistently.
    """
    depth = _depth(alpha, dy)
    thr = 0.5 * dy if threshold is None else float(threshold)
    wet = np.where(depth > thr)[0]
    if wet.size == 0:
        return 0.0
    return float((wet[-1] + 0.5) * dx)


def column_height(alpha: torch.Tensor, dy: float,
                  n_columns: int = 4, alpha_cut: float = 0.5) -> float:
    """Residual column height against the upstream wall [m]."""
    a = alpha.detach().cpu().numpy()[:, :max(1, int(n_columns))]
    rows = np.where((a > alpha_cut).any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    return float((rows[-1] + 0.5) * dy)


def run_dam_break(cfg: DamBreakConfig, solver: Phase0TwoPhaseSolver | None = None,
                  track: bool = True) -> dict:
    """Run the Martin & Moyce case and return the measured history.

    Returns numpy arrays ``t``, ``t_star``, ``front`` [m], ``front_star``,
    ``height`` [m], ``height_star`` (``h/h0``), ``volume`` [m^2],
    ``volume_drift`` (relative) plus run descriptors.  ``alpha_final`` is the
    final volume-fraction field (needed for the free-surface snapshots).
    """
    if solver is None:
        solver = Phase0TwoPhaseSolver(
            nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height,
            rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
            mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas,
            gravity=cfg.gravity, alpha_cfl=cfg.cfl,
            vof_bounds=cfg.vof_bounds)
    solver.set_alpha(initial_alpha(cfg))
    if cfg.pressure_init == "hydrostatic":
        solver.set_hydrostatic_pressure(cfg.h0, cfg.a)

    v0 = solver.liquid_volume()
    t_star_end = cfg.t_star_end * cfg.t_star_scale

    t_hist: list[float] = []
    front_hist: list[float] = []
    height_hist: list[float] = []
    volume_hist: list[float] = []
    div_hist: list[float] = []
    cg_hist: list[dict] = []

    started = time.perf_counter()
    diverged = False
    t = 0.0
    while t < t_star_end and solver.steps < cfg.max_steps:
        dt = min(solver.stable_dt(), t_star_end - t)
        try:
            info = solver.step(dt)
        except (RuntimeError, ValueError, FloatingPointError):
            diverged = True
            break
        t = float(solver.time)
        if not torch.isfinite(solver.alpha).all() or not torch.isfinite(
                solver.u).all():
            diverged = True
            break
        if track and (solver.steps % max(1, cfg.track_every) == 0
                      or t >= t_star_end):
            t_hist.append(t)
            front_hist.append(surge_front(solver.alpha, cfg.dx, cfg.dy))
            height_hist.append(column_height(solver.alpha, cfg.dy))
            volume_hist.append(info["volume"])
            div_hist.append(info["divergence_max"])
            cg_hist.append(info["cg"])

    elapsed = time.perf_counter() - started
    t_arr = np.asarray(t_hist, dtype=float)
    vol_arr = np.asarray(volume_hist, dtype=float)
    front_arr = np.asarray(front_hist, dtype=float)
    alpha_np = solver.alpha.detach().cpu().numpy()
    return {
        "config": asdict(cfg),
        "h0": cfg.h0,
        "density_ratio": cfg.density_ratio,
        "t": t_arr,
        "t_star": np.array([to_dimensionless_time(x, cfg.a, cfg.gravity)
                            for x in t_arr]),
        "front": front_arr,
        "front_star": np.array([to_dimensionless_front(x, cfg.a)
                                for x in front_arr]),
        "height": np.asarray(height_hist, dtype=float),
        "height_star": np.asarray(height_hist, dtype=float) / cfg.h0,
        "volume": vol_arr,
        "volume_initial": v0,
        "volume_drift": (vol_arr - v0) / v0,
        "divergence_max": np.asarray(div_hist, dtype=float),
        "cg": cg_hist,
        "alpha_final": alpha_np,
        "alpha_bounds": (float(alpha_np.min()), float(alpha_np.max())),
        "steps": solver.steps,
        "elapsed_s": elapsed,
        "diverged": diverged,
        "max_volume_residual": float(np.max(np.abs(vol_arr - v0)))
        if vol_arr.size else 0.0,
        "solver": solver,
    }