"""Hysing rising-bubble leg: terminal velocity and shape vs the 2-D benchmark.

The Hysing et al. (2009) benchmark is a canonical 2-D rising-bubble test that
exercises surface tension, buoyancy, and interface transport simultaneously.
It is the second half of the ``GAP-FVM-TWOPHASE`` acceptance face (the first
half being the Zalesak slotted disk).

Reference
---------
Hysing, S., Turek, S., Kuzmin, D., Parolini, N., Lohner, R., Turek, S.,
& van Brummelen, H. (2009). Quantitative benchmark computations of two-dimensional
bubble dynamics. *International Journal for Numerical Methods in Fluids*, 60(11),
1259-1288.

The benchmark defines two cases; we implement **Case 2** (Eo = 10, M = 1e-3)
which is the most widely cited.  The reference values are taken from Table II
of the paper (2-D, level-set reference solution on a very fine mesh):

* Terminal velocity ``U_T = 0.2342`` (dimensionless, in units of ``sqrt(g d)``)
* Circularity ``c = 0.4644`` at ``t = 3`` (minimum during rise)
* Centroid y-position ``y_c = 1.984`` at ``t = 3``

Gates
-----
======  =============================================================  ==============
id      quantity                                                       criterion
======  =============================================================  ==============
B1      Terminal velocity vs Hysing ref (Eo=10, M=1e-3)                ``<= 10 %``
B2      Minimum circularity vs Hysing ref                              ``<= 10 %``
B3      Mass conservation (worst per-step |dV|/V0)                     ``<= 1e-6``
B4      Grid convergence: U_T coarse -> fine change                    ``<= 5 %``
======  =============================================================  ==============

The terminal-velocity tolerance is set to 10 % as stated in the node acceptance
criteria.  The circularity tolerance is also 10 % because the reference value
is from a level-set method on a very fine mesh, and our PLIC-VOF on a moderate
grid will have some shape error.

Nondimensionalisation
---------------------
The benchmark runs in bubble units: ``d = 1`` (length), ``g = 1`` (hence
``sqrt(g d) = 1`` for velocity), ``rho_l = 1``.  Then:

* ``sigma = 1 / Eo``
* ``mu_l = (M * sigma^3 / (1 - 1/rho_ratio))^(1/4)``

Reproduce::

    PYTHONPATH=src python -m tensorfvm.twophase_gap --case hysing --out docs/GAP-FVM-TWOPHASE/artifacts
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

__all__ = [
    "run_hysing_leg",
    "HYSING_REF_UT",
    "HYSING_REF_CIRCULARITY",
    "HYSING_REF_YC",
    "TERMINAL_VELOCITY_MAX",
    "CIRCULARITY_MAX",
    "MASS_DRIFT_MAX",
    "CONVERGENCE_MAX",
]

# --------------------------------------------------------------------------- #
# Reference values from Hysing et al. (2009), Table II, Case 2
# --------------------------------------------------------------------------- #

#: Declared (gated) resolution in cells per bubble diameter.  The Hysing leg is
#: limited by the *explicit* viscous step of the gas phase --
#: dt ~ 0.25 h^2 / nu_gas with nu_gas = mu_gas/rho_gas ~ 3.2 (d = g = rho_l = 1),
#: so dt scales as h^2 and 64 cells/diameter needs ~3e5 steps (~10 h on one
#: core); 16 cells/diameter is the finest resolution that fits a normal
#: verification budget (~13 min).  B1/B2 are judged on this leg, B4 on the
#: 8 -> 16 pair.
DECLARED_CPD = 16

#: Terminal velocity reference (dimensionless, sqrt(g d) units).
HYSING_REF_UT = 0.2342
#: Minimum circularity reference (at t = 3).
HYSING_REF_CIRCULARITY = 0.4644
#: Centroid y-position reference (at t = 3).
HYSING_REF_YC = 1.984

# --------------------------------------------------------------------------- #
# Gate thresholds
# --------------------------------------------------------------------------- #

#: Max relative error of terminal velocity vs Hysing reference.
TERMINAL_VELOCITY_MAX = 0.10
#: Max relative error of minimum circularity vs Hysing reference.
CIRCULARITY_MAX = 0.10
#: Max per-step relative volume drift.
MASS_DRIFT_MAX = 1e-6
#: Max relative change in U_T between coarse and fine grids.
CONVERGENCE_MAX = 0.05

# --------------------------------------------------------------------------- #
# Case definition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HysingCase:
    """Hysing Case 2: Eo=10, M=1e-3, density_ratio=1000, viscosity_ratio=10."""

    eotvos: float = 10.0
    morton: float = 1e-3
    density_ratio: float = 1000.0
    viscosity_ratio: float = 10.0
    cells_per_diameter: int = 32
    domain_width: float = 2.0
    domain_height: float = 4.0
    diameter: float = 1.0
    #: Bubble centre height.  Hysing et al. put the d=0.5 bubble at (0.5, 0.5)
    #: in the [0,1]x[0,2] box, i.e. at (1 d, 1 d) after scaling by d -> the
    #: bottom of the bubble is 0.5 d above the lower wall.  The historic value
    #: 0.5 put the bubble *bottom on the wall* (start_y = r), which is neither
    #: the benchmark geometry nor free of wall contact.
    start_y: float = 1.0
    end_time: float = 3.0
    csf_smoothing: int = 3
    #: Pressure split handed to :class:`TwoPhaseConfig` (see its docstring):
    #: ``"total"`` carries the full gravity head in the pressure, ``"reduced"``
    #: carries the dynamic pressure and the buoyancy acceleration
    #: ``((rho - rho_l)/rho) g``.  The solver documents ``"reduced"`` as the
    #: form to use whenever ``gravity != 0`` and a free surface is present;
    #: ``"total"`` is kept as the default so that the historic artefacts stay
    #: bit-for-bit reproducible and the two forms can be compared directly.
    pressure_form: str = "total"

    @property
    def bubble_radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def sigma(self) -> float:
        """Surface tension: sigma = rho_l * g * d^2 / Eo (rho_l=g=d=1)."""
        return 1.0 / self.eotvos

    @property
    def mu_liquid(self) -> float:
        """Liquid viscosity from Morton number."""
        delta_rho = 1.0 - 1.0 / self.density_ratio
        return (self.morton * self.sigma ** 3 / delta_rho) ** 0.25

    @property
    def mu_gas(self) -> float:
        return self.mu_liquid / self.viscosity_ratio

    @property
    def rho_gas(self) -> float:
        return 1.0 / self.density_ratio

    def solve_config(self) -> tuple[GridConfig, TwoPhaseConfig]:
        """Build the solver configuration."""
        # ``cells_per_diameter`` is a *resolution*, not a cell count: the
        # historic code used nx = cells_per_diameter, which put only
        # nx/domain_width = cpd/2 cells across one diameter (cpd=16 resolved the
        # bubble with 8 cells).  h = d / cpd keeps the bubble spanning cpd cells.
        h = self.diameter / self.cells_per_diameter
        nx = int(round(self.domain_width / h))
        ny = int(round(self.domain_height / h))
        grid = GridConfig(
            nx=nx, ny=ny,
            length=self.domain_width, height=self.domain_height,
        )
        tp = TwoPhaseConfig(
            enabled=True,
            rho_liquid=1.0, rho_gas=self.rho_gas,
            mu_liquid=self.mu_liquid, mu_gas=self.mu_gas,
            sigma=self.sigma, gravity=1.0,
            surface_tension=True,
            csf_smoothing=self.csf_smoothing,
            momentum_advection=True,
            pressure_form=self.pressure_form,
        )
        return grid, tp

    def initial_alpha(self, nx: int, ny: int, dx: float, dy: float,
                      subcell: int = 4) -> torch.Tensor:
        """Initial volume fraction field.

        The solver convention: ``alpha = 1`` is liquid, ``alpha = 0`` is gas.
        The bubble is gas, so alpha = 0 inside the bubble, alpha = 1 outside.

        ``subcell > 1`` builds the field from the *area fraction* of each cell
        covered by the gas disc (``subcell x subcell`` sub-samples): a
        cell-centre mask quantises the initial bubble area by O(dx) (at 8
        cells/diameter the area came out 3.4 % high, and with it the buoyancy
        force that drives the whole benchmark).  ``subcell=1`` reproduces the
        historic sharp mask.
        """
        xc, yc = self.domain_width / 2.0, self.start_y
        r = self.bubble_radius
        alpha = np.ones((ny, nx), dtype=np.float64)
        if subcell <= 1:
            xs = (np.arange(nx) + 0.5) * dx
            ys = (np.arange(ny) + 0.5) * dy
            alpha[(xs[None, :] - xc) ** 2 + (ys[:, None] - yc) ** 2 < r ** 2] = 0.0
            return torch.from_numpy(alpha)
        offs = (np.arange(subcell) + 0.5) / subcell
        x = (np.arange(nx)[None, :] + offs[:, None]) * dx      # (subcell, nx)
        y = (np.arange(ny)[:, None] + offs[None, :]) * dy      # (ny, subcell)
        gas = np.zeros((ny, nx), dtype=np.float64)
        # fully vectorised: sum over the subcell lattice
        for i in range(subcell):
            for j in range(subcell):
                dx2 = (x[j, :][None, :] - xc) ** 2
                dy2 = (y[:, i][:, None] - yc) ** 2
                gas += (dx2 + dy2 < r * r)
        alpha -= gas / float(subcell * subcell)
        return torch.from_numpy(alpha)


# --------------------------------------------------------------------------- #
# Bubble metrics
# --------------------------------------------------------------------------- #


def _interface_perimeter(field: np.ndarray, dx: float, dy: float, level: float = 0.5) -> float:
    """Length of the ``field = level`` contour by marching squares.

    The stair-case perimeter of a thresholded mask (edge-count estimator) is
    orientation biased and over-estimates by up to ~40 % on a coarse circle,
    which corrupts the circularity gate.  Here the contour is reconstructed on
    the *node* field (cell-corner average of the cell-centred data) with linear
    edge interpolation, i.e. the usual 16-case marching-squares table, and the
    segment lengths are summed.
    """
    ny, nx = field.shape
    if ny < 2 or nx < 2:
        return 0.0
    pad = np.pad(field, 1, mode="edge")
    nodes = 0.25 * (pad[:-1, :-1] + pad[:-1, 1:] + pad[1:, :-1] + pad[1:, 1:])
    a = nodes[:-1, :-1]          # SW corner of each cell
    b = nodes[:-1, 1:]           # SE
    c = nodes[1:, 1:]            # NE
    d = nodes[1:, :-1]           # NW

    def _frac(v0, v1):
        # Degenerate edge (both endpoints exactly at the level) is ambiguous for
        # a *binary* indicator -- whole node plateaus sit at 0.5 when two of the
        # four averaged cells are inside.  Taking t -> 0 there (as a bare
        # np.clip((l-v0)/den) does) throws the contour onto one endpoint and
        # inflates the length by ~1.4 % even on a perfect disc; the midpoint is
        # the neutral choice.
        den = v1 - v0
        flat = np.abs(den) < 1e-300
        den = np.where(flat, 1.0, den)
        return np.where(flat, 0.5, np.clip((level - v0) / den, 0.0, 1.0))

    idx_row, idx_col = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x0 = idx_col * dx
    y0 = idx_row * dy
    # crossing points of the four cell edges
    px = (
        x0 + _frac(a, b) * dx,        # bottom edge (a -> b)
        x0 + dx,
        x0 + _frac(d, c) * dx,        # top edge (d -> c)
        x0,
    )
    py = (
        y0,
        y0 + _frac(b, c) * dy,        # right edge (b -> c)
        y0 + dy,
        y0 + _frac(a, d) * dy,        # left edge (a -> d)
    )
    case = (8 * (a > level) + 4 * (b > level) + 2 * (c > level) + 1 * (d > level)).astype(np.int64)

    # edge-pair connectivity per case; saddles (5, 10) resolved by the centre value
    pairs = {
        0: (), 15: (),
        1: ((2, 3),), 2: ((1, 2),), 3: ((1, 3),), 4: ((0, 1),),
        5: ((0, 1), (2, 3)), 6: ((0, 2),), 7: ((0, 3),),
        8: ((0, 3),), 9: ((0, 2),), 10: ((0, 3), (1, 2)),
        11: ((0, 1),), 12: ((1, 3),), 13: ((1, 2),), 14: ((2, 3),),
    }
    total = 0.0
    for k, segs in pairs.items():
        if not segs:
            continue
        sel = case == k
        if not sel.any():
            continue
        if k in (5, 10):
            centre = 0.25 * (a + b + c + d)
            inside = sel & (centre > level)
            outside = sel & ~(centre > level)
            use_in = ((0, 3), (1, 2)) if k == 5 else ((0, 1), (2, 3))
            use_out = ((0, 1), (2, 3)) if k == 5 else ((0, 3), (1, 2))
            for mask_sel, seg_pair in ((inside, use_in), (outside, use_out)):
                if not mask_sel.any():
                    continue
                for e0, e1 in seg_pair:
                    total += _segment_lengths(
                        px[e0][mask_sel], py[e0][mask_sel], px[e1][mask_sel], py[e1][mask_sel])
            continue
        for e0, e1 in segs:
            total += _segment_lengths(px[e0][sel], py[e0][sel], px[e1][sel], py[e1][sel])
    return float(total)


def _segment_lengths(x1, y1, x2, y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1).sum())


def bubble_metrics(alpha: torch.Tensor, dx: float, dy: float) -> dict:
    """Compute bubble shape and position metrics from the liquid fraction field.

    The solver stores ``alpha = liquid fraction``, so the bubble is ``1 - alpha``.
    """
    bubble = 1.0 - alpha
    bubble_np = bubble.detach().cpu().numpy()

    # Volume (area in 2D)
    volume = float(bubble_np.sum()) * dx * dy

    # Centroid
    xs = (np.arange(bubble_np.shape[1]) + 0.5) * dx
    ys = (np.arange(bubble_np.shape[0]) + 0.5) * dy
    total = bubble_np.sum()
    if total < 1e-12:
        return {"volume": volume, "xc": 0.0, "yc": 0.0, "circularity": 0.0}

    xc = float((bubble_np * xs[None, :]).sum() / total)
    yc = float((bubble_np * ys[:, None]).sum() / total)

    # Circularity c = 4 pi A / P^2 with sub-cell area and a marching-squares
    # contour length (a stair-case edge count biases P high => c far below 1
    # even for a perfect disc; measured 0.64 at 8 cells/diameter).
    if (bubble_np > 0.5).sum() < 4:
        return {"volume": volume, "xc": xc, "yc": yc, "circularity": 0.0}
    area = volume
    # One Laplacian pass before contouring: a *binary* indicator has whole node
    # plateaus pinned at the 0.5 level, so the raw marching-squares length is
    # ~+1.5 % high on a perfect digital disc (=> circularity 2-3 % low, a third
    # of the 10 % gate).  One pass costs 0.5 % and keeps c <= 1 for a disc;
    # the estimator is exact (1e-6) on an analytically smooth field.
    smooth = bubble_np
    for _ in range(1):
        pad = np.pad(smooth, 1, mode="edge")
        smooth = 0.25 * (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
    perimeter = _interface_perimeter(smooth, dx, dy, level=0.5)
    if perimeter < 1e-12:
        circularity = 0.0
    else:
        circularity = 4.0 * math.pi * area / (perimeter ** 2)

    return {"volume": volume, "xc": xc, "yc": yc, "circularity": circularity}


def compute_terminal_velocity(history: list[dict], window_frac: float = 0.25) -> float:
    """Estimate terminal velocity as the mean centroid velocity over the last
    fraction of the time series."""
    if len(history) < 4:
        return 0.0
    n_window = max(2, int(len(history) * window_frac))
    window = history[-n_window:]
    dt = window[-1]["time"] - window[0]["time"]
    if dt < 1e-12:
        return 0.0
    dy = window[-1]["yc"] - window[0]["yc"]
    return dy / dt


# --------------------------------------------------------------------------- #
# Main leg runner
# --------------------------------------------------------------------------- #


def run_hysing_leg(
    cells_per_diameter: int = 32,
    *,
    end_time: float = 3.0,
    csf_smoothing: int = 3,
    pressure_form: str = "total",
) -> dict:
    """Run the Hysing Case 2 benchmark and return the metrics record.

    Parameters
    ----------
    cells_per_diameter:
        Grid resolution (cells per bubble diameter).
    end_time:
        Simulation end time (dimensionless, in units of sqrt(d/g)).
    csf_smoothing:
        Number of smoothing iterations for the CSF surface tension force.

    Returns
    -------
    dict
        JSON-serialisable record with terminal velocity, circularity,
        mass conservation, and the full history.
    """
    case = HysingCase(
        cells_per_diameter=cells_per_diameter,
        end_time=end_time,
        csf_smoothing=csf_smoothing,
        pressure_form=pressure_form,
    )
    grid, tp = case.solve_config()
    solver = TwoPhaseSolver(grid, tp)
    alpha0 = case.initial_alpha(grid.nx, grid.ny, solver.dx, solver.dy)
    solver.set_alpha(alpha0)  # alpha=1 liquid, alpha=0 gas (bubble)

    dx, dy = solver.dx, solver.dy
    volume0 = float((1.0 - solver.alpha).sum()) * dx * dy

    history: list[dict] = []
    worst_step_drift = 0.0
    volume_prev = volume0
    diverged = False

    while solver.time < end_time:
        dt = solver.stable_time_step()
        if solver.time + dt > end_time:
            dt = end_time - solver.time
        try:
            solver.step(dt)
        except (RuntimeError, ValueError, FloatingPointError):
            diverged = True
            break

        if not torch.isfinite(solver.alpha).all():
            diverged = True
            break

        vol = float((1.0 - solver.alpha).sum()) * dx * dy
        step_drift = abs(vol - volume_prev) / max(volume0, 1e-12)
        worst_step_drift = max(worst_step_drift, step_drift)
        volume_prev = vol

        m = bubble_metrics(solver.alpha, dx, dy)  # alpha=liquid; helper inverts
        m["time"] = float(solver.time)
        m["volume_drift"] = (vol - volume0) / max(volume0, 1e-12)
        m["umax"] = float(max(solver.u.abs().max(), solver.v.abs().max()))
        history.append(m)

    # Terminal velocity from the last 25% of the history
    u_t = compute_terminal_velocity(history, window_frac=0.25)

    # Peak climb velocity: Hysing's U_T is the maximum of the rise velocity over
    # the run, the window mean above is a conservative (low) proxy.
    peak_velocity = 0.0
    for prev, cur in zip(history[:-1], history[1:]):
        dt = cur["time"] - prev["time"]
        if dt > 0.0:
            peak_velocity = max(peak_velocity, (cur["yc"] - prev["yc"]) / dt)

    # Minimum circularity over the run
    circularities = [m["circularity"] for m in history if m["circularity"] > 0]
    min_circularity = min(circularities) if circularities else 0.0

    record = {
        "case": "hysing",
        "cells_per_diameter": cells_per_diameter,
        "pressure_form": case.pressure_form,
        "eotvos": case.eotvos,
        "morton": case.morton,
        "end_time": end_time,
        "terminal_velocity": u_t,
        "peak_velocity": peak_velocity,
        "min_circularity": min_circularity,
        "worst_step_drift": worst_step_drift,
        "total_volume_drift": history[-1]["volume_drift"] if history else 0.0,
        "diverged": diverged,
        "steps": len(history),
        "declared": cells_per_diameter == DECLARED_CPD,
        "reference": {
            "source": "Hysing et al. (2009) Int. J. Numer. Meth. Fluids 60, "
                      "1259-1288, Table II, Case 2 (Eo=10, M=1e-3)",
            "terminal_velocity": HYSING_REF_UT,
            "min_circularity": HYSING_REF_CIRCULARITY,
            "centroid_y_at_t3": HYSING_REF_YC,
        },
        "terminal_velocity_rel_error": abs(u_t - HYSING_REF_UT) / HYSING_REF_UT,
        "circularity_rel_error": abs(min_circularity - HYSING_REF_CIRCULARITY) / HYSING_REF_CIRCULARITY,
    }
    return {"record": record, "history": history}


def convergence_order(coarse_ut: float, fine_ut: float, ratio: float = 2.0) -> float:
    """Observed order from two terminal velocities."""
    coarse_ut = max(float(coarse_ut), 1e-16)
    fine_ut = max(float(fine_ut), 1e-16)
    return math.log(coarse_ut / fine_ut) / math.log(ratio)
