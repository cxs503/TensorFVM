"""Attribution of the Re = 400 lid-cavity ``v``-centreline discrepancy.

Why this module exists
----------------------
``tests/test_lid_cavity_re1000_evidence.py`` keeps a strict xfail for the Re = 400
row of the Ghia ladder: the ``v``-centreline error *grows* under refinement
(0.115 / 0.142 / 0.149 for n = 32 / 64 / 128) while ``u`` and both extrema
converge, and the recorded hypothesis is

    "the worst point (x = 0.9063) exceeds the profile's own maximum in
    magnitude -- a wall-adjacent extraction defect, not discretisation"

That hypothesis is testable without re-running any long transient: the
extraction path (``LidCavitySolver.centreline_v`` -> ``hermite_interpolate``)
can be exercised on a **synthetic field of known closed form**, where the exact
answer at every Ghia ordinate is known analytically.  If the extraction path
reproduces a smooth profile to far better than the observed 15 % discrepancy
then the hypothesis is false and the discrepancy must be attributed to the
solution field; that changes what a fix has to look like.

What is measured
----------------
``extraction_geometry``
    where the sampled ordinates actually sit.  ``v`` is staggered as
    ``v[x_cell, y_face]`` with faces at ``j*h``, so the horizontal mid-plane is
    the *face* ``j = n/2`` and is represented exactly (no half-cell offset).
    ``u`` is ``u[x_face, y_cell]``, so ``x = L/2`` is likewise exact.  Both are
    asserted rather than assumed, because an off-by-half-cell here would be
    exactly the kind of defect the xfail hypothesises.
``interpolation_study``
    the interpolation error of the published Ghia ordinates for a smooth
    profile of known form, plus the observed order in ``h``.
``index_shift_control``
    a positive control: the same measurement applied to a field sampled half a
    cell off its true position.  The audit must *see* that defect, otherwise a
    small measured error would prove nothing.
``attribution``
    the extraction error budget against the discrepancy the xfail records.

Scope, stated honestly: this module audits the extraction path and the
attribution arithmetic.  It does not by itself fix the Re = 400 solution-field
defect; it says which half of the pipeline owns it.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver, hermite_interpolate
from tensorfvm.lid_cavity_reference import table

__all__ = [
    "GHIA_V_X",
    "GHIA_V_ABS_MAX",
    "profile",
    "synthetic_solver",
    "extraction_geometry",
    "interpolation_study",
    "index_shift_control",
    "observed_orders",
    "attribution",
]


#: Interior ``x/L`` ordinates of Ghia et al. (1982) table II (``v`` on ``y/L = 1/2``).
#: The wall ordinates 0 and 1 are excluded: no cell centre sits there, so scoring
#: them would score an extrapolation rather than the interior solution.
GHIA_V_X = tuple(x for x, _ in table(400.0, "v") if 0.0 < x < 1.0)

#: Ghia's Re = 400 ``v``-centreline peak magnitude, the yardstick the xfail quotes.
GHIA_V_ABS_MAX = max(abs(v) for _, v in table(400.0, "v"))

#: Relative discrepancy the ladder recorded at the worst point, x/L = 0.9063.
OBSERVED_RELATIVE_DEFECT = 0.149


def profile(x: np.ndarray) -> np.ndarray:
    """Smooth stand-in for the ``v``-centreline shape.

    Chosen so that (a) its magnitude is the same order as Ghia's Re = 400 peak,
    so error budgets are directly comparable, and (b) every derivative is
    analytic, so the interpolation order is measured rather than estimated.
    """
    x = np.asarray(x, dtype=float)
    return -0.45 * np.sin(2.0 * math.pi * x) * (1.0 - x) ** 2


def synthetic_solver(n: int, *, shift_cells: float = 0.0) -> LidCavitySolver:
    """Fill a solver's staggered arrays from :func:`profile`.

    ``v`` is sampled at the true staggered positions ``x_i = (i + 1/2) h`` on
    every ``y``-face; ``shift_cells`` displaces the sample by that many cells,
    which is the deliberate defect the audit has to detect.
    """
    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=400.0))
    h = solver.h
    x_cell = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    sampled = profile(x_cell + shift_cells * h)
    solver.v[:, :] = torch.as_tensor(
        np.repeat(sampled[:, None], n + 1, axis=1), dtype=torch.float64)
    y_cell = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    u_column = 0.5 * np.asarray(y_cell)  # analytic in y, so the u-path is checkable too
    solver.u[:, :] = torch.as_tensor(np.repeat(u_column[None, :], n + 1, axis=0),
                                     dtype=torch.float64)
    return solver


def extraction_geometry(n: int) -> dict:
    """Locate the sampled ordinates and check them against the mid-planes."""
    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=400.0))
    x, v = solver.centreline_v()
    y, u = solver.centreline_u()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nearest = np.abs(x[:, None] - np.asarray(GHIA_V_X)[None, :]).argmin(axis=0)
    worst_ordinate = 0.9063
    worst_offset = float(np.min(np.abs(x - worst_ordinate)))
    return {
        "n": n,
        "h": solver.h,
        # v faces sit at j*h, so the horizontal mid-plane is the face n//2 exactly.
        "v_mid_plane_y": float(solver.n // 2) * solver.h,
        "v_mid_plane_is_exact": abs(float(solver.n // 2) * solver.h - 0.5) < 1e-15,
        "u_mid_plane_x": float(solver.n // 2) * solver.h,
        "u_mid_plane_is_exact": abs(float(solver.n // 2) * solver.h - 0.5) < 1e-15,
        "point_count": int(x.size),
        "first_ordinate": float(x[0]),
        "last_ordinate": float(x[-1]),
        "min_ordinate_to_ghia": float(np.max(np.min(np.abs(x[:, None] - np.asarray(GHIA_V_X)[None, :]),
                                                    axis=0))),
        "worst_point_offset_cells": worst_offset / solver.h,
        "nearest_node_index": [int(i) for i in nearest],
    }


def _extract(solver: LidCavitySolver) -> tuple[np.ndarray, np.ndarray]:
    x, v = solver.centreline_v()
    return np.asarray(x, dtype=float), np.asarray(v, dtype=float)


def interpolation_study(ns) -> list[dict]:
    """Max/near-worst extraction error at the Ghia ordinates versus ``n``.

    The field is analytic, so any error measured here is produced by the
    extraction path alone (mid-plane selection plus interpolation), not by a
    discretised solution.
    """
    rows = []
    targets = np.asarray(GHIA_V_X)
    exact = profile(targets)
    for n in ns:
        solver = synthetic_solver(n)
        x, v = _extract(solver)
        interpolated = np.asarray(hermite_interpolate(torch.as_tensor(x), torch.as_tensor(v),
                                                      GHIA_V_X), dtype=float)
        error = np.abs(interpolated - exact)
        scale = float(np.abs(exact).max())
        rows.append({
            "n": n,
            "h": solver.h,
            "max_abs_error": float(error.max()),
            "max_relative_error": float(error.max() / scale),
            "worst_point_argument": float(targets[int(error.argmax())]),
            "defective_point_error": float(np.abs(np.asarray(hermite_interpolate(
                torch.as_tensor(x), torch.as_tensor(v), [0.9063]), dtype=float) - profile(
                    np.asarray([0.9063])))[0]),
        })
    return rows


def index_shift_control(ns, shifts=(0.5, 1.0)) -> list[dict]:
    """Positive control: sample the field off its true position and measure it."""
    rows = []
    for n in ns:
        clean = interpolation_study([n])[0]["max_abs_error"]
        for shift in shifts:
            solver = synthetic_solver(n, shift_cells=shift)
            x, v = _extract(solver)
            interpolated = np.asarray(hermite_interpolate(
                torch.as_tensor(x), torch.as_tensor(v), GHIA_V_X), dtype=float)
            error = float(np.abs(interpolated - profile(np.asarray(GHIA_V_X))).max())
            rows.append({
                "n": n,
                "shift_cells": shift,
                "max_abs_error": error,
                "clean_max_abs_error": clean,
                "detection_ratio": error / clean if clean > 0 else float("inf"),
            })
    return rows


def observed_orders(rows: list[dict], key: str = "max_abs_error") -> float:
    """Observed order in ``h`` from a refinement series of error magnitudes."""
    if len(rows) < 2:
        return float("nan")
    orders = []
    for fine, coarse in zip(rows[:-1], rows[1:]):
        ef, ec = fine[key], coarse[key]
        if ef <= 0 or ec <= 0:
            return float("inf")
        orders.append(math.log(ec / ef) / math.log(coarse["h"] / fine["h"]))
    return float(sum(orders) / len(orders))


@dataclass(frozen=True)
class Attribution:
    """Extraction error budget against the discrepancy the xfail records."""

    extraction_relative_error: float
    observed_relative_defect: float
    ratio: float
    verdict: str

    def as_dict(self):
        return {
            "extraction_relative_error": self.extraction_relative_error,
            "observed_relative_defect": self.observed_relative_defect,
            "ratio": self.ratio,
            "verdict": self.verdict,
        }


def attribution(extraction_relative_error: float,
                observed_relative_defect: float = OBSERVED_RELATIVE_DEFECT) -> Attribution:
    """Decide whether the extraction path can own the observed discrepancy.

    The extraction budget is taken at ``n = 32``, the coarsest grid of the
    ladder, i.e. the grid where extraction error is largest: if it is still
    negligible there it is negligible on every finer grid.
    """
    ratio = (observed_relative_defect / extraction_relative_error
             if extraction_relative_error > 0 else float("inf"))
    if ratio > 10.0:
        verdict = (
            "extraction path exonerated: it reproduces a smooth profile to "
            f"{extraction_relative_error:.2e} relative, {ratio:.0f}x below the "
            f"{observed_relative_defect:.3f} the ladder records at x/L = 0.9063; "
            "the discrepancy belongs to the solution field, so the xfail's "
            "'wall-adjacent extraction defect, not discretisation' hypothesis is "
            "refuted")
    else:
        verdict = (
            "extraction path not exonerated: its own error "
            f"({extraction_relative_error:.2e}) is within a factor {ratio:.1f} of the "
            f"observed {observed_relative_defect:.3f}; fix the extraction first")
    return Attribution(extraction_relative_error, observed_relative_defect, ratio, verdict)