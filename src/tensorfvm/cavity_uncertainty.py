"""Discretisation-uncertainty quantification for the lid-driven cavity case.

Why this module exists
----------------------
The case gate of ``CASE-FVM-CAVITY-RE1000`` (see
:mod:`tensorfvm.case_cavity_re1000`) answers one question: *how far is the
computed centreline from Ghia's tabulated data at the finest grid?*  That is a
reference-agreement check.  It is not the same as asking *how far the computed
solution is from the exact solution of the PDE*, because a grid that is not in
the asymptotic range can agree with a reference better than its own
discretisation error (error cancellation), and a reference has an uncertainty of
its own.

This module supplies the second answer from the solver's own grid sequence using
Roache's grid-convergence index (GCI) and, optionally, the same machinery along
the time-step axis:

* :func:`functionals_from_fields` — grid-independent solution functionals
  (``u_min`` on ``x = L/2``, ``v_max`` on ``y = L/2``, the streamfunction
  minimum and the kinetic energy) computed **from the stored face fields**, so a
  third party can re-derive every number without re-running the solver;
* :func:`richardson` — observed order ``p``, Richardson-extrapolated value,
  fine-grid GCI and a data-quality verdict (monotonicity, admissible ``p``);
  handles a non-uniform refinement ratio (Celik et al. 2008, eq. 9);
* :func:`time_step_spread` — the spread of the same functionals over a set of
  Courant numbers on one grid, i.e. the temporal contribution that the spatial
  study deliberately holds fixed;
* :func:`study` — assembles the full payload (per-functional tables, the
  reference offset, and the honest statement of what the data support).

References
----------
* P. J. Roache, *J. Fluids Eng.* 116 (1994) 405-413 (GCI).
* I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, *J. Fluids Eng.* 130 (2008)
  078001 (GCI procedure with data-quality checks).
* U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* 48 (1982) 387-411.

Nothing here is cavity-specific except the default functional set; the
estimators are generic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from .lid_cavity import LidCavityConfig, LidCavitySolver
from .lid_cavity_reference import VORTEX_CENTRE, table

__all__ = [
    "DataQuality",
    "TIME_STEP_TOLERANCE",
    "functionals_from_fields",
    "reference_functionals",
    "richardson",
    "study",
    "time_step_spread",
]

#: Safety factor of the GCI (Roache recommends 1.25 for a three-grid study of a
#: formally second-order scheme; 3.0 when only two grids are available).
DEFAULT_SAFETY = 1.25

#: Two runs on the same grid that differ only in the Courant number are accepted
#: as time-step independent when every functional agrees to this relative
#: tolerance.  It is set an order of magnitude below the spatial GCI of the
#: finest grid, which is what the temporal error has to be compared against.
TIME_STEP_TOLERANCE = 1e-3


@dataclass(frozen=True)
class DataQuality:
    """Flags that decide whether a GCI number may be quoted at all."""

    monotonic: bool
    order_admissible: bool
    note: str

    @property
    def usable(self) -> bool:
        return self.monotonic and self.order_admissible


# --------------------------------------------------------------------------- #
# solution functionals straight from the stored fields
# --------------------------------------------------------------------------- #
def functionals_from_fields(u: torch.Tensor, v: torch.Tensor,
                            reynolds: float = 1000.0,
                            lid_velocity: float = 1.0) -> dict[str, float]:
    """Grid-independent functionals of a converged field pair.

    ``u`` has shape ``(n + 1, n)`` (staggered vertical faces) and ``v`` has shape
    ``(n, n + 1)`` (horizontal faces).  ``u_min``/``v_max`` are taken on the face
    line that coincides with the benchmark cut (``x = L/2`` and ``y = L/2``,
    exact for even ``n``) and are therefore extremum values of the discrete
    solution, not values interpolated to somebody's sample table — which is what
    makes them usable as a convergence sequence.
    """
    u = torch.as_tensor(u).double()
    v = torch.as_tensor(v).double()
    n = int(u.shape[0] - 1)
    if tuple(v.shape) != (n, n + 1):
        raise ValueError(f"inconsistent staggered shapes {tuple(u.shape)} / {tuple(v.shape)}")
    h = 1.0 / n

    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=float(reynolds),
                                            lid_velocity=float(lid_velocity)))
    solver.u, solver.v = u.clone(), v.clone()

    centre_u = u[n // 2, :]
    centre_v = v[:, n // 2]
    j_min = int(torch.argmin(centre_u))
    i_max = int(torch.argmax(centre_v))
    psi = solver.streamfunction()
    psi_interior = psi[1:-1, 1:-1]
    idx = int(torch.argmin(psi_interior))
    i_psi, j_psi = divmod(idx, n - 2)

    u_c = 0.5 * (u[:-1, :] + u[1:, :])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    kinetic = float(0.5 * (u_c ** 2 + v_c ** 2).sum()) * h * h

    return {
        "u_min": float(centre_u.min()),
        "u_min_y": (j_min + 0.5) * h,
        "v_max": float(centre_v.max()),
        "v_max_x": (i_max + 0.5) * h,
        "psi_min": float(psi.min()),
        "vortex_x": (i_psi + 1.5) * h,
        "vortex_y": (j_psi + 1.5) * h,
        "kinetic": kinetic,
    }


#: Functionals that are compared with the reference and across time steps.  The
#: locations (``*_y`` / ``*_x``) are diagnostics, not convergence targets.
QUANTITIES = ("u_min", "v_max", "psi_min", "kinetic")


def reference_functionals(reynolds: float = 1000.0) -> dict[str, float]:
    """Reference values for the quantities that Ghia tabulates.

    ``u_min``/``v_max`` are read from the shared transcription
    (:mod:`tensorfvm.lid_cavity_reference`), never typed in here; ``psi_min`` and
    the kinetic energy are not tabulated by the paper and therefore have no
    reference entry (their GCI is still meaningful as a self-convergence number).
    """
    u_pts = [float(b) for _, b in table(reynolds, "u")]
    v_pts = [float(b) for _, b in table(reynolds, "v")]
    vx, vy = VORTEX_CENTRE[reynolds]
    return {"u_min": min(u_pts), "v_max": max(v_pts),
            "vortex_x": float(vx), "vortex_y": float(vy)}


# --------------------------------------------------------------------------- #
# Richardson extrapolation + GCI
# --------------------------------------------------------------------------- #
def _solve_observed_order(eps_coarse: float, eps_fine: float,
                          r1: float, r2: float) -> float:
    """Observed order for a (possibly) non-uniform refinement (Celik et al. 2008).

    ``eps_coarse = f2 - f1`` (between the two coarser grids), ``eps_fine = f3 - f2``
    with ``r1 = h1/h2`` and ``r2 = h2/h3``.  For a power-law error
    ``eps21/eps32 = (r1^p - 1) r2^p / (r2^p - 1)``, which is implicit in ``p`` when
    ``r1 != r2``; it is solved by bisection (the right-hand side increases
    monotonically with ``p``).  For ``r1 == r2 == r`` this is
    ``p = ln|eps_coarse/eps_fine| / ln r``.
    """
    a, b = abs(eps_coarse), abs(eps_fine)
    if a == 0.0 or b == 0.0 or r1 <= 1.0 or r2 <= 1.0:
        return float("nan")
    if math.isclose(r1, r2, rel_tol=1e-12):
        return math.log(a / b) / math.log(r1)

    def log_ratio(p: float) -> float:
        try:
            return math.log((r1 ** p - 1.0) * r2 ** p / (r2 ** p - 1.0))
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan")

    lo, hi = 1e-3, 10.0
    glo, ghi = log_ratio(lo) - math.log(a / b), log_ratio(hi) - math.log(a / b)
    if not (math.isfinite(glo) and math.isfinite(ghi)) or glo * ghi > 0.0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gmid = log_ratio(mid) - math.log(a / b)
        if not math.isfinite(gmid):
            return float("nan")
        if abs(gmid) < 1e-14 or hi - lo < 1e-14:
            return mid
        if gmid * glo < 0.0:
            hi = mid
        else:
            lo, glo = mid, gmid
    return 0.5 * (lo + hi)


def richardson(samples: Sequence[tuple[float, float]],
               safety: float = DEFAULT_SAFETY,
               p_bounds: tuple[float, float] = (0.5, 4.0),
               name: str = "functional") -> dict:
    """Observed order, Richardson extrapolation and fine-grid GCI.

    ``samples`` is a sequence of ``(n, value)`` with ``n`` the grid resolution
    (strictly increasing); the last three samples are used, which is the
    minimum a GCI can be built from.  Convergence must be monotonic — otherwise
    no order can be extracted and the result is reported as *not usable* rather
    than silently averaged.
    """
    rows = sorted(((float(n), float(f)) for n, f in samples), key=lambda r: r[0])
    if len(rows) < 3:
        raise ValueError("a GCI needs at least three grids")
    (n1, f1), (n2, f2), (n3, f3) = rows[-3:]
    h = [1.0 / n1, 1.0 / n2, 1.0 / n3]
    r1, r2 = h[0] / h[1], h[1] / h[2]
    eps21, eps32 = f2 - f1, f3 - f2

    monotonic = eps21 * eps32 > 0.0
    p = _solve_observed_order(eps21, eps32, r1, r2) if monotonic else float("nan")
    ok_order = math.isfinite(p) and p_bounds[0] <= p <= p_bounds[1]

    note = []
    if not monotonic:
        note.append("non-monotonic sequence: no observed order, GCI not quoted")
    elif not math.isfinite(p):
        note.append("observed order undetermined")
    elif not ok_order:
        note.append(f"observed order p={p:.2f} outside {p_bounds}: not in the asymptotic range")
    else:
        note.append(f"monotonic, observed order p={p:.2f}")

    out = {
        "quantity": name,
        "grids": [n1, n2, n3],
        "values": [f1, f2, f3],
        "refinement_ratios": [r1, r2],
        "difference_coarse": eps21,
        "difference_fine": eps32,
        "observed_order": p,
        "safety_factor": safety,
        "data_quality": DataQuality(monotonic, ok_order, "; ".join(note)).__dict__
        | {"usable": bool(monotonic and ok_order)},
    }
    if monotonic and math.isfinite(p) and abs(r2 ** p - 1.0) > 0.0:
        extrapolated = f3 + (f3 - f2) / (r2 ** p - 1.0)
        gci_fine = safety * abs((f3 - f2) / f3) / abs(r2 ** p - 1.0) if f3 else float("nan")
        gci_coarse = safety * abs((f2 - f1) / f1) / abs(r1 ** p - 1.0) if f1 else float("nan")
        out.update({
            "extrapolated": extrapolated,
            "fine_minus_extrapolated": f3 - extrapolated,
            "gci_fine_absolute": abs(gci_fine * f3) if math.isfinite(gci_fine) else float("nan"),
            "gci_fine_relative": gci_fine,
            "gci_coarse_relative": gci_coarse,
            "fine_grid_relative_error_estimate": abs((f3 - extrapolated) / extrapolated)
            if extrapolated else float("nan"),
        })
    return out


# --------------------------------------------------------------------------- #
# time-step (Courant) contribution
# --------------------------------------------------------------------------- #
def time_step_spread(runs: Mapping[str, Mapping[str, float]],
                     quantities: Iterable[str] = QUANTITIES,
                     tolerance: float = TIME_STEP_TOLERANCE) -> dict:
    """Spread of the functionals over runs that differ only in the time step.

    ``runs`` maps a label (e.g. ``"cfl=0.075"``) to a functional dict from
    :func:`functionals_from_fields`.  The largest relative spread of any
    quantity, referenced to its mean, is the temporal contribution to the
    discretisation error on that grid.  Comparing it with the spatial GCI is the
    point: if it is far smaller, the spatial study is not contaminated by the
    time step and the CFL choice only affects cost.
    """
    labels = list(runs)
    if len(labels) < 2:
        raise ValueError("a time-step study needs at least two runs")
    rows = {}
    for q in quantities:
        vals = [float(runs[lab][q]) for lab in labels if q in runs[lab]]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        rel = [(x - mean) / mean for x in vals]
        rows[q] = {
            "values": dict(zip([lab for lab in labels if q in runs[lab]],
                               [float(x) for x in vals])),
            "max_relative_spread": max(abs(x) for x in rel),
            "mean": mean,
        }
    worst = max((v["max_relative_spread"] for v in rows.values()), default=float("nan"))
    return {
        "runs": labels,
        "tolerance": tolerance,
        "quantities": rows,
        "max_relative_spread": worst,
        "time_step_independent": bool(math.isfinite(worst) and worst <= tolerance),
    }


# --------------------------------------------------------------------------- #
# full study
# --------------------------------------------------------------------------- #
def study(fields: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
          reynolds: float = 1000.0,
          safety: float = DEFAULT_SAFETY) -> dict:
    """Assemble the spatial (and optionally temporal) uncertainty payload.

    ``fields`` maps ``n`` to the stored ``(u, v)`` pair.  The time-step (Courant)
    contribution is a separate call to :func:`time_step_spread`, because those
    runs publish functionals rather than raw fields.
    """
    grids = sorted(int(n) for n in fields)
    per_grid = {n: functionals_from_fields(*fields[n], reynolds=reynolds) for n in grids}
    ref = reference_functionals(reynolds)

    quantities = {}
    for q in QUANTITIES:
        block = richardson([(n, per_grid[n][q]) for n in grids], safety=safety, name=q)
        if q in ref:
            block["reference"] = ref[q]
            block["fine_minus_reference_relative"] = (per_grid[grids[-1]][q] - ref[q]) / abs(ref[q]) \
                if ref[q] else float("nan")
            if "extrapolated" in block:
                block["extrapolated_minus_reference_relative"] = \
                    (block["extrapolated"] - ref[q]) / abs(ref[q]) if ref[q] else float("nan")
                # does the GCI band of the finest grid cover the reference offset?
                band = abs(block.get("gci_fine_relative", float("nan")))
                block["reference_within_gci_band"] = bool(
                    math.isfinite(band)
                    and abs(block["fine_minus_reference_relative"]) <= band)
        quantities[q] = block

    payload = {
        "case": "CASE-FVM-CAVITY-RE1000",
        "reynolds": float(reynolds),
        "grids": grids,
        "safety_factor": safety,
        "functionals": per_grid,
        "reference": ref,
        "quantities": quantities,
    }
    return payload


def statements(payload: Mapping) -> list[str]:
    """Plain-language statements the payload supports (and does not support)."""
    out = []
    for q in QUANTITIES:
        blk = payload["quantities"][q]
        if not blk["data_quality"]["usable"]:
            out.append(f"{q}: no GCI quoted ({blk['data_quality']['note']})")
            continue
        line = (f"{q}: p={blk['observed_order']:.2f}, extrapolated={blk['extrapolated']:.6g}, "
                f"GCI(finest)={100 * blk['gci_fine_relative']:.2f}%")
        if "reference" in blk:
            line += (f", finest vs reference={100 * blk['fine_minus_reference_relative']:+.2f}%, "
                     f"extrapolated vs reference={100 * blk['extrapolated_minus_reference_relative']:+.2f}%")
            if blk.get("reference_within_gci_band"):
                line += " (reference offset inside the GCI band)"
            else:
                line += " (reference offset OUTSIDE the GCI band -> grid/order or reference uncertainty dominates)"
        out.append(line)
    ts = payload.get("time_step")
    if ts:
        out.append(f"time step: max relative spread {100 * ts['max_relative_spread']:.4f}% "
                   f"over {ts['runs']} (tolerance {100 * ts['tolerance']:.2f}%)")
    return out