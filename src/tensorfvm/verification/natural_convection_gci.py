"""Grid-convergence study of the buoyant-cavity Nusselt number.

Motivation
----------
``docs/benchmarks/incompressible-verification.md`` §2.1 carries two marginal
verdicts for the differentially heated cavity (``Pr = 0.71``):

    Ra = 1e5, N = 32 : Nu error +3.4 %  -> FAIL (gate 3 %)
    Ra = 1e6, N = 64 : Nu error +3.1 %  -> FAIL (gate 3 %)

Both were reported on a single grid, so the failure could be either (a) plain
grid truncation that a refinement removes, or (b) a genuine bias of the discrete
operator.  This module supplies the machinery to tell those apart: it re-solves a
given ``(Ra, N)`` with the same staggered MAC solver used for the archived
evidence, measures ``Nu`` with two independent wall-flux estimators, and feeds
the ``(N, Nu)`` series into the repository's three-grid Richardson/GCI tool
(:mod:`tensorfvm.verification.richardson`).

Wall-flux estimators
--------------------
The staggered solver stores *cell averages* of ``theta``; the wall value is
Dirichlet.  Two reconstructions of ``-dtheta/dx`` at the wall are provided:

``half_cell``
    ``(theta_w - theta_P1) / (h/2)``.  This is exactly the flux the discrete
    energy equation uses, but as an estimator of the true flux it carries an
    ``O(h)`` truncation (``-theta'' h/4``), because it ignores the curvature
    across the first cell.

``quadratic``
    Quadratic fit through the wall value and the two first cell averages
    (``theta_P1`` over ``[0,h]``, ``theta_P2`` over ``[h,2h]``):

        ``b = (3 A - B/2) / h``, ``A = theta_P1 - theta_w``, ``B = theta_P2 - theta_P1``,

    so ``-b`` is a second-order accurate wall flux.  It is exact for quadratic
    profiles and reduces to the half-cell value when the profile is linear.

Reporting both makes the *estimator* truncation visible instead of silently
folding it into the flow error: their difference decays like ``O(h^2)`` on a
smooth field while a solution-side boundary-layer error does not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

import numpy as np

from .richardson import observed_order, richardson_gci

#: de Vahl Davis (1983) benchmark values (see ``incompressible_benchmarks``).
BENCHMARK_NU = {1.0e4: 2.243, 1.0e5: 4.519, 1.0e6: 8.800}
BENCHMARK_PSI = {1.0e4: 5.071, 1.0e5: 9.111, 1.0e6: 16.750}

GATE = 0.03  #: ``Nu`` acceptance gate of the incompressible verification surface


def _wall_columns(theta: np.ndarray, wall: str):
    """``(near, further, wall_value, sign)`` in the wall's inward coordinate.

    ``sign`` converts ``-dtheta/dxi`` (``xi`` pointing into the cavity) into the
    conventional positive ``Nu``: the outward normal of the cold wall is ``+x``,
    so the two walls carry opposite signs.
    """
    if wall == "hot":
        return theta[0, :], theta[1, :], 0.5, 1.0
    if wall == "cold":
        return theta[-1, :], theta[-2, :], -0.5, -1.0
    raise ValueError("wall must be 'hot' or 'cold'")


def nusselt_half_cell(theta: np.ndarray, h: float, wall: str = "hot") -> np.ndarray:
    """Local ``Nu`` from the half-cell (first-order) wall difference."""
    near, _further, wall_value, sign = _wall_columns(theta, wall)
    return sign * (wall_value - near) / (0.5 * h)


def nusselt_quadratic(theta: np.ndarray, h: float, wall: str = "hot") -> np.ndarray:
    """Local ``Nu`` from the quadratic reconstruction of the wall gradient."""
    near, further, wall_value, sign = _wall_columns(theta, wall)
    a = near - wall_value
    b = further - near
    return -sign * (3.0 * a - 0.5 * b) / h


def mean_nusselt_estimators(theta: np.ndarray, h: float) -> dict:
    """Mean ``Nu`` on both walls with both estimators plus the energy imbalance."""
    out = {}
    for wall, sign in (("hot", 1.0), ("cold", -1.0)):
        for name, fn in (("half_cell", nusselt_half_cell),
                         ("quadratic", nusselt_quadratic)):
            # A positive wall heat flux heats the cavity on the hot wall and
            # extracts it on the cold wall; the sign convention above already
            # gives a positive mean for both, so both are reported directly.
            out[f"nusselt_{wall}_{name}"] = float(np.mean(fn(theta, h, wall)))
        out[f"nusselt_{wall}"] = out[f"nusselt_{wall}_half_cell"]
    out["energy_imbalance_half_cell"] = abs(out["nusselt_hot_half_cell"]
                                            - out["nusselt_cold_half_cell"]) \
        / out["nusselt_hot_half_cell"]
    out["energy_imbalance_quadratic"] = abs(out["nusselt_hot_quadratic"]
                                            - out["nusselt_cold_quadratic"]) \
        / out["nusselt_hot_quadratic"]
    out["estimator_gap_relative"] = abs(out["nusselt_hot_quadratic"]
                                        - out["nusselt_hot_half_cell"]) \
        / out["nusselt_hot_half_cell"]
    return out


# --------------------------------------------------------------------- one run
@dataclass(frozen=True)
class RunRequest:
    rayleigh: float
    cells: int
    cfl: float = 0.25
    tolerance: float = 1e-7
    max_steps: int = 400_000

    @property
    def tag(self) -> str:
        return f"Ra{self.rayleigh:.0e}_N{self.cells}"


def run_case(request: RunRequest, *, on_check=None) -> dict:
    """Solve one ``(Ra, N)`` and return metrics + convergence ledger.

    The steady-state criterion is the solver's own per-step velocity change;
    ``converged`` is recorded verbatim so an under-marched run cannot be quoted
    as a grid-converged number.
    """
    from ..incompressible_benchmarks import natural_convection

    flow = natural_convection(request.rayleigh, cells=request.cells, cfl=request.cfl)
    state = flow.run_steady(tolerance=request.tolerance,
                            max_steps=request.max_steps,
                            check_every=100, on_check=on_check)
    h = flow.h
    nu = mean_nusselt_estimators(flow.theta, h)
    psi = flow.streamfunction()
    psi_ext = float(psi.max()) if abs(psi.max()) > abs(psi.min()) else float(psi.min())
    u_cell, v_cell = flow.cell_velocity()
    reference_nu = BENCHMARK_NU.get(float(request.rayleigh))
    reference_psi = BENCHMARK_PSI.get(float(request.rayleigh))
    metrics = {
        "cells": request.cells,
        "h": h,
        "rayleigh": float(request.rayleigh),
        "viscosity": float(flow.nu),
        "prandtl": float(flow.prandtl),
        **nu,
        "psi_extremum_abs": abs(psi_ext),
        "psi_extremum_signed": psi_ext,
        "u_max_abs": float(np.abs(u_cell).max()),
        "v_max_abs": float(np.abs(v_cell).max()),
        "divergence_max": state.max_divergence,
        "steps": state.steps,
        "time_step": state.dt,
        "time": float(state.history[-1][1]) if state.history else 0.0,
        "residual": state.residual,
        "converged": bool(state.converged),
        "residual_history": [list(map(float, row)) for row in state.history],
    }
    if reference_nu is not None:
        metrics["nusselt_reference"] = reference_nu
        metrics["nusselt_relative_error"] = abs(metrics["nusselt_hot_half_cell"]
                                                - reference_nu) / reference_nu
        metrics["nusselt_quadratic_relative_error"] = abs(
            metrics["nusselt_hot_quadratic"] - reference_nu) / reference_nu
    if reference_psi is not None:
        metrics["psi_reference"] = reference_psi
        metrics["psi_relative_error"] = abs(abs(psi_ext) - reference_psi) / reference_psi
    fields = {"theta": flow.theta, "u": flow.u, "v": flow.v, "p": flow.p,
              "streamfunction": psi}
    return {"tag": request.tag, "metrics": metrics, "fields": fields}


# ------------------------------------------------------------------- summary
def _series(records: list[dict], rayleigh: float, key: str) -> dict[int, float]:
    out = {}
    for rec in records:
        m = rec["metrics"]
        if abs(float(m["rayleigh"]) - rayleigh) > 1e-12 * rayleigh:
            continue
        value = m.get(key)
        if value is None:
            continue
        out[int(m["cells"])] = float(value)
    return out


def _refinement(values: dict[int, float]) -> dict:
    """Observed order + Richardson/GCI on the three finest grids (ratio 2)."""
    grids = sorted(values)
    if len(grids) < 3:
        return {"enough_grids": False, "grids": grids}
    coarse, medium, fine = grids[-3], grids[-2], grids[-1]
    if (fine / medium) != (medium / coarse):
        return {"enough_grids": False, "grids": grids,
                "reason": f"non-uniform refinement ratios ({coarse},{medium},{fine})"}
    spacings = [1.0 / fine, 1.0 / medium, 1.0 / coarse]
    triplet = [values[fine], values[medium], values[coarse]]
    order = observed_order(spacings, triplet)
    band = richardson_gci(fine=values[fine], medium=values[medium],
                          coarse=values[coarse], spacings=spacings,
                          theory_order=2.0)
    band = asdict(band) if is_dataclass(band) else dict(band)
    return {
        "enough_grids": True,
        "grids": grids,
        "triplet": {"coarse": [coarse, values[coarse]],
                    "medium": [medium, values[medium]],
                    "fine": [fine, values[fine]]},
        "observed_order": None if order != order else float(order),
        "richardson": band,
    }


def summarise(records: list[dict], *, gate: float = GATE) -> dict:
    """Grid-convergence verdict per Rayleigh number for the whole record set."""
    out = {"gate": gate, "families": {}}
    rayleighs = sorted({float(rec["metrics"]["rayleigh"]) for rec in records})
    for ra in rayleighs:
        block = {}
        for key in ("nusselt_hot_half_cell", "nusselt_hot_quadratic",
                    "psi_extremum_abs"):
            series = _series(records, ra, key)
            if not series:
                continue
            block[key] = {
                "values": {str(k): series[k] for k in sorted(series)},
                "errors_vs_reference": {},
                "refinement": _refinement(series),
            }
            reference = BENCHMARK_NU.get(ra) if key.startswith("nusselt") \
                else BENCHMARK_PSI.get(ra)
            if reference:
                block[key]["reference"] = reference
                block[key]["errors_vs_reference"] = {
                    str(k): abs(series[k] - reference) / reference
                    for k in sorted(series)
                }
        nu_half = block.get("nusselt_hot_half_cell", {})
        if nu_half:
            errors = nu_half["errors_vs_reference"]
            passing = sorted(int(k) for k, e in errors.items() if e <= gate)
            refinement = nu_half["refinement"]
            extrapolated = None
            if refinement.get("enough_grids"):
                rich = refinement["richardson"]
                extrapolated = rich.get("extrapolated")
            entry = {
                "finest_grid": max(int(k) for k in nu_half["values"]),
                "finest_error": errors[str(max(int(k) for k in nu_half["values"]))],
                "grids_passing_gate": passing,
                "extrapolated": extrapolated,
                "extrapolated_error": None,
            }
            if extrapolated is not None and nu_half.get("reference"):
                entry["extrapolated_error"] = abs(extrapolated
                                                 - nu_half["reference"]) \
                    / nu_half["reference"]
            block["verdict"] = entry
        out["families"][f"Ra{ra:.0e}"] = block
    return out