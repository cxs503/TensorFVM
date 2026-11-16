"""Third-code cross-validation of ``CASE-FVM-DAMBREAK-STOKER`` (Clawpack witness).

Why this module exists
----------------------
The dam-break entry of the case pool grades six criteria against the Ritter
(1892) / Stoker (1957) exact solutions, and by now (``VERIFICATION_DEBT_LEDGER``
D29-D32, the oracle-free gate, the momentum criteria G7/G8) the evidence chain is
deep -- but **every witness is inside TensorFVM**: the graded solver, the
acceptance readings, the ``oracle-free`` re-readings and the config-envelope
sweep all share this package's grid, boundary and commit discipline.  A
systematic defect common to all of them (wrong end-face boundary, a sign in the
momentum flux, a limiter that is "monotone by construction" in this code only)
would be *invisible*: it would pass everything.

D29 therefore recommended a separate ledger item,
``XVAL-FVM-DAMBREAK-THIRD-CODE`` (acceptance: an independent code family solves
the same case to L1 <= 5 %).  This module is that witness.  It runs Clawpack
5.14.0 (Classic wave-propagation, Fortran kernels -- an independent author,
limiter, wave decomposition and time stepping) with **two** independent Riemann
solvers (Roe + entropy fix, and HLLE) on the *same* initial data, the *same*
domain, the *same* transmissive end faces, the *same* CFL 0.4 and the *same*
``t* = 2``.  The two profiles are then compared

* on the same grid (**no interpolation**, which would hide a translation error);
* with the anchor's own measuring instrument (:func:`~tensorfvm.shallow_water.
  dam_break.fan_front_estimate`, :func:`~...plateau_from_depth_modes`,
  :func:`~...local_extrema_details`) applied to the *third-code* profile -- so
  the question answered is not "is TensorFVM right" but "does the published
  acceptance gate reproduce the verdict on a foreign code, or is the pass an
  artefact of this instrument?".

Deliverables of the study (:func:`third_code_study`):

* ``wet`` -- Stoker wet bed, TensorFVM vs Clawpack/Roe vs Clawpack/HLLE for
  ``n = 200 ... 1600``: cross-code relative depth L1, both codes' errors against
  the exact bore position and level step, monotonicity readings;
* ``dry`` -- Ritter dry bed: the exact dry limit is **not runnable** by Clawpack
  Classic (see below), recorded as an honest scope limit, not a pass;
* ``dry_limit`` -- a ladder in the downstream depth ``h1 = 0.2 ... 0.01`` where
  both codes do run: the bore front tends to the Ritter value ``2 c0 t`` and the
  two codes are compared at each rung;
* ``calibration`` -- injection probes on the *new* distance metric (translation,
  amplitude scaling, round-off noise) so that a sub-percent reading is known to
  be a property of the solutions and not of a blind instrument.

Honest scope limits (measured here, not assumed)
------------------------------------------------
Clawpack Classic is a *wet* shallow-water solver: on the anchor's exact dry-bed
initial data (``h = 0`` downstream) both its Roe and HLLE fluxes produce
non-finite depths, because neither is positivity preserving.  TensorFVM's HLL
solver survives that case (it counts its clips/flips), so the dry-bed criteria
remain **single-code** evidence and this module says so instead of substituting
a softened initial condition and calling it the same test.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..shallow_water.dam_break import (
    ACCEPTED_MIN_N,
    CASES,
    RITTER_DRY,
    STOKER_WET,
    TOLERANCE,
    fan_front_estimate,
    local_extrema_details,
    plateau_from_depth_modes,
    ritter_front_position,
)
from ..shallow_water.exact import ritter_state, stoker_constant_state, stoker_state
from ..shallow_water.swe1d import ShallowWater1D

__all__ = [
    "CFL",
    "DEFAULT_GRIDS",
    "DRY_LADDER",
    "TOLERANCE",
    "clawpack_available",
    "run_clawpack",
    "run_tensorfvm",
    "profile_readings",
    "relative_l1",
    "cross_code_distance",
    "third_code_study",
    "calibration_probes",
    "wettability_floor",
    "build_payload",
]

#: CFL held identical in both codes (also the pool's declared value).
CFL = 0.4
#: grids of the head-to-head table; ``ACCEPTED_MIN_N`` (800) is the anchor floor.
DEFAULT_GRIDS: Tuple[int, ...] = (200, 400, 800, 1600)
#: downstream depths of the dry-limit ladder (``h1 = 0`` is not runnable by the
#: third code; the ladder is how close to it an independent witness can reach).
DRY_LADDER: Tuple[float, ...] = (0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 1e-3)
#: amplitude above which a local extremum counts as a spurious wiggle (pool口径)
EXTREMUM_AMPLITUDE_FRACTION = 1e-4


# --------------------------------------------------------------------------- #
# third code (Clawpack) — imported lazily so the package stays usable without it
# --------------------------------------------------------------------------- #
def clawpack_available() -> bool:
    """True when ``clawpack.pyclaw`` and the shallow-water Riemann solvers import."""
    try:  # pragma: no cover - environment probe
        from clawpack import pyclaw, riemann

        _ = pyclaw.ClawSolver1D(riemann.shallow_roe_with_efix_1D)
        _ = pyclaw.ClawSolver1D(riemann.shallow_hlle_1D)
    except Exception:
        return False
    return True


_RIEMANN = {"roe": "shallow_roe_with_efix_1D", "hlle": "shallow_hlle_1D"}


def _riemann_solver(kind: str):
    from clawpack import riemann

    name = _RIEMANN.get(kind.lower())
    if name is None:
        raise ValueError(f"unknown Riemann solver {kind!r}; use one of {sorted(_RIEMANN)}")
    return getattr(riemann, name)


def _clawpack_version() -> str:
    try:  # pragma: no cover - environment probe
        import clawpack

        return str(getattr(clawpack, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "unavailable"


def run_clawpack(case, n: int, riemann: str = "roe", h_left: Optional[float] = None,
                 h_right: Optional[float] = None, cfl: float = CFL,
                 gravity: float = 9.81, t_end: Optional[float] = None
                 ) -> Dict[str, Any]:
    """Solve one dam-break initial value problem with Clawpack Classic.

    ``h_left``/``h_right`` default to the case's own states; passing ``h_right``
    explicitly produces the dry-limit ladder rungs (a Stoker problem with a thin
    downstream layer).  Returns cell centres, ``h``, ``hu``, ``u`` and the
    time the solver actually reached -- plus ``ok = False`` and the failure mode
    when the run goes non-finite (Clawpack is not positivity preserving).
    """
    from clawpack import pyclaw

    h_left = case.h0 if h_left is None else float(h_left)
    h_right = case.h1 if h_right is None else float(h_right)
    t_end = case.t_end if t_end is None else float(t_end)

    solver = pyclaw.ClawSolver1D(_riemann_solver(riemann))
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.bc_lower[0] = pyclaw.BC.extrap  # zero-gradient ends, as in swe1d
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.cfl_desired = cfl
    solver.cfl_max = 0.9

    dim = pyclaw.Dimension(case.x_min, case.x_max, int(n), name="x")
    domain = pyclaw.Domain(dim)
    state = pyclaw.State(domain, 2)
    state.problem_data["grav"] = gravity
    x = np.asarray(domain.grid.x.centers, dtype=float)
    state.q[0] = np.where(x < case.x_dam, h_left, h_right)
    state.q[1] = 0.0
    solution = pyclaw.Solution(state, domain)
    try:
        solver.evolve_to_time(solution, t_end)
    except Exception as exc:  # pragma: no cover - solver-level failure
        return {"code": f"clawpack-classic-{riemann}", "n": int(n), "ok": False,
                "reason": f"{type(exc).__name__}: {exc}"}

    h = np.array(solution.q[0], dtype=float)
    hu = np.array(solution.q[1], dtype=float)
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(hu))):
        bad = int(np.size(h) - np.count_nonzero(np.isfinite(h)))
        return {"code": f"clawpack-classic-{riemann}", "n": int(n), "ok": False,
                "reason": "non-finite solution (negative/nan depth: the flux is "
                          "not positivity preserving on a dry or near-dry bed)",
                "non_finite_cells": bad,
                "h_min_finite": float(np.nanmin(h[np.isfinite(h)])) if bad < h.size else None}
    wet = h > 0.0
    u = np.where(wet, hu / np.where(wet, h, 1.0), 0.0)
    return {"code": f"clawpack-classic-{riemann}", "n": int(n), "ok": True,
            "x": x, "h": h, "hu": hu, "u": u, "t": float(solution.t),
            "t_requested": float(t_end), "h_min": float(h.min()),
            "h_max": float(h.max()), "volume": float(h.sum() * (case.x_max - case.x_min) / h.size)}


def run_tensorfvm(case, n: int, h_left: Optional[float] = None,
                  h_right: Optional[float] = None, order: int = 2,
                  cfl: float = CFL, gravity: float = 9.81,
                  t_end: Optional[float] = None) -> Dict[str, Any]:
    """The graded solver, configured exactly as the pool entry declares it."""
    h_left = case.h0 if h_left is None else float(h_left)
    h_right = case.h1 if h_right is None else float(h_right)
    t_end = case.t_end if t_end is None else float(t_end)
    solver = ShallowWater1D(int(n), case.x_min, case.x_max, gravity=gravity,
                            cfl=cfl, order=order)
    solver.set_state(np.where(solver.x < case.x_dam, h_left, h_right))
    solver.advance(t_end)
    h = np.array(solver.h, dtype=float)
    hu = np.array(solver.hu, dtype=float)
    _, u = solver.primitive()
    return {"code": "tensorfvm-hll-muscl-ssprk3", "n": int(n), "ok": True,
            "x": np.array(solver.x, dtype=float), "h": h, "hu": hu,
            "u": np.array(u, dtype=float), "t": float(solver.time),
            "t_requested": float(t_end), "h_min": float(h.min()),
            "h_max": float(h.max()), "clipped_cells": int(solver.clipped_cells),
            "dry_flips": int(solver.dry_flips),
            "volume": float(h.sum() * (case.x_max - case.x_min) / h.size)}


# --------------------------------------------------------------------------- #
# readings and distances
# --------------------------------------------------------------------------- #
def _exact_profile(case, x: np.ndarray, t: float, h_left: float,
                   h_right: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact ``(h, u)`` on ``x`` at ``t`` (Ritter when ``h_right == 0``)."""
    if h_right <= 0.0:
        return ritter_state(x, t, h_left, case.gravity, case.x_dam)
    return stoker_state(x, t, h_left, h_right, case.gravity, case.x_dam)


def profile_readings(case, x: np.ndarray, h: np.ndarray, t: float,
                     h_left: Optional[float] = None,
                     h_right: Optional[float] = None) -> Dict[str, Any]:
    """The anchor's own instrument applied to *any* profile.

    Dry beds are read with :func:`fan_front_estimate` (front position from the
    exact straight-line rarefaction shape), wet beds with
    :func:`plateau_from_depth_modes` (plateau level / still-water level / bore
    crossing from the depth distribution only), and monotonicity with
    :func:`local_extrema_details`.  Nothing is fitted to the numerical profile
    beyond what those published readings already do.
    """
    h_left = case.h0 if h_left is None else float(h_left)
    h_right = case.h1 if h_right is None else float(h_right)
    amplitude = EXTREMUM_AMPLITUDE_FRACTION * h_left
    extrema = local_extrema_details(h, amplitude, x)
    out: Dict[str, Any] = {
        "n_extrema": len(extrema),
        "monotone": len(extrema) == 0,
    }
    if h_right <= 0.0:
        front = fan_front_estimate(x, h, t, case.gravity, h_left)
        exact_front = ritter_front_position(t, h_left, case.gravity, case.x_dam)
        out.update({"front": float(front["front"]), "front_exact": float(exact_front),
                    "front_error": _relative(front["front"], exact_front),
                    "front_window_cells": int(front.get("cells", 0))})
    else:
        stoker = stoker_constant_state(h_left, h_right, case.gravity)
        plateau = plateau_from_depth_modes(x, h)
        exact_bore = case.x_dam + stoker["shock_speed"] * t
        exact_step = stoker["h2"] / h_right
        out.update({
            "method": plateau["method"],
            "bore": float(plateau["bore_x"]), "bore_exact": float(exact_bore),
            "bore_error": _relative(plateau["bore_x"], exact_bore),
            "h2": float(plateau["h2"]), "h2_exact": float(stoker["h2"]),
            "step_ratio": float(plateau["step_ratio"]),
            "step_ratio_exact": float(exact_step),
            "step_error": _relative(plateau["step_ratio"], exact_step),
        })
    return out


def _relative(measured: float, exact: float) -> float:
    if not (math.isfinite(measured) and math.isfinite(exact)):
        return float("nan")
    if exact == 0.0:
        return abs(measured)
    return abs(measured - exact) / abs(exact)


def relative_l1(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """``int |a - b| dx / int |b| dx`` on a shared grid (trapezoid-free, cellwise)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"profiles must share a shape, got {a.shape} vs {b.shape}")
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    denom = float(np.sum(np.abs(b)) * dx)
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(a - b)) * dx / denom)


def cross_code_distance(profile_a: Dict[str, Any], profile_b: Dict[str, Any]
                        ) -> Dict[str, Any]:
    """Same-grid distance between two solutions of the same problem.

    No interpolation: a translation of the wave by less than ``dx`` shows up in
    the L1 distance instead of being absorbed by a resample.  Velocity is only
    compared where both codes are wet, and the momentum flux ``hu`` is reported
    because it is what the conservative update actually transports.
    """
    if profile_a["n"] != profile_b["n"]:
        raise ValueError("cross-code comparison requires identical grids "
                         f"(got {profile_a['n']} and {profile_b['n']})")
    x = profile_a["x"]
    both_wet = (profile_a["h"] > 0.0) & (profile_b["h"] > 0.0)
    du = np.abs(profile_a["u"][both_wet] - profile_b["u"][both_wet])
    return {
        "n": int(profile_a["n"]),
        "depth_l1": relative_l1(x, profile_a["h"], profile_b["h"]),
        "momentum_l1": relative_l1(x, profile_a["hu"], profile_b["hu"]),
        "velocity_l1_wet": relative_l1(x[both_wet], profile_a["u"][both_wet],
                                       profile_b["u"][both_wet]) if both_wet.any() else float("nan"),
        "velocity_max_abs_wet": float(du.max()) if du.size else float("nan"),
        "wet_cells_compared": int(both_wet.sum()),
    }


def _exact_l1(case, profile: Dict[str, Any], h_left: Optional[float] = None,
              h_right: Optional[float] = None) -> float:
    h_left = case.h0 if h_left is None else float(h_left)
    h_right = case.h1 if h_right is None else float(h_right)
    h_exact, _ = _exact_profile(case, profile["x"], profile["t"], h_left, h_right)
    return relative_l1(profile["x"], profile["h"], h_exact)


# --------------------------------------------------------------------------- #
# studies
# --------------------------------------------------------------------------- #
def _wet_row(case, n: int, profiles: Sequence[Dict[str, Any]],
             h_left: float, h_right: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {"n": int(n), "codes": {}}
    for prof in profiles:
        if not prof.get("ok"):
            row["codes"][prof["code"]] = {
                "ok": False,
                **{k: v for k, v in prof.items() if k not in ("code", "n", "ok")},
            }
            continue
        readings = profile_readings(case, prof["x"], prof["h"], prof["t"], h_left, h_right)
        readings.update({
            "ok": True,
            "depth_l1_vs_exact": _exact_l1(case, prof, h_left, h_right),
            "h_min": prof.get("h_min"),
            "clipped_cells": prof.get("clipped_cells"),
        })
        row["codes"][prof["code"]] = readings
    raws = [p for p in profiles if p.get("ok")]
    if len(raws) >= 2:
        base = raws[0]
        row["cross_code"] = {
            other["code"]: cross_code_distance(other, base) for other in raws[1:]
        }
        row["cross_code_baseline"] = base["code"]
        # this grid's own sensitivity: L1 distance of a one-cell translation, so a
        # measured distance can be read as "N cells of wave position" (a global L1
        # percentage alone is not resolution-aware)
        shifted = dict(base, h=np.roll(base["h"], 1), hu=np.roll(base["hu"], 1),
                       u=np.roll(base["u"], 1))
        row["one_cell_translation_l1"] = cross_code_distance(shifted, base)["depth_l1"]
        row["cell_width_m"] = float(np.mean(np.diff(base["x"])))
    return row


def third_code_study(grids: Iterable[int] = DEFAULT_GRIDS,
                     ladder: Iterable[float] = DRY_LADDER) -> Dict[str, Any]:
    """Full head-to-head study (wet table, dry scope limit, dry-limit ladder)."""
    grids = tuple(int(n) for n in grids)
    out: Dict[str, Any] = {
        "tolerance": TOLERANCE,
        "accepted_min_n": int(ACCEPTED_MIN_N),
        "cfl": CFL,
        "grids": list(grids),
        "third_code": f"clawpack {_clawpack_version()} (Classic wave propagation, MC limiter)",
        "available": clawpack_available(),
        "wet": {}, "dry": {}, "dry_limit": {},
    }
    if not out["available"]:  # pragma: no cover - environment dependent
        out["skipped"] = "clawpack not importable"
        return out

    wet = CASES["stoker-wet"]
    for n in grids:
        profiles = [run_tensorfvm(wet, n),
                    run_clawpack(wet, n, "roe"),
                    run_clawpack(wet, n, "hlle")]
        out["wet"][str(n)] = _wet_row(wet, n, profiles, wet.h0, wet.h1)

    dry = CASES["ritter-dry"]
    for n in grids:
        profiles = [run_tensorfvm(dry, n), run_clawpack(dry, n, "roe"),
                    run_clawpack(dry, n, "hlle")]
        out["dry"][str(n)] = _wet_row(dry, n, profiles, dry.h0, dry.h1)

    for h1 in ladder:
        rungs: Dict[str, Any] = {}
        for n in gridss(grids):
            profiles = [run_tensorfvm(wet, n, wet.h0, float(h1)),
                        run_clawpack(wet, n, "roe", wet.h0, float(h1)),
                        run_clawpack(wet, n, "hlle", wet.h0, float(h1))]
            rungs[str(n)] = _wet_row(wet, n, profiles, wet.h0, float(h1))
        out["dry_limit"][f"{h1:g}"] = rungs

    out["verdict"] = _verdict(out)
    return out


def gridss(grids: Sequence[int]) -> Tuple[int, ...]:
    """Ladder grids: the anchor floor and the finest table grid."""
    accepted = [n for n in grids if n >= ACCEPTED_MIN_N]
    return tuple(sorted({min(accepted) if accepted else max(grids), max(grids)}))


def _verdict(study: Dict[str, Any]) -> Dict[str, Any]:
    """Decision on the pool's own tolerance, from the archived numbers only."""
    tol = study["tolerance"]
    n_min = study["accepted_min_n"]
    rows = [(int(n), row) for n, row in study["wet"].items() if int(n) >= n_min]
    verdict: Dict[str, Any] = {"criteria": {}}
    if not rows:
        verdict["status"] = "withheld"
        verdict["reason"] = "no grid at or above the accepted minimum"
        return verdict

    worst_l1 = 0.0
    worst_l1_grid = None
    worst_equiv_cells = None
    worst_cell_width = None
    worst_reading = 0.0
    worst_reading_key = None
    for n, row in rows:
        for code, metrics in row["codes"].items():
            if not metrics.get("ok"):
                continue
            for key in ("bore_error", "step_error"):
                value = metrics.get(key)
                if value is not None and math.isfinite(value) and value > worst_reading:
                    worst_reading, worst_reading_key = value, f"{code}@{n}:{key}"
        per_cell = row.get("one_cell_translation_l1")
        for code, dist in row.get("cross_code", {}).items():
            if dist["depth_l1"] > worst_l1:
                worst_l1, worst_l1_grid = dist["depth_l1"], f"{code}@{n}"
                worst_cell_width = row.get("cell_width_m")
                worst_equiv_cells = (dist["depth_l1"] / per_cell) if per_cell else None

    verdict["criteria"]["XV1_cross_code_depth_l1"] = {
        "value": worst_l1, "limit": tol, "worst_at": worst_l1_grid,
        "equivalent_translation_cells": worst_equiv_cells,
        "cell_width_m": worst_cell_width,
        "equivalent_translation_m": (None if worst_cell_width is None
                                     or worst_equiv_cells is None
                                     else worst_equiv_cells * worst_cell_width),
        "pass": bool(worst_l1 <= tol),
        "statement": "TensorFVM vs an independent code family on the same grid: "
                     "relative depth L1 over the wet-bed case (worst accepted grid)",
    }
    verdict["criteria"]["XV2_gate_portability"] = {
        "value": worst_reading, "limit": tol, "worst_at": worst_reading_key,
        "pass": bool(worst_reading <= tol),
        "statement": "the anchor's headline readings (bore position, level step) "
                     "recomputed by the same instrument on the third-code profile",
    }
    monotone_fail = [f"{code}@{n}" for n, row in rows for code, m in row["codes"].items()
                     if m.get("ok") and not m.get("monotone", True)]
    verdict["criteria"]["XV3_instrument_monotone"] = {
        "failures": monotone_fail, "pass": not monotone_fail,
        "statement": "the monotone/no-wiggle criterion also holds on the foreign code",
    }
    dry_ok = [n for n, row in study["dry"].items()
              if all(m.get("ok") for m in row["codes"].values() if m)]
    verdict["criteria"]["XV4_dry_bed_third_code"] = {
        "runnable_grids": dry_ok, "pass": bool(dry_ok),
        "statement": "the anchor's exact dry-bed case is runnable by the third code "
                     "(it is not: Classic Clawpack is not positivity preserving), so "
                     "the dry-bed criteria stay single-code evidence",
    }
    mandatory = ("XV1_cross_code_depth_l1", "XV2_gate_portability", "XV3_instrument_monotone")
    verdict["status"] = "pass" if all(verdict["criteria"][k]["pass"] for k in mandatory) else "fail"
    return verdict


def calibration_probes(base: Dict[str, Any], case=None,
                       translations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
                       amplitudes: Sequence[float] = (0.01, 0.02, 0.05)
                       ) -> Dict[str, Any]:
    """Injection probes on :func:`cross_code_distance` (instrument calibration).

    A sub-percent reading is only meaningful if the metric reacts to defects of
    the size it claims to resolve, so the probes are a **ladder in defect size**
    and they are compared against the anchor's two headline readings measured on
    the *same* perturbed profile:

    * the wave is translated by ``1 ... 64`` cells (a pure roll of the discrete
      field -- the cleanest admissible shift);
    * the plateau is scaled by ``+1 / +2 / +5 %`` at unchanged velocity (an
      amplitude error, the class of defect the position readings are blind to);
    * round-off noise ``1e-9`` must not trip anything.

    The output states the metric's **detection power** in physical units:
    ``cells_per_percent`` (how many cells of translation one per-cent of L1
    distance buys), the translation at which the pool tolerance trips, and the
    same ladder read through the bore-position / level-step readings so a reader
    can see which instrument is sharpest.
    """
    case = STOKER_WET if case is None else case
    x = np.asarray(base["x"], dtype=float)
    dx = float(np.mean(np.diff(x)))
    h = np.asarray(base["h"], dtype=float)
    hu = np.asarray(base["hu"], dtype=float)
    u = np.asarray(base["u"], dtype=float)
    h0_max = float(np.max(np.abs(h)))
    tol = TOLERANCE
    probes: Dict[str, Any] = {}
    sensitivity: List[Dict[str, Any]] = []

    def _shifted(cells: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_s, hu_s, u_s = np.roll(h, cells), np.roll(hu, cells), np.roll(u, cells)
        h_s[:cells] = h[0]
        hu_s[:cells] = hu[0]
        u_s[:cells] = u[0]
        return h_s, hu_s, u_s

    for cells in translations:
        h_s, hu_s, u_s = _shifted(int(cells))
        dist = cross_code_distance(dict(base, h=h_s, hu=hu_s, u=u_s), base)
        probes[f"translate_{cells}_cells"] = dist
        readings = profile_readings(case, x, h_s, base["t"])
        sensitivity.append({
            "translation_cells": int(cells),
            "translation_m": float(cells * dx),
            "depth_l1": dist["depth_l1"],
            "bore_error": readings.get("bore_error"),
            "step_error": readings.get("step_error"),
        })

    plateau = h > 0.99 * h0_max
    for amp in amplitudes:
        h_s = np.where(plateau, h * (1.0 + amp), h)
        hu_s = np.where(plateau, hu * (1.0 + amp), hu)
        dist = cross_code_distance(dict(base, h=h_s, hu=hu_s, u=u), base)
        probes[f"plateau_amplitude_{100 * amp:g}pct"] = dist
        readings = profile_readings(case, x, h_s, base["t"])
        sensitivity.append({
            "plateau_amplitude": float(amp),
            "depth_l1": dist["depth_l1"],
            "bore_error": readings.get("bore_error"),
            "step_error": readings.get("step_error"),
        })

    roundoff = cross_code_distance(dict(base, h=h * (1.0 + 1e-9),
                                        hu=hu * (1.0 + 1e-9), u=u), base)
    probes["roundoff_1e-9"] = roundoff

    per_cell = probes["translate_1_cells"]["depth_l1"]
    trip = next((int(c) for c in sorted(translations)
                 if probes[f"translate_{c}_cells"]["depth_l1"] >= tol), None)
    return {
        "probes": probes,
        "sensitivity_ladder": sensitivity,
        "cell_width": dx,
        "tolerance": tol,
        "detection": {
            "depth_l1_per_cell_translation": per_cell,
            "translation_cells_to_reach_tolerance": trip,
            "translation_m_to_reach_tolerance": (None if trip is None else trip * dx),
            "roundoff_false_alarm": bool(roundoff["depth_l1"] > 1e-7),
            "statement": "depth_l1 grows linearly with the translation of the wave; "
                         "dividing a measured distance by this slope converts it into "
                         "'equivalent cells of wave translation', a resolution-aware "
                         "way to read a sub-percent agreement. The bore-position "
                         "reading in the same ladder shows which instrument is sharper.",
        },
    }


def wettability_floor(case=None, n: int = 1600,
                      depths: Sequence[float] = (5e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4),
                      codes: Sequence[str] = ("tensorfvm", "roe", "hlle")
                      ) -> Dict[str, Any]:
    """How thin a downstream layer each code can carry (the dry-bed approach).

    The anchor's dry-bed case has ``h1 = 0``; a code without a positivity
    preserving flux cannot run it.  This scan locates, for each code, the
    thinnest ``h1`` that still integrates to ``t* = 2`` -- the quantitative form
    of the scope limit quoted in the report.
    """
    case = STOKER_WET if case is None else case
    out: Dict[str, Any] = {"n": int(n), "depths": [float(d) for d in depths], "codes": {}}
    for depth in depths:
        row: Dict[str, Any] = {}
        for code in codes:
            if code == "tensorfvm":
                prof = run_tensorfvm(case, n, case.h0, float(depth))
            else:
                prof = run_clawpack(case, n, code, case.h0, float(depth))
            if not prof.get("ok"):
                row[code] = {"ok": False, "reason": prof.get("reason")}
                continue
            readings = profile_readings(case, prof["x"], prof["h"], prof["t"],
                                        case.h0, float(depth))
            row[code] = {"ok": True, "h_min": prof.get("h_min"),
                         "bore_error": readings.get("bore_error"),
                         "method": readings.get("method"),
                         "finite": bool(np.all(np.isfinite(prof["h"])))}
        out["codes"][f"{depth:g}"] = row
    out["floor"] = {
        code: min((float(d) for d, row in out["codes"].items()
                   if row.get(code, {}).get("ok")), default=None)
        for code in codes
    }
    return out


def build_payload(grids: Iterable[int] = DEFAULT_GRIDS,
                  ladder: Iterable[float] = DRY_LADDER) -> Dict[str, Any]:
    """Study + calibration probes, ready for JSON (all numbers programmatic)."""
    study = third_code_study(grids, ladder)
    if study.get("available"):
        wet = CASES["stoker-wet"]
        reference = run_clawpack(wet, int(max(tuple(grids))), "roe")
        study["calibration"] = calibration_probes(reference, CASES["stoker-wet"])
        study["wettability_floor"] = wettability_floor(n=int(max(tuple(grids))))
        study["cases"] = {k: v.summary() for k, v in CASES.items()}
        study["exact"] = {
            "ritter_front_x": float(ritter_front_position(wet.t_end, wet.h0, wet.gravity, wet.x_dam)),
            "stoker": {k: float(v) for k, v in
                       stoker_constant_state(wet.h0, wet.h1, wet.gravity).items()
                       if isinstance(v, (int, float))},
        }
    return study


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays for ``json.dump``."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


build_payload._jsonable = _jsonable  # type: ignore[attr-defined]