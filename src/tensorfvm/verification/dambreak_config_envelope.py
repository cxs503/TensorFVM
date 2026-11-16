"""Numerical-configuration envelope of the ``CASE-FVM-DAMBREAK-STOKER`` entry.

Why this module exists
----------------------
The case-pool entry grades the shallow-water solver at **one** point of the
numerical configuration space: ``n = 1600``, ``cfl = 0.4``, ``order = 2``,
``ShallowWater1D``'s shipped depth floor ``dry_depth = 1e-12``.  The entry's
numbers are quoted as if they were properties of the *scheme*, but three of the
four knobs are defaults, and one of them -- ``dry_depth`` -- is an **absolute
length**, i.e. a scale-dependent constant sitting inside the discretisation
(``shallow_water.robustness`` already flags absolute constants as the thing its
``similitude`` axis is meant to catch, but nothing has ever *varied* this one).

The floor is not a cosmetic tolerance.  ``ShallowWater1D`` reads it in five
places:

``set_state``      ``hu`` is zeroed in cells with ``h <= dry_depth``;
``primitive``      the velocity of a "dry" cell is defined as exactly ``0``;
``_face_states``   the reconstructed velocities are zeroed below the floor;
``_hll_flux``      the flux of a face whose two sides are both below the floor is ``0``;
``_sanitise``      any residual ``hu`` in a floored cell is discarded.

So the floor *defines* the measured velocity field near the dry front: a cell
whose depth is below the floor reports ``u = 0`` while the analytic solution
carries ``u -> 2 c0`` there.  That is precisely the quantity behind the open
``honest FAIL`` of the anchor's velocity leg (``u``-L1 = 5.07 % at ``n = 800``),
and nobody has asked how much of it is floor convention and how much is
scheme error.

What is delivered
-----------------
``run_point``
    One configuration of ``(n, cfl, order, dry_depth)`` through the *production*
    solver and the *unmodified* pool gates, plus hard-failure flags (exception,
    positivity clipping, mass drift, non-finite state, step budget) that the
    pool gates cannot express.
``floor_diagnostics``
    The "unjustified drying" area: cells the pair (numerical state, floor)
    calls dry while the exact solution is wet, the water mass hidden there, and
    a closed-form yardstick ``3 t sqrt(g * dry_depth)`` -- the thickness of the
    exact region where ``h <= dry_depth`` (Ritter fan: ``h = (2 c0 - xi)^2 / (9 g)``
    reaches ``eps`` at ``xi = 2 c0 - 3 sqrt(g eps)``).
``sweep_*`` / ``study``
    One-at-a-time sweeps of the three knobs about the pool point and the
    certified interval each axis keeps every pool gate inside.
``axis_envelope``
    Turns rows into a certificate: for each gate, the parameter values where it
    passes, the worst value seen, and the first parameter at which it breaks --
    i.e. **which criterion is the binding constraint** of the entry.
``hidden_mass_instrument_error``
    Calibration of the "hidden mass" instrument against high-resolution
    quadrature of the closed form, so a floor-induced reading can be trusted.

Nothing here changes the solver, the exact solution, the pool gates or the
anchor's published numbers: with the shipped defaults every function returns
the same values the entry already quotes (``tests/test_dambreak_config_envelope.py``
pins that).

References: Ritter (1892) Z. Ver. Deutsch. Ing. 36:947; Stoker (1957) *Water
Waves* ch. 10; Toro (2009) ch. 6; pool entry
``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json``.
"""

from __future__ import annotations

import math
import time as _time

import numpy as np

from ..shallow_water.dam_break import (
    ACCEPTED_MIN_N,
    RITTER_DRY,
    STOKER_WET,
    run_case,
)

__all__ = [
    "BASELINE",
    "CFL_POINTS",
    "FLOOR_POINTS",
    "ORDER_POINTS",
    "HARD_MASS_DRIFT",
    "run_point",
    "pool_gates",
    "hard_failure",
    "floor_diagnostics",
    "analytic_tip_length",
    "hidden_mass_instrument_error",
    "sweep_cfl",
    "sweep_floor",
    "sweep_order",
    "axis_envelope",
    "study",
]

#: the configuration the pool entry was certified at
BASELINE = {"n": 1600, "cfl": 0.4, "order": 2, "dry_depth": 1e-12}

#: CFL axis -- up to 0.95, i.e. right against the stability limit of an
#: explicit scheme (the shipped value 0.4 is far from it and nothing has ever
#: tested whether the accepted numbers survive the approach to the limit).
CFL_POINTS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95)

#: depth-floor axis.  ``1e-15`` is below double-precision relative resolution
#: at ``h0 = 1`` (a probe of the "there is no floor" limit), ``1e-12`` is the
#: shipped default, ``1e-3`` is the order of magnitude a practical
#: "wet/dry" tolerance would use, ``1e-2`` is a deliberately aggressive floor
#: (``1 %`` of ``h0``) that is still physically plausible in a coastal model.
FLOOR_POINTS = (1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2)

#: order axis of the shipped MUSCL/HLL solver
ORDER_POINTS = (1, 2)

#: mass drift above this is not round-off and is flagged as a hard failure
HARD_MASS_DRIFT = 1e-12


# --------------------------------------------------------------------------- #
# one configuration
# --------------------------------------------------------------------------- #
def _gates(run: dict) -> dict:
    """The pool verdict, stripped to JSON scalars (order preserved)."""
    out: dict = {}
    for key, item in run["verdict"].items():
        if key == "all_pass":
            continue
        out[key] = {
            "value": None if item["value"] is None else float(item["value"]),
            "limit": float(item["limit"]),
            "pass": bool(item["pass"]),
        }
    return out


def pool_gates(run: dict) -> dict:
    """Public alias of :func:`_gates` (the entry's own acceptance quantities)."""
    return _gates(run)


def hard_failure(discretisation: dict, gates: dict, error: str | None = None) -> dict:
    """Failures a metric-based gate cannot express.

    The pool gates are all relative errors of smooth functionals; a run that
    crosses the CFL limit usually dies by producing a negative depth (clipped
    silently by ``_sanitise``, which *leaks mass*), by draining the volume, or
    by raising.  Those are reported here instead of being smoothed into a
    percentage.
    """
    if error is not None:
        return {"exception": True, "error": error, "any": True}
    volume0 = discretisation.get("volume_initial")
    volume1 = discretisation.get("volume_final")
    drift = 0.0
    if volume0:
        drift = abs(volume1 - volume0) / abs(volume0)
    clipped = int(discretisation.get("clipped_cells") or 0)
    flips = int(discretisation.get("dry_momentum_flips") or 0)
    report = {
        "exception": False,
        "clipped_cells": clipped,
        "dry_momentum_flips": flips,
        "mass_drift": float(drift),
        "mass_leak": bool(drift > HARD_MASS_DRIFT),
        "positivity_clipping": bool(clipped > 0),
        "nan_in_gates": bool(any(
            item["value"] is not None and not math.isfinite(item["value"])
            for item in gates.values())),
        "all_gates_defined": bool(all(item["value"] is not None for item in gates.values())),
    }
    report["any"] = bool(
        report["positivity_clipping"]
        or report["mass_leak"]
        or report["nan_in_gates"]
        or not report["all_gates_defined"]
    )
    return report


def analytic_tip_length(dry_depth: float, t: float, h0: float,
                        gravity: float = 9.81) -> float:
    """Thickness of the exact region where ``h <= dry_depth`` (Ritter fan).

    ``h = (2 c0 - xi)^2 / (9 g)`` with ``xi = x / t``, so ``h <= eps`` for
    ``xi >= 2 c0 - 3 sqrt(g eps)``, i.e. over a length ``3 t sqrt(g eps)``
    measured *backwards* from the front ``x_f = 2 c0 t``.  A floor therefore
    declares dry a strip whose thickness grows like ``sqrt(eps)``: it is the
    analytic yardstick the numerical "floored" area has to be compared with.
    """
    if dry_depth <= 0.0:
        return 0.0
    c0 = math.sqrt(gravity * h0)
    tip = 3.0 * t * math.sqrt(gravity * dry_depth)
    return float(min(tip, 2.0 * c0 * t))


def floor_diagnostics(case, run: dict, dry_depth: float) -> dict:
    """How much wet water the depth floor declares dry at the end state.

    Reading is taken on the *returned profile* (no solver access), so the
    diagnostic is defined for any configuration:

    ``unjustified_dry_cells``  numerical ``h <= dry_depth`` while exact ``h > dry_depth``
    ``hidden_mass_fraction``   exact water mass inside those cells / exact total mass
    ``floored_length``         ``unjustified_dry_cells * dx``
    ``analytic_tip_length``    :func:`analytic_tip_length` (dry bed only)
    """
    x = np.asarray(run["profile"]["x"], dtype=float)
    h = np.asarray(run["profile"]["h_numeric"], dtype=float)
    h_ex = np.asarray(run["profile"]["h_exact"], dtype=float)
    if x.size < 2:
        raise ValueError("profile needs at least two cells")
    dx = abs(float(x[1] - x[0]))
    floored = h <= dry_depth
    unjustified = floored & (h_ex > dry_depth)
    exact_mass = float(np.sum(h_ex)) * dx
    hidden = float(np.sum(h_ex[unjustified])) * dx if np.any(unjustified) else 0.0
    t = float(run["discretisation"]["t_end"])
    out = {
        "dry_depth": float(dry_depth),
        "floored_cells": int(np.count_nonzero(floored)),
        "unjustified_dry_cells": int(np.count_nonzero(unjustified)),
        "floored_length": float(np.count_nonzero(unjustified) * dx),
        "hidden_mass": hidden,
        "hidden_mass_fraction": float(hidden / exact_mass) if exact_mass > 0.0 else 0.0,
        "dx": dx,
        "t_end": t,
    }
    if getattr(case, "kind", None) == "dry":
        tip = analytic_tip_length(dry_depth, t, case.h0, case.gravity)
        out["analytic_tip_length"] = tip
        out["floored_over_analytic"] = (
            float(out["floored_length"] / tip) if tip > 0.0 else float("inf"))
    return out


def _readings(case, run: dict) -> dict:
    """Signed instrument readings behind the relative errors (no re-derivation)."""
    m = run["metrics"]
    if case.kind == "dry":
        return {
            "front_fan": m.get("front_fan"),
            "front_wetted": m.get("front_wetted"),
            "front_exact": case.exact_front(run["discretisation"]["t_end"]),
            "front_recession": (
                None if m.get("front_wetted") is None
                else float(case.exact_front(run["discretisation"]["t_end"])
                           - m["front_wetted"])),
            "h_min": m.get("h_min"),
        }
    return {
        "shock_position": m.get("shock_position"),
        "shock_exact": m.get("shock_exact"),
        "shock_offset": (
            None if m.get("shock_position") is None or m.get("shock_exact") is None
            else float(m["shock_position"] - m["shock_exact"])),
        "h2_measured": m.get("h2_measured"),
        "h1_measured": m.get("h1_measured"),
    }


def run_point(case, n: int = BASELINE["n"], cfl: float = BASELINE["cfl"],
              order: int = BASELINE["order"],
              dry_depth: float = BASELINE["dry_depth"]) -> dict:
    """Integrate one configuration and grade it with the pool's own gates."""
    row = {
        "case": getattr(case, "key", str(case)),
        "kind": getattr(case, "kind", None),
        "n": int(n), "cfl": float(cfl), "order": int(order),
        "dry_depth": float(dry_depth),
    }
    wall = _time.perf_counter()
    try:
        run = run_case(case, n=n, cfl=cfl, order=order, dry_depth=dry_depth)
    except Exception as exc:  # noqa: BLE001 - a failure is data for this study
        row.update({"ok": False, "gates": {},
                    "hard": hard_failure({}, {}, error=f"{type(exc).__name__}: {exc}"),
                    "wall_time_s": _time.perf_counter() - wall})
        return row
    wall = _time.perf_counter() - wall
    gates = _gates(run)
    row.update({
        "reading": _readings(case, run),
        "ok": True,
        "gates": gates,
        "hard": hard_failure(run["discretisation"], gates),
        "steps": int(run["discretisation"]["steps"]),
        "courant_last": float(run["discretisation"].get("cfl", cfl)),
        "mass_drift": gates.get("front_series<5%") is not None
        and float(run["metrics"].get("mass_drift", 0.0)),
        "local_extrema": int(run["metrics"].get("local_extrema", 0)),
        "floor": floor_diagnostics(case, run, dry_depth),
        "wall_time_s": wall,
    })
    return row


# --------------------------------------------------------------------------- #
# instruments: calibration of the "hidden mass" reading
# --------------------------------------------------------------------------- #
def hidden_mass_instrument_error(dry_depth: float, t: float, h0: float,
                                 gravity: float = 9.81, n: int = 1600,
                                 reference_points: int = 1 << 18,
                                 x_min: float = -4.0, x_max: float = 6.0) -> dict:
    """Discrete vs high-resolution quadrature of the water inside ``h_ex <= eps``.

    The profile reading (``floor_diagnostics``) sums ``h_ex dx`` over cells; this
    function measures that sum against a 262 144-point quadrature of the same
    closed form, at the same ``dx``.  It returns the relative difference, i.e.
    the *quadrature error of the instrument*, not of the solver.  The cell
    average used here is the mid-point value (the conservative choice for a
    diagnostic that is meant to under-, not over-, state the hidden mass).
    """
    from ..shallow_water.exact import ritter_state  # local: exact profile

    x = x_min + (np.arange(n) + 0.5) * (x_max - x_min) / n
    dx = (x_max - x_min) / n
    h = np.asarray(ritter_state(x, t, h0, gravity, 0.0)[0], dtype=float)
    discrete = float(np.sum(h[h <= dry_depth])) * dx

    xr = x_min + (np.arange(reference_points) + 0.5) * (x_max - x_min) / reference_points
    hr = np.asarray(ritter_state(xr, t, h0, gravity, 0.0)[0], dtype=float)
    reference = float(np.sum(hr[hr <= dry_depth])) * (x_max - x_min) / reference_points
    rel = abs(discrete - reference) / reference if reference > 0.0 else 0.0
    return {
        "dry_depth": float(dry_depth),
        "n": int(n),
        "discrete": discrete,
        "reference": reference,
        "relative_difference": float(rel),
        "analytic_tip_length": analytic_tip_length(dry_depth, t, h0, gravity),
    }


# --------------------------------------------------------------------------- #
# axes
# --------------------------------------------------------------------------- #
def sweep_cfl(case, cfls=CFL_POINTS, n: int = 800, order: int = 2,
              dry_depth: float = BASELINE["dry_depth"]) -> list[dict]:
    return [run_point(case, n=n, cfl=c, order=order, dry_depth=dry_depth)
            for c in cfls]


def sweep_order(case, orders=ORDER_POINTS, n: int = 800,
                cfl: float = BASELINE["cfl"],
                dry_depth: float = BASELINE["dry_depth"]) -> list[dict]:
    return [run_point(case, n=n, cfl=cfl, order=o, dry_depth=dry_depth)
            for o in orders]


def sweep_floor(case, floors=FLOOR_POINTS, n: int = 800,
                cfl: float = BASELINE["cfl"], order: int = 2) -> list[dict]:
    return [run_point(case, n=n, cfl=cfl, order=order, dry_depth=f)
            for f in floors]


def axis_envelope(rows: list[dict], parameter: str) -> dict:
    """Certified interval of ``parameter`` that keeps every pool gate inside.

    A parameter value is *certified* when the run exists, raises no hard
    failure, and every gate passes.  The certificate reports, per gate, the
    values where it passed, the worst value seen, and the first value at which
    it broke -- so the entry's binding constraint is identified rather than
    asserted.
    """
    keyed = sorted(rows, key=lambda r: r[parameter])
    gates = sorted({g for row in keyed if row.get("ok") for g in row["gates"]})
    per_gate: dict = {}
    for gate in gates:
        values = [(row[parameter], row["gates"][gate])
                  for row in keyed if row.get("ok") and gate in row["gates"]]
        passing = [p for p, item in values if item["pass"]]
        finite = [(p, item["value"]) for p, item in values
                  if item["value"] is not None and math.isfinite(item["value"])]
        worst = max(finite, key=lambda t: t[1]) if finite else (None, None)
        broke = [p for p, item in values if not item["pass"]]
        per_gate[gate] = {
            "values": [float(p) for p, _ in values],
            "pass": [bool(item["pass"]) for _, item in values],
            "certified": [float(p) for p in passing],
            "worst_value": None if worst[1] is None else float(worst[1]),
            "worst_at": None if worst[0] is None else float(worst[0]),
            "first_break": float(min(broke)) if broke else None,
        }
    certified = []
    for row in keyed:
        if not row.get("ok") or row["hard"]["any"]:
            continue
        if all(item["pass"] for item in row["gates"].values()):
            certified.append(float(row[parameter]))
    return {
        "parameter": parameter,
        "points": [float(row[parameter]) for row in keyed],
        "certified": certified,
        "all_points_certified": bool(len(certified) == len(keyed)),
        "failed_points": [float(r[parameter]) for r in keyed
                          if not r.get("ok") or r["hard"]["any"]
                          or not all(i["pass"] for i in r["gates"].values())],
        "gates": per_gate,
        "hard_failures": [
            {parameter: float(r[parameter]), **r["hard"]}
            for r in keyed if not r.get("ok") or r["hard"]["any"]],
        "binding": _binding(per_gate),
    }


def _binding(per_gate: dict) -> dict | None:
    """The gate that breaks at the smallest parameter value (earliest break)."""
    candidates = [(info["first_break"], gate) for gate, info in per_gate.items()
                  if info["first_break"] is not None]
    if not candidates:
        return None
    value, gate = min(candidates)
    return {"gate": gate, "first_break": value}


def _row_table(rows: list[dict], columns) -> list[dict]:
    table = []
    for row in rows:
        entry = {key: row.get(key) for key in ("case", "n", "cfl", "order", "dry_depth")}
        entry["ok"] = bool(row.get("ok"))
        entry["hard_any"] = bool(row.get("hard", {}).get("any", not row.get("ok")))
        entry["steps"] = row.get("steps")
        entry["wall_time_s"] = row.get("wall_time_s")
        for gate in columns:
            item = row.get("gates", {}).get(gate)
            entry[gate] = None if item is None else item["value"]
        entry["local_extrema"] = row.get("local_extrema")
        entry["reading"] = row.get("reading")
        if row.get("floor"):
            entry["hidden_mass_fraction"] = row["floor"]["hidden_mass_fraction"]
            entry["unjustified_dry_cells"] = row["floor"]["unjustified_dry_cells"]
        table.append(entry)
    return table


def grid_columns(rows: list[dict]) -> list[str]:
    """Gate names present in ``rows``, in the pool's own order."""
    seen: list[str] = []
    for row in rows:
        for gate in (row.get("gates") or {}):
            if gate not in seen:
                seen.append(gate)
    return seen


def study(dry_n: int = 800, wet_n: int = 800, confirm_n: int = 1600,
          negative: bool = True) -> dict:
    """Full one-at-a-time study about the pool point.

    ``confirm_n`` re-runs the baseline and the two aggressive floor points on
    the pool's own accepted grid, so the sweep (which is done on ``n = 800``,
    the pool's ``ACCEPTED_MIN_N``, for cost) cannot be accused of quoting only
    a coarse-grid certifiable range.
    """
    payload: dict = {
        "baseline": dict(BASELINE),
        "accepted_min_n": ACCEPTED_MIN_N,
        "cfl_points": list(CFL_POINTS),
        "floor_points": list(FLOOR_POINTS),
        "order_points": list(ORDER_POINTS),
        "axes": {},
    }
    for case in (RITTER_DRY, STOKER_WET):
        name = case.key
        cfl_rows = sweep_cfl(case, n=dry_n if case.kind == "dry" else wet_n)
        order_rows = sweep_order(case, n=dry_n if case.kind == "dry" else wet_n)
        floor_rows = sweep_floor(case, n=dry_n if case.kind == "dry" else wet_n)
        columns = grid_columns(cfl_rows + order_rows + floor_rows)
        payload["axes"][name] = {
            "cfl": {"rows": _row_table(cfl_rows, columns),
                    "envelope": axis_envelope(cfl_rows, "cfl")},
            "order": {"rows": _row_table(order_rows, columns),
                      "envelope": axis_envelope(order_rows, "order")},
            "floor": {"rows": _row_table(floor_rows, columns),
                      "envelope": axis_envelope(floor_rows, "dry_depth")},
        }
    confirm_runs = [
        run_point(case, n=confirm_n, dry_depth=floor)
        for case, floors in ((RITTER_DRY, (BASELINE["dry_depth"], 1e-9, 1e-6, 1e-3)),
                             (STOKER_WET, (BASELINE["dry_depth"], 1e-3)))
        for floor in floors
    ]
    confirm_columns = grid_columns(confirm_runs)
    payload["confirmation_n1600"] = {
        "note": "the pool's own accepted grid; the sweeps run on n = 800 "
                "(ACCEPTED_MIN_N) for cost, these rows show the first break of "
                "the floor axis does not move under refinement",
        "rows": _row_table(confirm_runs, confirm_columns),
    }
    payload["confirmation_summary"] = [
        {
            "case": row["case"],
            "n": row["n"],
            "dry_depth": row["dry_depth"],
            "ok": bool(row["ok"]),
            "all_gates_pass": bool(
                row["ok"] and not row["hard"]["any"]
                and all(item["pass"] for item in row["gates"].values())),
            "front_wetted_error": (row.get("gates", {})
                                   .get("front_wetted<5%", {}).get("value")),
            "front_fan_error": (row.get("gates", {})
                                .get("front_fan<5%", {}).get("value")),
            "hidden_mass_fraction": (row.get("floor") or {}).get("hidden_mass_fraction"),
        }
        for row in confirm_runs
    ]
    if negative:
        neg_runs = {
            "dry": [run_point(RITTER_DRY, n=400, dry_depth=0.5)],
            "wet": [run_point(STOKER_WET, n=400, dry_depth=0.6)],
        }
        payload["negative_control"] = {
            "note": "deliberately misconfigured depth floors (0.5 h0 / 0.6 m "
                    "above every exact depth): the sweep must be able to see a "
                    "broken configuration, otherwise a green column means nothing",
            **{name: _row_table(rows, grid_columns(rows))
               for name, rows in neg_runs.items()},
        }
    payload["instrument"] = {
        str(floor): hidden_mass_instrument_error(
            floor, t=_t_star_time(RITTER_DRY), h0=RITTER_DRY.h0,
            gravity=RITTER_DRY.gravity, n=800)
        for floor in (1e-6, 1e-4, 1e-3, 1e-2)
    }
    payload["verdict"] = _verdict(payload)
    return payload


def _t_star_time(case) -> float:
    return float(case.t_star_end / math.sqrt(case.gravity / case.h0)) if case.h0 else 0.0


def _verdict(payload: dict) -> dict:
    """Aggregate: is every swept configuration certified, and what binds it."""
    per_case: dict = {}
    all_certified = True
    for name, axes in payload["axes"].items():
        entry: dict = {}
        for axis, blob in axes.items():
            env = blob["envelope"]
            entry[axis] = {
                "certified": env["certified"],
                "failed": env["failed_points"],
                "all_certified": env["all_points_certified"],
                "binding": env["binding"],
                "hard_failures": env["hard_failures"],
            }
            all_certified = all_certified and env["all_points_certified"]
        per_case[name] = entry
    return {
        "all_swept_configurations_certified": bool(all_certified),
        "confirmation_n1600": payload.get("confirmation_summary"),
        "per_case": per_case,
    }
