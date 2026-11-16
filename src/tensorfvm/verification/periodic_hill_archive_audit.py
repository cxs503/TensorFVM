"""Round 26: artefact integrity + bubble-*object* audit for ``CASE-FVM-PERIODIC-HILLS``.

Why this module exists
----------------------
Twenty-five rounds of this anchor produced a wall-shear based ``x_r/H``.  Three
independent audits already certified the pieces of that pipeline:

* round 17 certified the *zero-crossing operator* (:mod:`wall_shear_bubbles`),
* round 21/22 pinned *which* bubble the number belongs to,
* round 24 audited the *wall-normal derivative* that feeds it.

All three treat the wall-shear profile as a one-dimensional object and pair its
signed crossings into "bubbles".  Nothing in the tree ever asked whether a
paired crossing interval corresponds to a **closed recirculation cell in the
velocity field** -- i.e. whether "two bubbles" in the shear profile are two
vortices or one vortex whose wall signature is interrupted.

That question is not academic here.  On the round-20 laminar ``Re = 2800``
artefact the wall-shear profile pairs into **two** bubbles
(``0.40 -> 1.25`` and ``3.92 -> 8.26``) while the mean velocity field contains a
**single** connected reversed-flow cell that touches the wall on both sides of
``1.25``.  Round 23 then published the crest bubble (``1.254``, ``-76.8 %``)
as *the* measurement of the anchor.  If the two paired intervals are one
vortex, ``x_r/H`` is being measured on an object that does not exist, and the
published error is definition-dominated (``1.25`` vs ``8.26`` = ``130 %`` of the
reference) before it is physics-dominated.

Second gap, and the cheaper one to close: the case archive is a *directory of
``.npz`` files* used as evidence by every round, but no gate has ever checked
that those files carry a field at all.  **Six of the fourteen** archived
artefacts in ``docs/case-periodic-hills/`` are all-zero dumps (velocity, mean
velocity *and* wall shear): the laminar ``144x72/192x96/240x120`` rungs, the SST
``120x60/144x72`` rungs, and the aborted ``Re = 1400`` probe -- i.e. exactly the
sites of the two grid-ladder questions rounds 23 and 25 left open.  Because
their ``mean_tau_b`` is a zero array, the certified crossing operator reports
``status = "no_recirculation"`` with ``x_r/H = nan`` for them -- indistinguishable
from a genuinely attached flow unless the integrity gate runs first.  A round
that reads that as "no recirculation measured" would be reporting a
*measurement* of a run that never produced a field.
:func:`artifact_health` makes the state explicit so a ladder verdict can never
again be built on a degenerate dump.

What is here
------------
* :func:`artifact_health` -- usable / ``zero_field`` / ``non_finite`` /
  ``missing_keys``, with the evidence (``max|u|``, cell count, samples).
* :func:`recirculation_cells` -- 2-D connected components of ``u < 0`` on the
  stored mean field, **periodic in ``x``**, with per-cell wall-attachment spans
  and area.  This is the field-side (topology) route, independent of
  ``mean_tau_b``.
* :func:`wall_spans` -- wall-attachment intervals from the wall-adjacent cell
  row, reusing the certified crossing machinery (:func:`bubble_intervals`) so
  the two routes share their interpolation, not their input.
* :func:`bubble_object_audit` -- does the certified wall-shear bubble
  correspond to a closed field cell?  Reports the lift-off gap, the shear level
  inside it relative to the profile maximum, and where the same cell re-attaches
  to the wall further downstream.
* :func:`definition_routes` / :func:`definition_band` -- the anchor metric under
  three legitimate definitions (``wall_shear:first``, ``wall_shear:largest_area``,
  ``field:cell_last_wall_span``) with an explicit ``definition_sensitive`` flag
  against the case tolerance.
* :func:`ladder_summary` -- a grid ladder over *usable* rungs only, so an
  unusable rung can only ever produce ``insufficient_rungs``, never a
  convergence claim.

Everything is read-only over stored artefacts; no solver run is involved.
"""

from __future__ import annotations

import numpy as np

try:  # scipy is an optional ``[benchmark]``/``[dev]`` dependency
    from scipy import ndimage
except Exception:  # pragma: no cover - exercised only without scipy
    ndimage = None

from .periodic_hill_reference import X_R_OVER_H, X_R_TOLERANCE
from .wall_shear_bubbles import (
    bubble_intervals,
    period_of,
    separation_reattachment,
)

__all__ = [
    "ARTIFACT_USABLE",
    "ARTIFACT_ZERO_FIELD",
    "ARTIFACT_NON_FINITE",
    "ARTIFACT_MISSING_KEYS",
    "OBJECT_SUPPORTED",
    "OBJECT_INCOMPLETE",
    "OBJECT_NO_FIELD",
    "ROUTE_KEYS",
    "artifact_health",
    "wall_spans",
    "recirculation_cells",
    "bubble_object_audit",
    "definition_routes",
    "definition_band",
    "ladder_summary",
]

ARTIFACT_USABLE = "usable"
ARTIFACT_ZERO_FIELD = "zero_field"
ARTIFACT_NON_FINITE = "non_finite"
ARTIFACT_MISSING_KEYS = "missing_keys"

OBJECT_SUPPORTED = "object_supported"
OBJECT_INCOMPLETE = "object_incomplete"
OBJECT_NO_FIELD = "no_field_recirculation"

#: the metric definitions that may enter :func:`definition_band`
ROUTE_KEYS = (
    "wall_shear_first",
    "wall_shear_largest_area",
    "field_cell_first_wall_span",
    "field_cell_last_wall_span",
)

#: relative ``|u|`` level below which a velocity field is treated as empty
_ZERO_LEVEL = 1.0e-9
#: two numbers closer than this (in ``H``) are the same wall crossing
_MATCH_TOL = 1.5e-2


# --------------------------------------------------------------------------
# artefact integrity
# --------------------------------------------------------------------------
def artifact_health(dataset, *, u_key="mean_u", x_key="x_over_h",
                    tau_key="mean_tau_b", zero_level=_ZERO_LEVEL) -> dict:
    """Is a stored case artefact a measurement at all?

    Parameters
    ----------
    dataset : mapping or ``numpy.lib.npyio.NpzFile``
        Stored artefact (``.npz`` handle, or a dict with the same keys).
    u_key, x_key, tau_key : str
        Key names; the defaults are the round-9/20 convention.
    zero_level : float
        ``max|u|`` at or below which the field counts as empty.

    Returns
    -------
    dict with ``status`` (one of the module's ``ARTIFACT_*`` constants),
    ``usable`` (bool), ``max_abs_u``, ``n_cells``, ``wall_attached_columns``,
    ``reason``, and the grid size when it can be read.

    A ``zero_field`` is *not* "no recirculation": it is "no data".  The two are
    kept apart because every ladder rule in this module treats the first as
    missing and the second as a measurement.
    """
    keys = set(getattr(dataset, "files", dataset.keys()))
    out = {
        "status": ARTIFACT_MISSING_KEYS,
        "usable": False,
        "reason": "",
        "max_abs_u": None,
        "n_cells": None,
        "wall_attached_columns": None,
        "nx": None,
        "ny": None,
    }
    missing = [k for k in (u_key, x_key) if k not in keys]
    if missing:
        out["reason"] = f"missing keys: {missing}"
        return out
    u = np.asarray(dataset[u_key], dtype=float)
    x = np.asarray(dataset[x_key], dtype=float).ravel()
    if u.ndim != 2:
        out["reason"] = f"mean field is not 2-D (shape {u.shape})"
        return out
    out["nx"], out["ny"] = int(u.shape[1]), int(u.shape[0])
    if not np.all(np.isfinite(u)):
        out["status"] = ARTIFACT_NON_FINITE
        out["reason"] = "mean field contains non-finite values"
        return out
    max_abs = float(np.abs(u).max())
    out["max_abs_u"] = max_abs
    if max_abs <= zero_level:
        out["status"] = ARTIFACT_ZERO_FIELD
        out["reason"] = (f"max|u| = {max_abs:.3e} <= {zero_level:.1e}: the "
                         "artefact carries no velocity field (aborted/diverged "
                         "run), so it is missing data, not a measurement")
        return out
    cells = recirculation_cells(u, x)
    out["status"] = ARTIFACT_USABLE
    out["usable"] = True
    out["n_cells"] = len(cells)
    out["wall_attached_columns"] = int(np.count_nonzero(u[0] < 0.0))
    if tau_key not in keys:
        out["reason"] = f"usable field, but no {tau_key!r} (wall-shear route unavailable)"
    else:
        out["reason"] = "usable"
    return out


# --------------------------------------------------------------------------
# field-side (topology) route
# --------------------------------------------------------------------------
def _label_periodic(mask):
    """Connected components of ``mask`` (ny, nx) with periodic ``x``."""
    if ndimage is None:  # pragma: no cover
        raise RuntimeError("scipy is required for recirculation_cells")
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, 0
    # periodic adjacency: column 0 is a neighbour of column nx-1, so *every*
    # component touching either seam column is one component
    seam = set(np.unique(lab[:, 0][mask[:, 0]]).tolist()) | \
        set(np.unique(lab[:, -1][mask[:, -1]]).tolist())
    seam.discard(0)
    for v in seam:
        lab[lab == v] = min(seam)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    root = {}
    out = np.zeros_like(lab)
    for v in np.unique(lab):
        if v == 0:
            continue
        r = find(int(v))
        if r not in root:
            root[r] = len(root) + 1
        out[lab == v] = root[r]
    return out, len(root)


def wall_spans(wall_row, x_over_h, *, period=None) -> list:
    """Wall-attachment intervals (``u_wall < 0``) of one cell row.

    Uses :func:`bubble_intervals` -- the certified signed-crossing operator --
    on the *velocity* row rather than on ``mean_tau_b``, so the interpolation and
    the seam unwrapping are shared with the wall-shear route while the input is
    independent of the wall-normal derivative.
    """
    u = np.asarray(wall_row, dtype=float).ravel()
    x = np.asarray(x_over_h, dtype=float).ravel()
    period = period_of(x) if period is None else float(period)
    intervals, status = bubble_intervals(u, x, period)
    return [dict(b, status=status) for b in intervals]


def recirculation_cells(mean_u, x_over_h, *, threshold=0.0) -> list:
    """Connected reversed-flow cells (``u < threshold``) of a mean field.

    Parameters
    ----------
    mean_u : array_like, shape ``(ny, nx)``
        Mean streamwise velocity, row 0 = wall-adjacent cell row.
    x_over_h : array_like, shape ``(nx,)``
        Cell-centre abscissa in hill heights (strictly increasing, one period).

    Returns
    -------
    list of dict, sorted by descending area.  Each cell carries
    ``label``, ``area_cells``, ``x_min``/``x_max`` (cell-centre extent),
    ``row_min``/``row_max``, ``wall_attached``, ``wraps_seam`` and
    ``wall_spans`` -- the wall-attachment intervals of *that* cell, taken from
    the wall-adjacent row so a cell that lifts off the wall and touches down
    again reports two spans instead of ending early.
    """
    u = np.atleast_2d(np.asarray(mean_u, dtype=float))
    x = np.asarray(x_over_h, dtype=float).ravel()
    if u.shape[1] != x.size:
        raise ValueError(f"mean_u has {u.shape[1]} columns, x has {x.size}")
    period = period_of(x)
    mask = u < float(threshold)
    lab, n = _label_periodic(mask)
    if n == 0:
        return []
    wall_lab = lab[0]
    spans_all = wall_spans(u[0], x, period=period)
    cells = []
    for label in range(1, n + 1):
        sel = lab == label
        j, i = np.nonzero(sel)
        cell_spans = []
        for span in spans_all:
            lo, hi = span["separation"], span["reattachment"]
            # a wall span belongs to this cell if any wall column inside it is
            # labelled with this cell (the spans come from the same row)
            inside = np.where((x >= lo - 1e-12) & (x <= hi + 1e-12))[0]
            if inside.size and np.any(wall_lab[inside] == label):
                cell_spans.append(span)
        cells.append({
            "label": int(label),
            "area_cells": int(sel.sum()),
            "x_min": float(x[i].min()),
            "x_max": float(x[i].max()),
            "row_min": int(j.min()),
            "row_max": int(j.max()),
            "wall_attached": bool(np.any(wall_lab == label)),
            "wraps_seam": bool(np.any(lab[:, 0] == label) and np.any(lab[:, -1] == label)),
            "wall_spans": cell_spans,
        })
    cells.sort(key=lambda c: -c["area_cells"])
    return cells


def _cell_at_wall_x(cells, x_value, period):
    """Cells whose wall-attachment span contains ``x_value`` (mod ``period``)."""
    hit = []
    xv = float(x_value) % period
    for k, cell in enumerate(cells):
        for span in cell["wall_spans"]:
            lo, hi = span["separation"], span["reattachment"]
            if lo - _MATCH_TOL <= xv <= hi + _MATCH_TOL:
                hit.append(k)
                break
    return hit


def bubble_object_audit(tau, mean_u, x_over_h, *, primary="first",
                        period=None) -> dict:
    """Is the certified wall-shear bubble a closed recirculation cell?

    The certified operator (:func:`separation_reattachment`) answers "where does
    the selected bubble start and end **in the wall shear**".  This function
    asks the field whether that interval bounds a *cell*, i.e. whether the same
    connected reversed-flow region is wall-attached on both sides of the
    reattachment.  If it is, the crossing is a wall *lift-off* of a cell that
    continues downstream -- not the end of a vortex -- and the number must not
    be called "the reattachment length of the crest bubble".

    Returns
    -------
    dict with the certified ``wall_shear`` block, ``n_field_cells``,
    ``bubble_cells`` (indices of cells wall-attached anywhere inside the
    certified interval), ``verdict`` (``object_supported`` /
    ``object_incomplete`` / ``no_field_recirculation``), and for an incomplete
    object: ``continued_downstream``, ``next_wall_touchdown``, ``lift_off_band``
    and ``lift_off_shear_ratio`` (max ``|tau|`` in the lift-off band divided by
    max ``|tau|`` of the profile -- how much shear the "reattachment" rests on).
    """
    u = np.atleast_2d(np.asarray(mean_u, dtype=float))
    x = np.asarray(x_over_h, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    period = period_of(x) if period is None else float(period)
    ws = separation_reattachment(tau, x, period=period, primary=primary)
    out = {
        "primary": primary,
        "wall_shear": ws,
        "n_field_cells": None,
        "bubble_cells": [],
        "verdict": OBJECT_NO_FIELD,
        "continued_downstream": False,
        "next_wall_touchdown": None,
        "lift_off_band": None,
        "lift_off_shear_ratio": None,
        "field_reattachment_first_span": None,
        "field_reattachment_last_span": None,
    }
    if ws["status"] not in ("measured", "out_of_window"):
        return out
    cells = recirculation_cells(u, x)
    out["n_field_cells"] = len(cells)
    if not cells:
        return out
    lo, hi = ws["x_s_over_H"], ws["x_r_over_H"]
    inside = np.where((x >= lo - _MATCH_TOL) & (x <= hi + _MATCH_TOL))[0]
    hit = sorted({k for i in inside for k in _cell_at_wall_x(cells, x[i], period)})
    out["bubble_cells"] = hit
    if not hit:
        out["verdict"] = OBJECT_NO_FIELD
        return out

    cell = cells[hit[0]]
    spans = sorted(cell["wall_spans"], key=lambda s: s["separation"])
    out["field_reattachment_first_span"] = (
        float(spans[0]["reattachment"]) if spans else None)
    out["field_reattachment_last_span"] = (
        float(spans[-1]["reattachment"]) if spans else None)
    # a span that starts after the certified reattachment, on the same cell
    later = [s for s in spans if s["separation"] >= hi - _MATCH_TOL]
    if later:
        nxt = float(later[0]["separation"])
        out["continued_downstream"] = True
        out["next_wall_touchdown"] = nxt
        band = (hi, nxt)
        out["lift_off_band"] = [float(hi), float(nxt)]
        inner = (x >= band[0]) & (x <= band[1])
        denom = float(np.abs(tau).max())
        out["lift_off_shear_ratio"] = (
            float(np.abs(tau[inner]).max()) / denom if denom > 0 and np.any(inner)
            else None)
        out["verdict"] = OBJECT_INCOMPLETE
    else:
        out["verdict"] = OBJECT_SUPPORTED
    return out


# --------------------------------------------------------------------------
# definition band of the anchor metric
# --------------------------------------------------------------------------
def definition_routes(tau, mean_u, x_over_h, *, period=None) -> dict:
    """``x_r/H`` under three legitimate definitions, for one artefact."""
    u = np.atleast_2d(np.asarray(mean_u, dtype=float))
    x = np.asarray(x_over_h, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    period = period_of(x) if period is None else float(period)
    routes = {}
    for key, primary in (("wall_shear_first", "first"),
                         ("wall_shear_largest_area", "largest_area")):
        ws = separation_reattachment(tau, x, period=period, primary=primary)
        routes[key] = (float(ws["x_r_over_H"])
                       if ws["status"] in ("measured", "out_of_window")
                       else float("nan"))
    audit = bubble_object_audit(tau, u, x, primary="first", period=period)
    routes["field_cell_first_wall_span"] = audit["field_reattachment_first_span"]
    routes["field_cell_last_wall_span"] = audit["field_reattachment_last_span"]
    return {
        "routes": routes,
        "meta": {
            "n_cells": audit["n_field_cells"],
            "object_verdict": audit["verdict"],
            "continued_downstream": audit["continued_downstream"],
            "lift_off_shear_ratio": audit["lift_off_shear_ratio"],
            "next_wall_touchdown": audit["next_wall_touchdown"],
        },
    }


def definition_band(routes, *, reference=X_R_OVER_H,
                    tolerance=X_R_TOLERANCE) -> dict:
    """Spread of the candidate ``x_r/H`` values, with a sensitivity flag.

    ``routes`` is the ``"routes"`` mapping of :func:`definition_routes` (the
    ``"meta"`` block is excluded -- a cell count is not a length).

    ``flag`` is ``definition_sensitive`` when the spread exceeds twice the
    acceptance tolerance -- i.e. when *which* definition is used can move the
    verdict by more than the whole acceptance band.  In that case no single
    number may be quoted as "the" ``x_r/H`` without naming the definition.
    """
    if "routes" in routes and isinstance(routes["routes"], dict):
        routes = routes["routes"]
    vals = [v for k, v in routes.items() if k in ROUTE_KEYS
            and isinstance(v, (int, float)) and np.isfinite(v)]
    if not vals:
        return {"values": [], "n": 0, "min": None, "max": None,
                "spread": None, "spread_rel": None, "flag": "not_measured"}
    spread = float(max(vals) - min(vals))
    width = 2.0 * float(tolerance) * float(reference)
    return {
        "values": [float(v) for v in vals],
        "n": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "spread": spread,
        "spread_rel": spread / float(reference),
        "acceptance_width": width,
        "flag": ("definition_sensitive" if spread > width
                 else "definition_insensitive"),
    }


# --------------------------------------------------------------------------
# grid ladders over usable rungs only
# --------------------------------------------------------------------------
def ladder_summary(entries, *, min_rungs=3, reference=X_R_OVER_H,
                   tolerance=X_R_TOLERANCE) -> dict:
    """Coarse-to-fine summary of a grid ladder built from *usable* rungs.

    ``entries`` is an iterable of ``{"tag", "nx", "x_r_over_H", "usable"}``.
    Unusable rungs are reported but never enter the difference/verdict, so a
    ladder cannot be declared convergent by dropping the rungs that failed --
    the dropped rungs surface as ``n_excluded``.
    """
    rows, used, excluded = [], [], []
    for e in entries:
        row = {
            "tag": e.get("tag"),
            "nx": e.get("nx"),
            "x_r_over_H": (None if e.get("x_r_over_H") is None
                           else float(e["x_r_over_H"])),
            "usable": bool(e.get("usable", False)),
        }
        rows.append(row)
        if row["usable"] and row["x_r_over_H"] is not None \
                and np.isfinite(row["x_r_over_H"]):
            used.append(row)
        else:
            excluded.append(row)
    used.sort(key=lambda r: r["nx"])
    steps = []
    for a, b in zip(used, used[1:]):
        d = b["x_r_over_H"] - a["x_r_over_H"]
        steps.append({
            "from_nx": a["nx"], "to_nx": b["nx"],
            "delta": float(d),
            "delta_rel": float(d / a["x_r_over_H"]) if a["x_r_over_H"] else None,
        })
    out = {
        "rungs": rows,
        "n_usable": len(used),
        "n_excluded": len(excluded),
        "excluded": [r["tag"] for r in excluded],
        "steps": steps,
        "reference": float(reference),
    }
    if len(used) < min_rungs:
        out["verdict"] = "insufficient_rungs"
        out["reason"] = (f"{len(used)} usable rung(s) of {len(rows)}; a "
                         f"convergence claim needs >= {min_rungs}")
        out["x_r_over_H"] = None
    else:
        last = used[-1]
        out["verdict"] = "measured"
        out["x_r_over_H"] = last["x_r_over_H"]
        out["error_rel"] = ((last["x_r_over_H"] - reference) / reference)
        out["within_tolerance"] = bool(abs(out["error_rel"]) <= tolerance)
        out["finest_pair_change_rel"] = (steps[-1]["delta_rel"] if steps else None)
    return out