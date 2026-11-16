"""Refinement-ladder arithmetic for a benchmark whose reference is *itself* a grid solution.

Why this module exists (verification debt)
------------------------------------------
``CASE-FVM-CAVITY-RE1000`` grades the MAC finite-volume solver against the Ghia,
Ghia & Shin (1982) tables, which were produced on a ``129 x 129``
vorticity-streamfunction grid.  Every acceptance number therefore mixes two
things that a three-level ladder cannot separate:

1. the *discretisation error of our own solution*, and
2. the *offset of the tabulated reference from the exact solution*.

The repository's earlier uncertainty note (``UNCERTAINTY.md``) computed a
Roache/Celik GCI band from three grids (n = 32, 64, 128, refinement ratio 2) and
observed that the tabulated value sits inside that band.  Three grids give a
*single* contraction ratio, so that argument cannot test its own premise: a
sequence that agrees with the table only because two errors cancel looks exactly
like a sequence that converges to the table.

The functions here are the arithmetic that a fourth level makes available, and
that the third level could not:

* :func:`triple_order` - observed order from three levels with *unequal* spacing
  (n = 64, 128, 192 has ratios 2 and 1.5, so the textbook ``log2`` formula does
  not apply and the usual call would silently mis-report the order).
* :func:`relative_errors`, :func:`monotone_decreasing`, :func:`cancellation_ratio`
  - the same functional measured against *two different references*: the
  tabulated one and an independent high-accuracy one (``SPECTRAL_HIORDER.md``'s
  fourth-order spectral extrapolation).  A residual that is our own
  discretisation error must shrink against both; an offset in the reference
  shrinks against neither and *grows* against the table as the solution improves.
* :func:`ladder_verdict` - the disposition, which is allowed to answer
  ``unresolved``.

Everything is a pure function of numbers handed in by the caller, so the module
carries no knowledge of the cavity and is reusable for any ladder study against a
tabular benchmark.

Field-level helpers (:func:`centreline_profiles`, :func:`profile_errors`,
:func:`field_functionals`) are an *independent* re-derivation from the raw MAC
faces: the regression test ``tests/test_cavity_grid_ladder.py`` pins them to the
already-published re-scorer ``docs/CASE-FVM-CAVITY-RE1000/rescore_registered_gate.py``
to 1e-12, so two separately written implementations have to agree before any
ladder number is quoted.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "centreline_profiles",
    "streamfunction_vortex",
    "field_functionals",
    "profile_errors",
    "pair_order",
    "textbook_order",
    "relative_errors",
    "monotone_decreasing",
    "cancellation_ratio",
    "ladder_verdict",
]


# --------------------------------------------------------------------------- #
# 1. Raw MAC field -> benchmark functionals (independent re-derivation)
# --------------------------------------------------------------------------- #
def centreline_profiles(
    u_field: np.ndarray, v_field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centre coordinates plus ``u(y)`` on ``x = L/2`` and ``v(x)`` on ``y = L/2``.

    ``u_field`` is the normal-x velocity on the ``(n+1, n)`` vertical faces and
    ``v_field`` the normal-y velocity on the ``(n, n+1)`` horizontal faces, both on
    ``[0, L]^2`` with ``L = 1``.  The centreline of the *staggered* grid is a face
    column, i.e. exactly the stored data - no interpolation is needed to obtain it.
    """
    u = np.asarray(u_field, dtype=float)
    v = np.asarray(v_field, dtype=float)
    n = int(u.shape[1])
    if v.shape != (n, n + 1):
        raise ValueError(f"staggered shape mismatch: u {u.shape}, v {v.shape}")
    coords = (np.arange(n) + 0.5) / n
    return coords, u[n // 2, :].copy(), v[:, n // 2].copy()


def streamfunction_vortex(u_field: np.ndarray) -> dict:
    """Stream function by trapezoidal integration of ``u`` and its interior extremum.

    ``psi(y) = int_0^y u dy`` on the cell centres, integrated upwards from the
    no-slip wall.  The primary-vortex centre is the interior *minimum* of ``psi``.
    This is an estimator independent of the solver's own extremum search, so a
    disagreement between the two is evidence about the *estimator*, not the flow.
    """
    u = np.asarray(u_field, dtype=float)
    n = int(u.shape[1])
    uc = 0.5 * (u[:-1, :] + u[1:, :])                     # (x, y) on cell centres
    h = 1.0 / n
    psi = np.zeros_like(uc)
    for j in range(1, n):
        psi[:, j] = psi[:, j - 1] + 0.5 * h * (uc[:, j] + uc[:, j - 1])
    inner = psi[1:-1, 1:-1]
    i, j = np.unravel_index(int(np.argmin(inner)), inner.shape)
    return {
        "psi_min": float(inner.min()),
        "centre": [float((i + 1 + 0.5) * h), float((j + 1 + 0.5) * h)],
        "psi": psi,
    }


def field_functionals(u_field: np.ndarray, v_field: np.ndarray) -> dict:
    """Resolution-consistent functionals of one converged field.

    ``*_interior`` are the discrete extrema of the centreline profiles (what the
    grid can actually resolve); ``u_min_node`` / ``v_max_node`` are the values at
    the *tabulated* nodes, which is what the registered judge compares and which is
    therefore an interpolant-dependent quantity at coarse grids.
    """
    coords, u_line, v_line = centreline_profiles(u_field, v_field)
    psi = streamfunction_vortex(u_field)
    return {
        "n": int(len(coords)),
        "u_min_interior": float(u_line.min()),
        "v_max_interior": float(v_line.max()),
        "psi_min": psi["psi_min"],
        "vortex_centre": psi["centre"],
        "coords": coords,
        "u_line": u_line,
        "v_line": v_line,
    }


def profile_errors(
    u_field: np.ndarray,
    v_field: np.ndarray,
    *,
    scheme: str = "hermite",
    reynolds: float = 1000.0,
) -> dict:
    """Score one field against the Ghia tables with the registered interpolation rule.

    ``scheme="hermite"`` is the repository's judge rule; ``scheme="pchip"`` is the
    monotone alternative used to show the verdict does not depend on the
    interpolant.  Wall nodes are supplied as data (``u = 0/1``, ``v = 0``) so no
    interpolant ever extrapolates.
    """
    import torch

    from .lid_cavity import hermite_interpolate
    from .lid_cavity_reference import table

    coords, u_line, v_line = centreline_profiles(u_field, v_field)
    out: dict[str, Any] = {}
    walls = {"u": ((0.0, 0.0), (1.0, 1.0)), "v": ((0.0, 0.0), (1.0, 0.0))}
    for which, line in (("u", u_line), ("v", v_line)):
        refs = table(float(reynolds), which)
        targets = [p[0] for p in refs]
        ref_vals = [float(p[1]) for p in refs]
        if scheme == "hermite":
            got = [
                float(x)
                for x in hermite_interpolate(
                    torch.as_tensor(coords, dtype=torch.float64),
                    torch.as_tensor(line, dtype=torch.float64),
                    targets,
                )
            ]
        elif scheme == "pchip":
            from scipy.interpolate import PchipInterpolator

            (lo_x, lo_v), (hi_x, hi_v) = walls[which]
            xs = np.concatenate([[lo_x], coords, [hi_x]])
            ys = np.concatenate([[lo_v], line, [hi_v]])
            f = PchipInterpolator(xs, ys, extrapolate=False)
            got = [float(f(t)) for t in targets]
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown interpolation scheme {scheme!r}")
        j = int(np.argmin(ref_vals)) if which == "u" else int(np.argmax(ref_vals))
        out[f"max_abs_{which}"] = max(abs(g - r) for g, r in zip(got, ref_vals))
        out[f"{which}_extremum_node"] = float(got[j])
        out[f"{which}_extremum_ref"] = float(ref_vals[j])
        out[f"{which}_profile"] = got
    out["scheme"] = scheme
    return out


# --------------------------------------------------------------------------- #
# 2. Observed order: against an independent limit, and on a uniform ladder
# --------------------------------------------------------------------------- #
def pair_order(
    f_coarse: float,
    f_fine: float,
    n_coarse: int,
    n_fine: int,
    reference: float,
) -> dict:
    """Observed order of a *pair* of grids measured against a known limit.

    ``p = log(e_coarse / e_fine) / log(n_fine / n_coarse)`` with ``e = |f - reference|``.
    This is the only order estimator used for the fourth level, and deliberately so:

    * a three-level Richardson estimate assumes a *uniform* refinement ratio.  The
      rung n = 128 -> 192 has ``r = 1.5`` while 64 -> 128 has ``r = 2``, so the
      textbook ``log2(delta_1/delta_2)`` formula would mis-report the order (it
      returns 4.7 where the honest value is ~2.3);
    * no order can be extracted from a pair alone *without* a reference, because
      the leading coefficient C is unknown.  Since an independent high-accuracy
      reference exists here (``SPECTRAL_HIORDER.md`` fourth-order spectral
      extrapolation, own uncertainty ~1e-5 relative), using it removes C instead of
      assuming it away.

    Returns ``{"order": None, "oscillatory": True}`` if refining made the error
    worse, and flags pairs whose order exceeds an attainable bound rather than
    quoting the number as if it were a property of the scheme.
    """
    e_c = abs(float(f_coarse) - float(reference))
    e_f = abs(float(f_fine) - float(reference))
    ratio = float(n_fine) / float(n_coarse)
    if ratio <= 1.0:
        raise ValueError("fine grid must be finer")
    if e_f == 0.0:
        raise ValueError("zero error against the reference: order is unbounded")
    if e_c == 0.0:
        raise ValueError("coarse-grid error is zero: cancellation at zero, order undefined")
    if e_c <= e_f:
        return {
            "order": None,
            "oscillatory": True,
            "error_ratio": e_c / e_f,
            "errors": [e_c, e_f],
            "grids": [int(n_coarse), int(n_fine)],
        }
    p = math.log(e_c / e_f) / math.log(ratio)
    return {
        "order": float(p),
        "oscillatory": False,
        "error_ratio": e_c / e_f,
        "errors": [e_c, e_f],
        "grids": [int(n_coarse), int(n_fine)],
    }


def textbook_order(values: Sequence[float], grids: Sequence[int]) -> dict:
    """Roache two-pair order for a *uniform* ladder, with its assumption exposed.

    ``values`` coarse -> fine.  Requires equal refinement factors ``n_{i+1}/n_i``; the ratio
    actually seen is returned so a caller cannot silently apply it to a
    non-uniform rung.  Returns ``oscillatory=True`` when increments change sign.
    """
    if len(values) < 3 or len(grids) != len(values):
        raise ValueError("textbook_order needs at least three levels")
    n = [float(g) for g in grids]
    ratios = [n[i + 1] / n[i] for i in range(len(n) - 1)]   # refinement factor > 1
    if max(ratios) - min(ratios) > 1e-12:
        raise ValueError(f"ladder is not uniform: refinement ratios {ratios}")
    inc = [float(values[i + 1]) - float(values[i]) for i in range(len(values) - 1)]
    if any(a * b < 0.0 for a, b in zip(inc, inc[1:])):
        return {"order": None, "oscillatory": True, "increments": inc, "ratios": ratios}
    p = math.log(inc[0] / inc[-1]) / ((len(inc) - 1) * math.log(ratios[0]))
    return {
        "order": float(p),
        "oscillatory": False,
        "increments": inc,
        "ratios": ratios,
        "refinement": ratios[0],
    }


# --------------------------------------------------------------------------- #
# 3. Two references, one ladder
# --------------------------------------------------------------------------- #
def relative_errors(values: Sequence[float], reference: float) -> list[float]:
    """``|f_i - ref| / |ref|`` for every level (absolute reference for zero refs)."""
    ref = float(reference)
    if ref == 0.0:
        return [abs(float(v)) for v in values]
    return [abs(float(v) - ref) / abs(ref) for v in values]


def monotone_decreasing(values: Sequence[float], *, tol: float = 0.0) -> bool:
    """True when every step is non-increasing (``tol`` allows a flat step)."""
    v = [float(x) for x in values]
    return all(b <= a + tol for a, b in zip(v, v[1:]))


def cancellation_ratio(errors: Sequence[float]) -> float:
    """``e_finest / e_previous``: > 1 means refining made agreement *worse*.

    Against a reference that is itself grid-limited this is the fingerprint of
    error cancellation on the coarser grid, not of a solver defect - which is why
    the number is only meaningful next to the same functional measured against an
    independent reference.
    """
    e = [float(x) for x in errors]
    if len(e) < 2:
        raise ValueError("need at least two levels")
    if e[-2] == 0.0:
        raise ValueError("previous-level error is zero; ratio undefined")
    return e[-1] / e[-2]


def ladder_verdict(
    *,
    err_vs_table: Sequence[float],
    err_vs_limit: Sequence[float],
    gci_rel: Sequence[float],
    tolerance: float,
) -> dict:
    """Disposition of a multi-grid study with an independent high-accuracy reference.

    The four statements are kept separate on purpose; collapsing them into one
    "PASS" is exactly what let a three-level band be quoted as an asymptotic
    uncertainty.

    * ``gate_pass`` - every level's tabulated error inside ``tolerance``.
    * ``table_agreement_monotone`` - does refining improve agreement with the table?
    * ``limit_convergence_monotone`` - does refining improve agreement with the
      independent limit?  This is what a *solver* is required to do.
    * ``residual_exceeds_discretisation`` - is the finest tabulated error larger
      than the finest discretisation-uncertainty band?

    The verdict is ``reference_offset_dominates`` only when agreement with the
    independent limit improves while agreement with the table degrades and the
    residual exceeds our own band: then the residual cannot be our discretisation
    error.  Anything else stays ``unresolved`` (or ``discretisation_dominated``).
    """
    e_tab = [float(x) for x in err_vs_table]
    e_lim = [float(x) for x in err_vs_limit]
    gci = [float(x) for x in gci_rel]
    table_mono = monotone_decreasing(e_tab)
    limit_mono = monotone_decreasing(e_lim)
    exceeds = e_tab[-1] > gci[-1]
    gate = max(e_tab) <= float(tolerance)
    if (not table_mono) and limit_mono and exceeds:
        verdict = "reference_offset_dominates"
    elif table_mono and not exceeds:
        verdict = "discretisation_dominated"
    else:
        verdict = "unresolved"
    return {
        "gate_pass": bool(gate),
        "table_agreement_monotone": bool(table_mono),
        "limit_convergence_monotone": bool(limit_mono),
        "residual_exceeds_discretisation": bool(exceeds),
        "cancellation_finest": cancellation_ratio(e_tab),
        "finest_table_error": e_tab[-1],
        "finest_limit_error": e_lim[-1],
        "finest_gci": gci[-1],
        "tolerance": float(tolerance),
        "verdict": verdict,
    }