"""Gravity-driven dam-break leg: surge front and column height vs experiment.

The flow solver and the VOF transport are the repository's
(:class:`tensorfvm.vof_phase0.solver.Phase0TwoPhaseSolver` with the geometric
:class:`tensorfvm.vof_plic.PLICVOF`), driven by
:func:`tensorfvm.two_phase.vof.dam_break_history`.  This module adds the
*experimental* anchor and the numbers that make it auditable:

* **Reference**: Martin & Moyce (1952), Phil. Trans. R. Soc. A 244, 312 -- the
  classical dam-break experiment for an aspect-2 column, digitised in
  :mod:`tensorfvm.vof_phase0.anchors` (``t* = t sqrt(g/a)``,
  ``x* = (x_f - a)/a``, ``h* = h/h0``).  This is an independent external
  anchor: the gate is not compared against the code's own initial condition.
* **Observables**: leading-edge front (outermost cell carrying more than 2 % of
  a cell of liquid -- the physical wet front for a geometric reconstruction),
  half-cell front (sensitivity bound), residual column height over the four
  leftmost columns, liquid volume drift and ``alpha`` boundedness.
* **Window**: ``0.3 < t* <= t_end`` -- the very early transient is dominated by
  the column-collapse initialisation and is *not* where the comparison means
  anything (stated explicitly so the gate cannot be tuned after the fact).

The zero-gravity control (:func:`tensorfvm.two_phase.vof.static_interface_drift`)
is run as a negative control: with no force the interface must not move at all.
"""

from __future__ import annotations

import numpy as np

from ..two_phase.vof import dam_break_history, plic_solver, static_interface_drift
from ..vof_phase0.anchors import martin_moyce_front, martin_moyce_height
from ..vof_phase0.dam_break import DamBreakConfig

__all__ = ["run_dambreak_leg", "decompose_dambreak_error",
           "quantisation_floor", "resolvable_error_floor",
           "FRONT_ERROR_MAX", "HEIGHT_ERROR_MAX",
           "VOLUME_DRIFT_MAX", "CONTROL_MAX_ABS_CHANGE", "WINDOW",
           "LATE_T_STAR"]

#: Max relative front error against Martin & Moyce over the comparison window.
FRONT_ERROR_MAX = 0.15
#: Max relative column-height error against Martin & Moyce.
HEIGHT_ERROR_MAX = 0.15
#: Max relative liquid-volume drift of the transport during the run.
VOLUME_DRIFT_MAX = 0.01
#: Zero-gravity negative control: the interface must be frozen to round-off.
CONTROL_MAX_ABS_CHANGE = 1.0e-14
#: Dimensionless-time window of the quantitative comparison.
WINDOW = (0.3, 8.0)

#: Late-time window used by the decomposition diagnostics (``t* >= LATE_T_STAR``).
LATE_T_STAR = 2.0
#: Shift grid searched by :func:`decompose_dambreak_error` [t*].
SHIFT_GRID = (-1.5, 2.0, 0.01)


def decompose_dambreak_error(
    t_star,
    got,
    ref,
    *,
    late_t_star: float = LATE_T_STAR,
    shift_grid=SHIFT_GRID,
) -> dict:
    """Split a dam-break error into *phase lag* and *rate deficit*.

    The D1/D2 gates are max-relative-error criteria: they say *how far off* the
    curve is but not *why*, and a curve that starts late and then runs at the
    right speed is indistinguishable from one that runs too slowly.  This
    function is a **diagnostic, not a gate** -- it never changes a verdict, it
    only reports, from the same history arrays the gates use, the three
    separable components:

    ``phase_lag_t_star``
        the time shift ``Δ`` that, applied to the reference
        (``ref(t - Δ)``), minimises the RMS difference *in x\*\** units over the
        full window.  ``Δ > 0`` means the numerical curve is late.
    ``speed_ratio``
        least-squares slope of ``got`` over ``t* >= late_t_star`` divided by
        the slope of ``ref`` over the same window on the same samples
        (``> 1``: the late curve is steeper than the reference, i.e. the defect
        is a delayed release rather than a slow wave).
    ``verdict``
        ``"phase_lag"`` (ratio >= 0.95 and Δ >= 0.05), ``"rate_deficit"``
        (ratio < 0.85) or ``"mixed"``.

    The helper is a pure function of the sampled curves so it can be unit
    tested on synthetic histories with a known shift (see
    ``tests/test_dambreak_error_decomposition.py``).
    """
    t = np.asarray(t_star, dtype=float)
    g = np.asarray(got, dtype=float)
    r = np.asarray(ref, dtype=float)
    out = {
        "phase_lag_t_star": float("nan"),
        "rms_at_zero_shift": float("nan"),
        "rms_at_best_shift": float("nan"),
        "max_rel_at_best_shift": float("nan"),
        "speed_got": float("nan"),
        "speed_ref": float("nan"),
        "speed_ratio": float("nan"),
        "time_scale_zero_shift": float("nan"),
        "rms_at_best_scale": float("nan"),
        "shift_explained_fraction": float("nan"),
        "late_max_rel_error": float("nan"),
        "early_max_rel_error": float("nan"),
        "verdict": "unavailable",
        "reason": "",
    }

    usable = np.isfinite(t) & np.isfinite(g) & np.isfinite(r)
    if usable.sum() < 4:
        out["reason"] = "fewer than 4 finite samples"
        return out
    t, g, r = t[usable], g[usable], r[usable]
    order = np.argsort(t)
    t, g, r = t[order], g[order], r[order]

    def _rms(a, b):
        d = a - b
        d = d[np.isfinite(d)]
        return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")

    out["rms_at_zero_shift"] = _rms(g, r)

    lo, hi, step = shift_grid
    shifts = np.arange(lo, hi + 0.5 * step, step)
    best = None
    for s in shifts:
        # got(t) is compared with the reference taken at t - s: a positive
        # shift means the reference attained these values *earlier*.
        rs = np.interp(t - s, t, r, left=np.nan, right=np.nan)
        val = _rms(g, rs)
        if not np.isfinite(val):
            continue
        if best is None or val < best[0]:
            best = (val, float(s), rs)
    if best is None:
        out["reason"] = "no shift keeps the reference inside the history"
        return out
    out["rms_at_best_shift"] = best[0]
    out["phase_lag_t_star"] = best[1]

    # Falsification control for the reference's time normalisation: if the
    # digitised curve were normalised with sqrt(2 g / a) instead of sqrt(g / a)
    # (a common convention clash) the misfit would be absorbed by a *scale*
    # factor, not by a shift.  Report the best pure time scale at zero shift so
    # the two hypotheses can be told apart from the same history.
    best_scale = None
    for sc in np.arange(0.6, 2.2 + 0.5 * 0.005, 0.005):
        rs = np.interp(t / sc, t, r, left=np.nan, right=np.nan)
        val = _rms(g, rs)
        if not np.isfinite(val):
            continue
        if best_scale is None or val < best_scale[0]:
            best_scale = (val, float(sc))
    if best_scale is not None:
        out["time_scale_zero_shift"] = best_scale[1]
        out["rms_at_best_scale"] = best_scale[0]
    rs = best[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(g - rs) / np.maximum(np.abs(rs), 1.0e-6)
    out["max_rel_at_best_shift"] = float(np.nanmax(rel))

    late = t >= late_t_star
    if late.sum() >= 3:
        sg = float(np.polyfit(t[late], g[late], 1)[0])
        sr = float(np.polyfit(t[late], r[late], 1)[0])
        out["speed_got"] = sg
        out["speed_ref"] = sr
        if abs(sr) > 1.0e-12:
            out["speed_ratio"] = sg / sr
        with np.errstate(divide="ignore", invalid="ignore"):
            late_rel = np.abs(g[late] - r[late]) / np.maximum(np.abs(r[late]), 1e-6)
        out["late_max_rel_error"] = float(np.nanmax(late_rel))
    with np.errstate(divide="ignore", invalid="ignore"):
        early = t < late_t_star
        if early.any():
            early_rel = np.abs(g[early] - r[early]) / np.maximum(np.abs(r[early]), 1e-6)
            out["early_max_rel_error"] = float(np.nanmax(early_rel))

    ratio = out["speed_ratio"]
    lag = out["phase_lag_t_star"]
    if np.isfinite(ratio) and np.isfinite(lag):
        if ratio >= 0.95 and lag >= 0.05:
            out["verdict"] = "phase_lag"
        elif ratio < 0.85:
            out["verdict"] = "rate_deficit"
        else:
            out["verdict"] = "mixed"
        zero = out["rms_at_zero_shift"]
        explained = (100.0 * (1.0 - out["rms_at_best_shift"] / zero)
                     if zero > 1.0e-12 else float("nan"))
        out["shift_explained_fraction"] = (
            float(1.0 - out["rms_at_best_shift"] / zero)
            if zero > 1.0e-12 else 1.0)
        out["reason"] = (
            f"late-window slope {out['speed_got']:.3f} vs reference "
            f"{out['speed_ref']:.3f} (ratio {ratio:.3f}), best time shift "
            f"{lag:+.2f} t*: RMS {zero:.3f} -> "
            f"{out['rms_at_best_shift']:.3f} x* "
            f"({explained:.0f} % of the misfit is a pure time shift)")
    return out


def resolvable_error_floor(
    t_star, ref, *, quantum: float, gate: float,
) -> dict:
    """Smallest relative error the cell-quantised observable can exhibit.

    A half-cell front (or a cell-count column height) moves in whole cells, so
    the smallest non-zero relative error at a sample is ``quantum / |ref|``.
    Where that floor already exceeds the gate, *any* deviation violates the gate
    and the criterion is decided by quantisation noise instead of by model
    accuracy.  ``violating_fraction`` is the share of window samples in that
    state; a non-zero value means the gate cannot be read as an accuracy bound
    on that grid.
    """
    t = np.asarray(t_star, dtype=float)
    r = np.asarray(ref, dtype=float)
    ok = np.isfinite(t) & np.isfinite(r) & (np.abs(r) > 1.0e-9)
    if not ok.any():
        return {"quantum": float(quantum), "violating_fraction": float("nan")}
    floor = quantum / np.abs(r[ok])
    return {
        "quantum": float(quantum),
        "floor_at_window_start": float(floor[0]),
        "floor_max": float(np.max(floor)),
        "violating_fraction": float(np.mean(floor > gate)),
        "samples": int(ok.sum()),
    }


def quantisation_floor(
    *, dx: float, dy: float, a: float, h0: float, gate: float,
    ref_mean: float | None = None,
) -> dict:
    """Resolution floor of the two dam-break observables.

    Both observables are cell-quantised: the residual column height moves in
    whole ``dy`` steps (it is a cell count times ``dy``) and the half-cell front
    in whole ``dx`` steps, so a gate of ``15 %`` can only be *measured* if it
    contains several quantisation levels.  ``*_gate_levels`` is how many
    quantisation steps fit inside the gate (``< 3`` means the criterion is
    resolution-limited, not physics-limited).
    """
    h_q = dy / h0
    f_q = dx / a
    out = {
        "height_quantum_rel": float(h_q),
        "front_quantum_star": float(f_q),
        "height_gate_levels": float(gate / h_q) if h_q > 0 else float("inf"),
        "front_gate_levels": float("nan"),
    }
    if ref_mean is not None and abs(ref_mean) > 1.0e-12:
        f_rel = f_q / abs(ref_mean)
        out["front_quantum_rel"] = float(f_rel)
        out["front_gate_levels"] = float(gate / f_rel)
    return out



def run_dambreak_leg(
    nx: int = 120,
    ny: int = 40,
    *,
    t_star_end: float = 4.0,
    scheme: str = "plic",
    control_steps: int = 20,
    max_steps: int = 6000,
    well_balanced_surface: bool | None = None,
) -> dict:
    """Run the Martin & Moyce dam break and grade it against the experiment.

    ``scheme="plic"`` uses the geometric interface reconstruction (the scheme
    the node ships); ``scheme="algebraic"`` runs the FCT algebraic transport of
    :class:`tensorfvm.vof_phase0.AlgebraicVOF` as the comparison leg (expected
    to be more diffusive but still volume conserving).

    ``well_balanced_surface`` selects the free-surface pressure treatment:
    ``None`` (default) grades whatever the solver ships, ``True`` forces the
    ghost-fluid datum, ``False`` forces the historical centred gradient
    (interface face dropped from the operator = homogeneous Neumann), which is
    the negative control of ``ROUND1-FREE-SURFACE-DATUM.md``.  The resolved
    choice is written into the leg record.
    """
    if scheme not in ("plic", "algebraic"):
        raise ValueError(f"unknown scheme {scheme!r}")
    cfg = DamBreakConfig(nx=nx, ny=ny, t_star_end=t_star_end,
                         max_steps=max_steps)
    # ``None`` grades the solver default (the shipped free-surface treatment);
    # ``False`` reproduces the pre-round-1 homogeneous-Neumann surface as the
    # negative control.  The choice is written into the record so a leg artefact
    # can never be mistaken for the other treatment.
    solver_kwargs = ({}
                     if well_balanced_surface is None
                     else {"well_balanced_surface": bool(well_balanced_surface)})
    effective_wb = (True if well_balanced_surface is None
                    else bool(well_balanced_surface))
    surface_treatment = ("shipped-default" if well_balanced_surface is None
                         else ("well-balanced" if well_balanced_surface
                               else "centred-control"))
    if scheme == "plic":
        out = dam_break_history(cfg, solver=plic_solver(cfg, **solver_kwargs))
    else:
        from ..vof_phase0.dam_break import run_dam_break
        out = run_dam_break(cfg)
        # the algebraic driver does not record a two-threshold front; reuse the
        # half-cell convention of the chain by re-deriving it here.
        out = dict(out)
        out.setdefault("front_star", None)

    t_star = np.asarray(out["t_star"], dtype=float)
    height_star = np.asarray(out["height_star"], dtype=float)
    front_star = (np.asarray(out["front_star"], dtype=float)
                  if out.get("front_star") is not None else None)
    front_ref = np.array([martin_moyce_front(float(t)) for t in t_star])
    height_ref = np.array([martin_moyce_height(float(t)) for t in t_star])

    window = (t_star > WINDOW[0]) & (t_star <= min(t_star_end, WINDOW[1]))
    if not np.any(window):
        raise RuntimeError("no sample inside the comparison window")

    def _rel(got, ref):
        return np.abs(got - ref) / np.maximum(np.abs(ref), 1e-6)

    if front_star is not None:
        lead = _rel(front_star[window, 0], front_ref[window])
        half = _rel(front_star[window, 1], front_ref[window])
        front_gate = float(np.max(half))
        front_gate_name = "half-cell"
        rows = [[float(t), float(a), float(b), float(c), float(d)]
                for t, a, b, c, d in zip(t_star[window], front_star[window, 1],
                                         front_star[window, 0],
                                         front_ref[window],
                                         height_star[window])]
    else:
        lead = np.array([np.nan])
        half = np.array([np.nan])
        front_gate = float("nan")
        front_gate_name = "unavailable (algebraic driver records no front)"
        rows = [[float(t), float("nan"), float("nan"), float(f), float(h)]
                for t, f, h in zip(t_star[window], front_ref[window],
                                   height_star[window])]

    height_err = _rel(height_star[window], height_ref[window])
    volume_drift = np.asarray(out["volume_drift"], dtype=float)
    alpha = np.asarray(out["alpha_final"], dtype=float)
    bound_violation = float(max(0.0, alpha.max() - 1.0, -alpha.min()))

    # Negative control: frozen interface under zero gravity.
    control = None
    if scheme == "plic":
        from dataclasses import replace
        ctrl_cfg = replace(cfg, gravity=0.0, t_star_end=0.5, max_steps=control_steps)
        control = static_interface_drift(ctrl_cfg, n_steps=control_steps)

    # ---- diagnostics (never gates): why D1/D2 read what they read ----------
    front_for_diag = (front_star[:, 1] if front_star is not None
                      else np.full(t_star.shape, np.nan))
    in_window = window
    diagnostics = {
        "front": decompose_dambreak_error(t_star, front_for_diag, front_ref),
        "height": decompose_dambreak_error(t_star, height_star, height_ref),
        "quantisation": quantisation_floor(
            dx=float(cfg.dx), dy=float(cfg.dy), a=float(cfg.a),
            h0=float(cfg.h0), gate=HEIGHT_ERROR_MAX,
            ref_mean=float(np.mean(front_ref[in_window])),
        ),
        "resolvable": {
            "front": resolvable_error_floor(
                t_star[in_window], front_ref[in_window],
                quantum=float(cfg.dx / cfg.a), gate=FRONT_ERROR_MAX),
            "height": resolvable_error_floor(
                t_star[in_window], height_ref[in_window],
                quantum=float(cfg.dy / cfg.h0), gate=HEIGHT_ERROR_MAX),
        },
        "note": ("diagnostic only: D1/D2 verdicts above are unchanged; this "
                 "block decomposes the misfit into time shift vs rate deficit "
                 "and reports the cell-quantisation floor of the observables"),
    }

    record = {
        "case": "dambreak",
        "scheme": scheme,
        "grid": [int(nx), int(ny)],
        "surface_treatment": surface_treatment,
        "well_balanced_surface": effective_wb,
        "dx_m": float(cfg.dx),
        "dy_m": float(cfg.dy),
        "cell_size_ratio_dy_over_a": float(cfg.dy / cfg.a),
        "t_star_end": float(t_star_end),
        "steps": int(out["steps"]),
        "elapsed_s": float(out["elapsed_s"]),
        "diverged": bool(out["diverged"]),
        "front_max_rel_error": front_gate,
        "front_error_convention": front_gate_name,
        "front_max_rel_error_leading_edge": float(np.max(lead)),
        "front_mean_rel_error": float(np.mean(half)),
        "height_max_rel_error": float(np.max(height_err)),
        "volume_drift": float(np.max(np.abs(volume_drift))),
        "bound_violation": bound_violation,
        "vof_clip_volume": float(out.get("vof_clip_volume", 0.0)),
        "front_final_star": float(front_star[-1, 1]) if front_star is not None
        else float("nan"),
        "height_final_star": float(height_star[-1]),
        "control_max_abs_change": (float(control["max_abs_change"])
                                   if control else None),
        "control_volume_drift": (float(control["volume_drift"])
                                 if control else None),
        "reference": {
            "source": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312, "
                      "aspect-2 column, digitised curve (fig. 3); t* = "
                      "t sqrt(g/a), x* = (x_f - a)/a, h* = h/h0",
            "window_t_star": list(WINDOW),
        },
        "comparison": rows,
        "diagnostics": diagnostics,
    }
    return {"record": record,
            "history": {"t_star": t_star.tolist(),
                        "front_star": (front_star.tolist()
                                       if front_star is not None else None),
                        "height_star": height_star.tolist(),
                        "front_ref": front_ref.tolist(),
                        "height_ref": height_ref.tolist(),
                        "volume_drift": volume_drift.tolist()},
            "alpha_final": alpha}