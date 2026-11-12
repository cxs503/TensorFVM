"""Error-profile decomposition of the Martin & Moyce dam-break comparison.

Why this module exists
----------------------
Gate ``D1`` of :mod:`tensorfvm.twophase_gap.campaign` grades the dam-break surge
front with ``max_t |got - ref| / ref <= 15 %`` over the window
``0.3 < t* <= 4.0``.  A single maximum number, however, cannot tell *which
mechanism* is responsible for a failure: interface diffusion (a wrong front
**slope**), a startup lag (a right slope but a **shifted** curve), or a coarse
resolution are all collapsed into one scalar.  This module keeps the graded
scalar untouched and adds the missing decomposition, computed as a pure function
of the committed comparison rows, so that a round-2 fix can be aimed at the
dominant mechanism instead of at a guess.

Definitions (all dimensionless, ``t* = t sqrt(g/a)``, ``x* = (x_f - a)/a``)
----------------------------------------------------------------------------
``e(t*)``            ``|got - ref| / max(ref, 1e-6)`` -- the gate's own metric.
segment profile      ``max`` and ``mean`` of ``e`` over joined time segments,
                     reported with the time at which each maximum is attained.
developed phase      the same statistics restricted to ``t* >= 2``.
startup lag          best constant delay ``Delta`` in ``got(t*) ~ ref(t* - Delta)``
                     (grid search on ``max e``), reported next to the best
                     multiplicative time rescaling ``got(t*) ~ ref(t*/alpha)``.
                     Comparing the two residuals separates "shifted in time"
                     from "slowed down".
front speed ratio    least-squares slope of ``got`` divided by that of ``ref``
                     over a developed window.  A ratio near 1 with a large lag
                     means the propagation *rate* is right and only the start
                     is late.

Nothing here changes a gate or a tolerance; ``campaign.FRONT_ERROR_MAX`` and
``campaign.WINDOW`` are imported, not redefined, and
``tests/test_dambreak_profile.py`` asserts they are still the graded values.

Row layout of a leg's ``comparison`` block (produced by
:func:`tensorfvm.twophase_gap.dambreak_gate.run_dambreak_leg`)::

    [t*, got_half_cell_front, got_leading_edge_front, ref_front, got_height]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..vof_phase0.anchors import (
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
    martin_moyce_height,
)
from .campaign import FRONT_ERROR_MAX, HEIGHT_ERROR_MAX
from .dambreak_gate import WINDOW

__all__ = [
    "SEGMENTS",
    "DEVELOPED_MIN_T",
    "rel_error",
    "segment_profile",
    "developed_phase_stats",
    "startup_lag",
    "front_speed_ratio",
    "time_to_front_star",
    "build_profile",
    "profile_leg",
    "main",
]

#: Joined segments covering the gate's comparison window (upper bound inclusive).
SEGMENTS: tuple[tuple[float, float], ...] = (
    (0.3, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
)

#: Lower bound of the "developed phase" (post-startup) statistics.
DEVELOPED_MIN_T = 2.0

#: Dimensionless front displacement used by :func:`time_to_front_star`.
FRONT_STAR_PROBE = 1.0


def _ref_front(t: np.ndarray) -> np.ndarray:
    """Martin & Moyce surge front, vectorised (same anchor as the gate)."""
    return np.interp(t, MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X)


def _ref_height(t: np.ndarray) -> np.ndarray:
    return np.array([martin_moyce_height(float(v)) for v in t])


def rel_error(got: np.ndarray, ref: np.ndarray, floor: float = 1.0e-6) -> np.ndarray:
    """The gate's relative error ``|got - ref| / max(|ref|, floor)``."""
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return np.abs(got - ref) / np.maximum(np.abs(ref), floor)


def _rows(comparison: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    c = np.asarray(comparison, dtype=float)
    if c.ndim != 2 or c.shape[1] < 5:
        raise ValueError(
            f"comparison must be N x >=5, got shape {c.shape}")
    return {
        "t": c[:, 0],
        "front_half": c[:, 1],
        "front_lead": c[:, 2],
        "ref_front": c[:, 3],
        "height": c[:, 4],
    }


def segment_profile(
    comparison: Sequence[Sequence[float]],
    segments: Iterable[tuple[float, float]] = SEGMENTS,
) -> list[dict]:
    """Max/mean front + height error per time segment.

    Segments are half-open on the left, closed on the right, matching the gate's
    ``t* > WINDOW[0]`` window convention.  Empty segments are skipped.
    """
    d = _rows(comparison)
    out: list[dict] = []
    for lo, hi in segments:
        m = (d["t"] > lo) & (d["t"] <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        ef = rel_error(d["front_half"][m], d["ref_front"][m])
        eh = rel_error(d["height"][m], _ref_height(d["t"][m]))
        ts = d["t"][m]
        out.append({
            "t_from": float(lo),
            "t_to": float(hi),
            "n_samples": n,
            "front_max_rel_error": float(ef.max()),
            "front_max_at_t": float(ts[int(ef.argmax())]),
            "front_mean_rel_error": float(ef.mean()),
            "height_max_rel_error": float(eh.max()),
            "height_max_at_t": float(ts[int(eh.argmax())]),
            "height_mean_rel_error": float(eh.mean()),
        })
    return out


def developed_phase_stats(
    comparison: Sequence[Sequence[float]], t_min: float = DEVELOPED_MIN_T,
) -> dict:
    """Front/height statistics restricted to the developed phase ``t* >= t_min``.

    Reported as a *diagnostic*, never as a substitute for the gate: the gate
    window is fixed before the run and is not to be re-tuned after seeing
    results.
    """
    d = _rows(comparison)
    m = d["t"] >= t_min
    if not np.any(m):
        raise ValueError(f"no sample with t* >= {t_min}")
    ef = rel_error(d["front_half"][m], d["ref_front"][m])
    eh = rel_error(d["height"][m], _ref_height(d["t"][m]))
    return {
        "t_min": float(t_min),
        "n_samples": int(m.sum()),
        "front_max_rel_error": float(ef.max()),
        "front_mean_rel_error": float(ef.mean()),
        "front_max_at_t": float(d["t"][m][int(ef.argmax())]),
        "height_max_rel_error": float(eh.max()),
        "height_mean_rel_error": float(eh.mean()),
        "exceeds_gate": bool(ef.max() > FRONT_ERROR_MAX),
        "front_gate": float(FRONT_ERROR_MAX),
        "height_gate": float(HEIGHT_ERROR_MAX),
    }


def startup_lag(
    comparison: Sequence[Sequence[float]],
    *,
    t_min: float = 0.5,
    shift_max: float = 1.6,
    shift_points: int = 321,
    alpha_bounds: tuple[float, float] = (0.2, 3.0),
    alpha_points: int = 561,
) -> dict:
    """Fit ``got(t*) ~ ref(t* - Delta)`` and ``got(t*) ~ ref(t*/alpha)``.

    Both fits minimise the maximum relative error over ``t* >= t_min``.  A shift
    with a small residual and an ``alpha`` with a large one means the solution is
    *late*, not *slow*; the converse means a genuine rate deficit.
    """
    d = _rows(comparison)
    m = d["t"] >= t_min
    t = d["t"][m]
    got = d["front_half"][m]

    best_shift = None
    for delta in np.linspace(0.0, shift_max, shift_points):
        pred = _ref_front(np.maximum(t - delta, 0.0))
        e = rel_error(got, pred)
        rec = (float(delta), float(e.max()), float(e.mean()))
        if best_shift is None or rec[1] < best_shift[1]:
            best_shift = rec

    best_alpha = None
    for alpha in np.linspace(alpha_bounds[0], alpha_bounds[1], alpha_points):
        pred = _ref_front(t / alpha)
        e = rel_error(got, pred)
        rec = (float(alpha), float(e.max()), float(e.mean()))
        if best_alpha is None or rec[1] < best_alpha[1]:
            best_alpha = rec

    return {
        "t_min": float(t_min),
        "shift": {
            "best_t_star_delay": best_shift[0],
            "max_rel_error": best_shift[1],
            "mean_rel_error": best_shift[2],
        },
        "timescale": {
            "best_alpha": best_alpha[0],
            "max_rel_error": best_alpha[1],
            "mean_rel_error": best_alpha[2],
        },
        "shift_wins": bool(best_shift[1] < best_alpha[1]),
        "max_rel_error_reduction": float(
            best_alpha[1] - best_shift[1]),
    }


def front_speed_ratio(
    comparison: Sequence[Sequence[float]],
    *,
    t_lo: float = 1.0,
    t_hi: float = 4.0,
) -> dict:
    """Least-squares slope of the numeric front divided by that of the reference."""
    d = _rows(comparison)
    m = (d["t"] > t_lo) & (d["t"] <= t_hi)
    if int(m.sum()) < 3:
        raise ValueError("need >= 3 samples for a slope")
    s_num = float(np.polyfit(d["t"][m], d["front_half"][m], 1)[0])
    s_ref = float(np.polyfit(d["t"][m], d["ref_front"][m], 1)[0])
    return {
        "t_from": float(t_lo),
        "t_to": float(t_hi),
        "n_samples": int(m.sum()),
        "numeric_slope": s_num,
        "reference_slope": s_ref,
        "ratio": s_num / s_ref if s_ref else float("nan"),
    }


def time_to_front_star(
    comparison: Sequence[Sequence[float]], target: float = FRONT_STAR_PROBE,
) -> dict:
    """Dimensionless time at which each curve first reaches ``x* = target``.

    Linear interpolation between the bracketing samples; ``None`` when the curve
    never reaches the target inside the recorded window.
    """
    d = _rows(comparison)
    out: dict = {"target_front_star": float(target)}
    eps = 1.0e-12
    for name in ("front_half", "ref_front"):
        y = d[name]
        hit = np.nonzero(y >= target - eps)[0]
        if hit.size == 0:
            out[name] = None
            continue
        i = int(hit[0])
        if i == 0:
            out[name] = float(d["t"][0])
            continue
        t0, t1 = float(d["t"][i - 1]), float(d["t"][i])
        y0, y1 = float(y[i - 1]), float(y[i])
        frac = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
        out[name] = t0 + frac * (t1 - t0)
    if out.get("front_half") is not None and out.get("ref_front") is not None:
        out["delay_t_star"] = out["front_half"] - out["ref_front"]
    else:
        out["delay_t_star"] = None
    return out


def profile_leg(leg: Mapping, t_min: float = DEVELOPED_MIN_T) -> dict:
    """Full decomposition for one committed leg record."""
    comp = leg["comparison"]
    segments = segment_profile(comp)
    d = developed_phase_stats(comp, t_min=t_min)
    return {
        "case": leg.get("case", "dambreak"),
        "scheme": leg.get("scheme"),
        "grid": list(leg.get("grid", [])),
        "steps": int(leg.get("steps", 0)),
        "t_star_end": float(leg.get("t_star_end", float("nan"))),
        "gated_front_max_rel_error": float(leg["front_max_rel_error"]),
        "gated_height_max_rel_error": float(leg["height_max_rel_error"]),
        "gated_window": [float(WINDOW[0]), float(WINDOW[1])],
        "gated_front_limit": float(FRONT_ERROR_MAX),
        "segments": segments,
        "height_peak_segment_index": int(np.argmax(
            [seg["height_max_rel_error"] for seg in segments])),
        "developed": d,
        "startup": startup_lag(comp),
        "front_speed": front_speed_ratio(comp),
        "front_crossing": time_to_front_star(comp),
    }


def build_profile(legs: Mapping[str, Mapping], t_min: float = DEVELOPED_MIN_T) -> dict:
    """Decompose several legs and derive the cross-grid conclusions."""
    per_leg = {name: profile_leg(leg, t_min=t_min) for name, leg in legs.items()}
    verdicts: dict = {}

    segs = [v["segments"] for v in per_leg.values()]
    if segs:
        first = [s[0]["front_max_rel_error"] for s in segs if s]
        last = [s[-1]["front_max_rel_error"] for s in segs if s]
        verdicts["error_peaks_at_startup"] = bool(
            segs and all(
                s[0]["front_max_rel_error"] >= max(x["front_max_rel_error"] for x in s)
                for s in segs))
        verdicts["startup_to_developed_error_ratio"] = (
            float(np.mean(first) / np.mean(last)) if last and np.mean(last) else None)
        verdicts["developed_phase_still_exceeds_gate"] = bool(
            all(v["developed"]["exceeds_gate"] for v in per_leg.values()))
        verdicts["developed_phase_still_exceeds_height_gate"] = bool(
            all(v["developed"]["height_max_rel_error"] > HEIGHT_ERROR_MAX
                for v in per_leg.values()))
        verdicts["front_speed_ratio_developed"] = {
            name: v["front_speed"]["ratio"] for name, v in per_leg.items()}
        verdicts["shift_beats_timescale"] = {
            name: v["startup"]["shift_wins"] for name, v in per_leg.items()}
        verdicts["height_error_peaks_mid_transient"] = bool(
            all(v["height_peak_segment_index"] > 0 for v in per_leg.values()))
        verdicts["height_peak_segment_index"] = {
            name: v["height_peak_segment_index"] for name, v in per_leg.items()}
        verdicts["mechanism"] = _classify(per_leg)

    return {
        "anchor": "GAP-FVM-TWOPHASE",
        "kind": "dambreak_error_profile",
        "notes": (
            "Diagnostic decomposition of the committed dam-break legs. The graded "
            "gates D1/D2 (max relative error over the declared window) are "
            "unchanged and still FAIL; nothing here is a re-tuned tolerance."),
        "developed_min_t": float(t_min),
        "legs": per_leg,
        "verdicts": verdicts,
    }


def _classify(per_leg: Mapping[str, Mapping]) -> str:
    """One-line mechanism label from the per-leg fits (conservative)."""
    shift_ok = all(v["startup"]["shift_wins"] for v in per_leg.values())
    speed = [v["front_speed"]["ratio"] for v in per_leg.values()]
    speed_ok = all(0.9 <= s <= 1.25 for s in speed)
    if shift_ok and speed_ok:
        return ("delayed collapse: developed-phase front speed matches the "
                "experiment (ratio ~1.1-1.2) but the curve is shifted in time, and "
                "the column drains late through the mid-transient; interface "
                "diffusion of the front is not the dominant mechanism")
    if shift_ok:
        return ("partly startup lag: shift fits better than a time rescaling, but "
                "the developed-phase front slope is off")
    return "rate deficit: a time rescaling fits better than a pure delay"


def load_legs(artifacts: str | Path) -> dict[str, dict]:
    """Load every ``leg_dambreak_*.json`` in an artefacts directory."""
    art = Path(artifacts)
    legs: dict[str, dict] = {}
    for p in sorted(art.glob("leg_dambreak_*.json")):
        with p.open() as fh:
            legs[p.stem.replace("leg_dambreak_", "")] = json.load(fh)
    if not legs:
        raise FileNotFoundError(f"no leg_dambreak_*.json under {art}")
    return legs


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m tensorfvm.twophase_gap.dambreak_profile",
        description="Decompose the dam-break comparison error in time.")
    ap.add_argument("--artifacts", default="docs/GAP-FVM-TWOPHASE/artifacts")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <artifacts>/dambreak_profile.json)")
    ap.add_argument("--t-min", type=float, default=DEVELOPED_MIN_T)
    args = ap.parse_args(argv)

    legs = load_legs(args.artifacts)
    prof = build_profile(legs, t_min=args.t_min)
    out = Path(args.out) if args.out else Path(args.artifacts) / "dambreak_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prof, indent=1, sort_keys=True))

    for name, p in prof["legs"].items():
        print(f"{name}: gated front err {p['gated_front_max_rel_error']:.4f}, "
              f"developed (t*>={prof['developed_min_t']:.0f}) "
              f"max {p['developed']['front_max_rel_error']:.4f}, "
              f"shift {p['startup']['shift']['best_t_star_delay']:.2f} t* "
              f"(resid {p['startup']['shift']['max_rel_error']:.3f}), "
              f"speed ratio {p['front_speed']['ratio']:.3f}")
    print("mechanism:", prof["verdicts"]["mechanism"])
    print("wrote", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())