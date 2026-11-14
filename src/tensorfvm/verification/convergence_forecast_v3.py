"""Causal stopping-iteration forecast v3: curvature-gated quadratic log-residual model.

Background
----------
``docs/xval-round2/ROUND2.md`` credits the round-2 mechanism with ``0.00 %``
forecast error on six decaying cases.  ``docs/xval-round2/CAUSAL-AUDIT.md``
showed that number is an *ex post identity* (the solver had already stopped when
the forecast was requested) and that, measured causally on residual prefixes, the
round-2 mechanism is statistically indistinguishable from the v1 constant-rate
monitor: earliest-available forecast 1/6 inside 15 %, worst case 307 %.

What v3 does differently
------------------------
The audit's diagnosis was that the missing ingredient is a model of the
*acceleration* of a Krylov iteration, not of its instantaneous rate.  v3
therefore extrapolates ``log residual`` -- not ``log rate`` -- with a local
model whose curvature is the acceleration:

* **constant candidate** (v1 assumption)
  ``log r(k+m) = log r(k) + m b``, ``b`` = mean log-ratio of the last window.
* **quadratic candidate** (superlinear phase)
  ``log r(k+m) = log r(k) + m b + c [(k+m-j0)^2 - (k-j0)^2]``, ``(b, c)`` from a
  degree-2 least-squares fit of ``log r`` over a trailing window.  ``c < 0`` is
  the signature of a rate that keeps improving; ``c -> 0`` degenerates to the
  constant candidate.
* **curvature gate**: the quadratic candidate is used only if ``c < 0``, the
  slope is negative (a still-growing residual cannot certify a crossing) *and* a
  Student-t test on ``c`` from the fit residuals gives ``t_c <= -t_crit``
  (default ``t_crit = 2``, the conventional two-sigma level).  This is the
  lesson of the round-2 audit: an ungated extrapolation of acceleration
  over-predicts and made the forecast worse.  The gate is a standard
  significance test, not a threshold tuned on the case set.
* **anchored extrapolation**: predictions start from the *observed*
  ``log r(k)``; the fitted curve's offset at ``k`` is not trusted.
* **physical-rate gate**: the contraction factor over the final step before the
  crossing must lie in ``(0, 1)``, otherwise no forecast is published.
* **prediction interval**: the crossing is re-evaluated with the fitted
  parameter perturbed by its own standard error, which turns the point forecast
  into ``[lo, hi]`` at the requested confidence (default 68 % and 90 %).  The
  interval is what a scheduler can actually use when the point forecast is
  uncertain -- and on late-regime-change histories it is wide, which is honest.

Everything is prefix-only: no future residual touches the fit, the gate, or the
interval, so the numbers are exactly what an on-line diagnosis could publish.
Pure numpy/CPU, milliseconds per prefix, no solver call, no cluster.

Known limit (measured, not asserted)
------------------------------------
The residual trace is the *only* channel available (``docs/convergence-diagnosis-v2/
histories/*.json`` store ``iteration``/``residual`` only).  When GMRES enters a
terminal breakdown -- ratios approaching 1 and then collapsing within a few
iterations -- a smooth model of ``log residual`` cannot see the collapse coming.
Two of the six committed cases behave that way; v3 does not fix them and does
not pretend to (see ``docs/xval-round2/FORECAST-V3.md`` §5/§6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorfvm.convergence_diagnosis import (DiagnosisConfig,
                                             openfoam_converged)

#: trailing window (iterations) used to fit the quadratic log-residual model
QUAD_WINDOW = 10
#: window used by the constant-rate candidate (identical to the v1 monitor)
CONST_WINDOW = 5
#: two-sigma gate on the curvature of the quadratic candidate
T_CRIT = 2.0
#: Optional guard: reject the quadratic candidate when its own predicted crossing
#: lies farther than ``ratio * k`` ahead.  Tested (0.1 .. 1.0) against the
#: committed histories and **not shipped**: it removed the small-case gain at
#: lead 10 % without recovering the two long-lead failures, i.e. it traded one
#: failure for another.  ``None`` = disabled; the knob stays for re-testing.
HORIZON_RATIO: Optional[float] = None
#: confidence levels published for the interval forecast
CONFIDENCES: Tuple[float, ...] = (0.68, 0.90)
#: two-sided normal quantiles for the published confidence levels
_Z = {0.68: 1.0, 0.90: 1.6448536269514722}


@dataclass(frozen=True)
class WindowFit:
    """A fitted local model of ``log residual`` vs iteration."""

    kind: str                    # "constant" | "quadratic"
    b: float                     # slope d(log r)/dk
    c: float = 0.0               # curvature d2(log r)/dk2 (0 for constant)
    j0: float = 0.0              # abscissa centre used by the quadratic fit
    se_b: float = 0.0            # standard error of b
    se_c: float = 0.0            # standard error of c
    n_points: int = 0

    @property
    def t_c(self) -> float:
        """Student-t statistic of the curvature (0 for the constant model)."""
        if self.kind != "quadratic" or self.se_c <= 0.0:
            return 0.0
        return self.c / self.se_c

    @property
    def accelerating(self) -> bool:
        """True when the decay is speeding up at a two-sigma level."""
        return self.c < 0.0 and self.t_c <= -T_CRIT

    def as_dict(self) -> dict:
        return {"kind": self.kind, "b": self.b, "c": self.c, "j0": self.j0,
                "se_b": self.se_b, "se_c": self.se_c, "t_c": self.t_c,
                "n_points": self.n_points}


def _log_residuals(residuals: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(residuals, dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        return None
    return np.log(arr)


def fit_constant(residuals: Sequence[float],
                 window: int = CONST_WINDOW) -> Optional[WindowFit]:
    """Constant-rate candidate: mean log-ratio of the trailing window."""
    log_r = _log_residuals(residuals)
    if log_r is None:
        return None
    w = min(window, log_r.size - 1)
    if w < 1:
        return None
    diffs = np.diff(log_r[-(w + 1):])
    b = float(np.mean(diffs))
    if not math.isfinite(b) or b >= 0.0:
        return None
    se = float(np.std(diffs, ddof=1) / math.sqrt(w)) if w > 1 else 0.0
    return WindowFit("constant", b, se_b=se, n_points=w + 1)


def fit_quadratic(residuals: Sequence[float], iterations: Sequence[int],
                  window: int = QUAD_WINDOW) -> Optional[WindowFit]:
    """Quadratic candidate: degree-2 least squares of ``log r`` vs iteration."""
    log_r = _log_residuals(residuals)
    if log_r is None:
        return None
    n = min(window, log_r.size)
    if n < 5 or len(iterations) < n:
        return None
    j = np.asarray(iterations[-n:], dtype=float)
    y = log_r[-n:]
    j0 = float(j.mean())
    x = j - j0
    try:
        co = np.polyfit(x, y, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    model = np.poly1d(co)
    resid = y - model(x)
    dof = max(n - 3, 1)
    s2 = float(np.sum(resid ** 2)) / dof
    sxx = float(np.sum(x ** 4)) - float(np.sum(x ** 2)) ** 2 / n
    se_c = math.sqrt(s2 / sxx) if sxx > 0.0 else 0.0
    sxx_b = float(np.sum(x ** 2)) - float(np.sum(x)) ** 2 / n
    se_b = math.sqrt(s2 / sxx_b) if sxx_b > 0.0 else 0.0
    return WindowFit("quadratic", float(co[1]), float(co[0]), j0,
                     se_b=se_b, se_c=se_c, n_points=n)


def select_model(residuals: Sequence[float], iterations: Sequence[int],
                 window: int = QUAD_WINDOW, t_crit: float = T_CRIT,
                 long_window: Optional[int] = None,
                 t_crit_long: Optional[float] = None) -> Optional[WindowFit]:
    """Quadratic candidate when its curvature passes the gate, else constant.

    ``long_window`` adds a second, longer fit whose curvature must agree in sign
    (and, if ``t_crit_long`` is given, pass its own t-test).  A bend that only
    exists at the very tail of the prefix survives the short window but not the
    long one; requiring both to agree rejects exactly those spurious bends.
    """
    quad = fit_quadratic(residuals, iterations, window)
    # ``b >= 0`` would be a residual that is still growing: a negative curvature
    # there describes *decelerating divergence*, not convergence.  Extrapolating
    # it produces a fictitious crossing far in the future, so the sign of the
    # slope is part of the gate, not an afterthought.
    if quad is None or quad.c >= 0.0 or quad.b >= 0.0 or quad.t_c > -t_crit:
        return fit_constant(residuals)
    if long_window is not None:
        long = fit_quadratic(residuals, iterations, long_window)
        if long is None or long.c >= 0.0:
            return fit_constant(residuals)
        if t_crit_long is not None and long.t_c > -t_crit_long:
            return fit_constant(residuals)
    return quad


def _crossing(model: WindowFit, anchor: int, log_rk: float, log_target: float,
              horizon: int, c_override: Optional[float] = None) -> Optional[int]:
    """First ``m >= 1`` with predicted ``log r(anchor+m) <= log_target``."""
    if horizon < 1:
        return None
    steps = np.arange(1.0, horizon + 1.0)
    c = model.c if c_override is None else c_override
    pred = log_rk + model.b * steps
    if c:
        x = steps + anchor - model.j0
        x0 = anchor - model.j0
        pred = pred + c * (x ** 2 - x0 ** 2)
    hit = np.nonzero(pred <= log_target)[0]
    if hit.size == 0:
        return None
    return int(hit[0]) + 1


def _final_rate(model: WindowFit, anchor: int, log_rk: float, m: int,
                c_override: Optional[float] = None) -> float:
    """Predicted contraction factor over the step that reaches the crossing."""
    c = model.c if c_override is None else c_override
    def at(step: float) -> float:
        val = log_rk + model.b * step
        if c:
            x = step + anchor - model.j0
            x0 = anchor - model.j0
            val += c * (x ** 2 - x0 ** 2)
        return val
    prev = at(m - 1.0)
    return float(math.exp(at(float(m)) - prev))


def forecast(residuals: Sequence[float], iterations: Sequence[int],
             cfg: Optional[DiagnosisConfig] = None,
             window: int = QUAD_WINDOW, t_crit: float = T_CRIT,
             horizon_ratio: Optional[float] = HORIZON_RATIO,
             long_window: Optional[int] = None,
             t_crit_long: Optional[float] = None,
             confidences: Sequence[float] = CONFIDENCES) -> dict:
    """Forecast the OF-10 stopping iteration from a residual prefix.

    Returns a dict with ``forecast_iterations`` (absolute iteration index or
    ``None``), ``model`` (``constant``/``quadratic``/``converged``/``none``),
    ``rate``, ``interval`` (per confidence), ``fit`` and ``reason``.
    """
    cfg = cfg or DiagnosisConfig()
    out: dict = {"forecast_iterations": None, "model": "none", "rate": None,
                 "interval": {}, "fit": None, "horizon_gate": None, "reason": ""}
    n = len(residuals)
    if n < cfg.min_points:
        out["reason"] = f"need >= {cfg.min_points} residuals for a rate estimate"
        return out
    r0, r = float(residuals[0]), float(residuals[-1])
    k = n - 1
    if openfoam_converged(r, r0, cfg.tolerance, cfg.rel_tol):
        out.update({"forecast_iterations": int(iterations[k]), "model": "converged",
                    "reason": "OF-10 checkConvergence already satisfied"})
        return out
    log_r = _log_residuals(residuals)
    if log_r is None:
        out["reason"] = "non-positive or non-finite residual"
        return out
    log_target = math.log(cfg.threshold(r0))
    if log_r[k] <= log_target:
        out["reason"] = "residual already below the effective threshold"
        return out

    model = select_model(residuals, iterations, window, t_crit,
                         long_window, t_crit_long)
    if model is None:
        out["reason"] = "no candidate model could be fitted"
        return out
    max_iter = int(cfg.max_iter or 10_000)
    horizon = max(max_iter - k, 1)
    m = _crossing(model, k, float(log_r[k]), log_target, horizon)
    gate: dict = {"horizon_ratio": horizon_ratio, "applied": False}
    if model.kind == "quadratic" and horizon_ratio is not None:
        limit = horizon_ratio * max(k, 1)
        gate.update({"predicted_steps": m, "limit": limit})
        if m is None or m > limit:
            const = fit_constant(residuals)
            if const is not None:
                gate["applied"] = True
                model = const
                m = _crossing(model, k, float(log_r[k]), log_target, horizon)
    out["horizon_gate"] = gate
    out["fit"] = model.as_dict()
    if m is None:
        out.update({"model": model.kind,
                    "reason": f"no crossing predicted within {horizon} iterations"})
        return out
    rate = _final_rate(model, k, float(log_r[k]), m)
    if not (0.0 < rate < 1.0):
        out.update({"model": model.kind,
                    "reason": f"predicted final rate {rate:.4f} outside (0, 1)"})
        return out

    out.update({"forecast_iterations": int(iterations[k] + m),
                "model": model.kind, "rate": rate,
                "reason": f"{model.kind} crossing at +{m} iterations"})
    for conf in confidences:
        z = _Z.get(round(conf, 2))
        if z is None:
            continue
        out["interval"][f"{conf:.2f}"] = _interval(model, k, float(log_r[k]),
                                                   log_target, horizon, m, z)
    return out


def _interval(model: WindowFit, anchor: int, log_rk: float, log_target: float,
              horizon: int, m: int, z: float) -> Optional[Dict[str, int]]:
    """Crossing interval obtained by perturbing the fitted parameter by ``z*se``."""
    ends: List[int] = [m]
    if model.kind == "quadratic":
        for delta in (-z * model.se_c, z * model.se_c):
            c_eff = model.c + delta
            if c_eff >= 0.0:
                return None  # the perturbation removes the crossing: unbounded
            mm = _crossing(model, anchor, log_rk, log_target, horizon, c_eff)
            if mm is None:
                return None
            ends.append(mm)
    else:
        for delta in (-z * model.se_b, z * model.se_b):
            b_eff = model.b + delta
            if b_eff >= 0.0:
                return None
            shifted = WindowFit("constant", b_eff, se_b=model.se_b,
                                n_points=model.n_points)
            mm = _crossing(shifted, anchor, log_rk, log_target, horizon)
            if mm is None:
                return None
            ends.append(mm)
    return {"lo": int(anchor + min(ends)), "hi": int(anchor + max(ends))}


def causal_series(residuals: Sequence[float], iterations: Sequence[int],
                  cfg: Optional[DiagnosisConfig] = None, **kwargs) -> List[dict]:
    """Replay :func:`forecast` on every prefix (the causal scoring recorder)."""
    cfg = cfg or DiagnosisConfig()
    series: List[dict] = []
    for end in range(len(residuals)):
        res = forecast(residuals[: end + 1], iterations[: end + 1], cfg, **kwargs)
        series.append({
            "k": end,
            "verdict": ("converged" if res["model"] == "converged"
                        else "insufficient" if res["forecast_iterations"] is None
                        else "converging"),
            "ready": res["forecast_iterations"] is not None,
            "forecast": res["forecast_iterations"],
            "rate": res["rate"],
            "spread": None,
            "superlinear": res["model"] == "quadratic",
            "gate": res["fit"],
            "model": res["model"],
            "interval": res["interval"],
            "reason": res["reason"],
        })
    return series