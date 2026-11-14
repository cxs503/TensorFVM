"""Causal stopping-iteration forecast v4: walk-forward selection + calibrated intervals.

Why v4 exists
-------------
``docs/xval-round2/FORECAST-V3.md`` closed with two open debts:

1. the *point* forecast is only inside 15 % at lead <= 10 % (6/6, worst 6.2 %),
   and at lead 25 % the curvature-gated model v3 is **worse** than the v1
   constant-rate monitor (2/6 vs 4/5);
2. the published *prediction interval* covers 11.8 % / 19.4 % of the time at
   nominal 68 % / 90 %.  The interval only perturbs the *fitted parameters* by
   their own standard errors, so it measures parameter uncertainty and ignores
   model error -- which is the dominant error source here.

v4 addresses (2) directly and (1) partially, with one mechanism: **rolling-origin
(walk-forward) self-backtesting** of the forecast model, using only residuals
already observed at the time the forecast is issued.

Mechanism
---------
At prefix ``k`` the diagnosis owns ``residuals[0..k]``.  Every earlier origin
``j < k`` can be scored without peeking: fit the model on ``residuals[0..j]``
(exactly the fit that prefix ``j`` would have produced), extrapolate the
``log residual`` path forward, and compare with ``residuals[j+1..k]`` -- all of
which are *observed*.  This yields, per model family and per horizon ``h``, a
sample of realised extrapolation errors ``e(j, h) = pred - actual``.

Three products of that same error sample:

* **selection**: pick the family (v1 constant vs v3 curvature-gated) whose pooled
  multi-step error is smaller *for this solve*, instead of shipping one family
  for all cases;
* **scale law**: pool ``|e| / sqrt(h)`` and report its empirical quantile.  A
  random-walk scaling (error ~ sqrt(horizon)) is used, so the band also exists
  beyond the longest backtested horizon;
* **interval**: the crossing is recomputed with the predicted path shifted by
  ``±q_conf(h)``.  ``pred + q`` is the "residual stays higher than predicted"
  case and gives the late crossing ``hi``; ``pred - q`` gives ``lo``.  The
  published interval is the union of that band with the v3 parameter interval.

Cost is one fit per origin plus ``O(k)`` arithmetic per iteration: total
``O(n^2)`` per case, milliseconds-to-seconds on the committed histories, no
solver call, no cluster.

Honest scope
------------
The interval is only as good as the stationarity assumption of the error
distribution *within* one solve.  For the two cases whose prefix is a plateau
followed by a terminal breakdown (``poisson-n32-reltol1e-2``,
``cd-pe1000-gmres``) the backtest never observes a breakdown, so the band is
wide -- that is the honest statement, and the measured coverage/width trade-off
is reported in ``docs/xval-round2/FORECAST-V4.md`` rather than hidden.

Everything here is prefix-only: no future residual enters a fit, the selection,
the scale law or the interval.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorfvm.convergence_diagnosis import DiagnosisConfig, openfoam_converged
from tensorfvm.verification.convergence_forecast_v3 import (CONST_WINDOW,
                                                           QUAD_WINDOW, T_CRIT,
                                                           WindowFit,
                                                           fit_constant,
                                                           select_model)

#: model families compared by walk-forward selection
FAMILIES: Tuple[str, str] = ("constant", "v3")
#: confidence levels published as intervals
CONFIDENCES: Tuple[float, ...] = (0.68, 0.90)
#: minimum number of backtest origins before selection/calibration is trusted
MIN_ORIGINS = 8
#: floor on the number of error samples per horizon before a raw quantile is used
MIN_SAMPLES = 5


# ---------------------------------------------------------------------------
# fitted-path arithmetic (identical to the v3 extrapolation, kept explicit so
# the backtest can evaluate the path at any horizon)
# ---------------------------------------------------------------------------

def log_path(fit: WindowFit, anchor: int, log_rk: float, horizon: int) -> np.ndarray:
    """Predicted ``log residual`` at ``anchor+1 .. anchor+horizon``."""
    if horizon < 1:
        return np.empty(0)
    steps = np.arange(1.0, horizon + 1.0)
    pred = log_rk + fit.b * steps
    if fit.kind == "quadratic" and fit.c:
        x = steps + anchor - fit.j0
        x0 = anchor - fit.j0
        pred = pred + fit.c * (x ** 2 - x0 ** 2)
    return pred


def _first_crossing(pred: np.ndarray, log_target: float) -> Optional[int]:
    hit = np.nonzero(pred <= log_target)[0]
    if hit.size == 0:
        return None
    return int(hit[0]) + 1


def fit_family(family: str, residuals: Sequence[float],
               iterations: Sequence[int], window: int = QUAD_WINDOW,
               t_crit: float = T_CRIT) -> Optional[WindowFit]:
    """Fit one model family on a prefix (causal: prefix only)."""
    if family == "constant":
        return fit_constant(residuals, window=CONST_WINDOW)
    return select_model(residuals, iterations, window=window, t_crit=t_crit)


# ---------------------------------------------------------------------------
# walk-forward backtest accumulators
# ---------------------------------------------------------------------------

class _Errors:
    """Per-family walk-forward extrapolation errors (causal, incremental).

    The confidence band needs the *law* by which the extrapolation error grows
    with the horizon.  Asserting ``sqrt(h)`` (random walk) would be convenient
    but wrong here: on the committed histories the measured per-horizon RMS grows
    almost *linearly* in ``h`` (a bias, not a random walk: an extrapolation with a
    slightly wrong rate drifts linearly).  The exponent is therefore **measured**
    per solve -- a log-log least-squares fit of the per-horizon RMS -- and used to
    normalise the sample pool from which the confidence quantile is read.

    Everything is accumulated in ``O(1)`` per new sample apart from the capped
    pool (``2 * CAP`` entries, oldest half dropped).
    """

    #: cap on the pooled sample size (keeps the per-prefix quantile cheap)
    CAP = 20_000
    #: admissible range of the measured growth exponent
    P_RANGE = (0.25, 1.5)
    #: fallback exponent while the law cannot be measured yet
    P_DEFAULT = 0.5
    #: cap on a single horizon bucket
    BIN_CAP = 4_000
    #: minimum bucket size before its own quantile is used
    BIN_MIN = 60

    @staticmethod
    def bin_of(horizon: int) -> int:
        """Log2 bucket index of a horizon (bucket ``b`` covers ``2^(b-1) < h <= 2^b``)."""
        return int(math.ceil(math.log2(max(horizon, 1))))

    def __init__(self) -> None:
        self.sum_sq: Dict[str, float] = {f: 0.0 for f in FAMILIES}
        self.count: Dict[str, int] = {f: 0 for f in FAMILIES}
        self.pool_h: Dict[str, List[int]] = {f: [] for f in FAMILIES}
        self.pool_e: Dict[str, List[float]] = {f: [] for f in FAMILIES}
        #: horizon-bucketed pools (log2 bins): the error distribution is *not*
        #: horizon-independent, so the quantile is read per bucket when the bucket
        #: has enough samples and falls back to the pooled value otherwise.
        self.bin_h: Dict[str, Dict[int, List[int]]] = {f: {} for f in FAMILIES}
        self.bin_e: Dict[str, Dict[int, List[float]]] = {f: {} for f in FAMILIES}
        #: per-horizon running moments (both the exponent fit and the check figure)
        self.h_sum_sq: Dict[str, Dict[int, float]] = {f: {} for f in FAMILIES}
        self.h_count: Dict[str, Dict[int, int]] = {f: {} for f in FAMILIES}

    # -- accumulation -------------------------------------------------------
    def add(self, family: str, horizon: int, error: float) -> None:
        h = max(int(horizon), 1)
        e = abs(float(error))
        if not math.isfinite(e):
            return
        self.sum_sq[family] += e * e
        self.count[family] += 1
        self.h_sum_sq[family][h] = self.h_sum_sq[family].get(h, 0.0) + e * e
        self.h_count[family][h] = self.h_count[family].get(h, 0) + 1
        ph, pe = self.pool_h[family], self.pool_e[family]
        ph.append(h)
        pe.append(e)
        if len(ph) > 2 * self.CAP:
            del ph[: self.CAP]
            del pe[: self.CAP]
        b = self.bin_of(h)
        bh = self.bin_h[family].setdefault(b, [])
        be = self.bin_e[family].setdefault(b, [])
        bh.append(h)
        be.append(e)
        if len(bh) > 2 * self.BIN_CAP:
            del bh[: self.BIN_CAP]
            del be[: self.BIN_CAP]

    def n_origins(self, family: str) -> int:
        return self.count[family]

    def pooled_rmse(self, family: str) -> Optional[float]:
        n = self.count[family]
        if n < MIN_SAMPLES:
            return None
        return math.sqrt(self.sum_sq[family] / n)

    # -- measured scale law -------------------------------------------------
    def growth_profile(self, family: str) -> Dict[int, dict]:
        """Raw per-horizon RMS of the error (no scale law assumed)."""
        prof: Dict[int, dict] = {}
        for h in sorted(self.h_count[family]):
            n = self.h_count[family][h]
            if n < MIN_SAMPLES:
                continue
            prof[h] = {"n": n, "rms": math.sqrt(self.h_sum_sq[family][h] / n),
                       "sqrt_h": math.sqrt(float(h))}
        return prof

    def exponent(self, family: str) -> float:
        """Measured ``p`` in ``RMS(h) ~ h**p`` (log-log least squares)."""
        prof = self.growth_profile(family)
        if len(prof) < 3:
            return self.P_DEFAULT
        hs = np.asarray(sorted(prof), dtype=float)
        rms = np.asarray([prof[int(h)]["rms"] for h in hs], dtype=float)
        keep = rms > 0.0
        if keep.sum() < 3:
            return self.P_DEFAULT
        slope = float(np.polyfit(np.log(hs[keep]), np.log(rms[keep]), 1)[0])
        lo, hi = self.P_RANGE
        return min(max(slope, lo), hi)

    def unit_quantile(self, family: str, conf: float,
                      exponent: Optional[float] = None) -> Optional[float]:
        """Quantile of ``|e| / h**p`` over the sample pool."""
        hs = self.pool_h[family]
        if len(hs) < MIN_SAMPLES:
            return None
        p = self.exponent(family) if exponent is None else exponent
        unit = np.asarray(self.pool_e[family], dtype=float) / np.power(
            np.asarray(hs, dtype=float), p)
        return float(np.quantile(unit, conf))

    def binned_quantile(self, family: str, conf: float, horizon: int) -> Optional[float]:
        """Quantile of ``|e|`` at this horizon, from its bucket when it is big enough."""
        p = self.exponent(family)
        b = self.bin_of(horizon)
        be = self.bin_e[family].get(b)
        if be is not None and len(be) >= self.BIN_MIN:
            return float(np.quantile(np.asarray(be), conf))
        unit = self.unit_quantile(family, conf, p)
        if unit is None:
            return None
        return unit * float(max(horizon, 1) ** p)

    def horizon_quantile(self, family: str, conf: float,
                         horizons: np.ndarray) -> Optional[np.ndarray]:
        """Error quantile at each horizon (bucket-wise, pooled scale law as fallback)."""
        out = np.empty(horizons.size, dtype=float)
        cache: Dict[int, Optional[float]] = {}
        for i, h in enumerate(horizons):
            hh = int(h)
            b = self.bin_of(hh)
            if b not in cache:
                cache[b] = self.binned_quantile(family, conf, hh)
            v = cache[b]
            if v is None:
                return None
            out[i] = v
        return out


# ---------------------------------------------------------------------------
# per-prefix forecast
# ---------------------------------------------------------------------------

def _interval_from_path(model: WindowFit, family: str, anchor: int,
                        log_rk: float, log_target: float, horizon: int,
                        errors: _Errors, confidences: Sequence[float]) -> dict:
    """Crossing intervals whose band is the walk-forward extrapolation error."""
    pred = log_path(model, anchor, log_rk, horizon)
    if pred.size == 0:
        return {}
    hs = np.arange(1, horizon + 1, dtype=float)
    out: dict = {}
    for conf in confidences:
        q = errors.horizon_quantile(family, conf, hs)
        if q is None:
            continue
        lo_hit = np.nonzero(pred - q <= log_target)[0]
        hi_hit = np.nonzero(pred + q <= log_target)[0]
        if lo_hit.size == 0:
            continue
        lo = int(lo_hit[0]) + 1
        if hi_hit.size:
            hi, unbounded = int(hi_hit[0]) + 1, False
        else:
            hi, unbounded = int(horizon), True
        out[f"{conf:.2f}"] = {"lo": int(anchor + lo), "hi": int(anchor + hi),
                              "unbounded_hi": unbounded}
    return out


def forecast_from_prefix(residuals: Sequence[float], iterations: Sequence[int],
                         errors: _Errors, cfg: Optional[DiagnosisConfig] = None,
                         window: int = QUAD_WINDOW, t_crit: float = T_CRIT,
                         confidences: Sequence[float] = CONFIDENCES) -> dict:
    """v4 forecast at the last available prefix (uses ``errors`` only)."""
    cfg = cfg or DiagnosisConfig()
    out: dict = {"forecast_iterations": None, "model": "none", "rate": None,
                 "interval": {}, "interval_v3style": {}, "selection": None,
                 "reason": ""}
    n = len(residuals)
    if n < cfg.min_points:
        out["reason"] = f"need >= {cfg.min_points} residuals"
        return out
    r0, r = float(residuals[0]), float(residuals[-1])
    k = n - 1
    if openfoam_converged(r, r0, cfg.tolerance, cfg.rel_tol):
        out.update({"forecast_iterations": int(iterations[k]),
                    "model": "converged",
                    "reason": "OF-10 checkConvergence already satisfied"})
        return out
    if r <= 0.0 or not math.isfinite(r):
        out["reason"] = "non-positive or non-finite residual"
        return out
    log_rk = math.log(r)
    log_target = math.log(cfg.threshold(r0))
    if log_rk <= log_target:
        out["reason"] = "residual already below the effective threshold"
        return out

    # ---- walk-forward model selection (causal) -----------------------------
    score = {f: errors.pooled_rmse(f) for f in FAMILIES}
    ready = {f: errors.n_origins(f) >= MIN_ORIGINS and score[f] is not None
             for f in FAMILIES}
    n_origins = max(errors.n_origins(f) for f in FAMILIES)
    if all(ready.values()):
        family = min(FAMILIES, key=lambda f: score[f])          # type: ignore[arg-type]
        basis = "walk_forward_backtest"
    elif any(ready.values()):
        family = next(f for f in FAMILIES if ready[f])
        basis = "partial_backtest"
    else:
        family = "v3"
        basis = "insufficient_backtest (default)"
    out["selection"] = {"family": family, "basis": basis, "n_origins": n_origins,
                        "rmse": {f: score[f] for f in FAMILIES}}

    model = fit_family(family, residuals, iterations, window, t_crit)
    if model is None:
        out["reason"] = "no candidate model could be fitted"
        return out
    horizon = max(int(cfg.max_iter or 10_000) - k, 1)
    pred = log_path(model, k, log_rk, horizon)
    m = _first_crossing(pred, log_target)
    if m is None:
        out.update({"model": model.kind,
                    "reason": f"no crossing within {horizon} iterations"})
        return out
    rate = float(math.exp(float(pred[m - 1] - (pred[m - 2] if m >= 2 else log_rk))))
    if not (0.0 < rate < 1.0):
        out.update({"model": model.kind,
                    "reason": f"final rate {rate:.4f} outside (0, 1)"})
        return out
    out.update({"forecast_iterations": int(iterations[k] + m),
                "model": model.kind, "rate": rate,
                "reason": f"{model.kind} crossing at +{m}"})
    out["interval"] = _interval_from_path(model, family, k, log_rk, log_target,
                                          horizon, errors, confidences)
    out["interval_v3style"] = _parameter_interval(model, k, log_rk, log_target,
                                                 horizon, m, confidences)
    return out


def _parameter_interval(model: WindowFit, anchor: int, log_rk: float,
                        log_target: float, horizon: int, m: int,
                        confidences: Sequence[float]) -> dict:
    """v3's parameter-only interval, recomputed here for a like-for-like arm."""
    z_of = {0.68: 1.0, 0.90: 1.6448536269514722}
    out: dict = {}
    for conf in confidences:
        z = z_of.get(round(conf, 2))
        if z is None:
            continue
        ends = [m]
        ok = True
        if model.kind == "quadratic" and model.se_c > 0.0:
            for delta in (-z * model.se_c, z * model.se_c):
                c_eff = model.c + delta
                if c_eff >= 0.0:
                    ok = False
                    break
                mm = _first_crossing(
                    log_path(WindowFit("quadratic", model.b, c_eff, model.j0),
                             anchor, log_rk, horizon), log_target)
                if mm is None:
                    ok = False
                    break
                ends.append(mm)
        elif model.kind == "constant":
            for delta in (-z * model.se_b, z * model.se_b):
                b_eff = model.b + delta
                if b_eff >= 0.0:
                    ok = False
                    break
                mm = _first_crossing(
                    log_path(WindowFit("constant", b_eff), anchor, log_rk, horizon),
                    log_target)
                if mm is None:
                    ok = False
                    break
                ends.append(mm)
        if ok:
            out[f"{conf:.2f}"] = {"lo": int(anchor + min(ends)),
                                  "hi": int(anchor + max(ends))}
    return out


# ---------------------------------------------------------------------------
# causal replay of the whole walk
# ---------------------------------------------------------------------------

def causal_series(residuals: Sequence[float], iterations: Sequence[int],
                  cfg: Optional[DiagnosisConfig] = None,
                  window: int = QUAD_WINDOW, t_crit: float = T_CRIT,
                  confidences: Sequence[float] = CONFIDENCES,
                  errors_out: Optional[dict] = None) -> List[dict]:
    """Replay v4 over every prefix, accumulating the backtest as it goes.

    For prefix ``k`` only errors of origins ``j`` with ``j + h <= k`` exist in
    the accumulators, i.e. only *observed* residuals are used: the replay is
    exactly what an online diagnosis could have published at that iteration.
    """
    cfg = cfg or DiagnosisConfig()
    res = np.asarray(residuals, dtype=float)
    its = np.asarray(iterations, dtype=int)
    n = res.size
    log_r = np.log(np.where(res > 0.0, res, np.nan))
    errors = _Errors()
    # fits are needed for every origin: compute lazily and cache
    fits: Dict[str, List[Optional[WindowFit]]] = {f: [None] * n for f in FAMILIES}
    paths: Dict[str, List[Optional[np.ndarray]]] = {f: [None] * n for f in FAMILIES}
    series: List[dict] = []
    for k in range(n):
        if k > 0 and np.isfinite(log_r[k]):
            j = k - 1                       # new origin becomes scorable at k
            if j >= cfg.min_points - 1:
                for family in FAMILIES:
                    fit = fits[family][j]
                    if fit is None:
                        fit = fit_family(family, res[: j + 1], its[: j + 1],
                                         window, t_crit)
                        fits[family][j] = fit
                        paths[family][j] = (None if fit is None else
                                            log_path(fit, j, float(log_r[j]),
                                                     n - 1 - j))
                    path = paths[family][j]
                    if path is not None:
                        errors.add(family, 1, float(path[0] - log_r[j + 1]))
                # older origins gain one horizon each
                for jj in range(cfg.min_points - 1, j):
                    h = k - jj
                    for family in FAMILIES:
                        path = paths[family][jj]
                        if path is not None and path.size >= h:
                            errors.add(family, h, float(path[h - 1] - log_r[k]))
        rec = forecast_from_prefix(res[: k + 1], its[: k + 1], errors, cfg,
                                   window, t_crit, confidences)
        series.append({
            "k": k,
            "verdict": ("converged" if rec["model"] == "converged"
                        else "insufficient" if rec["forecast_iterations"] is None
                        else "converging"),
            "ready": rec["forecast_iterations"] is not None,
            "forecast": rec["forecast_iterations"],
            "rate": rec["rate"],
            "spread": None,
            "superlinear": rec["model"] == "quadratic",
            "gate": None,
            "model": rec["model"],
            "interval": rec["interval"],
            "interval_v3style": rec["interval_v3style"],
            "selection": rec["selection"],
            "reason": rec["reason"],
        })
    if errors_out is not None:
        errors_out["errors"] = errors
    return series


def coverage(series: Sequence[dict], stop: int, conf: str,
             field: str = "interval") -> Optional[bool]:
    """Does the published interval at the *last ready* prefix contain ``stop``?"""
    for row in reversed(series):
        if not row["ready"]:
            continue
        iv = (row.get(field) or {}).get(conf)
        if not iv:
            return None
        return bool(iv["lo"] <= stop <= iv["hi"])
    return None