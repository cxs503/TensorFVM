"""Round-3: Phase-aware convergence forecast with quadratic rate extrapolation.

Motivation
----------
The causal audit of round-2 (``docs/xval-round2/CAUSAL-AUDIT.md``) found that
the linear ``log(rate)`` extrapolation does **not** improve causal forecast
accuracy over v1: at the first gate opening, both v1 and v2 achieve only 1/6
pass at the 15 % threshold, with worst-case error 307 %.

Root cause: GMRES superlinear convergence has a characteristic *two-phase*
shape — an initial "building" phase where the Krylov subspace is assembled
(rate is poor or even worsening), followed by a "superlinear" phase where the
rate improves.  A single linear fit to all observed rates cannot capture this
curvature, and extrapolating from the building phase produces wildly optimistic
forecasts.

Round-3 solution
----------------
1. **Phase detection**: Classify the convergence state into three phases:
   - ``building``: rate is increasing (worsening) or too few observations;
   - ``superlinear``: rate has been decreasing for ≥ ``min_improve_windows``
     consecutive windows, with a statistically significant trend;
   - ``plateau``: rate is stable (not clearly improving or worsening).

2. **Quadratic rate extrapolation**: In the superlinear phase, fit
   ``log(rate) = a·k² + b·k + c`` to capture the *deceleration* of rate
   improvement.  The quadratic term prevents the linear model's tendency to
   over-extrapolate the acceleration indefinitely.

3. **Phase-aware forecast**:
   - Building phase → use current tail rate (v1 conservative behavior);
   - Superlinear phase → use quadratic extrapolation with physical clamping;
   - Plateau phase → use current tail rate.

4. **Causal hold-out gate**: The quadratic model must beat the constant-rate
   model on a held-out portion of the history (same causal discipline as
   round-2's ``_walk_forward_accepts``).

Acceptance criteria
-------------------
* C1: Rate error ≤ 15 % on all 6 decaying cases (inherited, unchanged).
* C2: Causal forecast error ≤ 15 % at 25 % remaining lead: target ≥ 5/6
      (round-2 achieved 4/6).
* C3: Causal forecast error ≤ 15 % at first gate opening: target ≥ 2/6
      (round-2 achieved 1/6).
* C4: Verdict agreement 8/8 (inherited, unchanged).
* C5: No regression on non-decaying cases.

Reference: Same 8-case suite as round-2, causal audit discipline.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..convergence_diagnosis import (ConvergenceMonitor, DiagnosisConfig,
                                     openfoam_converged)
from ..convergence_xval import (ReferenceStop, _mk_reference,
                                convection_diffusion_operator, poisson_operator,
                                RATE_TOL, FORECAST_TOL)
from .convergence_xval_round2 import (SuperlinearFit, _tail_rate,
                                      _walk_forward_accepts)

# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

#: Minimum number of consecutive windows with decreasing rate to declare
#: superlinear phase.
MIN_IMPROVE_WINDOWS = 4

#: Minimum number of rate observations for phase detection.
MIN_RATE_OBS = 6


@dataclass(frozen=True)
class ConvergencePhase:
    """Detected convergence phase with supporting evidence."""
    phase: str  # "building" | "superlinear" | "plateau"
    confidence: float  # 0..1, how confident we are in the classification
    n_improving: int  # consecutive windows with decreasing rate
    rate_trend: Optional[float]  # slope of rate vs iteration (negative = improving)

    @property
    def is_superlinear(self) -> bool:
        return self.phase == "superlinear" and self.confidence > 0.5

    @property
    def is_building(self) -> bool:
        return self.phase == "building"

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "confidence": self.confidence,
            "n_improving": self.n_improving,
            "rate_trend": self.rate_trend,
        }


def _detect_phase(rate_history: List[Tuple[int, float]],
                  min_improve: int = MIN_IMPROVE_WINDOWS) -> ConvergencePhase:
    """Detect the convergence phase from the rate history.

    The phase is determined by the recent trend in the rate:
    - If the rate has been decreasing for ``min_improve`` consecutive windows,
      we are in the superlinear phase.
    - If the rate is increasing or oscillating, we are in the building phase.
    - If the rate is stable (small spread), we are in the plateau phase.
    """
    if len(rate_history) < MIN_RATE_OBS:
        return ConvergencePhase("building", 0.0, 0, None)

    rates = [r for _, r in rate_history]
    iters = [k for k, _ in rate_history]

    # Count consecutive decreasing rates at the end
    n_improving = 0
    for i in range(len(rates) - 1, 0, -1):
        if rates[i] < rates[i - 1]:
            n_improving += 1
        else:
            break

    # Fit linear trend to all rates
    k_arr = np.array(iters, dtype=float)
    r_arr = np.array(rates, dtype=float)
    A_mat = np.vstack([k_arr, np.ones_like(k_arr)]).T
    result = np.linalg.lstsq(A_mat, r_arr, rcond=None)
    slope = float(result[0][0])
    residuals_fit = r_arr - (result[0][0] * k_arr + result[0][1])
    ss_res = np.sum(residuals_fit ** 2)
    ss_tot = np.sum((r_arr - np.mean(r_arr)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    # Phase classification
    if n_improving >= min_improve and slope < -1e-6:
        # Superlinear: rate consistently decreasing with significant trend
        confidence = min(1.0, r_squared * (n_improving / min_improve))
        return ConvergencePhase("superlinear", confidence, n_improving, slope)
    elif abs(slope) < 1e-4:
        # Plateau: rate is stable
        rate_spread = (max(rates[-5:]) - min(rates[-5:])) / max(np.mean(rates[-5:]), 1e-20)
        if rate_spread < 0.05:
            return ConvergencePhase("plateau", 0.8, n_improving, slope)
        else:
            return ConvergencePhase("building", 0.3, n_improving, slope)
    else:
        # Building: rate is increasing (worsening) or oscillating
        return ConvergencePhase("building", 0.6, n_improving, slope)


# ---------------------------------------------------------------------------
# Quadratic rate extrapolation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadraticRateFit:
    """Quadratic fit to log(rate) vs iteration: log(rate) = a·k² + b·k + c."""
    a: float
    b: float
    c: float
    r_squared: float
    n_points: int
    extrapolated_rate: float  # predicted rate at max_iter

    @property
    def is_valid(self) -> bool:
        """Check if the fit produces a physical extrapolated rate."""
        return 0.05 <= self.extrapolated_rate <= 0.99

    def as_dict(self) -> dict:
        return {
            "a": self.a, "b": self.b, "c": self.c,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "extrapolated_rate": self.extrapolated_rate,
            "is_valid": self.is_valid,
        }


def _fit_quadratic_rate(iterations: List[int], rates: List[float],
                        max_iter: int) -> Optional[QuadraticRateFit]:
    """Fit log(rate) = a·k² + b·k + c using least squares.

    The quadratic term captures the *deceleration* of rate improvement:
    as the Krylov subspace fills, the rate improvement slows down and
    eventually plateaus.  A pure linear fit would over-extrapolate the
    acceleration, producing unphysically low rates.
    """
    if len(iterations) < 4:
        return None

    log_rates = []
    ks = []
    for k, r in zip(iterations, rates):
        if r > 0.0 and r < 1.0:
            log_rates.append(math.log(r))
            ks.append(float(k))

    if len(ks) < 4:
        return None

    k_arr = np.array(ks, dtype=float)
    lr_arr = np.array(log_rates, dtype=float)

    # Quadratic fit: log(rate) = a·k² + b·k + c
    A_mat = np.vstack([k_arr ** 2, k_arr, np.ones_like(k_arr)]).T
    result = np.linalg.lstsq(A_mat, lr_arr, rcond=None)
    a, b, c = result[0]
    predicted = A_mat @ result[0]
    ss_res = np.sum((lr_arr - predicted) ** 2)
    ss_tot = np.sum((lr_arr - np.mean(lr_arr)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    # Extrapolate to max_iter
    k_forecast = float(max_iter)
    log_rate_forecast = a * k_forecast ** 2 + b * k_forecast + c
    extrapolated_rate = math.exp(log_rate_forecast)

    # Clamp to physical range
    extrapolated_rate = max(0.05, min(0.99, extrapolated_rate))

    return QuadraticRateFit(
        a=float(a), b=float(b), c=float(c),
        r_squared=float(r_squared),
        n_points=len(ks),
        extrapolated_rate=extrapolated_rate,
    )


def _quadratic_holdout_accepts(residuals: Sequence[float],
                               rate_history: Sequence[Tuple[int, float]],
                               holdout: int = 5,
                               margin: float = 0.9) -> Tuple[bool, dict]:
    """Hold-out test for the quadratic model vs constant-rate model.

    Same causal discipline as round-2's ``_walk_forward_accepts``: fit on
    data up to ``k0 = n - 1 - holdout``, score on the held-out portion.
    """
    n = len(residuals)
    info: dict = {"holdout": holdout, "accepted": False, "reason": ""}
    k0 = n - 1 - holdout
    sub = [(k, r) for (k, r) in rate_history if k <= k0]
    if n < holdout + 5 or k0 < 4 or len(sub) < 4:
        info["reason"] = "history too short for quadratic hold-out test"
        return False, info

    fit = _fit_quadratic_rate([s[0] for s in sub], [s[1] for s in sub], k0)
    if fit is None:
        info["reason"] = "quadratic fit unavailable"
        return False, info

    rate_const = sub[-1][1]
    if not (0.0 < rate_const < 1.0) or residuals[k0] <= 0.0:
        info["reason"] = "non-decaying hold-out window"
        return False, info

    # Score both models on held-out iterations
    log_r0 = math.log(residuals[k0])
    err_quad: List[float] = []
    err_con: List[float] = []
    log_quad = log_con = log_r0

    for j in range(k0 + 1, n):
        # Quadratic model: log rate(j) = a·(j-1)² + b·(j-1) + c
        log_rate_j = fit.a * (j - 1) ** 2 + fit.b * (j - 1) + fit.c
        log_quad += log_rate_j
        log_con += math.log(rate_const)
        actual = math.log(residuals[j]) if residuals[j] > 0 else log_con
        err_quad.append((log_quad - actual) ** 2)
        err_con.append((log_con - actual) ** 2)

    rmse_quad = math.sqrt(float(np.mean(err_quad)))
    rmse_con = math.sqrt(float(np.mean(err_con)))
    info.update({
        "rmse_quadratic": rmse_quad,
        "rmse_constant": rmse_con,
        "ratio": rmse_quad / rmse_con if rmse_con > 0 else float("inf"),
        "k0": k0,
        "r_squared": fit.r_squared,
    })
    accepted = rmse_quad <= margin * rmse_con
    info["accepted"] = bool(accepted)
    info["reason"] = ("quadratic model wins the hold-out" if accepted
                      else "constant-rate model wins or ties the hold-out")
    return accepted, info


# ---------------------------------------------------------------------------
# Round-3 diagnosis
# ---------------------------------------------------------------------------

def _diagnose_v3(residuals: List[float], cfg: DiagnosisConfig,
                 rate_history: List[Tuple[int, float]],
                 max_iter: int, backtest: bool = True) -> dict:
    """Round-3 diagnosis with phase-aware forecast.

    Returns a dict with the same fields as v1/v2 Diagnosis, plus
    phase and quadratic fit info.
    """
    r = residuals[-1]
    k = len(residuals) - 1
    r0 = residuals[0]
    thr = cfg.threshold(r0)
    converged = openfoam_converged(r, r0, cfg.tolerance, cfg.rel_tol)
    rate, spread = _tail_rate(residuals, cfg.window)

    if converged:
        return {
            "verdict": "converged", "converged": True, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": k, "forecast_ready": rate is not None,
            "reason": "OF-10 checkConvergence satisfied",
            "phase": None, "quadratic_fit": None,
        }

    if len(residuals) < cfg.min_points or rate is None:
        return {
            "verdict": "insufficient", "converged": False, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": None, "forecast_ready": False,
            "reason": f"need >= {cfg.min_points} residuals for a rate estimate",
            "phase": None, "quadratic_fit": None,
        }

    ratios = np.array([residuals[i + 1] / residuals[i]
                       for i in range(len(residuals) - 1)][-cfg.window:])
    if rate >= 1.0:
        verdict = "diverging" if bool(np.all(ratios >= 1.0)) else "oscillating"
        return {
            "verdict": verdict, "converged": False, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": None, "forecast_ready": False,
            "reason": f"tail rate {rate:.4f} >= 1 over {cfg.window} ratios",
            "phase": None, "quadratic_fit": None,
        }

    # Confidence gate (same as v1/v2)
    decade_ok = (r0 > 0) and (r <= r0 * 10.0 ** (-cfg.min_decades))
    rate_prev, _ = _tail_rate(residuals[:-cfg.window], cfg.window)
    stable = (rate_prev is not None and rate_prev > 0.0
              and abs(rate - rate_prev) / rate_prev <= cfg.rate_stability_tol)
    ready = (spread is not None and spread <= cfg.ratio_spread_tol
             and decade_ok and stable)

    forecast: Optional[int] = None
    optimistic: Optional[int] = None
    pessimistic: Optional[int] = None
    phase_info = None
    quadratic_fit = None
    quad_gate = None

    if ready and r > thr:
        # Detect convergence phase
        if len(rate_history) >= MIN_RATE_OBS:
            phase = _detect_phase(rate_history)
            phase_info = phase.as_dict()

            if phase.is_superlinear:
                # Try quadratic extrapolation
                iters = [h[0] for h in rate_history]
                rates = [h[1] for h in rate_history]
                qfit = _fit_quadratic_rate(iters, rates, max_iter)

                accept = True
                gate_info = None
                if qfit is not None and qfit.is_valid and backtest:
                    accept, gate_info = _quadratic_holdout_accepts(
                        residuals, rate_history)
                    quad_gate = gate_info

                if qfit is not None and qfit.is_valid and accept:
                    quadratic_fit = qfit
                    # Blend: 60% quadratic, 40% current rate (more conservative than v2's 70/30)
                    effective_rate = 0.6 * qfit.extrapolated_rate + 0.4 * rate
                    effective_rate = max(0.05, min(0.99, effective_rate))
                else:
                    # Fall back to current rate
                    effective_rate = rate
            else:
                # Building or plateau phase: use current rate
                effective_rate = rate
        else:
            effective_rate = rate

        def steps(rho: float) -> Optional[int]:
            if not (0.0 < rho < 1.0):
                return None
            remaining = math.log(thr / r) / math.log(rho)
            return k + int(math.ceil(remaining))

        forecast = steps(effective_rate)

        if rate_prev is not None and rate_prev > 0.0:
            optimistic = steps(min(effective_rate, rate_prev))
            pessimistic = steps(max(effective_rate, rate_prev))
        else:
            optimistic = pessimistic = forecast

    elif ready:
        forecast = optimistic = pessimistic = k

    if ready and cfg.max_iter is not None and forecast is not None and forecast > cfg.max_iter:
        return {
            "verdict": "stagnating", "converged": False, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": forecast, "forecast_ready": True,
            "reason": f"forecast {forecast} exceeds budget {cfg.max_iter}",
            "forecast_optimistic": optimistic, "forecast_pessimistic": pessimistic,
            "phase": phase_info, "quadratic_fit": quadratic_fit.as_dict() if quadratic_fit else None,
            "quadratic_gate": quad_gate,
        }

    if ready:
        return {
            "verdict": "converging", "converged": False, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": forecast, "forecast_ready": True,
            "reason": f"tail rate {rate:.4f} standing to {cfg.rate_stability_tol:.2f} (spread {spread:.3f})",
            "forecast_optimistic": optimistic, "forecast_pessimistic": pessimistic,
            "phase": phase_info, "quadratic_fit": quadratic_fit.as_dict() if quadratic_fit else None,
            "quadratic_gate": quad_gate,
        }

    return {
        "verdict": "insufficient", "converged": False, "iterations": k,
        "residual": r, "initial_residual": r0, "threshold": thr,
        "rate": rate, "ratio_spread": spread,
        "forecast_iterations": None, "forecast_ready": False,
        "reason": "confidence gate not open",
        "phase": None, "quadratic_fit": None,
    }


# ---------------------------------------------------------------------------
# Causal audit for round-3
# ---------------------------------------------------------------------------

def v3_forecasts(iterations: Sequence[int], residuals: Sequence[float],
                 cfg: DiagnosisConfig, backtest: bool) -> List[dict]:
    """Replay the round-3 diagnosis causally (rate_history built from prefix only)."""
    out: List[dict] = []
    for k in range(len(residuals)):
        prefix = residuals[: k + 1]
        rate_history: List[Tuple[int, float]] = [
            (iterations[i], out[i]["rate"])
            for i in range(len(out))
            if out[i]["rate"] is not None
            and (out[i].get("spread") is None or out[i].get("spread", 1) <= cfg.ratio_spread_tol)
        ]
        diag = _diagnose_v3(prefix, cfg, rate_history, cfg.max_iter, backtest=backtest)
        rate, spread = _tail_rate(prefix, cfg.window)
        out.append({"k": k, "verdict": diag["verdict"],
                    "ready": bool(diag["forecast_ready"]),
                    "forecast": diag["forecast_iterations"], "rate": rate,
                    "spread": spread,
                    "phase": diag.get("phase"),
                    "quadratic": bool(diag.get("quadratic_fit")),
                    "gate": diag.get("quadratic_gate")})
    return out


def run_causal_audit_v3(histories: Dict[str, dict]) -> dict:
    """Run the causal audit for round-3 on the same 8-case suite."""
    from pathlib import Path
    from tensorfvm.convergence_diagnosis import DiagnosisConfig

    CFG_KWARGS = dict(max_iter=2000, window=5, min_points=6, ratio_spread_tol=0.20,
                      min_decades=1.0, rate_stability_tol=0.05)
    NON_DECAYING = ("cd-pe10-cg-misuse", "poisson-n32-richardson")
    LEAD_FRACTIONS = (0.5, 0.25, 0.1)

    def true_stop(residuals, tol, rel):
        r0 = residuals[0]
        for i in range(1, len(residuals)):
            if openfoam_converged(residuals[i], r0, tol, rel):
                return i
        return None

    def first_causal(series):
        for row in series:
            if row["ready"] and row["forecast"] is not None:
                return row
        return None

    def error_at_lead(series, stop, fraction):
        target = int(round(stop * (1.0 - fraction)))
        for row in series:
            if row["k"] >= target and row["ready"] and row["forecast"] is not None:
                if row["k"] >= stop:
                    return None
                return {"k": row["k"], "forecast": row["forecast"],
                        "error": abs(row["forecast"] - stop) / stop,
                        "lead": stop - row["k"]}
        return None

    rows = []
    for key, hist in histories.items():
        iterations = [h["iteration"] for h in hist["history"]]
        residuals = [h["residual"] for h in hist["history"]]
        tol, rel = hist["tolerance"], hist["rel_tol"]
        cfg = DiagnosisConfig(tolerance=tol, rel_tol=rel, **CFG_KWARGS)
        stop = true_stop(residuals, tol, rel)

        v3_series = v3_forecasts(iterations, residuals, cfg, backtest=True)

        row = {
            "case": key, "solver": hist["solver"], "n_samples": len(residuals),
            "stopping_iteration": stop, "decaying": key not in NON_DECAYING,
        }
        if stop:
            first = first_causal(v3_series)
            arm = {
                "verdict_at_end": v3_series[-1]["verdict"],
                "first": first,
                "quadratic_triggers": sum(1 for r in v3_series if r["quadratic"]),
                "gate_accepts": sum(1 for r in v3_series if r["gate"] and r["gate"].get("accepted")),
                "gate_tests": sum(1 for r in v3_series if r["gate"]),
            }
            if first is not None:
                arm["first_error"] = abs(first["forecast"] - stop) / stop
                arm["first_lead"] = stop - first["k"]
            for frac in LEAD_FRACTIONS:
                arm[f"at_lead_{frac}"] = error_at_lead(v3_series, stop, frac)
            row["arm_v3"] = arm
        else:
            row["arm_v3"] = {"verdict_at_end": v3_series[-1]["verdict"]}
        rows.append(row)

    # Summary
    decaying = [r for r in rows if r["decaying"] and r["stopping_iteration"]]
    n_first_pass = sum(1 for r in decaying
                       if r["arm_v3"].get("first_error") is not None
                       and r["arm_v3"]["first_error"] <= 0.15)
    n_lead25_pass = sum(1 for r in decaying
                        if r["arm_v3"].get("at_lead_0.25") is not None
                        and r["arm_v3"]["at_lead_0.25"]["error"] <= 0.15)
    n_lead10_pass = sum(1 for r in decaying
                        if r["arm_v3"].get("at_lead_0.1") is not None
                        and r["arm_v3"]["at_lead_0.1"]["error"] <= 0.15)

    meta = {
        "n_cases": len(rows),
        "n_decaying": len(decaying),
        "criterion": "|forecast - true stopping iteration| / stopping iteration <= 15%",
        "first_causal_forecast_within_15pct": {"v3": n_first_pass},
        "within_15pct_at_lead_25pct": {"v3": n_lead25_pass},
        "within_15pct_at_lead_10pct": {"v3": n_lead10_pass},
        "note": "Round-3: phase-aware forecast with quadratic rate extrapolation",
    }

    return {"cases": rows, "meta": meta}


# ---------------------------------------------------------------------------
# Main driver (same case suite as round-2)
# ---------------------------------------------------------------------------

CASES_V3 = [
    ("poisson-n32-tol1e-8", "poisson", "gmres", 32, 0, 1e-8, 0, 0, 2000),
    ("poisson-n64-tol1e-8", "poisson", "gmres", 64, 0, 1e-8, 0, 0, 2000),
    ("poisson-n64-tol1e-12", "poisson", "gmres", 64, 0, 1e-12, 0, 0, 2000),
    ("poisson-n32-reltol1e-2", "poisson", "gmres", 32, 0, 1e-12, 0.01, 0, 2000),
    ("cd-pe10-gmres", "cd", "gmres", 64, 10, 1e-10, 0, 0, 2000),
    ("cd-pe1000-gmres", "cd", "gmres", 64, 1000, 1e-10, 0, 0, 2000),
    ("cd-pe10-cg-misuse", "cd", "cg", 32, 10, 1e-10, 0, 0, 2000),
    ("poisson-n32-richardson", "poisson", "richardson", 32, 0, 1e-8, 0, 1.2, 2000),
]


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    HIST_DIR = ROOT / "docs" / "convergence-diagnosis-v2" / "histories"
    OUT_DIR = ROOT / "docs" / "xval-round3"

    histories = {p.stem: json.loads(p.read_text())
                 for p in sorted(HIST_DIR.glob("*.json"))}

    result = run_causal_audit_v3(histories)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "causal_audit_v3.json").write_text(json.dumps(result, indent=2))

    # Print summary
    print("Round-3 Causal Audit Summary:")
    print(json.dumps(result["meta"], indent=2))
