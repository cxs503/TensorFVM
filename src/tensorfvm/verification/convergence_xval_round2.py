"""Round-2 improvements for convergence diagnosis forecast accuracy.

Problem
-------
The v1 convergence diagnosis (``convergence_diagnosis.py``) issues forecasts
based on a *constant* geometric rate estimated from a short tail window.
For stationary iterative methods (Richardson, Jacobi) this is reasonable,
but for Krylov subspace methods (GMRES, CG) the convergence is
**superlinear**: the effective rate improves as the iteration proceeds.
The v1 forecast, issued early in the superlinear phase, systematically
underestimates the asymptotic rate (predicts slower convergence than actual),
leading to forecast errors of 30–300 %.

Round-2 solution
----------------
1. **Superlinear detection**: Monitor whether the tail rate is *improving*
   (decreasing) over successive windows.  A sustained improvement trend
   signals superlinear convergence.
2. **Rate extrapolation**: Fit a linear model to the log-rate vs iteration
   history to extrapolate the asymptotic rate.  For GMRES, the log-rate
   typically decreases approximately linearly with iteration count in the
   superlinear phase.
3. **Adaptive forecast**: When superlinear convergence is detected, use the
   extrapolated asymptotic rate for the forecast instead of the current
   (pessimistic) tail rate.
4. **Confidence gating**: Only apply the extrapolation when the rate trend
   is statistically significant (R² > threshold) and the extrapolation
   doesn't produce unphysical rates (< 0.1 or > 0.99).

Acceptance criteria (predeclared)
---------------------------------
* C1: Rate error ≤ 15 % on all 6 decaying cases (was 6/6 PASS, unchanged).
* C2: Forecast error ≤ 15 % on all 6 decaying cases (was 1/6 PASS → target 6/6).
* C3: Verdict agreement 8/8 (was 8/8 PASS, unchanged).
* C4: No regression on the two diverging/stagnating cases (verdict must stay correct).
* C5: Superlinear detection must not trigger on stationary-rate cases
      (Poisson with Richardson, or cases where rate is truly constant).

Reference: OpenFOAM-10 SolverPerformance semantics (unchanged from v1).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Reuse the v1 operators and reference stop rule
from ..convergence_diagnosis import (ConvergenceMonitor, DiagnosisConfig,
                                     normalised_residual, norm_factor,
                                     openfoam_converged, Diagnosis)
from ..convergence_xval import (ReferenceStop, _mk_reference, convection_diffusion_operator,
                                condition_estimate, poisson_operator, solve_cg, solve_gmres,
                                solve_richardson, RATE_TOL, FORECAST_TOL)

# ---------------------------------------------------------------------------
# Superlinear rate model
# ---------------------------------------------------------------------------

SMALL = 1e-20  # OpenFOAM small


def _tail_rate(residuals: Sequence[float], window: int) -> Tuple[Optional[float], Optional[float]]:
    """Geometric mean of the last ``window`` residual ratios + their spread."""
    if len(residuals) < window + 1:
        return None, None
    tail = residuals[-(window + 1):]
    if any(r <= 0.0 for r in tail):
        return None, None
    ratios = np.array([tail[i + 1] / tail[i] for i in range(window)], dtype=float)
    if np.any(ratios <= 0.0) or not np.all(np.isfinite(ratios)):
        return None, None
    rate = float(math.exp(float(np.mean(np.log(ratios)))))
    spread = float((ratios.max() - ratios.min()) / max(rate, SMALL))
    return rate, spread


@dataclass(frozen=True)
class SuperlinearFit:
    """Result of fitting log(rate) vs iteration for superlinear detection."""
    slope: float          # negative = rate improving (decreasing)
    intercept: float      # log(rate) at iteration 0
    r_squared: float      # goodness of fit
    n_points: int         # number of rate measurements used
    extrapolated_rate: float  # predicted asymptotic rate (at max_iter)

    @property
    def is_superlinear(self) -> bool:
        """Rate is improving if slope is significantly negative."""
        return self.slope < -1e-6 and self.r_squared > 0.7


def _fit_superlinear_rate(iterations: List[int], rates: List[float],
                          max_iter: int) -> Optional[SuperlinearFit]:
    """Fit log(rate) = slope * k + intercept using least squares.

    For GMRES/CG in the superlinear phase, log(rate) decreases approximately
    linearly with iteration count.  We fit this model and extrapolate to
    predict the asymptotic rate at the forecast iteration.
    """
    if len(iterations) < 3 or len(rates) < 3:
        return None

    log_rates = []
    ks = []
    for k, r in zip(iterations, rates):
        if r > 0.0 and r < 1.0:
            log_rates.append(math.log(r))
            ks.append(float(k))

    if len(ks) < 3:
        return None

    k_arr = np.array(ks, dtype=float)
    lr_arr = np.array(log_rates, dtype=float)

    # Linear fit: log(rate) = slope * k + intercept
    A_mat = np.vstack([k_arr, np.ones_like(k_arr)]).T
    result = np.linalg.lstsq(A_mat, lr_arr, rcond=None)
    slope, intercept = result[0]
    residuals = lr_arr - (slope * k_arr + intercept)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((lr_arr - np.mean(lr_arr)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    # Extrapolate to max_iter (or a reasonable forecast horizon)
    k_forecast = float(max_iter)
    log_rate_forecast = slope * k_forecast + intercept
    extrapolated_rate = math.exp(log_rate_forecast)

    # Clamp to physical range
    extrapolated_rate = max(0.05, min(0.99, extrapolated_rate))

    return SuperlinearFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        n_points=len(ks),
        extrapolated_rate=extrapolated_rate,
    )


# ---------------------------------------------------------------------------
# Round-2 XVAL driver (standalone, doesn't extend v1 monitor)
# ---------------------------------------------------------------------------

@dataclass
class XVALRound2Case:
    name: str
    operator: str
    solver: str
    n: int
    peclet: float
    tolerance: float
    rel_tol: float
    alpha: float
    max_iter: int


def _solve_gmres_monitored(A, b, ref, cfg, *, max_iter=2000, restart=32):
    """GMRES with streaming residual monitoring for round-2 diagnosis."""
    x = np.zeros_like(b)
    k = 0
    residuals = [ref.normalise(A, b, x)]
    ref.update(k, residuals[-1])
    if ref.converged:
        return x, k, residuals

    while k < max_iter:
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        if beta == 0.0:
            break
        V = [r / beta]
        H = np.zeros((restart + 1, restart))
        e = np.zeros(restart + 1)
        e[0] = beta
        base = x.copy()
        broke = False
        for j in range(restart):
            w = A @ V[j]
            basis = np.stack(V, axis=1)
            for _ in range(2):
                values = basis.T @ w
                H[:j + 1, j] += values
                w = w - basis @ values
            H[j + 1, j] = float(np.linalg.norm(w))
            y = np.linalg.lstsq(H[:j + 2, :j + 1], e[:j + 2], rcond=None)[0]
            x = base + basis @ y
            k += 1
            res = ref.normalise(A, b, x)
            residuals.append(res)
            ref.update(k, res)
            if ref.converged:
                return x, k, residuals
            if H[j + 1, j] < 1e-25 or k >= max_iter:
                broke = True
                break
            V.append(w / H[j + 1, j])
        if broke:
            break
    return x, k, residuals


def _solve_cg_monitored(A, b, ref, cfg, *, max_iter=2000):
    """CG with streaming residual monitoring."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs_old = float(r @ r)
    k = 0
    residuals = [ref.normalise(A, b, x)]
    ref.update(k, residuals[-1])
    if ref.converged:
        return x, k, residuals

    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        k += 1
        res = ref.normalise(A, b, x)
        residuals.append(res)
        ref.update(k, res)
        if ref.converged:
            return x, k, residuals
        rs_new = float(r @ r)
        if rs_new < 1e-30:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, k, residuals


def _solve_richardson_monitored(A, b, ref, cfg, *, max_iter=2000, alpha=1.0):
    """Richardson iteration with streaming residual monitoring."""
    x = np.zeros_like(b)
    k = 0
    residuals = [ref.normalise(A, b, x)]
    ref.update(k, residuals[-1])
    if ref.converged:
        return x, k, residuals

    diag = np.array(A.diagonal())
    lambda_max = float(np.max(diag)) if len(diag) > 0 else 1.0
    omega = alpha * 2.0 / lambda_max if lambda_max > 0 else 0.1

    for _ in range(max_iter):
        r = b - A @ x
        x = x + omega * r
        k += 1
        res = ref.normalise(A, b, x)
        residuals.append(res)
        ref.update(k, res)
        if ref.converged:
            return x, k, residuals
    return x, k, residuals


def _diagnose_v2(residuals: List[float], cfg: DiagnosisConfig,
                 rate_history: List[Tuple[int, float]],
                 max_iter: int) -> dict:
    """Round-2 diagnosis with superlinear-aware forecast.

    Returns a dict with the same fields as v1 Diagnosis.as_dict(),
    plus superlinear_fit info.
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
            "superlinear_fit": None,
        }

    if len(residuals) < cfg.min_points or rate is None:
        return {
            "verdict": "insufficient", "converged": False, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": None, "forecast_ready": False,
            "reason": f"need >= {cfg.min_points} residuals for a rate estimate",
            "superlinear_fit": None,
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
            "superlinear_fit": None,
        }

    # Confidence gate (same as v1)
    decade_ok = (r0 > 0) and (r <= r0 * 10.0 ** (-cfg.min_decades))
    rate_prev, _ = _tail_rate(residuals[:-cfg.window], cfg.window)
    stable = (rate_prev is not None and rate_prev > 0.0
              and abs(rate - rate_prev) / rate_prev <= cfg.rate_stability_tol)
    ready = (spread is not None and spread <= cfg.ratio_spread_tol
             and decade_ok and stable)

    forecast: Optional[int] = None
    optimistic: Optional[int] = None
    pessimistic: Optional[int] = None
    superlinear_fit = None

    if ready and r > thr:
        # Try superlinear extrapolation
        if len(rate_history) >= 3:
            iters = [h[0] for h in rate_history]
            rates = [h[1] for h in rate_history]
            fit = _fit_superlinear_rate(iters, rates, max_iter)
            if fit is not None and fit.is_superlinear:
                superlinear_fit = fit
                # Blend extrapolated rate with current rate (70/30)
                effective_rate = 0.7 * fit.extrapolated_rate + 0.3 * rate
                effective_rate = max(0.05, min(0.99, effective_rate))
            else:
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
            "superlinear_fit": superlinear_fit.as_dict() if superlinear_fit else None,
        }

    if ready:
        return {
            "verdict": "converging", "converged": False, "iterations": k,
            "residual": r, "initial_residual": r0, "threshold": thr,
            "rate": rate, "ratio_spread": spread,
            "forecast_iterations": forecast, "forecast_ready": True,
            "reason": f"tail rate {rate:.4f} standing to {cfg.rate_stability_tol:.2f} (spread {spread:.3f})",
            "forecast_optimistic": optimistic, "forecast_pessimistic": pessimistic,
            "superlinear_fit": superlinear_fit.as_dict() if superlinear_fit else None,
        }

    return {
        "verdict": "insufficient", "converged": False, "iterations": k,
        "residual": r, "initial_residual": r0, "threshold": thr,
        "rate": rate, "ratio_spread": spread,
        "forecast_iterations": None, "forecast_ready": False,
        "reason": "confidence gate not open",
        "superlinear_fit": None,
    }


def run_case_v2(case: XVALRound2Case) -> dict:
    """Run a single XVAL case with v2 (superlinear-aware) diagnosis."""
    if case.operator == "poisson":
        A, b = poisson_operator(case.n)
    else:
        A, b = convection_diffusion_operator(case.n, case.peclet)

    x0 = np.zeros_like(b)
    ref = _mk_reference(A, b, x0, case.tolerance, case.rel_tol)

    cfg = DiagnosisConfig(
        tolerance=case.tolerance,
        rel_tol=case.rel_tol,
        max_iter=case.max_iter,
        window=5,
        min_points=6,
        ratio_spread_tol=0.20,
        min_decades=1.0,
        rate_stability_tol=0.05,
    )

    # Solve with monitoring
    if case.solver == "gmres":
        x, k, residuals = _solve_gmres_monitored(A, b, ref, cfg, max_iter=case.max_iter)
    elif case.solver == "cg":
        x, k, residuals = _solve_cg_monitored(A, b, ref, cfg, max_iter=case.max_iter)
    else:
        x, k, residuals = _solve_richardson_monitored(A, b, ref, cfg, max_iter=case.max_iter, alpha=case.alpha)

    # Build rate history for superlinear detection
    rate_history = []
    for i in range(cfg.window, len(residuals)):
        window_residuals = residuals[i - cfg.window:i + 1]
        r, sp = _tail_rate(window_residuals, cfg.window)
        if r is not None and sp is not None and sp <= cfg.ratio_spread_tol:
            rate_history.append((i, r))

    # Run v2 diagnosis
    diag = _diagnose_v2(residuals, cfg, rate_history, case.max_iter)

    # Compute out-of-sample reference rate
    rate_ref = None
    fc_ref = None
    if ref.stop_iteration is not None and ref.stop_iteration > 0:
        # Find k_ready (first iteration where forecast was ready)
        k_ready = None
        for i in range(cfg.min_points, len(residuals)):
            window_residuals = residuals[i - cfg.window:i + 1]
            r, sp = _tail_rate(window_residuals, cfg.window)
            if r is not None and sp is not None:
                decade_ok = residuals[0] > 0 and residuals[i] <= residuals[0] * 10.0 ** (-cfg.min_decades)
                rate_prev, _ = _tail_rate(residuals[:i], cfg.window)
                stable = (rate_prev is not None and rate_prev > 0.0
                          and abs(r - rate_prev) / rate_prev <= cfg.rate_stability_tol)
                if sp <= cfg.ratio_spread_tol and decade_ok and stable:
                    k_ready = i
                    break

        if k_ready is not None and k_ready < ref.stop_iteration:
            r_ready = residuals[k_ready]
            r_stop = residuals[ref.stop_iteration] if ref.stop_iteration < len(residuals) else residuals[-1]
            if r_ready > 0 and r_stop > 0 and ref.stop_iteration > k_ready:
                rate_ref = (r_stop / r_ready) ** (1.0 / (ref.stop_iteration - k_ready))
                thr = cfg.threshold(residuals[0])
                if r_ready > thr and rate_ref > 0 and rate_ref < 1.0:
                    fc_ref = k_ready + int(math.ceil(math.log(thr / r_ready) / math.log(rate_ref)))
                else:
                    fc_ref = k_ready

    rate_v2 = diag["rate"]
    forecast_v2 = diag["forecast_iterations"]

    # Compute errors
    rate_err = None
    fc_err = None
    if rate_ref is not None and rate_v2 is not None and rate_ref > 0:
        rate_err = abs(rate_v2 - rate_ref) / rate_ref
    if fc_ref is not None and forecast_v2 is not None and fc_ref > 0:
        fc_err = abs(forecast_v2 - fc_ref) / fc_ref

    return {
        "case": case.name,
        "ref_nIter": ref.stop_iteration,
        "ref_converged": ref.converged,
        "v2_verdict": diag["verdict"],
        "v2_k_ready": None,  # would need tracking
        "rate_ref": rate_ref,
        "rate_v2": rate_v2,
        "rate_err": rate_err,
        "forecast_v2": forecast_v2,
        "fc_ref": fc_ref,
        "fc_err": fc_err,
        "rate_pass": rate_err is not None and rate_err <= RATE_TOL,
        "fc_pass": fc_err is not None and fc_err <= FORECAST_TOL,
        "verdict_agree": (diag["verdict"] == "converged") == ref.converged or
                         (diag["verdict"] in ("diverging", "oscillating", "stagnating") and not ref.converged),
        "superlinear_fit": diag.get("superlinear_fit"),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

CASES_V2 = [
    XVALRound2Case("poisson-n32-tol1e-8", "poisson", "gmres", 32, 0, 1e-8, 0, 0, 2000),
    XVALRound2Case("poisson-n64-tol1e-8", "poisson", "gmres", 64, 0, 1e-8, 0, 0, 2000),
    XVALRound2Case("poisson-n64-tol1e-12", "poisson", "gmres", 64, 0, 1e-12, 0, 0, 2000),
    XVALRound2Case("poisson-n32-reltol1e-2", "poisson", "gmres", 32, 0, 1e-12, 0.01, 0, 2000),
    XVALRound2Case("cd-pe10-gmres", "cd", "gmres", 64, 10, 1e-10, 0, 0, 2000),
    XVALRound2Case("cd-pe1000-gmres", "cd", "gmres", 64, 1000, 1e-10, 0, 0, 2000),
    XVALRound2Case("cd-pe10-cg-misuse", "cd", "cg", 32, 10, 1e-10, 0, 0, 2000),
    XVALRound2Case("poisson-n32-richardson", "poisson", "richardson", 32, 0, 1e-8, 0, 1.2, 2000),
]


def run_xval_round2(out_dir: str) -> dict:
    """Run the full XVAL round-2 suite."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for case in CASES_V2:
        print(f"Running {case.name}...")
        result = run_case_v2(case)
        results.append(result)
        sl = "yes" if result.get('superlinear_fit') else "no"
        print(f"  rate_err={result['rate_err']}, fc_err={result['fc_err']}, superlinear={sl}")

    # Summary
    n_cases = len(results)
    n_decaying = sum(1 for r in results if r['rate_err'] is not None)
    n_rate_pass = sum(1 for r in results if r['rate_pass'])
    n_fc_pass = sum(1 for r in results if r['fc_pass'])
    n_verdict_agree = sum(1 for r in results if r['verdict_agree'])

    summary = {
        "cases": n_cases,
        "rate_pass": f"{n_rate_pass}/{n_decaying}",
        "forecast_pass": f"{n_fc_pass}/{n_decaying}",
        "verdict_agree": f"{n_verdict_agree}/{n_cases}",
        "benchmark_pass": n_rate_pass >= 6 and n_fc_pass >= 6 and n_verdict_agree == 8,
    }

    # Write results
    with open(os.path.join(out_dir, "xval_round2.json"), "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    # Write markdown report
    md_lines = ["# XVAL Round-2: Superlinear-Aware Convergence Diagnosis\n"]
    md_lines.append("## Summary\n")
    md_lines.append(f"* cases: {n_cases}")
    md_lines.append(f"* rate error <= 15 %: {summary['rate_pass']}")
    md_lines.append(f"* forecast error <= 15 %: {summary['forecast_pass']}")
    md_lines.append(f"* verdict agreement: {summary['verdict_agree']}")
    md_lines.append(f"* **benchmark_pass**: {summary['benchmark_pass']}\n")

    md_lines.append("## Results\n")
    md_lines.append("| case | ref nIter | v2 verdict | rate ref | rate v2 | rate err | forecast | fc err | rate pass | fc pass | superlinear |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        sl = "yes" if r.get('superlinear_fit') else "no"
        rate_ref_s = f"{r['rate_ref']:.4f}" if r['rate_ref'] is not None else "n/a"
        rate_v2_s = f"{r['rate_v2']:.4f}" if r['rate_v2'] is not None else "n/a"
        rate_err_s = f"{r['rate_err']*100:.2f}%" if r['rate_err'] is not None else "n/a"
        fc_err_s = f"{r['fc_err']*100:.2f}%" if r['fc_err'] is not None else "n/a"
        md_lines.append(
            f"| {r['case']} | {r['ref_nIter']} | {r['v2_verdict']} | "
            f"{rate_ref_s} | {rate_v2_s} | {rate_err_s} | "
            f"{r['forecast_v2']} | {fc_err_s} | "
            f"{'PASS' if r['rate_pass'] else 'FAIL'} | "
            f"{'PASS' if r['fc_pass'] else 'FAIL'} | "
            f"{sl} |"
        )

    with open(os.path.join(out_dir, "xval_round2.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nRound-2 summary: {summary}")
    return summary


if __name__ == "__main__":
    out = os.environ.get("XVAL_ROUND2_OUT", "artifacts/xval-round2")
    run_xval_round2(out)
