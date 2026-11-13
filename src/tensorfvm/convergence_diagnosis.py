"""Convergence diagnosis (v2) for iterative finite-volume linear solves.

The module answers three questions about a running (or finished) Krylov
iteration that a residual ledger alone cannot:

* **Where would it stop?**  The stopping rule is not re-invented here: it is a
  semantic port of OpenFOAM-10's ``SolverPerformance::checkConvergence``
  (``src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.C:60-88``),
  i.e. *stop when the final residual drops below ``tolerance`` **or** below
  ``relTol * initialResidual``* (the second test is active only for
  ``relTol > small``).  Likewise the residual normalisation follows
  ``LduMatrix<...>::solver::normFactor``
  (``.../LduMatrix/LduMatrixSolver.C:167-186``), which divides the residual
  ``L1`` norm by ``sum|A psi - sumA*avg(psi)| + sum|b - sumA*avg(psi)|``.
* **How fast is it actually converging now?**  A tail-window geometric rate
  ``rho`` estimated from the *asymptotic* part of the history, with the
  relative spread of the successive ratios as an explicit confidence gate.
* **Is it worth waiting?**  The remaining iteration budget is extrapolated
  from ``rho``; if the extrapolation exceeds the declared budget the series is
  classified as stagnating, and non-decaying series are classified as
  diverging (monotone growth) or oscillating (non-monotone growth).

Design notes / boundaries
-------------------------
* No control flow of the host solver changes: the monitor only *observes*.
  A host that wants to act on the verdict must read :meth:`Diagnosis.verdict`.
* The extrapolation is only published while the confidence gate is open
  (``ratio_spread <= ratio_spread_tol`` over the tail window).  A closed gate
  yields ``verdict='insufficient'`` and no forecast instead of a fabricated
  number — an unverifiable prediction is worse than none.
* ``rho`` is a *local* rate.  For CG/GMRES the rate improves as the iteration
  proceeds (superlinear phase), so forecast issued early is biased low; the
  harness in :mod:`tensorfvm.convergence_xval` measures that out-of-sample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: ``SolverPerformance<scalar>::small_`` (OpenFOAM-10, SolverPerformance.H).
SMALL = 1e-20

VERDICTS: Tuple[str, ...] = (
    "converged",
    "converging",
    "stagnating",
    "diverging",
    "oscillating",
    "insufficient",
)


def stabilise(value: float, small: float = SMALL) -> float:
    """OpenFOAM's ``stabilise``: clamp ``|value| < small`` onto ``small``."""
    return small if abs(value) < small else float(value)


def norm_factor(Apsi: np.ndarray, b: np.ndarray, row_sums: np.ndarray,
                psi: np.ndarray) -> float:
    """OpenFOAM-10 residual normalisation factor (``LduMatrixSolver.C:167-186``)."""
    tmp = row_sums * float(np.mean(psi))
    total = float(np.sum(np.abs(Apsi - tmp)) + np.sum(np.abs(b - tmp)))
    return stabilise(total)


def normalised_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray,
                        row_sums: np.ndarray, factor: float) -> float:
    """``gSumMag(r)/normFactor`` — the residual OpenFOAM reports and tests."""
    Apsi = A @ x
    return float(np.sum(np.abs(b - Apsi))) / factor


def openfoam_converged(final_residual: float, initial_residual: float,
                       tolerance: float, rel_tol: float) -> bool:
    """Semantic port of ``SolverPerformance::checkConvergence`` (OF-10)."""
    if final_residual < tolerance:
        return True
    return bool(rel_tol > SMALL and final_residual < rel_tol * initial_residual)


@dataclass(frozen=True)
class DiagnosisConfig:
    """Thresholds.  Defaults are the ones used by the XVAL harness."""

    tolerance: float = 1e-8
    rel_tol: float = 0.0
    max_iter: Optional[int] = None
    window: int = 5
    min_points: int = 6
    ratio_spread_tol: float = 0.20
    divergence_factor: float = 1e2
    min_decades: float = 1.0
    rate_stability_tol: float = 0.05

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.min_points < self.window + 1:
            raise ValueError("min_points must be >= window + 1")
        if not (self.ratio_spread_tol > 0.0):
            raise ValueError("ratio_spread_tol must be positive")

    def threshold(self, initial_residual: float) -> float:
        """Effective level the OF-10 disjunction is satisfied below."""
        return max(self.tolerance, self.rel_tol * initial_residual)


@dataclass(frozen=True)
class Diagnosis:
    verdict: str
    converged: bool
    iterations: int
    residual: float
    initial_residual: float
    threshold: float
    rate: Optional[float]
    ratio_spread: Optional[float]
    forecast_iterations: Optional[int]
    forecast_ready: bool
    reason: str
    forecast_optimistic: Optional[int] = None
    forecast_pessimistic: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
            "initial_residual": self.initial_residual,
            "threshold": self.threshold,
            "rate": self.rate,
            "ratio_spread": self.ratio_spread,
            "forecast_iterations": self.forecast_iterations,
            "forecast_optimistic": self.forecast_optimistic,
            "forecast_pessimistic": self.forecast_pessimistic,
            "forecast_ready": self.forecast_ready,
            "reason": self.reason,
        }


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


class ConvergenceMonitor:
    """Streaming convergence diagnosis for one linear solve.

    >>> mon = ConvergenceMonitor(DiagnosisConfig(tolerance=1e-6))
    >>> for k in range(200):
    ...     mon.record(k, 0.8 ** k)          # doctest: +SKIP
    >>> mon.diagnosis().verdict              # doctest: +SKIP
    'converged'
    """

    def __init__(self, config: Optional[DiagnosisConfig] = None, *,
                 initial_residual: Optional[float] = None,
                 iteration_offset: int = 0) -> None:
        self.config = config or DiagnosisConfig()
        self.iterations: List[int] = []
        self.residuals: List[float] = []
        self.initial_residual = initial_residual
        self.iteration_offset = int(iteration_offset)
        self.first_forecast: Optional[int] = None
        self.first_forecast_iteration: Optional[int] = None
        self.first_flag: Optional[Tuple[str, int]] = None
        self._last = Diagnosis("insufficient", False, 0, float("nan"), float("nan"),
                               self.config.tolerance, None, None, None, False,
                               "no residual recorded")

    # ------------------------------------------------------------------ record
    def record(self, iteration: int, residual: float) -> Diagnosis:
        r = float(residual)
        if not math.isfinite(r) or r < 0.0:
            raise ValueError(f"non-physical residual {residual!r}")
        self.iterations.append(int(iteration))
        self.residuals.append(r)
        if self.initial_residual is None:
            self.initial_residual = r
        self._last = self._diagnose()
        verdict, flag_it = self._last.verdict, self._last.iterations
        if self.first_flag is None and verdict in ("stagnating", "diverging", "oscillating"):
            self.first_flag = (verdict, flag_it)
        if self._last.forecast_ready and self.first_forecast is None:
            self.first_forecast = self._last.forecast_iterations
            self.first_forecast_iteration = flag_it
        return self._last

    # --------------------------------------------------------------- diagnose
    def diagnosis(self) -> Diagnosis:
        return self._last

    def _diagnose(self) -> Diagnosis:
        cfg = self.config
        r = self.residuals[-1]
        k = self.iterations[-1]
        r0 = float(self.initial_residual if self.initial_residual is not None else r)
        thr = cfg.threshold(r0)
        converged = openfoam_converged(r, r0, cfg.tolerance, cfg.rel_tol)
        rate, spread = _tail_rate(self.residuals, cfg.window)

        if converged:
            return Diagnosis("converged", True, k, r, r0, thr, rate, spread,
                             k, rate is not None, "OF-10 checkConvergence satisfied")

        if len(self.residuals) < cfg.min_points or rate is None:
            return Diagnosis("insufficient", False, k, r, r0, thr, rate, spread,
                             None, False,
                             f"need >= {cfg.min_points} residuals for a rate estimate")

        ratios = np.array([self.residuals[i + 1] / self.residuals[i]
                           for i in range(len(self.residuals) - 1)][-cfg.window:])
        if rate >= 1.0:
            verdict = "diverging" if bool(np.all(ratios >= 1.0)) else "oscillating"
            return Diagnosis(verdict, False, k, r, r0, thr, rate, spread, None,
                             False, f"tail rate {rate:.4f} >= 1 over {cfg.window} ratios")

        # Confidence gate.  Three independent conditions, all data-driven:
        #  G1 the decaying regime has been entered (>= min_decades below r0);
        #  G2 the tail ratios are tight (spread <= ratio_spread_tol);
        #  G3 the rate is *standing*: the previous window over the same history
        #     agrees with the tail window to within rate_stability_tol.  G3 is
        #     what stops an early superlinear transient from being extrapolated
        #     as if it were the asymptotic rate.
        decade_ok = (r0 > 0) and (r <= r0 * 10.0 ** (-cfg.min_decades))
        rate_prev, _ = _tail_rate(self.residuals[:-cfg.window], cfg.window)
        stable = (rate_prev is not None and rate_prev > 0.0
                  and abs(rate - rate_prev) / rate_prev <= cfg.rate_stability_tol)
        ready = (spread is not None and spread <= cfg.ratio_spread_tol
                 and decade_ok and stable)
        forecast: Optional[int] = None
        optimistic: Optional[int] = None
        pessimistic: Optional[int] = None
        if ready and r > thr:
            def steps(rho: float) -> Optional[int]:
                if not (0.0 < rho < 1.0):
                    return None
                return k + int(math.ceil(math.log(thr / r) / math.log(rho)))
            forecast = steps(rate)
            optimistic = steps(min(rate, rate_prev))
            pessimistic = steps(max(rate, rate_prev))
        elif ready:
            forecast = optimistic = pessimistic = k
        if ready and cfg.max_iter is not None and forecast is not None and forecast > cfg.max_iter:
            return Diagnosis("stagnating", False, k, r, r0, thr, rate, spread, forecast,
                             True,
                             f"forecast {forecast} exceeds budget {cfg.max_iter}",
                             optimistic, pessimistic)
        if ready:
            return Diagnosis("converging", False, k, r, r0, thr, rate, spread, forecast,
                             True,
                             f"tail rate {rate:.4f} standing to "
                             f"{cfg.rate_stability_tol:.2f} (spread {spread:.3f})",
                             optimistic, pessimistic)
        gate = ("decade" if not decade_ok else
                "stability" if not stable else "spread")
        return Diagnosis("insufficient", False, k, r, r0, thr, rate, spread, forecast,
                         False, f"confidence gate closed on {gate}")

    # ------------------------------------------------------------------ helpers
    def history(self) -> List[dict]:
        return [{"iteration": i, "residual": r}
                for i, r in zip(self.iterations, self.residuals)]

    def report(self) -> dict:
        return {"diagnosis": self.diagnosis().as_dict(),
                "config": dict(vars(self.config)),
                "first_forecast": self.first_forecast,
                "first_forecast_iteration": self.first_forecast_iteration,
                "first_flag": (None if self.first_flag is None
                               else {"verdict": self.first_flag[0],
                                     "iteration": self.first_flag[1]}),
                "history": self.history()}


def diagnose_series(iterations: Sequence[int], residuals: Sequence[float],
                    config: Optional[DiagnosisConfig] = None) -> Diagnosis:
    """Diagnose a stored residual history (batch form of the streaming monitor)."""
    if len(iterations) != len(residuals):
        raise ValueError("iterations and residuals must have equal length")
    if not iterations:
        raise ValueError("empty residual history")
    mon = ConvergenceMonitor(config)
    for k, r in zip(iterations, residuals):
        mon.record(k, r)
    return mon.diagnosis()


def load_history(path: str) -> Tuple[List[int], List[float]]:
    """Read ``[{"iteration": i, "residual": r}, ...]`` or ``[[i, r], ...]``."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get("history", raw.get("residuals", []))
    iterations, residuals = [], []
    for row in raw:
        if isinstance(row, dict):
            iterations.append(int(row["iteration"]))
            residuals.append(float(row["residual"]))
        else:
            iterations.append(int(row[0]))
            residuals.append(float(row[1]))
    return iterations, residuals


def _demo_history(rate: float, n: int = 60, r0: float = 1.0) -> Tuple[List[int], List[float]]:
    return list(range(n)), [r0 * rate ** k for k in range(n)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensorfvm-convergence-diagnosis",
        description="Convergence diagnosis (v2) for iterative FVM linear solves")
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose", help="diagnose a stored residual history")
    diag.add_argument("--history", help="JSON: [{iteration, residual}, ...]")
    diag.add_argument("--demo", default="",
                      help="measurement: diagnose a synthetic rate, e.g. 0.9")
    diag.add_argument("--tolerance", type=float, default=1e-8)
    diag.add_argument("--rel-tol", type=float, default=0.0)
    diag.add_argument("--max-iter", type=int, default=None)
    diag.add_argument("--window", type=int, default=5)
    diag.add_argument("--ratio-spread", type=float, default=0.20)
    diag.add_argument("--out", help="write the full report JSON here")

    xval = sub.add_parser("xval", help="cross-validate v2 against OpenFOAM-10 rules")
    xval.add_argument("--out", required=True, help="artifact directory")
    xval.add_argument("--max-iter", type=int, default=2000)
    xval.add_argument("--figures", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "diagnose":
        config = DiagnosisConfig(tolerance=args.tolerance, rel_tol=args.rel_tol,
                                 max_iter=args.max_iter, window=args.window,
                                 ratio_spread_tol=args.ratio_spread)
        if args.history:
            iterations, residuals = load_history(args.history)
        elif args.demo:
            iterations, residuals = _demo_history(float(args.demo))
        else:
            parser.error("diagnose needs --history or --demo")
        mon = ConvergenceMonitor(config)
        for k, r in zip(iterations, residuals):
            mon.record(k, r)
        payload = mon.report()
        text = json.dumps(payload["diagnosis"], indent=2, sort_keys=True)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        return 0 if payload["diagnosis"]["converged"] else 2

    from . import convergence_xval
    payload = convergence_xval.run_xval(out_dir=args.out, max_iter=args.max_iter,
                                        figures=args.figures)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["benchmark_pass"] else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())