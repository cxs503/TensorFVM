# Adversarial Review: XVAL-FVM-M18737-ROUND2 Convergence Diagnosis Chain

> **Reviewer**: tp-qw36-1 (generalist席), mission m21131
> **Date**: 2026-01-15
> **Scope**: v1 → round-2 → v3 → v4 convergence diagnosis chain in TensorFVM
> **Independence**: This review was conducted independently of the original authors.
> The reviewer did not participate in writing the original code or reports.

---

## 1. Problem Contract & Intended Behavior

The convergence diagnosis module answers three questions at each iteration of a
linear solver (GMRES, CG, Richardson):

1. **Will it converge?** (verdict: converged / diverging / oscillating / stagnating / insufficient)
2. **At what rate?** (geometric contraction factor ρ)
3. **When will it stop?** (forecast iteration count)

**Acceptance criteria** (from feature ledger): forecast error < 15% vs reference
stopping iteration (OpenFOAM-10 `checkConvergence` semantics).

**Key constraint**: The diagnosis only has access to the residual history
(`iteration`, `residual` pairs). It does not have access to Ritz values,
operator spectra, or per-cycle convergence factors.

---

## 2. Execution Path Reconstruction

### v1 (constant-rate monitor)
- Fits a constant geometric rate from the last `window` residual ratios
- Forecasts stopping iteration by extrapolating at that constant rate
- **Assumption**: convergence rate is stationary

### round-2 (superlinear-aware)
- Detects superlinear convergence by fitting `log(rate)` vs iteration
- Extrapolates asymptotic rate and blends 70/30 with current tail rate
- **Added**: `_walk_forward_accepts` hold-out gate (D3 fix) to reject harmful extrapolation
- **Finding**: The hold-out gate prevents harm but does not create new predictive power

### v3 (curvature-gated quadratic)
- Fits `log(residual)` (not `log(rate)`) with a quadratic model
- Curvature `c < 0` with t-test gate (`t_c ≤ -2`) selects quadratic over constant
- Anchored extrapolation from observed `log r(k)`, not fitted offset
- **Finding**: Better at lead ≤ 10% (6/6), worse at lead 25% (2/6 vs v1's 4/6)

### v4 (walk-forward selection + calibrated intervals)
- Rolling-origin backtest: at prefix k, score every earlier origin j < k
- Selects model family (constant vs v3) by pooled multi-step RMS
- Measures error growth exponent `p` (found p ∈ [0.86, 1.50], not 0.5)
- Calibrated intervals using binned quantiles
- **Finding**: Recovers v3's 25% lead regression; interval coverage 71.3%/87.9% at nominal 68%/90%

---

## 3. High-Impact Failure Mode Analysis

### 3.1 Invalid Inputs

| Input | v1 | round-2 | v3 | v4 | Verdict |
|---|---|---|---|---|---|
| Empty residuals | Returns "need ≥ 6" | Same | Returns "none" | Returns "none" | ✅ Handled |
| Single residual | Same | Same | Same | Same | ✅ Handled |
| Negative residual | `openfoam_converged` returns True | Same | `_log_residuals` returns None | Same | ⚠️ Silent pass |
| NaN residual | `openfoam_converged` returns False | Same | `_log_residuals` returns None | Same | ✅ Handled |
| Inf residual | Same | Same | Same | Same | ✅ Handled |
| Zero residual | `openfoam_converged` returns True | Same | `_log_residuals` returns None | Same | ✅ Handled |
| Non-monotone (oscillating) | Verdict "oscillating" | Same | Returns "none" | Returns "none" | ✅ Handled |

**Finding**: Negative residuals pass `openfoam_converged` as converged. This is
consistent with OpenFOAM's `checkConvergence` (which also doesn't check sign),
but residuals should always be non-negative by construction (they are norms).
This is a **defensive coding gap**, not a correctness issue in practice.

### 3.2 Boundary Conditions

| Boundary | Behavior | Verdict |
|---|---|---|
| `min_points = 6` exactly | Gate opens, rate computed | ✅ Correct |
| `min_points = 5` | Gate stays closed | ✅ Correct |
| Rate exactly at tolerance (15%) | `rate_err <= RATE_TOL` → PASS | ✅ Correct (inclusive) |
| Forecast exactly at stopping iteration | `fc_err = 0` → PASS | ✅ Correct |
| Horizon = 0 in v4 | Returns empty array | ✅ Correct |
| Pool size < MIN_SAMPLES (5) | Returns None | ✅ Correct |

### 3.3 Resource Exhaustion

| Resource | Risk | Mitigation | Verdict |
|---|---|---|---|
| Memory (v4 backtest) | O(n²) per case, pool capped at 20,000 | Pool capping, bin capping at 4,000 | ✅ Mitigated |
| CPU time | O(n²) fits per case | Milliseconds on committed histories | ✅ Acceptable |
| History length | Up to 2001 points | No explicit limit, but O(n²) grows | ⚠️ No hard limit |

### 3.4 Compatibility

| Aspect | Status | Verdict |
|---|---|---|
| Python 3.11+ | Tested | ✅ |
| numpy only (no scipy for v3/v4) | Confirmed | ✅ |
| No solver call | Confirmed | ✅ |
| No cluster needed | Confirmed | ✅ |

### 3.5 Rollback / Partial Failure

| Scenario | Behavior | Verdict |
|---|---|---|
| v4 backtest fails mid-run | Returns partial results | ⚠️ No atomicity guarantee |
| Artifact write fails | No partial JSON | ✅ Atomic (single write) |
| Test artifact missing | `pytest.fail` (not skip) | ✅ Correct |

---

## 4. Test Quality Assessment

### 4.1 Tests That Assert Invariants (Good)

| Test | Invariant | Verdict |
|---|---|---|
| `test_forecast_metric_is_ex_post_identity` | Ex-post forecast = reference by construction | ✅ Strong |
| `test_ex_post_metric_flagged_as_identity` | Audit flags identity metric | ✅ Strong |
| `test_forecast_is_strictly_causal` | All forecasts use prefix-only data | ✅ Strong |
| `test_interval_brackets_point_forecast` | lo ≤ point ≤ hi | ✅ Strong |
| `test_growth_exponent_is_measured_not_assumed` | p ≠ 0.5 | ✅ Strong |
| `test_artifact_matches_rerun` | Committed artifact = fresh run | ✅ Strong |
| `test_holdout_gate_removes_the_harmful_extrapolation` | v2g ≤ v2 where both fire | ✅ Strong |

### 4.2 Tests That Merely Execute Code (Weak)

| Test | Issue | Recommendation |
|---|---|---|
| `test_superlinear_detected_on_gmres` | Asserts `n_superlinear >= 0` (always true) | Strengthen to check actual detection |
| `test_gate_is_causal` | Only checks `gates` is non-empty | Add assertion on causal ordering |

### 4.3 Missing Test Coverage

| Gap | Risk | Recommendation |
|---|---|---|
| Negative residual in `openfoam_converged` | Silent pass on invalid input | Add test for negative residual |
| v4 with very long history (>10,000 points) | O(n²) performance | Add performance test |
| v4 with all-constant residuals | Edge case for model selection | Add test |
| v3 with exactly 5 points (minimum for quadratic) | Boundary condition | Add test |
| Integration: diagnosis + actual FVM solver | End-to-end behavior | Add integration test |

---

## 5. Claims vs Evidence Binding

### 5.1 Supported Claims

| Claim | Evidence | Verdict |
|---|---|---|
| "Ex-post forecast is identity" | `test_forecast_metric_is_ex_post_identity` + audit JSON | ✅ Supported |
| "Causal forecast 1/6 at earliest gate" | `test_first_causal_forecast_is_not_accurate_at_early_prefixes` | ✅ Supported |
| "v4 interval coverage 71.3%/87.9%" | `causal_forecast_v4.json` + `test_committed_artifact_supports_the_report_claims` | ✅ Supported |
| "Growth exponent p ∈ [0.86, 1.50]" | `test_growth_exponent_is_measured_not_assumed` + audit JSON | ✅ Supported |
| "D1/D2/D3 fixes" | Code changes + test coverage | ✅ Supported |

### 5.2 Overstated Claims

| Claim | Reality | Verdict |
|---|---|---|
| ROUND2.md "forecast error 0.00%, 6/6 PASS" | Ex-post identity, not predictive capability | ⚠️ Corrected in FINAL-REPORT.md |
| "Superlinear detection improves forecast" | Causal audit shows no improvement over v1 | ⚠️ Corrected in CAUSAL-AUDIT.md |
| "v3 better than v1" | Only at lead ≤ 10%; worse at lead 25% | ⚠️ Clarified in FORECAST-V3.md |

### 5.3 Honest FAILs (Not粉饰)

| FAIL | Status | Verdict |
|---|---|---|
| Causal forecast < 15% at earliest gate (1/6) | Documented in FINAL-REPORT.md | ✅ Honest |
| v3 worse than v1 at lead 25% | Documented in FORECAST-V3.md | ✅ Honest |
| v3 interval coverage 11.8%/19.4% | Documented in FORECAST-V3.md | ✅ Honest |
| v4 interval width 85%/193% (median) | Documented in FORECAST-V4.md | ✅ Honest |

---

## 6. Counterexamples & Reproduction

### 6.1 Reproduction of Causal Audit

```bash
cd /root/tp-agent/work/m21131
PYTHONPATH=src python docs/xval-round2/causal_forecast_audit.py
PYTHONPATH=src python docs/xval-round2/causal_forecast_v4.py
PYTHONPATH=src python -m pytest tests/test_convergence_causal_audit.py \
    tests/test_convergence_forecast_v4.py -v
```

**Result**: All tests pass, audit JSON matches committed artifact.

### 6.2 Counterexample: v3 at Lead 25%

The `poisson-n32-reltol1e-2` case has a residual trace that approaches ratio ≈ 1
for the first 36 iterations, then collapses in the last 12. At lead 25% (k ≈ 36),
the quadratic fit sees a near-flat log-residual and extrapolates poorly:

- v1 error: 70.8%
- v3 error: 70.8% (same — curvature gate rejects quadratic)
- v4 error: 70.8% (same — backtest selects constant)

This is not a bug; it's a fundamental limitation of the residual-only channel.
The collapse is not predictable from the prefix.

---

## 7. Severity-Ranked Findings

### Blocking (Correctness)

| # | Finding | Severity | Evidence | Remediation |
|---|---|---|---|---|
| B1 | None found | — | — | — |

### Non-Blocking (Quality)

| # | Finding | Severity | Evidence | Remediation |
|---|---|---|---|---|
| Q1 | `test_superlinear_detected_on_gmres` asserts `>= 0` (tautology) | Low | Test code | Strengthen assertion |
| Q2 | `test_gate_is_causal` only checks non-empty | Low | Test code | Add causal ordering check |
| Q3 | No hard limit on history length in v4 | Low | O(n²) complexity | Add configurable limit |
| Q4 | Negative residual passes `openfoam_converged` | Low | Edge case test | Add sign check (defensive) |
| Q5 | No integration test with actual FVM solver | Medium | Test inventory | Add end-to-end test |

---

## 8. Checks Not Run

| Check | Reason | Status |
|---|---|---|
| 3D operator convergence diagnosis | Not in scope (2D only) | Not run |
| Non-linear (SIMPLE) inner iteration diagnosis | Not in scope (linear only) | Not run |
| Strong anisotropy operator | Not in scope | Not run |
| Performance benchmark on >10,000 point history | Not run (no such history in repo) | Not run |
| Cross-solver comparison (e.g., PETSc, Trilinos) | Not in scope | Not run |

---

## 9. Disposition

**Approve with notes.**

The convergence diagnosis chain (v1 → round-2 → v3 → v4) is well-engineered,
thoroughly tested, and honestly documented. The key findings are:

1. **No blocking correctness issues** found.
2. **Honest FAILs are properly documented** — the "<15% at all lead times"
   acceptance criterion cannot be met with residual-only data, and this is
   clearly stated in all reports.
3. **Code quality is high** — prefix-only causality is enforced, artifacts are
   reproducible, and tests assert invariants (not just execution).
4. **Minor quality improvements** identified (Q1-Q5 above) but none are blocking.

### Recommended Remediation

1. **Q1/Q2**: Strengthen weak test assertions (low effort, high value).
2. **Q5**: Add integration test with actual FVM solver (medium effort, high value).
3. **Feature ledger**: Update status to reflect honest conclusion — the "<15%"
   criterion is met only at lead ≤ 10% (6/6), not at all lead times.

---

## 10. Independence Limitations

- The reviewer did not have access to the original design discussions or
  decision logs.
- The review was conducted on the committed code and reports; no access to
  development history or intermediate drafts.
- The review assumes the residual histories in `docs/convergence-diagnosis-v2/histories/`
  are correct and representative.
- The review did not independently verify the OpenFOAM-10 `checkConvergence`
  semantics (assumed correct per the original authors' port).
