# PREDECLARATION r8 — repaired WENO7 probe (frozen before measurement)

> Round 6 (`docs/XVAL-FVM-M18737-ROUND2/R6-ORDER-AUDIT-CLOSURE.md`, ledger D22) falsified the
> WENO7 probe as a reference: NaN at nx=200/800, peak overshoot +3.9 % at nx=1600, i.e. the
> probe was diagnostic-only. Its stated next step ① was *"fix the boundary/positivity handling
> and lock it with a shock-containing convergence test"*. This round executes exactly that and
> then re-asks the attribution question with a probe that is allowed to be trusted.

## 0. Diagnosis (established before the criteria below were frozen)

Instrumented replay of the failing run (`nx=200`, CFL 0.4) shows the first inadmissible state is
**not** a reconstructed face state but a **cell average in SSP-RK stage 1** (one cell, p = −0.0054).
The face-state positivity fallback can never see such a state; the null result of round 6 is
therefore a *missing stage-level admissibility mechanism*, not a boundary-ghost defect.

## 1. Repair mechanism (one change, conservative)

`pp_stage()` limits the **flux**, not the state: with `T_ho` the WENO7 tendency and `T_lo` the
first-order LLF tendency, each stage is written as

```
U_stage(theta) = P + theta * dU_ho + (1 - theta) * dU_lo,      P admissible,
```

and `theta` is the largest value keeping every cell admissible, obtained in closed form
(`rho(theta)` affine, `q = 2 rho E - m^2` quadratic ⇒ per-cell quadratic root, minimised over
cells). Consequences that are *properties, not hopes*:

* **conservative** — `theta` rescales a flux difference, so fluxes telescope (mass is conserved to
  round-off). A state blend would not be: measured 5.5e-5 relative mass drift at nx=200 ⇒ rejected;
* **terminating** — `theta = 0` is the first-order LLF step, admissible for CFL ≤ 1/2 (we run 0.4);
  the release-by-halving loop can therefore always reach an admissible update;
* **inert on smooth data** — `theta = 1` exactly wherever no cell is at risk, so the entropy-wave
  order is not traded away (criterion S2).

## 2. Frozen criteria (thresholds fixed before the ladder was run)

| id | claim to be tested | frozen threshold |
|---|---|---|
| S1 | the repaired probe is robust | every rung ∈ {400, 800, 1600, 3200}: `step_status=complete`, final state admissible, `pp_failures=0`, `first_order_inadmissible=0`, relative mass drift ≤ 1e-12 |
| S2 | accuracy is preserved | entropy-wave order (limiter on) ≥ 5.0 on sizes 80…1280 **and** the limiter inert there (`pp_limited_stages=0`, `theta_min=1`) |
| S3 | no oscillation blow-up at matched grid | `|peak7 − peak5| / peak5 ≤ 1.0 %` for nx ≥ 800 |
| S4 | cross-order difference is discretisation error | window L1(rho7(nx) − rho5(nx)) order over 800→1600→3200 ≥ 0.5, value ≤ 1.0 % at 3200 |
| S5 | attribution | D7(nx) = L1(rho7(nx) − rho5@12800) ≤ D5(nx) = L1(rho5(nx) − rho5@12800) for all nx ≥ 800 |
| S6 | verdict discipline | certified-<1 % is **not** advanced unless S1–S5 all pass *and* eps_ref7(3200) < 0.5 %; otherwise it stays `False` and the round is `WITHHELD` |

Window x ∈ [0.4, 1.0] and the conservative restriction convention are inherited unchanged from
rounds 2–7 (no re-tuning of the comparison instrument in the same round as the repair).

## 3. Declared limitation (stated up front, not buried)

WENO7 shares SSP-RK3 and the Rusanov flux with the WENO5 reference. It probes **reconstruction
order**, not code independence; cross-*code* independence remains the finite-volume
MUSCL/HLLC (TensorFVM) vs finite-difference WENO5 comparison. A PASS here may not be reported as
"independently verified".

## 4. Negative control

The pre-repair code path (`positivity_limiter=False`) is re-run on the rungs round 6 saw fail, so
that "the repair is what fixed it" is an observation rather than an assertion.
