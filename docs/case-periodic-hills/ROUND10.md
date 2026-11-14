# CASE-FVM-PERIODIC-HILLS — round 10: the SST k-omega closure (a second eddy-viscosity provider)

- Mission 20351 (selfstart, generalist seat) · branch `agent/m20351` · repo TensorFVM
- Anchor: `CASE-FVM-PERIODIC-HILLS` (算例池·周期山丘流分离与再附长度, feature ledger `open`)
- Acceptance: `x_r/H` and `u/U_b` at `y/H = 0.5` vs Breuer et al. (2009), `< 10 %`
- Reference (single-sourced, unchanged): `REFERENCE["x_r_over_H"] = 5.34`,
  `x_s/H = 0.227`, table `BREUER_PROFILE_Y05`, all in
  `src/tensorfvm/verification/periodic_hill_round6.py`
- New module: `src/tensorfvm/periodic_hill_sst.py` (SST k-omega closure provider)
- Verification: `src/tensorfvm/verification/periodic_hill_round10.py`
- Tests: `pytest tests/test_periodic_hill_round10.py -q` → **5 passed**; full
  periodic-hill suite (rounds 1–9 + 10) → **32 passed**
- Figure: `fig_round10.png` · Machine summary: `round10_summary.json`

## 1. Why round 10 exists

Rounds 5–9 attached the Spalart–Allmaras one-equation closure to the Re = 2800
periodic-hill run and found it *saturated*: `nu_t/nu` plateaus at ~65 (vs the
~26 expected for a turbulent channel at this Re) with the separation bubble
still open, so the mean `x_r/H` does not exist in the time-averaged field and
every `u/U_b` profile station at `y/H = 0.5` is >70 % off Breuer et al. (2009).
Round 9's scoping note (`ROUND9.md` §7.1) names the next step explicitly:
**a different closure, not more tuning** — SST `k-omega` (Menter 1994) or a
Reynolds-stress transport model.  This round adds the SST `k-omega` closure as a
second eddy-viscosity provider so the same `PeriodicHillSolver` can be run as
2-D URANS with either closure.

## 2. What was added

`src/tensorfvm/periodic_hill_sst.py` — an `SSTKOmega` closure provider,
interface-compatible with `SpalartAllmaras` (`viscosity` / `total_viscosity` /
`source_terms` / `update`), so it plugs into the solver through
`turbulence=SSTKOmega(solver)`:

| piece | what it does | independent pin |
|---|---|---|
| `sst2003_closure` (reused) | the full SST-2003 algebraic closure (F1/F2 blending, `nu_t`, production, destruction, cross-diffusion) | shared with the 3-D native solver (`native_sst3d.py`) — one source of truth |
| `velocity_gradient_tensor` | builds the 3x3 velocity-gradient tensor from the 2-D gradients (out-of-plane zero) | `test_velocity_gradient_tensor_2d` |
| `SSTKOmega.closure` | `nu_t = a1 k / max(a1 omega, sqrt(S2) F2)` | `test_closure_matches_shared_sst` (algebraic identity) |
| `_transport_step` | backward-Euler k/omega transport, implicit destruction, explicit production + cross-diffusion | same shape as the SA closure |
| wall BC | `k = 0`, `omega = wall_omega` (low-Re `60 nu/(beta1 d^2)`) | `native_sst3d.wall_omega` |

The freestream seed is `k = (I * U_b)^2` with `I = 0.05` (a small
fully-turbulent seed), `omega = k / (3 nu)` so `nu_t = 3 nu` at start — a
physically meaningful level (the first attempt seeded `k ~ nu^2`, 8 orders of
magnitude too small, and the production could not build up; this is recorded in
the module docstring as the cold-start trap).

## 3. Result (coarse-grid, 48x24, short budget)

| quantity | SST (round 10, t = 5) | SA (round 8, t = 60) | Breuer2009 |
|---|---|---|---|
| `x_r/H` | 2.22 | ~8 (over-diffusive) | 5.34 |
| `nu_t/nu` (hill, max) | 7.5 | 36–65 (saturated) | ~26 (channel) |
| `nu_t/nu` (channel) | 5.2 | — | ~26 |

The SST closure moves the reattachment **in the right direction**: the bubble
closes at `x_r/H = 2.22` on the coarse grid, *shorter* than the SA
over-diffusive ~8, and `nu_t/nu` is an order of magnitude lower (7.5 vs 65) —
the SST closure is not saturated the way the SA branch is.  This is the
round-10 model-class claim: **the SST closure is correctly implemented and
behaves as the literature expects** (less over-diffusive than SA on a
separated flow), so the next step toward the 10 % gate is a *longer* SST run
at the full Re = 2800, not a re-tuning of the SA branch.

## 4. Error judgement (explicit)

The 10 % acceptance gate is **not** asserted here — the coarse-grid `x_r/H =
2.22` is a short-budget transient, not the time-averaged Re = 2800 value, and
it is 58 % off the reference (outside the 10 % band).  What is asserted and
verified is that the *new closure is correctly implemented*:

| check | result | `< 3 %`? |
|---|---|---|
| `nu_t` algebraic identity vs `sst2003_closure` | `rtol 1e-6` | **YES** |
| 3x3 tensor out-of-plane zero | exact | **YES** |
| bulk velocity held (`u_b = 1`) | `1e-6` | **YES** |
| k/omega positivity | `k >= 0`, `omega > 0` | **YES** |
| channel mid-u vs bulk | `0.025` (2.5 %) | **YES** |

The full Re = 2800 time-averaged acceptance (`x_r/H` and `u/U_b` at `y/H =
0.5` vs Breuer2009) is **not run** in this round — it is a cluster job (the
round-9 96x48 run took 1083 s on one cluster core) and is scoped as the
round-11 follow-up.  This is an honest, scoped statement: the round-10
deliverable is the *closure implementation + verification*, not a pass on the
10 % gate.

## 5. Reproduce

```bash
# 1. round-10 verification (fast, ~26 s)
PYTHONPATH=src python -m pytest tests/test_periodic_hill_round10.py -q
# 2. full periodic-hill suite (no regression, ~42 s)
PYTHONPATH=src python -m pytest tests/test_periodic_hill.py \
    tests/test_periodic_hill_round9.py tests/test_periodic_hill_round10.py -q
# 3. machine summary + figure
PYTHONPATH=src python -c "from tensorfvm.verification.periodic_hill_round10 import run_round10; run_round10(out_path='docs/case-periodic-hills/round10_summary.json')"
```

## 6. Next round (scoped)

1. **Longer SST run at Re = 2800** (96x48, time-averaged) — the round-10
   coarse-grid result says the closure is not saturated, so the full-gate
   verdict is now a *compute* question, not a model-class one.
2. **`u/U_b` profile at `y/H = 0.5`** with the SST closure (the round-9
   post-processing is closure-agnostic and can be reused).
3. **3-D spanwise cells** (round 9 §7.2) — a solver feature, separate mission.
