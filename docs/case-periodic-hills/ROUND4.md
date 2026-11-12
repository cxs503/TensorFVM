# CASE-FVM-PERIODIC-HILLS — round 4: the Re = 2800 blow-up was three transport defects, not the hill

- Mission: 16765 (selfstart, TensorFVM) · branch `agent/m16765` · anchor `CASE-FVM-PERIODIC-HILLS` (feature ledger, `open`)
- Driver: `python -m tensorfvm.verification.periodic_hill_round4 docs/case-periodic-hills` (8 criteria, ~2 min)
- Machine summary: `docs/case-periodic-hills/round4_summary.json` · Figure: `fig_round4.png`
- Artefacts: `runs/periodic-hill-round4/{g48.jsonl, a48_mean.npz, a96_mean.npz, g48_field_*.npz}`
- Tests: `tests/test_periodic_hill_round4.py` (plus the round-1/2/3 suites, 22 tests)

## 1. What rounds 1–3 left behind

Rounds 1–3 recorded an honest FAIL: the plane channel (`hill_amplitude = 0`) was
clean to 3 % and `max|div u| ~ 1e-15`, while the *hill* at `Re = 2800` went NaN
inside 1000 steps on 48x24 and inside 500 on 64x32. Round 3 fixed two genuine
defects in the pressure-gradient operator and still could not run the case.
Round 4 stopped guessing and isolated the divergence with two ablations:

| ablation (48x24, Re=2800, dt=0.01) | outcome |
|---|---|
| shipped round-3 solver | `max|u|` 1.41 → 1.96 → **NaN at step 1068** |
| convection term replaced by 0 (Stokes-like) | **stable**, `max|u|` 1.67, `t = 20` |
| `hill_amplitude = 0` (plane channel), convection on | **stable**, `div = 1e-15` |

The convection term was the only energy source, and it needed a *curved wall*
(`flux_n != 0`) to fire. Two bugs, both in `PeriodicHillSolver._convection`:

**(viii) non-conservative east-face weighting.** The outflow and the inflow of
one east face were both divided by the **donor** cell's volume. On this
terrain-following mesh the column thickness — hence the cell volume — varies by a
factor **1.49** between crest and trough, so the two sides of a face were weighted
differently: the operator no longer annihilated a constant field on a
divergence-free flux (`max|C(1)| = 4.13` against `max|div flux| = 2.8e-09`), i.e.
it was a volume source of kinetic energy.

**(ix) inverted eta-face sign.** `out` is the *net outflow per unit volume*, so the
cell a face flux **leaves** must gain `+flux*face/V`. The eta branch had the signs
swapped, making the wall-normal direction an **anti-upwind** (negative-diffusion)
operator: convection pumped variance into the field instead of smoothing it. On
the plane channel `flux_n` vanishes identically, which is exactly why that case
never showed the bug.

**(x) the bulk-velocity control was a marginally unstable oscillator.** The law
`G += relaxation (u_b* - u_b)/dt` is a proportional controller on a plant that
responds to `G` with inertia. Linearising (`du_b/dt = G - lam u_b`, explicit `G`
update) gives `G_{n+2} - 2 G_{n+1} + (1+relaxation) G_n = 0`, i.e. roots with
`|lambda|^2 = 1 + relaxation > 1` for **every** `relaxation > 0`. Measured
symptom: `G` wanders to `-22 … -110` where the physical force is `4e-03`, and the
flow is hammered until it overflows (an overflow trace, not a physics trace — this
is what rounds 2/3 were recording as "divergence"). The force now comes from the
*affine* impulse predictor, `us(G) = us(0) + G M^{-1} 1`, solved once per build:
`G = (u_b* - u_b(us(0)))/bulk(M^{-1}1)` is an algebraic **constraint**, not a
loop, and `body_force_relaxation` is demoted to an under-relaxation whose bulk
error decays by `(1 - relaxation)` per step (measured first-step ratio
**0.5002** for `relaxation = 1/2`).

## 2. Gates (pre-declared, machine-evaluated)

| criterion | measured (round 4) | verdict |
|---|---|---|
| C1 conservative convection annihilates constants on a div-free flux | `max|C(1)| = 1.5e-09` (flux div 1.5e-09) | PASS |
| C2 negative control: round-3 assembly on the same flux | `max|C(1)| = 4.27` | PASS (not vacuous) |
| C3 matches an independently assembled upwind stencil | mismatch `0.0` | PASS |
| C4 bulk velocity imposed, not chased | `|u_b-1| = 2.7e-09`; filter ratio 0.5002 | PASS |
| C5 plane-channel gate survives | `u_max/u_b` 1.34589 vs 1.34637 → **0.036 %**, `max|div| = 6e-16` | PASS |
| C6 Re=2800, 48x24, dt=0.01, 30000 steps (`t=300`) | finite throughout, `|u_b-1| < 1e-9` | PASS |
| C7 `x_r/H`, `x_s/H` vs Breuer 2009 (5.34, 0.227), 10 % | `x_r/H = 8.137` (**+52 %**), `x_s/H = 0.631` (+178 %) | **FAIL (honest)** |
| C8 grid sensitivity of `x_r/H`, 48x24 vs 96x48 | 8.137 vs 8.135 → **0.025 %** | reported |

## 3. Why C7 fails — and why that is the useful result

The mean separation bubble of this solver closes at `x_r/H = 8.14` instead of
5.34, and refining 48x24 → 96x48 moves it by **0.025 %**: the answer is *grid
converged and still wrong by 52 %*. The bubble is not under-resolved — it is a
**model-class** difference. `Re = 2800` over these hills is a turbulent case;
Breuer et al. (2009) is (U)RANS/LES/DNS with a closure, while this solver is 2D
laminar Navier–Stokes with no turbulence model. A laminar solver keeps the shear
layer attached far too long, exactly as round 1 predicted in prose
(`docs/case-periodic-hills/README.md`, "no turbulence model, so `x_r/H` will most
likely overshoot"). So the anchor's 10 % criterion is **unreachable inside the
current solver class**; it needs either a turbulence closure or the case to be
re-anchored on a laminar-validated quantity (see §5).

## 4. Reproduction

```bash
PYTHONPATH=src python -m tensorfvm.verification.periodic_hill_round4 docs/case-periodic-hills
# time-averaged field (about 90 s per grid, single core)
python /tmp/avg_run.py 48 24 10000 20000 0.01 a48     # spin-up t=100, average t=100..300
```

## 5. Left open (withheld, not claimed)

1. **Turbulence closure** (mixing length / k-epsilon / SA) is the only route to
   the Breuer numbers; that is a new capability, not a defect fix — propose
   `GAP-FVM-TURBULENCE-CLOSURE`.
2. `x_r/H` from the *instantaneous* trace fluctuates in `[2.4, 5.5]` (the 2D flow
   at `Re = 2800` is unsteady, `max|u|` oscillating 1.19–1.31), so any single
   snapshot is meaningless; the numbers above are 20000-sample time averages.
3. The `x_s/H` definition on a *slanted* wall is ambiguous (the streamwise shear
   component changes sign on the slope); only the last `- -> +` crossing is used
   for `x_r/H` and the first `+ -> -` crossing for `x_s/H`, both listed in
   `round4_summary.json.criteria.C7…`/`mean_field` for auditability.
4. Cell-level divergence of the *cell-centred* velocity is still `O(1e-1)` on the
   hill mesh (the divergence-free quantity is the face flux, `2e-09`). This is the
   collocated non-adjointness documented in round 3 and is not fixed here; a
   staggered (MAC) layout is the structural cure and remains the round-5 item.
5. No comparison against the Breuer *velocity profiles* — no tabulated reference
   in the tree; the profile figure is our own output only, for eyeball sanity.