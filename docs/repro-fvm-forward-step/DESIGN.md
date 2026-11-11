# REPRO-FVM-FORWARD-STEP — pre-registered design

**Status: pre-registration.** Written *before* the production grid-triple run, so the
tolerances and graded quantities below are not chosen after seeing results. Deviations are
recorded at the bottom.

## 1. Decision this experiment supports

Can TensorFVM's compressible Euler path reproduce the *quantitative* physics of Woodward &
Colella (1984) problem 2 — Mach 3 flow over a forward-facing step — to better than 8 % on
grid-independent shock quantities, and is the embedded-wall (solid mask) treatment
conservative? If yes, the step case can enter the benchmark pool as a graded case; if no,
the failure stage and reason are recorded instead.

Alternatives considered: (a) reuse an existing TensorFVM case — none exists in the repo
(`grep -ri "forward.step|woodward|colella"` returned nothing, and the platform
knowledge-base engine was offline at :8021); (b) run the case through an external solver —
out of scope for this repository's remit, and the mission asks to use or extend TensorFVM.

## 2. Case definition (from the paper)

Woodward & Colella, *J. Comput. Phys.* **54** (1984) 115–173, problem 2 (“a Mach 3 wind
tunnel with a step”): tunnel length 3, height 1; step height 0.2 starting at x = 0.6 and
extending to the outflow end; uniform inflow M = 3, γ = 1.4, ρ = 1.4, p = 1 (so a = 1,
u = 3); reflecting walls; impulsively started from the uniform state, reported at t = 4.

Reference availability, stated honestly: the paper publishes **density contour plots** for
this problem, not tables. No number is therefore transcribed from it. Every reference value
is evaluated in code from the exact Rankine–Hugoniot and isentropic relations for the
paper's stated inflow state (`normal_shock_reference`), which is the zero-hand-copy
substitute: M = 3 ⇒ p2/p1 = 10.333333, ρ2/ρ1 = 3.857143, u2/u1 = 0.259259, M2 = 0.475191,
p02/p01 = 0.328344, p02/p1 = 12.060965, ρ02/ρ1 = 4.307487.

## 3. Hypotheses and metrics

H1 (graded): the near-normal part of the shock that stands in front of the step face
satisfies the normal-shock jump for the uniform M = 3 inflow.
H2 (graded): the stream that stagnates on the step face carries the post-shock total
pressure (isentropic deceleration, no second shock).

Unit of analysis: cell averages on the stagnation streamline y = h/2 (H1) and the maximum
over the step-face cells (H2). Measurement definitions are fixed in
`tensorfvm.compressible.forward_step` (`measure_shock_jump`, `stagnation_state`); the H1
window is **min(3 dx, 0.02)** behind the half-jump crossing, i.e. it stays inside the
standoff region on every grid of the study, and the value is the line fit of that window
evaluated at the shock.

## 4. Graded criteria (tolerance fixed in advance)

| id | quantity | reference | tolerance |
|---|---|---|---|
| C1 | p2/p1 | 10.333333 | 8 % |
| C2 | ρ2/ρ1 | 3.857143 | 8 % |
| C3 | u2/u1 | 0.259259 | 8 % |
| C4 | M2 | 0.475191 | 8 % |
| C5 | p_stag/p1 | 12.060965 | 8 % |
| C6 | ρ_stag/ρ1 | 4.307487 | 8 % |
| C7 | mass budget drift (fluid + open-end ledger vs initial) | 0 | 1e-6 |
| C8 | energy budget drift | 0 | 1e-2 |

C7/C8 are conservation identities, not literature numbers; C8 is looser than C7 because
the solver's documented positivity repair is non-conservative by construction and fires on
a few cells per step at strong expansions (measured 0.16 cells/step at dx = 0.025), whereas
the embedded-wall treatment itself closes to machine precision (see the ablation below).

Grid of record: 480 × 160 (dx = dy = 0.00625). Grids 120 × 40 and 240 × 80 are run for the
convergence check; they are not used to relax any tolerance.

Reported but **not** graded, with the a-priori reason:

* mass-flux imbalance between x = 0.05 and x = 2.95 — the recirculation behind the step is
  unsteady at t = 4, so a global quasi-steady balance has no exact reference value;
* shock standoff distance and step-face force — the paper gives contours, not numbers;
* window-scale sensitivity — reported to show how much of C1–C4 is the sampling choice.

## 5. Controls, confounders, environment

* Control (ablation): the same run with the wall-flux projection off (`project_wall_flux=
  False`), 60 × 20, t = 0.5, shows the mirrored states alone leak mass at the wall; the
  projected version is the treatment under test.
* Geometry confounder removed: `step_x` and `step_h` are placed on cell faces for every
  grid, so the block wall is exact and grid refinement does not move the wall.
* Nuisance variables: limiter (mc), CFL (0.4), order (2) held fixed; only dx = dy changes.
* Environment: CPU only, float64, PyTorch; the control node was heavily loaded
  (load average > 300 on 128 cores), so the grid triple runs through the cluster queue
  (`run_cluster`), which is the documented route for > 10 min of compute.
* Safety/resource limits: 7200 s job timeout; the three grids take ≈ 1 + 5 + 25 min.

## 6. Stopping rule

Stop and report honest-FAIL (with fail stage) if (i) a graded quantity misses its tolerance
on the grid of record, or (ii) positivity repair destroys the budget beyond C8, or (iii) the
run does not reach t = 4 within the job timeout. No post-hoc tolerance relaxation: a miss is
reported as a miss.

## 7. Deviations (filled in after the run)

See `docs/repro-fvm-forward-step.md` §5.