# REPRO-FVM-FORWARD-STEP — design addendum (repair round 1, m15542)

Written before the 240x80 / 480x160 runs of this round were available.  The
pre-registration in `DESIGN.md` stands unchanged: same geometry, same inflow
state, same t = 4, same graded quantities, same 8 % tolerance, same grid of
record **480 x 160**.  Nothing below relaxes a tolerance.

## 1. Measurement defect fixed (declared deviation)

`measure_shock_jump` was meant to sample *post-shock* cells: the window is
`[x_s + dx, x_s + min(3dx, 0.02)]`.  On a grid where that window is narrower
than one cell (dx > 0.02, i.e. every grid coarser than ~150 x 50) the fallback
selected the first cell whose **centre** is `>= x_s`.  When the half-jump point
falls in the left half of the transition cell, that centre *is* the transition
cell, so the reported state was a mixture of pre- and post-shock — the 60x20
run of the previous round reported p2/p1 = 6.18 and u2/u1 = 0.4915 that way.
The fallback now requires the cell's upstream face to be at or behind `x_s`.

Consequence for the record: the previous round's `-31 %/-25 %/+76 %/+83 %`
numbers were a measurement artefact on top of an under-resolved run.  On the
same 120 x 40, t = 4 field the repaired measurement gives p2/p1 +1.30 %,
rho2/rho1 -1.29 %, u2/u1 -11.53 %, M2 -12.66 %, p_stag/p1 -2.79 %,
rho_stag/rho1 -2.52 %.  Both the definition change and the before/after numbers
are in this document; the tolerance is untouched.

## 2. Solver robustness (the grid of record did not run at all)

180 x 60 died at t ~= 0.67 with `non-positive time step` (dt = NaN): the strong
expansion over the step roof drove cells to the density floor, a floored density
carries the near-vacuum momentum, u explodes, the next step is NaN.  Fixed with
(a) clamping of the reconstructed face states to the positive floors and
(b) a counted troubled-cell repair from usable fluid neighbours.  Without (2)
the pre-registered 480 x 160 could never have produced a number.

## 3. What is still pending in this round (honest status)

* 240 x 80 and 480 x 160, t = 4: launched locally (16 threads, load average
  > 200 on the 128-core control node) and on the cluster queue (job cj1160,
  queued).  Neither had completed when this round was closed.
  Command: `python scripts/forward_step_study.py --out
  docs/repro-fvm-forward-step/run --grids 240x80,480x160 --t-end 4`, followed by
  `python -m tensorfvm.compressible.forward_step_benchmark --runs
  docs/repro-fvm-forward-step/run` and `scripts/forward_step_report.py`.
* Therefore the *benchmark verdict* is not claimed: on the one grid that did
  complete (120 x 40) 4 of the 6 graded quantities are inside 8 % and u2/u1 and
  M2 are not.  fail_stage = `grid_of_record_not_run`; failure_reason =
  `compute budget / queue`, not a numerical failure of the case.

## 4. Scale of the graded run (may need resize on this node)

120 x 40, t = 4: 2834 steps, 133 s at 4 threads.  480 x 160 is ~16x the cells
and ~4x the steps, i.e. hours of CPU on one socket; the cluster queue is the
documented route.
