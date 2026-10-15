# 30P30N accuracy work — persistent notes

(Rebuilt after a sandbox reset wiped the untracked work/ directory; everything below is
committed on branch arena/e7f3f162-tensorfvm.  Large artifacts — meshes, checkpoints, logs,
run directories — are regenerated, see work/.gitignore.)

## Target and status
* HLPW4 / BANC 30P30N, chord 1, Re 9e6, alpha 8.1 deg (experiment M 0.2, NASA LTPT):
  exp Cl 3.1354; Cp_min slat -2.38, main -5.76, flap -3.18; exp main-upper Cf (U_inf units)
  0.00642 @x .45, 0.00528 @.85, flap 0.00708 @.975.  Typical 2-D RANS-SA Cl 3.05-3.15.
* Archived run: Cl 2.47063 (-21 %), 70 it, first-order stage only, limit-cycles — do not continue it.
* Best so far (mesh m3, krylov): Cl 2.47-2.50 on TWO different steady branches (see below);
  NOT a converged physical answer.  Never quote these Cl without the caveat.

## Environment (after a reset)
    bash work/env/setup.sh           # venv /home/user/.venv-tfvm + gmsh GL/X stubs
    export LD_LIBRARY_PATH=/home/user/.stublibs   # for any gmsh call
    OMP_NUM_THREADS=1 /home/user/.venv-tfvm/bin/python ...
RAM 3.9 GB, 2 CPUs.  m2b/m3 solver 1.5-1.8 GB, NACA 62k-cell solver 1.1 GB, diagnostics 0.6 GB.
Processes do not advance between turns; poll logs.  History-live rows lack coefficients; read Cl
from the JSON lines of the .log.  `--tolerance 1e-5` never triggers (momentum L1 is far-field
dominated) — judge by velocity_change / Cl history.

## Code changes on the branch
* 73762c8  Gmsh BL fan apex triangles touch the wall at one vertex and never get no-slip
  (|u| up to 2.6 U, Cp -59 within 1e-6 c of the wall) -> reader collapses them into a
  2e-6 fillet; no fans on the sampled leading edges.  (ROOT CAUSE #1)
* 1138760  O(N) linear solvers (`linear_backend='krylov'`: momentum spilu(1e-3,fill 5)+GMRES,
  pressure pyamg Ruge-Stueben, SA spilu; GMRES callback_type 'pr_norm'; ILU(0) fails for
  pressure), cached LU option, wake refinement (Fields 4/5/6), cross-mesh initialisation
  (`--initialize-from DIR --interpolate-initialization`; DIR needs producer.json, config.json,
  checkpoint.npz, mesh.msh; `--resume` refuses when mesh hashes differ).
* 260fbf2  spline resampling of the official polylines (`--max-turn 1.5,1.5,2`), the 12 deg
  polygon facets of the LE gave a Cp/Cf sawtooth that tripped/separated the BL (ROOT CAUSE #2);
  `config.single_element` for plate/NACA meshes (only the `main` wall group required).
* 6f539fc  `--sa-linear-upwind` (bounded linear-upwind deferred correction for nu_tilde).

## Mesh recipes (scripts/mesh_hlpw30p30n.py; Gmsh BL is a robustness lottery with resampled profiles)
* M0  `--layer-thickness 0.001` -> 58,861 cells (3.3-3.9 s/it krylov, 1.2 GB)
* m2b `--layer-thickness 0.005 --near-size 0.003 --wake-size 0.02 --wake-distance 0.5 --layer-ratio 1.18` -> 133,923
* m3  = m2b + `--max-turn 1.5,1.5,2` -> 143,669 cells, 37 layers to 0.005, first layer 2e-6 (8-12 s/it alone)
  ORIGINAL polylines: near 0.003 works, 0.002 fails.  SuperLU: COLAMD best, MMD_AT_PLUS_A
  pathologically slow, NATURAL OOMs.  Raw gmsh cell order != solver cell order (use reader centres).

## Run history (Cl)
* M0: peak 2.7222 (it 201) -> 2.345 plateau.   m2b: peak 2.5515 (481), 2.5362 (521).
* m3-run1 (from freestream): long laminar bubble on the main LE (x .044-.108, Cp plateau -4.13);
  2.4859 (401) -> 2.4769 (511).  Branch A.
* m3-run2 (interpolated from m2b it 530): attached nose, Cp_min -4.85/-4.90, Cf 0.025-0.029 @x .044
  -> 0.0028-0.0046 @.07 -> 0.0039 @.14 -> 0.0016-0.0022 after the knee -> 0.0010-0.0015 at .2-.45;
  2.5734 (1) -> 2.5127 (101) -> 2.4688 (201), Cd .104.  Branch B.
* m3-run3 (branch B + nu_tilde seeded in the nose BLs): nose chi identical to run2 after 70/120 it,
  Cl 2.5028 -> 2.4401 (121) still falling.  Seeding REJECTED.
  => hysteresis: two discrete branches, both ~20 % low.

## Experimental main-upper Cp (alpha 8.1, cp_exp.dat zone "main" = file lines 41-95, sorted by x)
  -4.75 @.0438  -5.31 @.0448  -5.68 @.0465  -5.75 @.049  -5.76 @.053  -5.69 @.058  -5.58 @.064
  -5.40 @.070   -5.11 @.0855  -4.79 @.095   -4.55 @.1075 -4.29 @.1225 -4.17 @.1325 -4.15 @.1425
  -4.08 @.1525  -3.63 @.165   -3.13 @.18    -2.81 @.20   -2.52 @.225  -2.40 @.25   -2.08 @.30
  -1.93 @.35    -1.81 @.40    -1.71 @.45    -1.64 @.50   -1.59 @.55   -1.53 @.60   -1.46 @.70
  -1.43 @.80    -1.37 @.85;  lower +0.55..+1.05 (x .058-.155), +0.78 @.30, +0.66 @.525, +0.74 @.75.
The main upper surface has a genuine knee at x .14-.175 (slope 13.6 -> 5.3 deg, curvature -7.7/c);
the experiment survives dCp = +1.3 over 0.05 c there.  Branch B: -4.9 peak, -4.1 @.07, -3.55 @.10,
-3.15 @.134, -3.03 @.151, -2.64 @.168, -2.18 @.20, -1.82 @.25, -1.60 @.30, -1.33 @.45 -> deficit
1.0-1.3 for x .07-.15, 0.4-0.6 downstream; local cf at .45 0.00047 vs ~0.0025 expected.

## Verification / rejected hypotheses
* Flat plate (TMR, Re 5e6, symmetry run-up, 116x72): Cf +0.1..+0.35 % vs TMR incompressible
  (0.0027292 @x .97), u+ log law within +0.1..0.25 -> SA, wall treatment, LSQ gradients and
  convection are right for an attached ZPG turbulent BL.  Inlet-at-the-LE meshes give +1.3..4.5 %
  (virtual origin).  SIMPLE from uniform flow needs 1400-3900 it on the plate.
  Momentum-integral check on the plate: d(theta) = int Cf/2 dx within 4 %.
* REJECTED as the 30P30N cause: BJ limiter activity census; triangle diffusion; wall distance;
  LSQ weighting; "SA bug"; iteration-limited transition (frozen-velocity SA unchanged); ft2;
  raising nu_tilde_inf (1-D: laminar for any chi_inf at delta 8e-5); local nu_tilde seeding;
  first-order vs second-order nu_tilde convection (plate < 0.1 %).  1-D frozen-profile SA tests are
  not representative of 2-D SA (TMR plate transitions at Re_delta 350-800 via the LE singularity).
* NOTE: wall-cell chi = kappa y+ (0.1-0.4 at y+ 0.3-1) is NORMAL for a turbulent SA BL — the
  earlier "laminar nose" reading of wall-cell chi was wrong; judge by chi_max/Cf instead.

## NACA 0012 isolation test (work/naca0012) — the decisive result so far
TMR SA validation case, Re 6e6, alpha 10, CFL3D Cl 1.0909 Cd 0.01231 (7 codes 1.089-1.100 /
0.01225-0.01245); alpha 0 Cd 0.00819; alpha 15 Cl 1.5461.  Reference Cp/Cf tables in
work/naca0012/reference.json (CFL3D is turbulent from the LE: Cf 0.028 at x 3e-7, peak 0.031).
Mesh mesh-a (make_naca_mesh.py defaults): 626 wall points (facets <= 1.5 deg turn, <= 0.004,
1e-3 near the TE), first layer 2e-6, 37 layers to 0.005 (ratio 1.18), near 0.004, 0.03 to 0.5 c,
circle R 500 -> 54k cells (first incarnation 62k).  Driver run_naca.py, 3.5 s/it, 1.1 GB.
Result (first incarnation, it 300-400, Cl flat): Cl 0.904-0.905 (-17 %), Cd 0.032 (3x),
Cp_min -4.3 vs -5.66, upper Cf 0.55-0.6x CFL3D for x .002-.2 (local cf 0.65-0.85x), 2.5x at x .9.
* Far field is clean: circulation constant (Cl-equivalent 0.91) from r 0.7 to 40 c, residual
  uniform flow at 9.99 deg -> no starting vortex / BC problem; the deficit is produced at the airfoil.
* Cd_p ~0.026 = Cl*tan(1.65 deg): the missing leading-edge suction (thin-airfoil LE thrust
  2 pi alpha^2 ~ 0.19) explains both the drag excess and the lift loss.
* Profiles: law of the wall fine; BL much thicker than it should be: Re_theta 928 @x .01,
  3405 @.05, ~14000 @.2 (delta99 1.5e-2 at x .2).  Von Karman momentum integral violated:
  the momentum deficit grows ~2x faster between x .01 and .2 than Cf + pressure gradient allow
  (plate closes to 4 %) => numerical momentum loss in curved / pressure-gradient BL cells —
  a discretisation issue, not the turbulence model, not the multi-element setup.
IMPORTANT CORRECTION (second incarnation, 2026-10-15): "Cl flat at 0.905 for it 300-360" was NOT
convergence.  Rerun on mesh-a (54k): Cl 0.509 @90, 0.870 @270, 0.900 @480, 0.914 @600 and still
rising ~0.0012/10 it; Cp_min deepening (-3.75 @250 -> -4.51 @600); Cf(.05) rising 0.0053 -> 0.0072
(ref 0.0096) while the rear upper BL weakens (Cf(.6) 0.0032 -> 0.0008, ref 0.00235).  The SIMPLE
transient of the BL/circulation system takes >1000 iterations on these meshes (U/p/SA relaxation
0.5/0.3/0.3).  Momentum-integral checks are only meaningful near convergence:
* alpha 0 (mesh-a0, 54k): at it 200 the ratios LHS/RHS were 0.24-0.72 (BL "too young");
  at it 500 they are 0.83-1.09 (closed), theta(.3) 3.7e-4 -> 5.3e-4, Cf .05/.1 = CFL3D within 1 %,
  Cf .3 +6.5 %, .6 +17 %, .9 +32 % and still falling, Cd 0.0117 @150 -> 0.0089 @500 (ref 0.00819),
  Cp_min -0.407 (ref -0.413).  => alpha 0 is converging toward the reference.
* alpha 10 at it 600: ratios 1.5-2.0 for x .01-.1 (inside the quad layer), local cf ratio 1.05 at
  the nose, 0.84 @.05, 0.75 @.1, 0.67 @.2, 0.56 @.3; H 1.56; theta too large.  Unknown yet
  whether this persists at convergence (runs resumed: a10 -> 1500 it, a0 -> 1000 it).
Also noted: profiles.py theta/delta99 are contaminated once delta99 exceeds the quad layer (0.005)
and by the slow potential-flow variation (alpha 0, x > .4: delta99 0.04-0.16 nonsense).
Meshes generated for the next experiments: mesh-n (first layer 1e-6, ratio 1.12, 66,564 cells),
mesh-t (near .002, max_turn .75 deg, 1192 wall points, 98,318 cells), mesh-thick0 (alpha 0 split,
quad layer 0.04 c, ratio 1.12, 69 layers, 75,050 cells; run-a0-thick stopped at it 350 to free
a core; resume with the same command, 500 -> more).
UPDATE (same day, later): the NACA deficit is a SLOW TRANSIENT, not a discretisation defect.
* alpha 0 baseline (0.5/0.3/0.3) at it 800: Cd 0.0084 (ref 0.00819), Cf .05/.1/.3/.6 within 2 %
  of CFL3D, Cf .9 +7 % and still falling; momentum integral closes (0.83-1.09).  With relaxation
  0.85/0.9 it reaches Cd 0.00814 at it 760 and keeps falling (Cf .9 0.0018 < ref 0.0020; 4
  reversed TE faces) -> the converged alpha 0 Cd may undershoot by a few %.
* alpha 10 baseline: Cl 0.914 @600, dip to 0.898 @880 (TE separation sets in, 60 reversed faces
  x > .85), 0.914 @1050; front Cf keeps healing: Cf(.05) 0.0069 @560 -> 0.0093 @1040 -> 0.0095
  @1200 (ref 0.0096), Cf(.1) 0.0046 -> 0.0068 (ref 0.0070), Cf(.3) 0.0021 -> 0.0035 (ref 0.0040),
  Cf(.6) 0.0007 -> 0.0010 (ref 0.00235).  Resumed with 0.7/0.7 from it 1050 (run-a10-r7):
  Cl 0.938 @1220 rising 0.0015/10 it, Cp_min -4.77.  The forward BL "momentum loss" at it 400-600
  was the BL still developing from the impulsive start (the BL state propagates downstream over
  hundreds of iterations); the rear TE separation is (at least partly) the consequence of the
  transient excess theta arriving there.
* Convergence acceleration test (alpha 0, forks from it 650, 110-150 it): P_RELAX 0.5 = identical
  to 0.3 (pressure is not the bottleneck); ANDERSON=5 (MultiElementFlowSolver.step mixing) =
  identical (always rejected/ineffective); U/SA relaxation 0.7/0.7 = 1.8x faster; 0.85/0.9 = 2.75x
  faster, stable (dU 3e-3 flat).  => use U_RELAX 0.85 SA_RELAX 0.9 (P 0.3) from now on; the
  30P30N m3 runs (500 it at 0.5/0.3/0.3) were far from converged — their Cl 2.44-2.5 and the
  "two branches" are transient states, not solutions.
* CONVERGED alpha 10 on mesh-a (run-a10-r9, 0.85/0.9 from scratch, Cl flat after ~1000 it;
  r7 from the baseline follows the same trajectory => path independent): Cl 1.028 (-5.8 %),
  Cd 0.0181 (ref 0.0123), Cp_min -5.20 @x .003 (ref -5.66), upper Cf = CFL3D within 1-2 % for
  x .002-.2 (local cf ratio 1.04-1.09 because U_e is 3-4 % low), momentum integral closes
  (0.99-1.13); but Cf falls below CFL3D from x .3 on: 0.87 @.4, 0.80 @.5, 0.73 @.6, 0.57 @.8,
  0.36 @.9, separation at x .977 (21 reversed faces).  => the front BL is right, the remaining
  deficit is the rear upper BL (delta 1-4 cm, far outside the 0.005 quad layer, resolved by
  isotropic 0.004 triangles) and the resulting TE separation / circulation loss.
* Linear-solver cost: budgets were |b|*1e-11 absolute (pressure 5e-12, momentum 3e-9) -> 90-145
  GMRES iterations per solve on 30P30N m4 (17 s/it).  Added solver.linear_budget_scale
  (external_cached.py; driver --linear-budget-scale; run_naca env LINEAR_BUDGET_SCALE) and
  LU_REFRESH env.  m4 with --lu-refresh-interval 1 --linear-budget-scale 1000: 9.9 s/it
  (momentum 6 its 0.9 s x2, pressure 23 its 1.8 s, SA 15 its 0.6 s x3; ~4.7 s/it is Python
  assembly/metrics overhead).  NACA 54k: solves are cheap anyway (3.1 -> 2.8 s/it); refresh 1
  is slower there (setup cost).  Trajectories identical to printed precision.
* The driver refuses --resume when numerical settings change (provenance gate) -> new output dir.
* Running now: work/30p30n/m4-run2 (m4 = m3 recipe with --wake-size 0.03, 118,895 cells;
  alpha 8.1, 0.85/0.9/0.3, refresh 1, budget 1e3, 3000 it, ~8 h) and run-a10-thick (mesh-thick10:
  quad layer 0.04 c, 69 layers ratio 1.12, 75k cells, 0.85/0.9) to test the rear-BL hypothesis.
Plan: (1) if the thick layer fixes the NACA rear BL/Cl, the 30P30N needs thick quad layers too
(limited by the slat/main and main/flap gaps -> at least on the main upper surface aft of the
slat and on the flap; or a wake/BL anisotropic triangle size field); (2) report m4-run2 only after
>= 1500 iterations with a flat Cl.

## 2026-10-15 (cont.) — thick-layer verdict, 30P30N relaxation limits, zoned layers, far-field grading
* m4-run2 (0.85/0.9 from scratch) DIVERGED (log work/30p30n/m4-run2-diverged.log).  The 30P30N
  also diverges at 0.85/0.9 when started from a developed 300-it state (m4-stage2 attempt: dU x4
  per 10 it, Cl -266 by it 61; log m4-stage2-diverged-0.85.log).  0.7/0.7/0.3 from the 300-it
  state is stable (m4-stage2.log: 70 it, dU 3e-2..7e-2 falling, Cl 2.52-2.53, Cd 0.205 -> 0.17;
  stopped at it 70 to free a core).  => 30P30N recipe: stage 1 = 300 it at 0.5/0.3/0.3 from
  scratch, stage 2 = --initialize-from stage1 at 0.7/0.7/0.3 (new dir; --resume refuses setting
  changes).  High relaxation that works on NACA is NOT usable here.
* NACA run-a10-thick CONVERGED (mesh-thick10: 0.04 c quad layer, 75k cells, 0.85/0.9, Cl flat
  from it 1300): Cl 1.0780 (-1.2 % vs CFL3D 1.0909; mesh-a 0.005 layer gave -5.8 %), Cd 0.0186
  (Cd_p 0.01217, Cd_f 0.00644), Cp_min -5.43 (ref -5.66), upper Cf/CFL3D 1.03-1.07 for x .002-.9
  (1.11-1.15 at .9-.95), attached to the TE (1 reversed face).  => ROOT CAUSE #4 CONFIRMED: the
  rear BL must live inside the quad layer; isotropic triangles outside a thin layer cost -27 %
  Cf(.6), -64 % Cf(.9), TE separation and -5 % Cl.  Tool: work/naca0012/cd_decomp.py.
* Remaining NACA alpha-10 discrepancy is a NOSE/OUTER-FLOW effect: pressure drag 0.0122 vs ~0.006
  expected, Cl_p/Cd_p bins (integrating CFL3D's Cp on the same geometry) put the whole excess at
  x < 0.02 (upper +0.0033, lower x < .005 +0.0064); the stagnation Cp is 1.021 (physical max 1.0,
  CFL3D 1.005 incl. M=0.15) and the total-pressure coefficient p+|u|^2 in the OUTER flow is
  1.015-1.03 at 0.3-2 c from the airfoil (0.966 at 3-5 c, 1.000 at 50 c).  Cause: the NACA mesh
  script's background grading reached far_size 25 at 25 c => cell size ~ distance (0.19 c at 1 c,
  1.8 c at 5 c), i.e. (h/r)^2 ~ 4-16 % outer-flow discretisation error.  The 30P30N writer grades
  to 100 c (h ~ 0.25 d); its m4 300-it field has Cp_total errors < 0.1 % upstream (transient junk
  above the main element aside).  Fix: `grading_distance` 400 (h ~ 0.03 + 0.06 d, +25k cells):
  make_naca_mesh.py arg 13, mesh_hlpw30p30n.py --grading-distance.  Test running:
  work/naca0012/run-a10-ff (mesh-thick10-ff, 101,446 cells, 0.85/0.9, 1500 it).
* Zoned quad layers for the 30P30N (hlpw30p30n.layer_zones + write_mesh_geo thick_layer;
  --thick-layer): 0.04 c on the main upper surface 0.06 < x < 0.855 and the flap upper surface
  0.905 < x < 1.115 (both flap sides for x > 0.96); 0.005 c elsewhere (gaps, slat, cove, TE fans
  — a 0.04 layer on the blunt flap TE gave 88 deg wall non-orthogonality, rejected by
  mesh_quality).  Gmsh is a lottery: thick 0.04/ratio 1.15 -> "Edge not recovered"; 1.16 works.
  m5: `--layer-thickness 0.005 --thick-layer 0.04 --near-size 0.003 --wake-size 0.03
  --wake-distance 0.5 --layer-ratio 1.16 --max-turn 1.5,1.5,2` -> 128,043 cells (m5/zones.png).
  m6 = m5 + `--grading-distance 400` -> 152,665 cells (ready, unused until run-a10-ff reports).
* Runs: m4-stage1 done (300 it, Cl 2.5210, Cd 0.2134 — transient); m5-stage1 running (same
  settings, ~11 s/it); next m5-stage2 (or m6) at 0.7/0.7/0.3, 2500 it (~7 h).
