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
Plan: switch suspects one at a time on the NACA (run_naca.py env switches UNBOUNDED, NO_STRESS,
BLEND, SA_LINEAR_UPWIND; normal/tangential refinement via make_naca_mesh.py), check with
compare.py and profiles.py (momentum-integral ratio), then return to 30P30N.
