# TensorFVM MRF (rotating reference frame) verification

Driver: `PYTHONPATH=src python -m tensorfvm.verification.rotating_frame`  
Module: `src/tensorfvm/rotating_frame.py`  
Tests: `tests/test_rotating_frame.py`  
Criteria: **18/18 PASS** (wall 2.1 s, generated 2026-11-12T14:36:43Z)

## 1. Case introduction (physics, geometry, parameters)

A single rotating reference frame (MRF) is the standard industrial model for a propeller or rotor: the flow is solved in a frame rotating with the rotor, so the blade becomes stationary and the momentum equation gains the Coriolis term `2 Omega x u` and the centrifugal term `Omega x (Omega x r)`. Three configurations are anchored, each chosen because it exposes a different part of that operator.

| case | rotation axis | what it can and cannot see |
|---|---|---|
| A. rotating plane Poiseuille | along the wall normal `+y` | Coriolis drives a **secondary flow** `w`; a solver without Coriolis returns `w = 0` and fails by 100 %% |
| B. rotating channel | along the span `+z` (classical rotating channel, Johnston et al. 1972) | the **laminar mean profile is unchanged**; the only signature is a **cubic pressure correction** in `y` |
| C. circular Couette in a rotating frame | cylinder axis `+z` (any frame angular velocity) | MRF **frame invariance**: absolute flow and torque independent of `omega_frame` |


Case A parameters (fixed, pre-declared): channel height `h = 1.0` m, `rho = 1.0` kg/m^3, `nu = 0.050` m^2/s, driving gradient `-dp/dx = G = 1.0` Pa/m, frame `omega = 2.00` rad/s; wall layer `sqrt(nu/(2 omega)) = 0.1118` m.

Case C parameters: `r1 = 1.0` m, `r2 = 2.0` m, `omega_inner = 1.0` rad/s, `omega_outer = 0.0` rad/s, `rho = 800` kg/m^3, `mu = 0.800` Pa s; frames `omega_frame in {0, 1.0, 1.7}` rad/s.

## 2. What was done (method, flow, tooling)

1. The frame operator is implemented once (`RotatingFrame`): Coriolis, centrifugal, frame transforms, and the modified pressure `p_m = p - rho/2 |Omega x r|^2` that absorbs the centrifugal term. The sign convention is written down in the module docstring and pinned by the tests.
2. Case A closed form is derived by writing `F = u - i w`, which turns the coupled real pair into `nu F'' + 2 i omega F = -G/rho`, then solved with the no-slip conditions: `F = F_p [1 - cosh(a(y-h/2))/cosh(a h/2)]`, `F_p = i G/(2 rho omega)`, `a = sqrt(-2 i omega / nu)`.
3. That closed form is verified **three independent ways**: analytically against its own ODEs (C4), numerically against an adaptive collocation BVP (`scipy.integrate.solve_bvp`, C4b), and against a second-order finite-volume solve of the coupled `(u, w)` system on a grid ladder (C5/C5b). A derivation checked only against itself is not evidence, so the cross-checks are part of the deliverable.
4. Case B: the exact laminar solution is inert (parabolic `u`), and the Coriolis force shows up only in the wall-normal balance `p_y = -2 rho omega u`. Both halves are asserted (C7a/C7b).
5. Case C: circular Couette is solved in three frames on the conservative form `d/dr[r^3 d(v/r)/dr] = 0`; the absolute velocity and Taylor's torque must be frame invariant (C8), and the rotating-frame radial balance is evaluated with the implemented operators (C10).
6. Three **control** criteria quantify what a wrong implementation looks like (C6, C7c, C10b): an anchor a wrong solver can also pass is not an anchor.

## 3. Results

| id | value | gate | pass | what it pins down |
|---|---|---|---|---|
| C1_inert_limit_is_exact | 0.000e+00 | 0.000e+00 (max) | PASS | omega=0 gives identically zero sources and p_m == p (bitwise), not 'small' |
| C2_frame_kinematics | 1.777e-10 | 1.000e-06 (max) | PASS | 2*Omega x u components (axis-swap trap), abs<->rel roundtrip, grad(|Omega x r|^2/2) = -Omega x (Omega x r) by central differences |
| C3_solid_body_rotation_exact | 0.000e+00 | 1.000e-14 (max) | PASS | u_rel = 0 with p = rho/2 |Omega x r|^2: Coriolis 0, p_m 0, source 0 (machine zero) |
| C3b_solid_body_full_momentum_residual | 3.628e-10 | 1.000e-06 (max) | PASS | -grad p/rho - Omega x (Omega x r) = 0 for the solid-body pressure (FD gradient) |
| C4_closed_form_satisfies_odes | 2.220e-16 | 1.000e-13 (max) | PASS | nu u'' + 2 omega w = -G/rho and nu w'' - 2 omega u = 0 with analytic derivatives |
| C4b_closed_form_vs_independent_bvp | 9.121e-15 | 1.000e-08 (max) | PASS | collocation BVP (scipy solve_bvp) vs closed form on 401 points |
| C5_rotating_channel_vs_closed_form | 1.270e-05 | 3.000e-02 (max) | PASS | finite-volume (u,w) vs closed form, N=640: u and secondary flow w < 3% |
| C5b_rotating_channel_second_order | 1.974e+00 | 1.800e+00 (min) | PASS | observed order from N=320/640, streamwise and spanwise |
| C6_anchor_rejects_no_coriolis_solver | 1.000e+00 | 5.000e-01 (min) | PASS | a solver without Coriolis must FAIL the w anchor (measured error >= 50%) |
| C7a_spanwise_axis_leaves_mean_flow_inert | 1.329e-14 | 1.000e-12 (max) | PASS | rotation along the span: parabolic u reproduced to round-off for any omega |
| C7b_spanwise_axis_pressure_correction | 0.000e+00 | 1.000e-12 (max) | PASS | 2 omega u + p_y/rho = 0 with the implemented Coriolis y component |
| C7c_controls_reject_wrong_coriolis | 1.000e+00 | 5.000e-01 (min) | PASS | dropping or sign-flipping Coriolis must violate C7b by >= 50% |
| C8a_couette_absolute_profile_vs_exact | 8.229e-07 | 3.000e-02 (max) | PASS | v_theta = A r + B/r from every frame, n=320 |
| C8b_mrf_frame_invariance | 4.146e-13 | 1.000e-12 (max) | PASS | absolute profile and torque identical for omega_frame = 0, 1.0, 1.7 rad/s |
| C8c_frozen_rotor_statement | 0.000e+00 | 1.000e-15 (max) | PASS | omega_frame = omega_inner makes the inner wall's relative velocity exactly 0 |
| C9_couette_torque_vs_taylor_closed_form | 3.052e-06 | 3.000e-02 (max) | PASS | M/L = 4 pi mu r1^2 r2^2 |omega_i - omega_o| / (r2^2 - r1^2) at n=320 |
| C10a_frame_radial_balance_exact | 1.981e-16 | 1.000e-14 (max) | PASS | conv + 2 Omega x u + grad p/rho - Omega x (Omega x r) = 0 |
| C10b_controls_reject_missing_frame_terms | 1.000e+00 | 5.000e-02 (min) | PASS | dropping the centrifugal term or flipping Coriolis must violate C10a by >= 5% |

### 3.1 Case A: rotating plane Poiseuille (rotation along the wall normal)

| N | rel $L_2$ error $u$ | rel $L_2$ error $w$ |
|---|---|---|
| 40 | 2.520e-03 | 4.757e-04 |
| 80 | 7.178e-04 | 1.268e-04 |
| 160 | 1.926e-04 | 3.287e-05 |
| 320 | 4.990e-05 | 8.376e-06 |
| 640 | 1.270e-05 | 2.115e-06 |

Observed order (N=320 -> 640): $u$ 1.97, $w$ 1.99; from N=80 -> 160: $u$ 1.90, $w$ 1.95. Closed-form ODE residual (analytic derivatives): $u$-balance 2.22e-16, $w$-balance 1.11e-16; independent collocation BVP agrees to 9.12e-15.

### 3.2 Case C: circular Couette in rotating frames

| frame $\omega_f$ (rad/s) | abs err vs exact | rel err vs exact | torque coeff | inner-wall $v_{rel}$ |
|---|---|---|---|---|
| 0.000 | 8.229e-07 | 8.229e-07 | -16.75521 | 1.00e+00 |
| 1.000 | 8.229e-07 | 3.445e-07 | -16.75521 | 0.00e+00 |
| 1.700 | 8.229e-07 | 1.855e-07 | -16.75521 | 7.00e-01 |

Absolute-profile spread across the three frames 4.15e-13, torque spread 2.54e-13. Taylor torque closed form `M/L = 4 pi mu r1^2 r2^2 |omega_i - omega_o| / (r2^2 - r1^2)` = -13.4041 N m/m vs numerical -13.4042 N m/m.

### 3.3 Discriminating power (controls)

* A solver that drops Coriolis returns $w \equiv 0.0e+00$ against an anchor maximum of 0.2711 -> relative error 1.000, i.e. it fails C5/C6 as it must.

* Case B, dropping Coriolis: wall-normal balance residual 1.000; sign-flipped Coriolis: 2.000 (gate 0.5).

* Case C rotating-frame radial balance, dropping the centrifugal term: 1.000; flipping Coriolis: 2.000 (gate 0.05).

## 4. Figure

`fig_rotating_frame.png`: (a) case A closed form vs finite volume (N=80) for $u$ and the secondary flow $w$; (b) grid convergence of both components against the closed form with the $N^{-2}$ reference and the 3 % gate; (c) case C, absolute and relative azimuthal profiles from three rotating frames on top of the single exact inertial profile -- the curves coincide, which *is* the MRF statement.

## 5. Notes, traps and limits

* **The anchor direction matters more than the model.** In case B (classical rotating channel) the *velocity* field is unchanged by rotation, so a rotating-laminar-channel comparison cannot detect a missing Coriolis term at all -- only the pressure can. Testing MRF with case B alone would have produced a green suite over a broken operator; that is why case A (secondary flow) is the primary anchor.
* **The mirror-ghost boundary row is a silent trap.** Writing the wall no-slip as a mirror ghost `u_{-1} = -u_0` gives `(u_1 - 3u_0)/dy^2`, which is *not* exact for a parabola and degrades the boundary row to first order; measured effect: 1.3e-5 relative error on the parabola anchor at N=320 instead of round-off (1.3e-14 after switching to the one-sided three-point row). The quadratic anchor is exactly what exposes it -- a decaying-error plot would have hidden it.
* **Two regimes in case A.** For small $\omega$ the secondary flow is linear in $\omega$ ($u$ changes only at $O(\omega^2)$, measured: 100x for a 10x $\omega$), so a weakly rotating run cannot be used to argue Coriolis is negligible. For large $\omega$, $u \sim 1/\omega$ and $|w|/|u|$ saturates near 3.31. The grid requirement is set by $\sqrt{\nu/(2\omega)}$, not by `h`.
* **What is NOT delivered.** No 3-D time-accurate rotating-frame field solver, no sliding/mixing plane, no rotating-machinery *experiment* comparison (propeller open-water curves). The operators are 3-D and ready to be coupled, the anchors are 1-D reduced systems. Recorded as open in `docs/VERIFICATION_DEBT_LEDGER.md` (D10).

## 6. Error judgement vs the 3 % criterion

| key quantity | reference | relative error | inside 3 pct |
|---|---|---|---|
| case A streamwise $u$ (N=640) | closed form | 1.27e-05 | **yes** |
| case A secondary flow $w$ (N=640) | closed form | 2.11e-06 | **yes** |
| case A closed form vs independent BVP | collocation | 9.12e-15 | **yes** |
| case B parabolic $u$ (N=320) | closed form | 1.33e-14 | **yes** |
| case C absolute $v_\theta$ (n=320) | closed form | 8.23e-07 | **yes** |
| case C torque $M/L$ (n=320) | Taylor closed form | 3.05e-06 | **yes** |

All anchored quantities are far inside the 3 pct band (worst case 1.27e-05). Quantities that are *not* judged because no independent reference exists here are reported as WITHHELD rather than passed: the rotating-frame pressure field itself for case A (only its gradient enters the balance, anchored via C7b) and any rotating-machinery experimental comparison.
