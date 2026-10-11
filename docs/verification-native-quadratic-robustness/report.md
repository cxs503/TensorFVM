# Native quadratic FVM: coarse-grid robustness audit

## Scope

These are independent 27-cell (108-unknown) mathematical/solver probes using the fixed skew-box ABC problem: amplitude 0.5, wavenumber 0.7, viscosity 0.1, density 1. No interior analytic field initializes the quadratic solver. The source is the original numerical kernel archived by the ABC run, not the subsequently edited production API. This directory was archived **after completion**, and does not claim that these probe scripts or their copied source tree were snapshotted at probe launch.

The original production zero-start n=3 run **failed** and remains failed. A numerical warm start also fails with the unmodified solver. An independent backtracking experiment succeeds on the same equations; this is not production qualification or a SUBOFF drag/Cp benchmark. The separate 6/12/24 study retains its own acceptance scope.

## Original failure and matrix diagnosis

The original zero-start run completes three nonlinear iterations before GMRES fails. Its third momentum residual is 1.9410434e10, current reconstructed divergence 6529.2502, and face-flux inconsistency 706.4908: nonlinear divergence precedes the linear failure.

The initial corrected matrix has rank 108, condition 373.443857, and direct original-action relative residual 2.148321e-15. GMRES with fresh baseline LU needs 19 iterations for either restart 60 or 108.

The captured failed matrix also has rank 108, condition 37984.221952; it is not singular. The fresh-preconditioned matrix is strongly nonnormal (reported commutator norm ratio 0.916480).

| Preconditioner | Restart | Iterations | Original-action true relative residual | Status |
|---|---:|---:|---:|---|
| cached | 60 | 1800 | 1.737323e-04 | failed |
| cached | 108 | 210 | 2.541834e-12 | passed |
| fresh | 60 | 46 | 3.358548e-12 | passed |
| fresh | 108 | 46 | 3.358548e-12 | passed |

Dense direct solve has an explicit-matrix residual 4.73e-16, but replay through the original affine-difference action gives 1.175205e-06. At this enormous nonlinear state, subtracting the frozen affine source introduces action linearity error 1.371852e-09, and unit-basis explicit-matrix/action discrepancy 3.766533e-09. Linear refresh alone does not fix nonlinear robustness.

## Numerical warm start: preserved failure

The original linear-reconstruction solver starts from zero and converges with unchanged tolerances (velocity error 0.32181156%, pressure 24.72296072%). Only its computed velocity/pressure initialize the quadratic solver. With original relaxation and first-step policy, the quadratic run completes six iterations before failing:

`Quadratic current-equation GMRES failed: info=30, true=0.0020906107928477945, iterations=1800`

Histories, raw baseline fields and failed quadratic fields are preserved. No failed field is treated as an accepted solution.

## Independent residual-decreasing backtracking experiment

The same numerical warm start, analytic force and boundary data are used. The existing quadratic linear solve remains unchanged. Every update starts with omega=0.8, including the first; up to 16 trials halve omega and require Armijo decrease (coefficient 1e-4) in a norm comprising the actual corrected momentum residual, volume-weighted stored divergence and RMS face-flux inconsistency. Each trial replays the current quadratic equations; reference-field errors never select a trial. The normal production convergence gates still require every state change, current momentum residual, stored and reconstructed divergence and face-flux consistency below 1e-9. Linear tolerance remains 1e-11. No pressure gauge fitting or acceptance-limit adjustment occurs.

This independent experiment converges in 52 iterations, 8.847628 seconds. Errors are original centroid point-value, volume-weighted relative L2; pressure removes independent volume means, as in the original benchmark.

| Metric | Error |
|---|---:|
| velocity | 0.02587374% |
| pressure | 1.50125541% |
| velocity_x | 0.02691746% |
| velocity_y | 0.02666529% |
| velocity_z | 0.02391845% |

All reported metrics are below 3%, and the nonlinear gates pass. **This is a successful standalone algorithm probe, not evidence that the original production solver now handles n=3.** There is no mesh-convergence claim from one grid and no physical SUBOFF qualification.

## Zero-start backtracking: preserved failure

The same backtracking algorithm was also tested with zero velocity and constant pressure gauge, without the numerical warm start. Three updates decrease the actual merit (3.496684 → 2.142110 → 2.034398 → 2.030469), requiring omega 0.8, 0.025 and 0.0015625. On iteration four, every one of 16 trials fails the decrease condition, down to omega 2.44140625e-5. The rejected trial restores the last accepted state. The run stops as **not converged**, with velocity error 76.6398% and pressure error 4218.5201%; no failed iterate is qualified.

Thus the successful standalone algorithm requires the combination of an explicitly documented, converged numerical baseline warm start and residual-decreasing backtracking. It does not demonstrate robust default zero-start behavior. A production implementation should preserve the numerical initialization's complete solver history and raw fields; it must not initialize from analytic interior truth.

## Evidence and reproduction

The copied source tree is byte-identical to the previously frozen producer kernel. Probe scripts preserve their executed bytes and original absolute snapshot paths. To reproduce in the same workspace, run the archived Python scripts from the repository root with `/data/TensorFEM/.venv/bin/python` and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. They write diagnostic output under `/tmp`. The warm-start script must run before the linesearch script, which loads its computed baseline NPZ; both perform genuine numerical solves. Matrix scripts use the original failed solver kernel and all recorded tolerances.

`manifest.json` binds all archived scripts, configurations, raw NPZ matrices/fields, complete histories/logs and the copied source tree. The runtime package versions are recorded there. No production file was changed by these probes.
