# Standalone FVM accuracy first

Development priority is standalone FVM, followed by FVM–LBM. New LBM coupling
work is deferred until the corresponding FVM case passes its own accuracy and
conservation gates. Existing state conversion contracts are preparatory only.

## Acceptance order

1. Verify independent three-dimensional nonlinear momentum and pressure fields,
   including every velocity component, true current residuals and face fluxes.
2. Verify mesh convergence on actual body-fitted/mixed meshes. SUBOFF's baseline
   manufactured pressure errors remain 5.4999% and 4.2777%; both fail 3%.
3. Verify pure FVM overset Stokes and nonlinear velocity/pressure equations,
   excluding HOLE values from gradients and setting only one global gauge.
   Current 3D overset results verify interpolation, not a fluid solver.
4. Validate no-slip laminar cases, wall-resolved/wall-function turbulence and
   physical drag/Cp against independent references, with mesh sensitivity.
5. Start FVM–LBM joint time advancement after these FVM gates pass.

## Current quadratic FVM development

`native_quadratic_flow3d.py` integrates quadratic pressure area means, shared
symmetric stress and consistent Rhie–Chow correction into the actual equations.
Matrix-free GMRES solves the corrected operator; the baseline LU is only a
preconditioner. Independent prescribed boundary velocity means are distinct
from face-centroid values. The internal advection covariance is a second-order
approximation, not exact integration of arbitrary nonlinear velocity fields.
Interior forcing remains the independently derived analytic pointwise force.

Residual-decreasing backtracking records rejected trials and preserves the old
state when no trial decreases. A solution at roundoff can be accepted only if
all original residual and field-change gates pass. The nonlinear coarse case
requires explicitly recorded initialization from an independently converged
baseline numerical solve; this is not initialization from analytic truth.
Mixed velocity/pressure outlets, bounded Newton linearization and physical SST
are not supported by this new quadratic path yet. Their absence is a concrete
remaining gate, not experimental validation.

[Failure and robustness evidence](../verification-native-quadratic-robustness/report.md)
retains zero-start failure, failed numerical warm start without backtracking,
and independent successful warm-start/backtracking evidence. Do not transfer
one case's acceptance to another mesh or physical operating condition.

## Completed standalone verification

[The production three-grid quadratic ABC report](../verification-native-abc3d-quadratic-warmstart/quadratic-report/report.md)
passes 84/84 report gates. Actual pressure errors are 1.501255%, 0.559752%
and 0.143199%; velocity vector errors are 0.025874%, 0.001804% and
0.000387%. Every velocity component also passes 3% on every grid. All three
nonlinear solves pass the original residual and field-change tolerances.
Initialization uses archived baseline numerical solutions from zero.

[The original zero-start series](../verification-native-abc3d-quadratic/quadratic-report/report.md)
is explicitly failed overall because its coarse solve failed. Fine-grid success
is not substituted for that failed series. The numerical warm start and
residual-decreasing backtracking are part of the accepted configuration.

[Final regression log](pytest-final.txt): 321 tests and 45 subtests passed;
three existing PyTorch warnings. [Current-equation replay](current-equation-replay.json)
rebuilds the frozen producer operators and evaluates all four original equation
residuals from raw numerical fields independently of saved solver histories.
These results qualify this manufactured configuration only.

The separate zero-start 6/12/24 refinement run also completed on its frozen
pre-backtracking producer. At 13,824 control volumes its actual pressure error
is 0.0375142%, velocity vector error 0.0000610359%, with convergence after
18 nonlinear iterations (818.67 s for the finest run). This does not resolve
the original 27-cell zero-start failure or qualify the SUBOFF mesh.
[Refinement report](../verification-native-abc3d-quadratic-refinement/quadratic-report/report.md)
and [independent residual replay](refinement-current-equation-replay.json)
retain this distinct configuration.

## Actual SUBOFF quadratic blocker

[The first mixed-mesh quadratic attempt](../verification-native-suboff-mms/quadratic-coarse-attempt-01/failure-explanation.md)
failed its first current-equation GMRES solve after 1,800 inner iterations,
true residual 1.78421e-5. Zero nonlinear updates completed. Saved velocity and
pressure are exactly the copied baseline numerical initial fields; the pressure
error remains 5.49992%. The already-fresh baseline LU is not adequate here.
A quadratic-operator block/Schur preconditioner and mixed-mesh robustness checks
precede another costly production run. Backtracking alone cannot solve an inner
linear failure. This failed attempt is preserved and does not count as accepted.
