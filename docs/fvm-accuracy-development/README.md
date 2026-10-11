# Standalone FVM accuracy first

Development priority is standalone FVM, followed by FVM–LBM. New LBM coupling
work is deferred until the corresponding FVM case passes its own accuracy and
conservation gates. Existing state conversion contracts are preparatory only.

## Acceptance order

1. Verify independent three-dimensional nonlinear momentum and pressure fields,
   including every velocity component, true current residuals and face fluxes.
2. Verify mesh convergence on actual body-fitted/mixed meshes. SUBOFF's original linear-reconstruction baseline
   manufactured pressure errors remain 5.4999% and 4.2777%; both fail 3%.
3. Validate single-domain no-slip laminar cases, wall-resolved/wall-function
   turbulence and physical drag/Cp against independent references, with mesh
   sensitivity. Complete the SUBOFF body-fitted baseline before adding overset.
4. Verify pure FVM overset Stokes and nonlinear velocity/pressure equations,
   excluding HOLE values from gradients and setting only one global gauge.
   Current 3D overset results verify interpolation, not a fluid solver.
5. Start FVM–LBM joint time advancement after these FVM gates pass.

## Current quadratic FVM development

`native_quadratic_flow3d.py` integrates quadratic pressure area means, shared
symmetric stress and consistent Rhie–Chow correction into the actual equations.
The legacy matrix-free iteration uses a baseline LU preconditioner and lags
part of the baseline high-order advection source. Final acceptance replays the
complete current equations. The new opt-in `assembled` path builds the full
frozen-flux quadratic operator, including the complete linear advection term;
only the nonlinear velocity covariance is frozen for each nonlinear update.
Independent prescribed boundary velocity means are distinct
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

## Actual SUBOFF quadratic solve

[The first mixed-mesh quadratic attempt](../verification-native-suboff-mms/quadratic-coarse-attempt-01/failure-explanation.md)
failed its first current-equation GMRES solve after 1,800 inner iterations,
true residual 1.78421e-5. Zero nonlinear updates completed. Saved velocity and
pressure are exactly the copied baseline numerical initial fields; the pressure
error remains 5.49992%. The already-fresh baseline LU is not adequate here.
The new full sparse quadratic operator is checked against the original actual
equation residual on both warped triangular faces and planar polygon faces,
with mixed hex/tet control volumes and nonzero boundary sources. On the original
11,648-cell SUBOFF mesh, its numerical warm-start run converges in 15 updates:
velocity vector **0.0112074%**, all velocity components below **0.015405%**,
pressure **0.524758%**. Independent frozen-producer replay passes 19/19 gates.
The wall geometry and original triangulated faces remain unchanged in this run.
Backtracking alone cannot solve an inner linear failure. This failed attempt
is preserved and does not count as accepted.

[Single-domain report, real mesh and field/reference comparison](../verification-native-suboff-mms/single-domain-report.md)
separates this success from the smaller merged-face zero-start failure.
No actual SUBOFF drag/Cp, turbulence or mesh convergence qualification follows
from one manufactured run. [Full regression](pytest-single-domain.txt): 375 tests
and 45 subtests pass; [additional provenance checks](pytest-single-domain-audit.txt):
8 targeted tests pass after binding the initial producer convergence ledger.

## Single-domain SUBOFF face geometry

The immediate mesh is a complete conforming fluid domain: structured wall
layers, wedge nose/tail caps and external Gmsh tetrahedra. It contains no
overset interpolation. A new 6,085-cell mesh provides a smaller development
level alongside the existing 11,648/24,284-cell meshes.

Optional `--merge-coplanar-faces` restores two compatible triangles to a planar
quad only when owner, neighbor and boundary patch agree. Different exterior
tetrahedron neighbors keep their separate interface triangles. Cell geometry,
volumes and conservative area sums are preserved. Nonplanar or perpendicular
pairs are rejected. Quadratic face moments and manufactured boundary fluxes
are integrated over the actual polygon.

On the original 11,648-cell mesh, the wall maximum cell-to-face normal angle
changes from 88.774 degrees to 12.498 degrees (including tip triangles);
the 95th percentile becomes 3.932 degrees. Much of the original wall metric
came from offset triangle centroids on planar parent quads. This geometric
improvement alone does not establish pressure accuracy or physical validity.

On the separate 6,085-cell merged-face mesh, zero initialization fails at update
17 and remains archived. Initialization from its 100-step linear numerical field
(explicitly recorded as not yet converged) then converges in 16 quadratic updates:
velocity **0.0158491%**, pressure **0.772845%**, 19/19 independent audit gates pass.
The two successful SUBOFF cases have different face policies; they do not form
a certified same-configuration mesh refinement series.

The actual centroid comparison also exposes a local tail pressure hotspot in
the 6,085-cell result. Small cell volumes can hide large local errors in the
global volume-weighted L2 norm. Local pressure and wall Cp are not qualified;
regional error checks and a fixed physical pressure-scale maximum-error check
must precede any claim that SUBOFF is accurate or ready for overset flow.

## Larger standalone SUBOFF grids and current limits

[The refinement report](../verification-native-suboff-mms/mesh-refinement-report.md)
now compares converged 6,085 / 11,648 / 24,284-cell triangulated-face quadratic
runs with unchanged continuum forcing and convergence tolerances. Pressure L2
decreases **0.759395% → 0.524758% → 0.400184%**. The latest independent audit
passes **19/19** gates; velocity-vector error is **0.0076091%**.

Maximum pressure error divided by the fixed MMS scale `rho U²` decreases to
**2.54578%**, with zero cells above that scale's 3% threshold. Wall-owner relative
pressure L2 remains **8.36176%**, the fixed near-body window **4.06459%**, and
the fixed tail window **23.4110%**. These denominators differ; the case is not
declared locally or physically qualified. Refinement helps, but does not yet
resolve all pressure accuracy gates.

New **61,282 / 172,768-cell** conforming meshes pass geometry and prescribed-P2
gradient/Hessian reconstruction checks. Neither has a full NS solution yet.
The 24,284-cell solve took **3995.77 s**, with **17.465 GiB** observed peak RSS;
3594.30 s was spent in its first full linear solve including assembly. Evaluate
sparse ordering and reusable preconditioning before advancing full flow on
61,282 cells, and improve the tail tetrahedron transition. A separate Schur
linear probe passes its true residual gate in 583 iterations; that does not
enable or qualify the nonlinear production path.

The historical/current frozen-source probe gives identical face fields, stress,
current equation residuals, consistent fluxes and one random full-matrix action
on the archived 11,648-cell case. Numerical hashes differ only in reviewed
branches/guards; the comparison tool requires an explicit bound compatibility
probe for mixed source versions. [Targeted checks](pytest-suboff-refinement.txt):
**19 passed**. Numerical discretization source was not changed in this update.
