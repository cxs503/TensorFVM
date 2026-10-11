# Body-fitted component mesh and Cartesian background

Current scope follows the corrected user request: mesh generation, hole cutting,
fringe placement and bidirectional overset connectivity. Turbulence-model
selection, physical sphere/cube flow and new FVM–LBM solver coupling follow
these geometric gates.

## Implemented geometry inputs

The spherical component keeps radial structured wedge layers. The generator
accepts an explicit first radial height, fits a geometric progression to the
outer radius, and rejects invalid or unrepresentable spacing. It does not
silently change the requested first height. Surface angular subdivision,
radial layer count, outer extent and Cartesian cell count remain separate inputs.
Outer-layer/background spacing ratios and actual native mesh quality are saved.
The fitted growth ratio is measured, not a turbulence resolution guarantee.

`wall_geometry='circumscribed'` scales the wall out until every triangular facet
plane excludes the analytic sphere. Thin fluid layers then stay outside the
analytic solid. This is a conservative planar approximation with a measured
radial geometry error, not exact curved geometry. The first height is measured
between radial vertex levels, not assumed to equal wall-normal cell-centre distance.

## Two failed layouts retained

1. [Original graded layout](../verification-overset3d-graded/report.md): the
   uniform mesh split radius produced a coarse orphan receiver. No nearest-point
   extrapolation or orphan suppression was introduced.
2. [Adjusted overlap with inscribed wall](../verification-overset3d-graded-overlap/report.md):
   bidirectional maps were constructed, but nonlinear errors were not monotonic.
   Thin layers also had centres inside the analytic sphere (180/320/320 body HOLE
   cells), so this layout does not preserve the intended wall layers.

Changing overlap alone is insufficient to repair the inscribed thin-wall issue.
The corrected profile uses an explicitly circumscribed wall and angular
subdivisions 2/2/3. Raw failed outputs and producer snapshots are retained.

## Geometric acceptance before flow

- Positive native volumes, facet closure and face projection.
- Measured wall geometry deviation; no body HOLE cells erasing the wall layers.
- Every fringe receiver has an actual containing ACTIVE donor polyhedron.
- Only ACTIVE donor centres; nonnegative normalized weights.
- Entire donor tetrahedron is separated from the analytic solid.
- Constant/affine reproduction; three-level nonlinear interpolation refinement.
- Two-direction point interpolation errors below 3% on the finest grid.

These criteria qualify geometry and point connectivity only. They do not establish
conservative transfer of cell averages or Navier–Stokes accuracy.

## Next geometry implementation

Reuse SUBOFF's structured shell as the component block, without requiring the
outer tetrahedral fill. Add a SUBOFF solid adapter with verified whole-cell
blanking and entire donor-tetrahedron clearance. A nearest surface distance or
an edge-only test is insufficient. Check thin layers, nose/tail closure, local
spacing jumps and donor coverage before accepting the generalized geometry.

The standalone FVM overset solver then needs HOLE-free reconstruction stencils,
a single global pressure gauge, fringe constraints, and conservation/pressure
correction. Existing SI exchange contracts remain preparatory; LBM execution is
not part of this mesh milestone.

## Completed corrected profile

[Three-level geometry/connectivity report](../verification-overset3d-graded-circumscribed/report.md)
and [actual overlaid meshes with near-wall zoom](../verification-overset3d-graded-circumscribed/wall-layer-plots/report.md).

| Level | Cartesian cells | Body-fitted cells | Radial layers | First radial height | Body → background L2 | Background → body L2 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2,744 | 3,200 | 10 | 0.020000 | 0.100536% | 0.203154% |
| 2 | 5,832 | 3,840 | 12 | 0.015556 | 0.091827% | 0.127516% |
| 3 | 10,648 | 17,920 | 14 | 0.012727 | 0.037491% | 0.079507% |

All three levels preserve the complete body layers with zero body HOLE cells.
Maximum wall radial approximation errors are 1.8074%, 1.8074%, 0.4549%.
Both interpolation errors decrease; constants and affine fields reproduce to
roundoff. No orphan suppression, inactive donor use or extrapolation was added.
[Independent raw-array audit](../verification-overset3d-graded-circumscribed/independent-audit/result.json)
replays native containing cells, entire donor tetrahedron clearance, wall facet
planes, first height, wall radial error and nonlinear interpolation.
[Targeted regression](graded-mesh-tests.txt): 13 tests passed, including rejection
of the failed layout and preservation of thin circumscribed wall layers.

Reproduce with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python scripts/verify_overset3d.py --first-height .02 --split-radius 1.4 \
  --fringe-width .3 --wall-geometry circumscribed --angular-levels 2 2 3 \
  --output EMPTY_DIRECTORY
```

This is a spherical static geometric milestone. SUBOFF solid clearance, moving
overset, conservative remapping and fluid-equation acceptance remain separate work.
