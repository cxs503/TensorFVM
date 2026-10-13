# SUBOFF structured near-wall / unstructured exterior mesh

## Geometry and use

Bare AFF-1 mother profile from TensorLBM (source SHA256 `435bd062a90d6881784734fa614f033e880b310111200436e8b13950fbccae02`). The original experiment offset table remains independently unverified. Length 4.35610010 m. Surface coordinates use cosine-spaced axial stations with the nearest interior samples replaced by the three exact polynomial junctions (avoiding nearly coincident points), uniform periodic azimuth, and finite-difference meridian normals. Both tips close on the axis using wedges, rather than repeated degenerate hexahedra. Normal layer distances are measured from the same original surface, without accumulating marching-normal drift.

Actual mesh: **576 near-wall hexes**, **72 cap wedges**, **2779 exterior tetrahedra**. 3 layers, first height 0.00129254 L = 0.00563044 m, growth 1.129243234657234, total thickness 0.00440037 L. These are geometry-development parameters, not a Reynolds/y+ design.

Optional exterior tail sizing: `null`. This prescribes a continuous Gmsh Ball size field around the outer tail tip; actual neighbor volume ratios and flow errors require separate checks. The structured layers and imposed interface are retained.

Gmsh 4.15.2 meshes the exterior of the imposed layer skin inside a [-1,3] × [-1,1] × [-1,1] L box. This is genuine unstructured tetrahedral generation. Every outer structured quad is divided on its smallest-vertex-ID diagonal. The exact triangles are imposed on Gmsh and shared in the finite-volume importer: **408 verified internal triangles**. No overset interpolation or pyramid transition is used. The native volume mesh retains hex/wedge cell types; only FV integration faces are triangulated.

```python
from tensorfvm.native_mesh3d import NativeMesh3D
mesh = NativeMesh3D.from_gmsh('suboff-hybrid.msh', triangulate_faces=True)
```

The opt-in mode treats warped quads as two planar facets, not as bilinear curved faces. File node tags are sorted before triangulation to preserve diagonal choices across round-trips. The default planar-face importer remains strict.

## Mesh evidence

![Actual structured wall and native facet sections](mesh.png)

Green: hex facets; blue: cap-wedge facets; gray: exterior tetrahedral facets. The lower panels intersect actual native facets with z=0; they do not project off-plane edges. Displayed diagonals reflect the actual FV face representation. No pressure or velocity solution is shown.

## Verification

- Minimum volume: 1.53476e-08 L³.
- Maximum relative cell closure: 1.66944e-16.
- Minimum S·d: 6.7662e-09.
- Maximum nonorthogonality: 87.8557 degrees.
- Maximum of affine reconstruction / gradient / divergence audit errors: 9.14602e-13.
- Arbitrary internal interface flux global cancellation: 1.77636e-15.
- Fluid + faceted body volume vs box-volume error: 0 L³.
- Sampled wall-node radial error (including explicitly closed tip): 1.38778e-17 L.
- Geometry gates: **True**; numerical/physical flow qualified: **False**.

Full per-metric values are in `summary.json`; `.msh` and `native-geometry.npz` retain native geometry. Source snapshots and SHA256 manifest bind this execution. The PNG/PDF are mesh figures, not a flow report.

## Limits and next gates

This is a thin-layer mesh feasibility result. Surface facet chord error, normal-offset discretization, high aspect ratio, layer/exterior size jump and tail quality need refinement studies before viscous use. Maximum nonorthogonality alone does not certify suitability for turbulence. Fixed outer-domain dimensions are not a domain-independence study. No attachments, y+, wall model, viscous momentum solution, GPU qualification, force/Cp convergence or <3% claim. Layer thickness >0.02 L is rejected pending offset collision controls; candidate meshes fail immediately on invalid geometric projections.

TensorLBM `dg_curv.py` motivated layer height/growth and local-normal geometry, but its DG geometric factors and exterior coupling are not copied into this FVM geometry. SUBOFF DG-band tests do not certify this mesh.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_hybrid_mesh.py --output NEW_EMPTY_DIRECTORY --axial 18 --azimuthal 12 --layers 3 --first-height 0.00129254088520279 --growth 1.129243234657234 --far-size 0.5333333333333333
```
