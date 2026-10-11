# Three-dimensional Cartesian / body-fitted overset verification

## Scope and method

The background is a native Cartesian hexahedral grid on [-3,3]^3. The component grid has radially structured wedge layers between radii 1 and 2.6, using an icosphere angular triangulation. Planar facets approximate the spherical wall. A radius-1 solid is excluded. Background centroid classification is hole below radius 1.5, fringe below 1.8, otherwise active; shell centres beyond radius 2.1 are fringe. Background cells whose Cartesian boxes intersect the radius-1 solid additionally become HOLE by an exact closest-point sphere test. The remaining split/fringe masks use centroid radii. This is not a cut-cell method or a moving-grid calculation.

Both transfer directions require a containing active native polyhedron and a convex tetrahedron of active donor centres. A tangent separating half-space protects the entire interpolation tetrahedron from the sphere. No nearest-neighbour fallback or orphan suppression is used. Weights reproduce constants and affine fields. The independently prescribed nonlinear vector field is fixed across all levels; formulas and coordinates are in config.json. Angular subdivision levels are 1, 2 and 2: the final refinement changes Cartesian and radial resolution without further angular refinement. This is a joint-grid interpolation diagnostic, not a uniform-h order study or Navier–Stokes spatial convergence. Errors use receiver-cell-volume-weighted relative L2 on fringe point values. These weights define a diagnostic norm, not conservative remapping of volume averages.

## Results

|Level|Background cells|Shell cells|Shell → background L2|Background → shell L2|Time s|
|---|---:|---:|---:|---:|---:|
|1|failed|failed|ValueError: orphan receiver 663: no solid-safe convex donor tetrahedron|—|5.87|
|2|5832|3840|0.125316%|0.131065%|20.72|
|3|10648|4480|0.106011%|0.084288%|29.93|

Three-level error decrease: **False**. Constants/affine reproduction and active-donor invariants: **False**. Fine-level two-way interpolation errors below 3%: **False**. Full raw metrics, both donor maps, native geometry, classifications and fields are archived for independent replay.

### Level 2 real geometry and interpolation diagnostic

![Actual mesh classification](level-2/mesh-classification.png)

![Fringe interpolation errors](level-2/fringe-interpolation-error.png)

### Level 3 real geometry and interpolation diagnostic

![Actual mesh classification](level-3/mesh-classification.png)

![Fringe interpolation errors](level-3/fringe-interpolation-error.png)

## Limits and reproduction

No Navier–Stokes equations, pressure correction, turbulence, LBM populations, time stepping, moving-solid hole cutting or conservative shared interface flux are solved in this verification. The error plots show point interpolation of a prescribed analytic vector field; they are not velocity or pressure flow contours. The generic native mesh adapter uses a sphere-specific solid-clearance contract; a SUBOFF solid-clearance adapter remains necessary. FVM/LBM coupling requires common units and physical coordinates, shared time advancement, consistent pressure/density conversion, population reconstruction and conservative interface flux. Those are separate acceptance gates.

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_overset3d.py --output EMPTY_DIRECTORY`. Producer sources were copied before execution. Every archived output except the manifest itself has a SHA-256 binding in manifest.json.

## Actual radial layer distribution

Radial mode: geometric growth fitted to exact shell thickness. First-layer height is a geometry input; no wall stress, y+ or turbulence resolution is inferred. Per-level wall-layer-geometry.json records every radius/height, growth ratio, outer spacing relative to Cartesian spacing and actual mesh quality.

|Level|First height|Outer height|Outer/background spacing|Maximum nonorthogonality (degrees)|
|---|---:|---:|---:|---:|
|2|0.0155556|0.427711|1.28313|8.3579|
|3|0.0127273|0.376908|1.38199|8.3579|

Reproduce the recorded radial mode with `--first-height 0.02` when set; omit it for uniform spacing.
