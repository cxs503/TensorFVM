NativeViscous3D wall-closure re-audit
=====================================

Scope
-----
``tensorfvm.native_viscous3d.NativeViscous3D`` supplies the shared-face diffusion
operator and the symmetric viscous traction matrix used by
``native_collocated3d``, ``native_rans3d`` and ``native_sst3d``. This page records
an independent re-test of the accuracy claims made by the earlier audit
(mission m2530), whose harness never reached this repository.

Library
-------
``src/tensorfvm/verification/native_viscous_order3d.py``

* ``cube_mesh(divisions, warp=...)``: orthogonal or interior-warped unit cube of
  hexahedra with triangulated faces.
* ``scalar_field(points, mode)``: analytic manufactured scalar field plus its
  exact gradient. ``trig3d`` is the multi-dimensional field of the m2530 audit,
  ``trig1d`` is a one-dimensional control that is exact in y and z.
* ``wall_closure_metrics(mesh, mu, mode=...)``: norm-based wall flux error,
  per-face mean relative normal-derivative error (plain and denominator-floored),
  and the constant-mode residual of the assembled fixed-value matrix.
* ``affine_exactness(mesh)``: negative control; must stay at machine precision.
* ``family(...)`` / ``orders(...)``: refinement families and observed orders.

Reproduce
---------
::

    PYTHONPATH=src python scripts/audit_native_viscous_order3d_wall_closure.py --publish-path /tmp/report.json
    PYTHONPATH=src python -m pytest tests/test_native_viscous_order3d.py -q

Artefacts land in ``scripts/artifacts/native_viscous_order3d/`` (``results.json``,
``convergence.png``, ``wall_error_map.png``, ``report_body.md``).

What it establishes
-------------------
1. On a genuinely multi-dimensional field the wall normal-derivative flux
   converges at order ~1.1-1.3, so the m2530 first-order verdict stands; its
   concrete 6.92% / 3.55% entries are *not* reproduced under a norm-based
   metric (2.41% at 8x8x8, 1.54% at 12x12x12), and the 4x4x4 base mesh
   already sits at 5.92%.
2. The same closure is second order (order ~2.0) on the one-dimensional
   control, so the defect is inherently transverse and cannot be repaired by a
   curvature term along the centre-to-face vector. A ``d.H.d/2`` Dirichlet-face
   correction with a least-squares cell Hessian was implemented during this
   audit, measured (1-D unchanged, coarse-grid error ten times worse because the
   Hessian is only O(h) accurate, 3-D rate unchanged) and deliberately not
   merged.
3. The constant mode of the assembled operator decays with refinement
   (0.385 -> 0.236 -> 0.181). The Dirichlet datum of this closure enters only
   through the right-hand side, so ``scalar_diffusion_matrix`` and
   ``momentum_diffusion_matrix`` describe the zero-jump (mirror) operator and
   become singular as h -> 0. A fixed-value linear system assembled from them
   leaves the constant mode unconstrained, which is why the earlier
   "Dirichlet diffusion solution has no convergence order" experiment cannot
   measure the closure; that row should be read as a conditioning result.

Known limitation / follow-up
----------------------------
The wall reconstruction needs a least-squares quadratic gradient at the wall to
reach second order in three dimensions, and the diffusion matrices need an
explicit Dirichlet anchoring term before they are used as linear systems. Both
are structural changes and are not part of this audited snapshot.