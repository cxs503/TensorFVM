Tensor ecosystem integration
============================

TensorFVM is an independently usable finite-volume fluid solver. This document
defines its intended boundary with the shared Tensor ecosystem; it does not
claim that every proposed adapter or coupled physical model is already implemented.

Responsibilities
----------------

* Own finite-volume discretization, SIMPLE pressure--velocity coupling, mesh
  handling, boundary conditions, solver configuration and FVM-specific
  verification/benchmark cases.
* Export documented fields, forces and run artifacts for downstream workflows.
* Keep solver-native examples runnable without requiring the shared platform.

Boundaries with sibling projects
--------------------------------

* **TensorPhysics** coordinates workflows, solver adapters, multiphysics timing,
  execution and provenance; it must not duplicate FVM numerical kernels.
* **TensorPlatform** provides shared dataset/catalog, lineage, permissions and
  AI model lifecycle services.
* **TensorLBM** is a separate fluid solver family. Similar output quantities do
  not imply identical discretization, boundary treatment or validation.
* **TensorFEM** and **TensorDEM** exchange structural fields or particle loads
  only through explicit, documented coupling interfaces.

Coupling requirements
---------------------

Each exchanged field must declare units, coordinate frame, mesh location,
shape, physical time, sign convention and interpolation/integration method.
Fluid-to-structure or fluid-to-particle transfer should test resultant force and
moment conservation where applicable. A runnable coupling fixture is not, by
itself, evidence of validated engineering accuracy.

Recommended integration path
----------------------------

#. Keep standalone FVM benchmark cases independent of platform services.
#. Define a versioned adapter for configuration, stepping, diagnostics and
   artifact export.
#. Register run manifests and artifact references with TensorPhysics.
#. Register datasets and derived AI experiments with TensorPlatform.
#. Add multiphysics use cases only after solver-specific and interface tests pass.

Documentation convention
------------------------

This repository uses reStructuredText and Sphinx-style documentation. Keep
README.rst and existing .rst cross-references intact unless a deliberate,
tested documentation migration is approved. Ecosystem documentation should
follow the same convention; do not mechanically convert files to Markdown.

Repository map
--------------

* `TensorPlatform <https://github.com/cxs503/TensorPlatform>`_ — shared data and AI services.
* `TensorPhysics <https://github.com/cxs503/TensorPhysics>`_ — scientific workflow orchestration and multiphysics coordination.
* `TensorLBM <https://github.com/cxs503/TensorLBM>`_ — lattice Boltzmann solver.
* `TensorFVM <https://github.com/cxs503/TensorFVM>`_ — finite-volume solver.
* `TensorFEM <https://github.com/cxs503/TensorFEM>`_ — finite-element solver.
* `TensorDEM <https://github.com/cxs503/TensorDEM>`_ — discrete-element solver.
