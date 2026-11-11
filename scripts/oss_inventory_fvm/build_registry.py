"""Build the TensorFVM open-source FVM inventory registry (machine-readable).

Two evidence channels are kept strictly separate and both are recorded per
entry:

``metadata``
    Measured, in this run, from the public GitHub REST API (stars, SPDX licence,
    last push, language, archived flag).  ``metadata.source == "github_api"``
    means measured; ``"unresolved"`` / ``"commercial"`` means we could not (or
    may not) measure it and the fields stay ``null``.

``curation``
    Expert annotation (family, discretisation, capability tags, tier, gap
    targets).  ``curation.capability_evidence == "curated_knowledge"`` states up
    front that these tags were NOT re-derived by reading the upstream sources in
    this run; the accompanying report lists which claims need a source-reading
    pass before adoption.  Nothing here is reported as verified capability.

Usage::

    python scripts/oss_inventory_fvm/build_registry.py            # fetch + write
    python scripts/oss_inventory_fvm/build_registry.py --offline  # no network

The script never writes anything outside ``docs/oss-inventory-fvm/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "oss-inventory-fvm" / "registry.json"
API = "https://api.github.com/repos/{slug}"
TIERS = {
    "T1": "absorb now: directly closes a P0/P1 TensorFVM gap; licence allows "
          "reading and re-implementing; upstream implementation is well tested",
    "T2": "reference design: same paradigm, read for algorithm/validation "
          "design and cross-check numbers, not scheduled for porting",
    "T3": "context only: non-FVM paradigm, closed source, or licence forbids "
          "code reuse; kept for the state of the art and benchmark provenance",
}

# ---------------------------------------------------------------------------
# curated corpus (28 projects).  ``slugs`` are tried in order; first hit wins.
# ---------------------------------------------------------------------------
PROJECTS = [
    {
        "id": "openfoam-foundation",
        "name": "OpenFOAM (Foundation, OpenFOAM-dev)",
        "slugs": ["OpenFOAM/OpenFOAM-dev"],
        "family": "OpenFOAM lineage",
        "discretisation": "unstructured collocated FVM, polyhedral cells",
        "capabilities": [
            "unstructured_mesh", "vof_interface_compression", "rhoCentralFoam_kt",
            "mrf_sliding_ami", "dynamic_amr", "adjoint", "les_ras", "mpi_decomposition",
            "implicit_steady", "combustion_species",
        ],
        "tier": "T1",
        "tier_rationale": "closest paradigm match to TensorFVM (collocated FVM, "
                          "VOF, compressible) and the richest set of the missing "
                          "P0/P1 mechanisms (AMI/MRF, dynamic AMR, implicit steady)",
        "gap_targets": ["GAP-FVM-MRF-SLIDING-AMI", "GAP-FVM-IMPLICIT-DUAL-TIME",
                        "GAP-FVM-MPI-DECOMP", "GAP-FVM-AMR-QUADTREE"],
        "absorb": "MULES algebraic VOF limiter + AMI/GGI sliding interface "
                  "topology + dynamicRefineFvMesh hanging-node flux correction",
        "license_risk": "GPL-3.0: read-only, re-implement from the papers, do not copy code",
    },
    {
        "id": "openfoam-esi",
        "name": "OpenFOAM (ESI / OpenCFD distribution)",
        "slugs": [],
        "homepage": "https://develop.openfoam.com/",
        "upstream_note": "ESI distributes via its own GitLab (develop.openfoam.com); "
                         "no canonical GitHub repository exists",
        "family": "OpenFOAM lineage",
        "discretisation": "unstructured collocated FVM",
        "capabilities": [
            "unstructured_mesh", "vof_interface_compression", "compressible_vof",
            "mrf_sliding_ami", "adjoint", "les_ras", "mpi_decomposition",
            "implicit_steady", "overset",
        ],
        "tier": "T1",
        "tier_rationale": "same paradigm; ESI branch carries the overset and "
                          "compressible-VOF machinery TensorFVM still lacks at "
                          "production quality",
        "gap_targets": ["GAP-FVM-MRF-SLIDING-AMI", "GAP-FVM-IMPLICIT-DUAL-TIME",
                        "GAP-FVM-LOWMACH-PRECOND"],
        "absorb": "overset interpolation weights + compressibleInterFoam pressure "
                  "equation with phase-change source",
        "license_risk": "GPL-3.0 (read-only)",
    },
    {
        "id": "foam-extend",
        "name": "foam-extend",
        "slugs": ["Unofficial-Extend-Project-Mirror/foam-extend-foam-extend-4.0"],
        "homepage": "https://sourceforge.net/projects/foam-extend/",
        "upstream_note": "canonical hosting is SourceForge; the GitHub mirror "
                         "carries one release branch",
        "family": "OpenFOAM lineage",
        "discretisation": "unstructured collocated FVM",
        "capabilities": ["unstructured_mesh", "implicit_coupled", "fsi_immersed",
                         "vof_interface_compression"],
        "tier": "T2",
        "tier_rationale": "adds coupled implicit solvers and immersed-boundary FSI "
                          "that the Foundation branch lacks; useful design reference",
        "gap_targets": ["GAP-FVM-IMPLICIT-DUAL-TIME"],
        "absorb": "block-coupled implicit assembly pattern",
        "license_risk": "GPL-3.0 (read-only)",
    },
    {
        "id": "solids4foam",
        "name": "solids4foam",
        "slugs": ["solids4foam/solids4foam"],
        "family": "OpenFOAM lineage",
        "discretisation": "unstructured FVM solid mechanics",
        "capabilities": ["fsi_partitioned", "solid_mechanics_fvm"],
        "tier": "T2",
        "tier_rationale": "the partitioned FSI coupling contract TensorFVM would "
                          "have to satisfy once 6DOF/elastic coupling is validated",
        "gap_targets": [],
        "absorb": "partitioned coupling data contract (traction/displacement fields)",
        "license_risk": "GPL-3.0 (read-only)",
    },
    {
        "id": "su2",
        "name": "SU2",
        "slugs": ["su2code/SU2"],
        "family": "compressible density-based",
        "discretisation": "unstructured vertex-centred FVM",
        "capabilities": ["compressible", "implicit_dual_time", "low_mach_precond",
                         "adjoint", "turbulence_sa_sst", "amg_krylov",
                         "objective_optimisation"],
        "tier": "T1",
        "tier_rationale": "the reference open implementation of the two P0 "
                          "compressible gaps (implicit dual time, low-Mach "
                          "preconditioning) with published verification cases",
        "gap_targets": ["GAP-FVM-IMPLICIT-DUAL-TIME", "GAP-FVM-LOWMACH-PRECOND",
                        "GAP-FVM-AMG-KRYLOV"],
        "absorb": "Weiss-Smith/Turkel preconditioning matrix + LU-SGS "
                  "linearisation + discrete adjoint boundary conditions",
        "license_risk": "LGPL-2.1 (read-only; re-implementation safest)",
    },
    {
        "id": "amrex",
        "name": "AMReX",
        "slugs": ["AMReX-Codes/amrex"],
        "family": "block-structured AMR",
        "discretisation": "block-structured FVM with embedded boundaries",
        "capabilities": ["block_amr", "embedded_boundary", "mpi_decomposition",
                         "gpu_offload", "subcycling"],
        "tier": "T1",
        "tier_rationale": "canonical blueprint for the AMR gap: level/flux "
                          "regridding, refluxing and time subcycling contracts",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE", "GAP-FVM-MPI-DECOMP"],
        "absorb": "refluxing (flux register) conservation across coarse-fine "
                  "interfaces + regridding/clustering strategy",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "basilisk",
        "name": "Basilisk",
        "slugs": [],
        "homepage": "http://basilisk.fr/",
        "upstream_note": "distributed by darcs from basilisk.fr; no GitHub "
                         "repository, so metadata cannot be measured here",
        "family": "adaptive quadtree/octree",
        "discretisation": "quadtree/octree cell-centred FVM with wavelet adaptivity",
        "capabilities": ["octree_amr", "vof_geometric", "multiphase", "surface_tension",
                         "mpi_decomposition", "shallow_water"],
        "tier": "T1",
        "tier_rationale": "the cheapest route to conservative adaptive VOF: "
                          "wavelet error estimator + balanced quadtree refinement",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE"],
        "absorb": "wavelet-based refinement criterion + VOF flux conservation "
                  "across level boundaries",
        "license_risk": "GPL-3.0 (read-only)",
    },
    {
        "id": "gerris",
        "name": "Gerris",
        "slugs": ["gerris/gerris", "Gerris/gerris"],
        "family": "adaptive quadtree/octree",
        "discretisation": "adaptive quadtree FVM, VOF",
        "capabilities": ["octree_amr", "vof_geometric", "multiphase", "mpi_decomposition"],
        "tier": "T2",
        "tier_rationale": "predecessor of Basilisk; documented refinement/flux "
                          "balancing maths is easier to read than the newer code",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE"],
        "absorb": "GCI/Gerris adaptive VOF test-suite as acceptance anchors",
        "license_risk": "GPL-2.0 (read-only)",
    },
    {
        "id": "code-saturne",
        "name": "Code_Saturne",
        "slugs": ["code-saturne/code_saturne"],
        "family": "general unstructured",
        "discretisation": "unstructured co-located FVM, polyhedral cells",
        "capabilities": ["unstructured_mesh", "polyhedral_cells", "les_ras",
                         "mpi_decomposition", "implicit_steady", "combustion_species"],
        "tier": "T1",
        "tier_rationale": "industrial-grade unstructured FVM with published "
                          "verification book; the model for our mesh-quality and "
                          "linear-solver acceptance gates",
        "gap_targets": ["GAP-FVM-MPI-DECOMP", "GAP-FVM-AMG-KRYLOV"],
        "absorb": "gradient/limiter treatment on highly non-orthogonal polyhedra",
        "license_risk": "GPL-2.0 (read-only)",
    },
    {
        "id": "clawpack",
        "name": "Clawpack (incl. AMRClaw, PyClaw)",
        "slugs": ["clawpack/clawpack"],
        "family": "hyperbolic finite volume",
        "discretisation": "finite-volume wave-propagation, logically rectangular AMR",
        "capabilities": ["hyperbolic_fv", "amr", "riemann_solvers", "well_balanced"],
        "tier": "T2",
        "tier_rationale": "clean reference for conservative AMR bookkeeping and "
                          "for manufactured/analytic hyperbolic solutions",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE"],
        "absorb": "wave-propagation AMR flux registers + exact-solution test suite",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "cfl3d",
        "name": "CFL3D",
        "slugs": ["nasa/CFL3D"],
        "family": "compressible density-based",
        "discretisation": "structured cell-centred FVM",
        "capabilities": ["compressible", "implicit_steady", "lu_sgs", "turbulence_sa_sst"],
        "tier": "T2",
        "tier_rationale": "canonical LU-SGS implicit steady scheme and its "
                          "documented convergence behaviour; good sanity reference",
        "gap_targets": ["GAP-FVM-IMPLICIT-DUAL-TIME"],
        "absorb": "LU-SGS sweep ordering + CFL ramping schedule",
        "license_risk": "NASA open source agreement (permissive-ish, read-only)",
    },
    {
        "id": "fun3d",
        "name": "FUN3D",
        "slugs": [],
        "homepage": "https://fun3d.larc.nasa.gov/",
        "upstream_note": "released through the NASA software catalogue under a "
                         "restricted agreement; no public repository",
        "family": "compressible density-based",
        "discretisation": "unstructured vertex-centred FVM",
        "capabilities": ["compressible", "implicit_steady", "adjoint", "amg_krylov",
                         "overset", "moving_mesh"],
        "tier": "T3",
        "tier_rationale": "reference only: distribution and licensing are "
                          "restricted, and the published verification data are "
                          "the useful part",
        "gap_targets": ["GAP-FVM-IMPLICIT-DUAL-TIME"],
        "absorb": "published adjoint/implicit verification data for cross-checks",
        "license_risk": "restricted (US export control); no code reuse",
    },
    {
        "id": "cart3d",
        "name": "Cart3D",
        "slugs": [],
        "homepage": "https://www.nas.nasa.gov/publications/software/docs/cart3d/",
        "upstream_note": "restricted NASA release; no public repository",
        "family": "compressible density-based",
        "discretisation": "Cartesian cut-cell FVM with AMR",
        "capabilities": ["cut_cell", "amr", "compressible", "inviscid_adjoint"],
        "tier": "T3",
        "tier_rationale": "cut-cell AMR design is interesting but the solver is "
                          "inviscid and restricted; context only",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE"],
        "absorb": "cut-cell merge/volume-conservation rules",
        "license_risk": "restricted; no code reuse",
    },
    {
        "id": "pelec",
        "name": "PeleC",
        "slugs": ["AMReX-Combustion/PeleC"],
        "family": "block-structured AMR",
        "discretisation": "block-structured compressible FVM + AMR",
        "capabilities": ["block_amr", "compressible", "combustion_species",
                         "gpu_offload", "mpi_decomposition"],
        "tier": "T2",
        "tier_rationale": "the reference for chemistry-coupled AMR once the "
                          "combustion gap is scheduled; also shows AMR cost models",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE", "GAP-FVM-COMBUSTION-SPECIES"],
        "absorb": "operator-split stiff chemistry integration contract",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "pelelmex",
        "name": "PeleLMeX",
        "slugs": ["AMReX-Combustion/PeleLMeX"],
        "family": "block-structured AMR",
        "discretisation": "low-Mach variable-density FVM + AMR",
        "capabilities": ["block_amr", "low_mach", "combustion_species",
                         "implicit_diffusion", "gpu_offload"],
        "tier": "T2",
        "tier_rationale": "low-Mach projection + AMR combination is exactly the "
                          "structure our low-Mach gap will need",
        "gap_targets": ["GAP-FVM-LOWMACH-PRECOND", "GAP-FVM-COMBUSTION-SPECIES"],
        "absorb": "low-Mach pressure-correction loop with spectral deferred "
                  "correction for chemistry",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "incflo",
        "name": "INCflo",
        "slugs": ["AMReX-Codes/INCflo"],
        "family": "block-structured AMR",
        "discretisation": "incompressible FVM with embedded boundaries",
        "capabilities": ["block_amr", "incompressible", "embedded_boundary", "gpu_offload"],
        "tier": "T2",
        "tier_rationale": "shows how the incompressible projection we already have "
                          "is carried across AMR level boundaries",
        "gap_targets": ["GAP-FVM-AMR-QUADTREE"],
        "absorb": "projection at coarse-fine interfaces without pressure drift",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "chombo",
        "name": "Chombo",
        "slugs": [],
        "homepage": "https://commons.lbl.gov/display/chombo/Chombo+-+Software+for+Adaptive+Solutions+of+Partial+Differential+Equations",
        "upstream_note": "hosted by LBNL (SourceForge mirror); no verified "
                         "GitHub repository",
        "family": "block-structured AMR",
        "discretisation": "block-structured AMR (FVM + FD)",
        "capabilities": ["block_amr", "amr_elliptic_solvers", "mpi_decomposition"],
        "tier": "T2",
        "tier_rationale": "the AMR multigrid/elliptic bookkeeping reference; "
                          "older but unusually well documented",
        "gap_targets": ["GAP-FVM-AMG-KRYLOV"],
        "absorb": "level-by-level multigrid restriction/prolongation on AMR hierarchies",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "mfix",
        "name": "MFIX / MFiX-Exa",
        "slugs": [],
        "homepage": "https://mfix.netl.doe.gov/",
        "upstream_note": "NETL hosts MFIX on its own GitLab instance; no "
                         "verified GitHub repository",
        "family": "multiphase",
        "discretisation": "unstructured FVM two-fluid + DEM",
        "capabilities": ["two_fluid_model", "cfd_dem", "multiphase", "mpi_decomposition"],
        "tier": "T2",
        "tier_rationale": "the open TFM/CFD-DEM implementation closest to our "
                          "ice-floe / slurry problems",
        "gap_targets": [],
        "absorb": "drag-closure plug-in interface + fluid-particle momentum exchange ledger",
        "license_risk": "public domain / permissive (verify before reuse)",
    },
    {
        "id": "openqbmm",
        "name": "OpenQBMM",
        "slugs": ["OpenQBMM/OpenQBMM"],
        "family": "multiphase",
        "discretisation": "OpenFOAM-based FVM population balance",
        "capabilities": ["population_balance", "quadrature_moments", "multiphase"],
        "tier": "T2",
        "tier_rationale": "cavitation/bubble population modelling reference; "
                          "relevant if bubble-size spectra enter the ice/water work",
        "gap_targets": [],
        "absorb": "moment-inversion closure structure",
        "license_risk": "GPL-3.0 (read-only)",
    },
    {
        "id": "truchas",
        "name": "Truchas",
        "slugs": ["truchas/truchas"],
        "family": "multiphase",
        "discretisation": "unstructured FVM multiphase + heat transfer",
        "capabilities": ["multiphase", "phase_change", "radiation", "mpi_decomposition"],
        "tier": "T2",
        "tier_rationale": "phase-change + radiation coupling in one FVM code; the "
                          "radiation gap has no better open reference",
        "gap_targets": ["GAP-FVM-RADIATION-P1"],
        "absorb": "P1/enclosure radiation source-term coupling into energy equation",
        "license_risk": "BSD-3-Clause (verify)",
    },
    {
        "id": "reef3d",
        "name": "REEF3D",
        "slugs": ["REEF3D/REEF3D"],
        "family": "free-surface marine",
        "discretisation": "structured Cartesian FVM, level-set free surface",
        "capabilities": ["free_surface_levelset", "waves", "sediment",
                         "mpi_decomposition", "fsi"],
        "tier": "T2",
        "tier_rationale": "marine free-surface FVM with published wave/sediment "
                          "cases; direct source of benchmark anchors for our "
                          "wave tank work",
        "gap_targets": [],
        "absorb": "level-set reinitialisation + wave-generation/absorption anchors",
        "license_risk": "GPL-3.0 (read-only)",
    },
    {
        "id": "precice",
        "name": "preCICE",
        "slugs": ["precice/precice"],
        "family": "coupling infrastructure",
        "discretisation": "n/a (partitioned coupling library)",
        "capabilities": ["fsi_partitioned", "conservative_mapping", "coupling_schemes"],
        "tier": "T2",
        "tier_rationale": "the de-facto open coupling contract for partitioned "
                          "FSI/CHT; informs our cross-solver exchange primitives",
        "gap_targets": ["GAP-FVM-MPI-DECOMP"],
        "absorb": "conservative interpolation (nearest-projection) + implicit "
                  "coupling iteration",
        "license_risk": "LGPL-3.0 (read-only)",
    },
    {
        "id": "pyfoam",
        "name": "PyFoam",
        "slugs": [],
        "homepage": "https://openfoamwiki.net/index.php/Contrib/PyFoam",
        "upstream_note": "hosted on SourceForge; no verified GitHub repository",
        "family": "tooling",
        "discretisation": "n/a (case/run tooling)",
        "capabilities": ["case_tooling", "regression_runs"],
        "tier": "T2",
        "tier_rationale": "case-directory contract + automated run harvesting; "
                          "useful for our benchmark provenance",
        "gap_targets": [],
        "absorb": "run-record harvesting / case manifest layout",
        "license_risk": "GPL-2.0 (read-only)",
    },
    {
        "id": "fipy",
        "name": "FiPy",
        "slugs": ["usnistgov/fipy"],
        "family": "python prototyping",
        "discretisation": "generic cell-centred FVM on unstructured meshes",
        "capabilities": ["generic_pde_fvm", "unstructured_mesh", "diffusion_reaction"],
        "tier": "T2",
        "tier_rationale": "pure-Python FVM with an unusually clean term/coefficient "
                          "assembly API; good cross-check partner for scalar "
                          "transport and phase-field equations",
        "gap_targets": [],
        "absorb": "term-wise (TransientTerm/DiffusionTerm) coefficient assembly "
                  "for manufactured-solution tests",
        "license_risk": "public domain / BSD-like (verify)",
    },
    {
        "id": "gmsh",
        "name": "Gmsh",
        "slugs": ["live-clones/gmsh"],
        "homepage": "https://gmsh.info/",
        "upstream_note": "official hosting is gitlab.onelab.info; the GitHub "
                         "entry used here is a read-only mirror",
        "family": "meshing",
        "discretisation": "n/a (mesh generator)",
        "capabilities": ["mesh_generation", "polyhedral_cells", "mesh_quality"],
        "tier": "T2",
        "tier_rationale": "source of polyhedral/hybrid meshes for mesh-quality "
                          "and non-orthogonality acceptance cases",
        "gap_targets": [],
        "absorb": "mesh-quality metrics we can consume from .msh files",
        "license_risk": "GPL-2.0 (tool use only; output meshes are unrestricted)",
    },
    {
        "id": "nektar",
        "name": "Nektar++",
        "slugs": [],
        "homepage": "https://www.nektar.info/",
        "upstream_note": "official hosting is a GitLab instance; no verified "
                         "GitHub repository",
        "family": "high-order (non-FVM)",
        "discretisation": "spectral/hp discontinuous Galerkin",
        "capabilities": ["high_order_dg", "hp_adaptivity"],
        "tier": "T3",
        "tier_rationale": "not FVM; only useful as a high-order accuracy target "
                          "and for cross-solver comparison of smooth solutions",
        "gap_targets": [],
        "absorb": "hp-error estimators as accuracy yardsticks",
        "license_risk": "MIT (compatible) but paradigm mismatch",
    },
    {
        "id": "pyfr",
        "name": "PyFR",
        "slugs": ["PyFR/PyFR"],
        "family": "high-order (non-FVM)",
        "discretisation": "flux reconstruction",
        "capabilities": ["flux_reconstruction", "gpu_offload"],
        "tier": "T3",
        "tier_rationale": "GPU-native high order; context on where the ecosystem "
                          "is heading, not directly absorbable",
        "gap_targets": [],
        "absorb": "GPU backend abstraction ideas",
        "license_risk": "BSD-3-Clause (compatible)",
    },
    {
        "id": "petsc",
        "name": "PETSc (incl. hypre bindings)",
        "slugs": ["petsc/petsc"],
        "family": "linear algebra",
        "discretisation": "n/a (Krylov/AMG library)",
        "capabilities": ["amg_krylov", "mpi_decomposition", "preconditioners"],
        "tier": "T1",
        "tier_rationale": "do not re-implement AMG: the P1 linear-solver gap is "
                          "closed faster by binding PETSc/hypre than by writing "
                          "our own multigrid hierarchy",
        "gap_targets": ["GAP-FVM-AMG-KRYLOV"],
        "absorb": "GMRES/BiCGStab + BoomerAMG via a thin Python binding",
        "license_risk": "BSD-2-Clause (compatible)",
    },
    {
        "id": "hypre",
        "name": "hypre",
        "slugs": ["hypre-space/hypre"],
        "family": "linear algebra",
        "discretisation": "n/a (algebraic multigrid)",
        "capabilities": ["amg_krylov", "mpi_decomposition"],
        "tier": "T1",
        "tier_rationale": "BoomerAMG is the de-facto AMG for unstructured FVM; "
                          "a binding is the minimal-change fix for our gap",
        "gap_targets": ["GAP-FVM-AMG-KRYLOV"],
        "absorb": "boomeramg parameter set validated on our Poisson cases",
        "license_risk": "Apache-2.0/MIT (compatible)",
    },
    {
        "id": "fluent",
        "name": "Ansys Fluent",
        "slugs": [],
        "family": "commercial",
        "discretisation": "unstructured FVM",
        "capabilities": ["unstructured_mesh", "mrf_sliding_ami", "combustion_species",
                         "radiation", "amr"],
        "tier": "T3",
        "tier_rationale": "closed source; used only as an industrial acceptance "
                          "baseline for the same test cases",
        "gap_targets": [],
        "absorb": "nothing (benchmark numbers only)",
        "license_risk": "commercial; no source access",
    },
]

GAP_BLUEPRINTS = [
    {
        "id": "GAP-FVM-AMR-QUADTREE",
        "priority": "P0",
        "title": "Adaptive mesh refinement with conservative coarse-fine flux refluxing",
        "current_state": "no quadtree/octree or hanging-node machinery: uniform "
                         "grids only, refinement studies are done by re-running "
                         "finer uniform meshes",
        "evidence_in_tensorfvm": [
            "src/tensorfvm/mesh3d.py", "src/tensorfvm/native_mesh3d.py"],
        "evidence_note": "grep -riE 'quadtree|octree|hanging|coarsen' over "
                         "src/tensorfvm returns no AMR data structure, only prose "
                         "in docstrings",
        "reference_repos": ["amrex", "basilisk", "gerris", "clawpack", "pelec"],
        "acceptance": "Sod on a 2-level AMR hierarchy (effective 400 cells) must "
                      "match the uniform 400-cell solution in L1(rho) within 5% "
                      "and conserve mass to 1e-12 via the flux-register ledger; "
                      "cell count must drop by >=3x vs uniform 400",
        "external_anchor": "Sod 1978 exact Riemann solution (same anchor as the "
                           "uniform-grid leg) + AMReX/Basilisk refluxing formulation",
        "effort": "L",
    },
    {
        "id": "GAP-FVM-IMPLICIT-DUAL-TIME",
        "priority": "P0",
        "title": "Implicit steady / dual-time integration (LU-SGS or Newton-Krylov)",
        "current_state": "compressible solver is explicit forward-Euler with "
                         "CFL ~0.4; steady states are reached by pseudo-time "
                         "marching, so high-Re and low-Mach cases are step-limited",
        "evidence_in_tensorfvm": [
            "src/tensorfvm/compressible/solver.py",
            "src/tensorfvm/compressible/cases.py"],
        "evidence_note": "EulerSolver advances with explicit time stepping "
                         "(cfl=0.4 default in cases.sod); grep -niE "
                         "'implicit|newton|lusgs' in src/tensorfvm/compressible "
                         "hits only the exact-Riemann Newton iteration",
        "reference_repos": ["su2", "cfl3d", "openfoam-foundation", "foam-extend"],
        "acceptance": "on the same Sod setup restarted to a steady channel case, "
                      "implicit CFL>=10 must reach the explicit converged solution "
                      "within 1e-6 (L2, rho) using >=10x fewer time steps; a "
                      "regression test pins the step-count ratio",
        "external_anchor": "Jameson & Turkel (1981) LU-SGS implicit formulation "
                           "(as implemented in SU2/CFL3D) + the existing explicit "
                           "converged state as the comparison state",
        "effort": "L",
    },
    {
        "id": "GAP-FVM-LOWMACH-PRECOND",
        "priority": "P0",
        "title": "Low-Mach preconditioning (Weiss-Smith / Turkel) for the compressible solver",
        "current_state": "no preconditioning matrix: acoustic stiffness at "
                         "Ma<0.1 makes the explicit compressible path unusable "
                         "for ship-scale flows, so those cases fall back to the "
                         "incompressible solver",
        "evidence_in_tensorfvm": [
            "src/tensorfvm/compressible/solver.py",
            "src/tensorfvm/compressible/riemann.py"],
        "evidence_note": "no low-Mach/preconditioning term anywhere in "
                         "src/tensorfvm/compressible (grep 'precondition' hits "
                         "only unrelated iterative-solver prose elsewhere)",
        "reference_repos": ["su2", "pelelmex", "code-saturne"],
        "acceptance": "low-Mach Gresho vortex (Ma=0.05) must preserve the "
                      "analytic pressure/density field within 2% and reduce the "
                      "acoustic iteration count by >=5x vs unpreconditioned at "
                      "the same CFL",
        "external_anchor": "Gresho & Chan 1990 vortex (analytic, incompressible "
                           "limit) + SU2 preconditioning formulation",
        "effort": "M",
    },
    {
        "id": "GAP-FVM-MRF-SLIDING-AMI",
        "priority": "P1",
        "title": "Rotating-frame (MRF) and arbitrary-mesh-interface (AMI) sliding boundaries",
        "current_state": "no MRF source term and no sliding/AMI interface: "
                         "rotating parts (propellers, rotating cylinders) cannot "
                         "be represented except by re-meshing",
        "evidence_in_tensorfvm": ["src/tensorfvm/moving_mesh.py",
                                  "src/tensorfvm/six_dof.py"],
        "evidence_note": "grep -riE 'sliding|MRF' over src/tensorfvm hits only "
                         "the 'sliding ice' wall comment in ice_channel.py",
        "reference_repos": ["openfoam-foundation", "openfoam-esi"],
        "acceptance": "2D Taylor-Couette: torque on the inner cylinder from the "
                      "AMI/MRF path vs the analytic Couette torque within 3% "
                      "at two gap resolutions",
        "external_anchor": "analytic Taylor-Couette torque (Couette 1890; "
                           "Batchelor 1967, §4.9): T = 4*pi*mu*L*omega*ri^2*ro^2/"
                           "(ro^2-ri^2)",
        "effort": "M",
    },
    {
        "id": "GAP-FVM-MPI-DECOMP",
        "priority": "P1",
        "title": "Distributed-memory domain decomposition with halo exchange",
        "current_state": "solves are single-process; torch.distributed is only "
                         "consulted to pick a device, there is no domain "
                         "decomposition or halo exchange for the FVM fields",
        "evidence_in_tensorfvm": ["src/tensorfvm/runtime.py"],
        "evidence_note": "runtime.py queries torch.distributed.is_initialized() "
                         "for device selection only; no halo/exchange code in "
                         "src/tensorfvm (grep 'halo' hits nothing)",
        "reference_repos": ["amrex", "openfoam-foundation", "code-saturne", "precice"],
        "acceptance": "lid-cavity Re=1000 split over 2 and 4 ranks must reproduce "
                      "the serial Ghia-u-centreline within 1e-6 L2 and keep "
                      "parallel efficiency >=0.7 at 4 ranks on the cluster",
        "external_anchor": "Ghia et al. 1982 centreline data + serial TensorFVM run",
        "effort": "L",
    },
    {
        "id": "GAP-FVM-AMG-KRYLOV",
        "priority": "P1",
        "title": "Algebraic multigrid + Krylov linear solver backend",
        "current_state": "the only multigrid is the geometric V-cycle inside the "
                         "immersed masked-Poisson solver; no general AMG/Krylov "
                         "for the unsymmetric pressure/momentum systems",
        "evidence_in_tensorfvm": ["src/tensorfvm/immersed/projection.py"],
        "evidence_note": "grep -riE 'multigrid|vcycle|amg|petsc|hypre' over "
                         "src/tensorfvm hits only immersed/projection.py (masked "
                         "geometric V-cycle) and docstrings",
        "reference_repos": ["petsc", "hypre", "su2", "code-saturne", "chombo"],
        "acceptance": "256^2 Poisson and an advection-diffusion system: AMG-PCG "
                      "iteration count must grow <=1.3x from 64^2 to 256^2 and "
                      "solve to 1e-10 in fewer than 30 iterations",
        "external_anchor": "Brandt (1977) multigrid h-independence + BoomerAMG "
                           "convergence data (Baker et al. 2012)",
        "effort": "M",
    },
    {
        "id": "GAP-FVM-COMBUSTION-SPECIES",
        "priority": "P2",
        "title": "Multi-species transport with stiff Arrhenius chemistry",
        "current_state": "the compressible solver carries at most two materials "
                         "with a stiffened-gas EOS; there is no species "
                         "transport, no reaction source and no operator splitting "
                         "for stiff chemistry",
        "evidence_in_tensorfvm": ["src/tensorfvm/compressible/eos.py",
                                  "src/tensorfvm/compressible/solver.py"],
        "evidence_note": "grep -riE 'combust|chemistry|species' hits only the "
                         "two-material 'species' bookkeeping in the compressible "
                         "package",
        "reference_repos": ["pelec", "pelelmex", "openfoam-foundation"],
        "acceptance": "1D premixed CH4/air flame: laminar flame speed within 10% "
                      "of the Cantera reference at two resolutions",
        "external_anchor": "Cantera (Goodwin et al. 2017) laminar flame speed "
                           "for the same mechanism",
        "effort": "L",
    },
    {
        "id": "GAP-FVM-RADIATION-P1",
        "priority": "P2",
        "title": "Radiative heat transfer (P1 / discrete ordinates) coupled to the energy equation",
        "current_state": "no radiation model: the energy equation has conduction "
                         "and convection only",
        "evidence_in_tensorfvm": ["src/tensorfvm/thermal/__init__.py",
                                  "src/tensorfvm/cht"],
        "evidence_note": "no radiation module; docs/verification-gap-audit-m7555.md "
                         "already lists radiation as an unverified capability gap",
        "reference_repos": ["truchas", "openfoam-foundation"],
        "acceptance": "1D grey parallel plates: net radiative flux vs the "
                      "analytic exchange factor within 2%; coupling conserves "
                      "total energy to 1e-10",
        "external_anchor": "analytic view-factor exchange for infinite parallel "
                           "plates (Incropera & DeWitt 2007, Ch. 13)",
        "effort": "M",
    },
]


def fetch_metadata(slug: str, timeout: float = 8.0) -> dict:
    """Return measured GitHub metadata for *slug*, or an unresolved record."""
    req = urllib.request.Request(
        API.format(slug=slug),
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "tensorfvm-oss-inventory/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        return {"source": "unresolved", "slug": slug, "http_status": exc.code}
    except Exception as exc:  # network down, DNS, timeout ...
        return {"source": "unresolved", "slug": slug,
                "error": f"{type(exc).__name__}: {exc}"}
    lic = data.get("license") or {}
    return {
        "source": "github_api",
        "slug": data.get("full_name", slug),
        "url": data.get("html_url"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "language": data.get("language"),
        "license_spdx": lic.get("spdx_id"),
        "license_name": lic.get("name"),
        "pushed_at": data.get("pushed_at"),
        "archived": data.get("archived"),
        "default_branch": data.get("default_branch"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build(offline: bool = False, pause: float = 0.0,
          cache_path: str | None = None, refresh: bool = False) -> dict:
    """Assemble the registry; reuse measured metadata from *cache_path*.

    GitHub's anonymous rate limit is 60 requests/hour, so a rebuild must not
    re-query entries that were already measured with the same slug.
    """
    cache: dict[str, dict] = {}
    if cache_path and not refresh and Path(cache_path).exists():
        try:
            previous = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            cache = {r["id"]: r["metadata"] for r in previous.get("repos", [])
                     if r.get("metadata", {}).get("source") == "github_api"}
        except Exception:
            cache = {}
    repos = []
    for project in PROJECTS:
        record = {k: v for k, v in project.items() if k != "slugs"}
        metadata = {"source": "not_attempted"}
        if not project["slugs"]:
            metadata = {"source": "commercial",
                        "note": "closed source, no repository"} if project["id"] == "fluent" else {
                "source": "upstream_not_github",
                "homepage": project.get("homepage"),
                "note": project.get("upstream_note", "no GitHub repository"),
            }
        elif not offline:
            cached = cache.get(project["id"])
            wanted = {slug.lower() for slug in project["slugs"]}
            if cached and cached.get("slug", "").lower() in wanted:
                metadata = cached
            else:
                for slug in project["slugs"]:
                    metadata = fetch_metadata(slug)
                    if metadata.get("source") == "github_api":
                        break
                    if pause:
                        time.sleep(pause)
        else:
            metadata = {"source": "offline",
                        "note": "fetch skipped (--offline); slugs recorded only",
                        "slugs_tried": project["slugs"]}
        record["metadata"] = metadata
        record["curation"] = {
            "capability_evidence": "curated_knowledge",
            "note": "capability tags are expert curation for triage, not a "
                    "source-reading audit performed in this run",
        }
        repos.append(record)

    ids = {r["id"] for r in repos}
    for gap in GAP_BLUEPRINTS:
        unknown = [r for r in gap["reference_repos"] if r not in ids]
        if unknown:
            raise SystemExit(f"{gap['id']} references unknown repos: {unknown}")

    return {
        "schema_version": "1.0",
        "generated_by": "scripts/oss_inventory_fvm/build_registry.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_repo": "TensorFVM",
        "mission": 14887,
        "feature_slug": "FEAT-OSS-INVENTORY-FVM",
        "tiers": TIERS,
        "evidence_policy": {
            "measured": "metadata.* fields are fetched from the GitHub REST API "
                        "in this run; source=='github_api' marks measured values",
            "curated": "curation.capability_evidence=='curated_knowledge' marks "
                       "tags that were NOT verified by reading upstream sources "
                       "in this run; adoption requires a source-reading pass",
            "unresolved": "source=='unresolved' entries keep null metadata and "
                          "must not be quoted as current project facts",
        },
        "counts": {
            "repos": len(repos),
            "by_tier": {t: sum(1 for r in repos if r["tier"] == t) for t in TIERS},
            "measured_metadata": sum(1 for r in repos
                                     if r["metadata"].get("source") == "github_api"),
            "gaps": len(GAP_BLUEPRINTS),
        },
        "repos": repos,
        "gaps": GAP_BLUEPRINTS,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds between API calls (rate-limit friendly)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached metadata and re-query every slug")
    args = ap.parse_args(argv)
    registry = build(offline=args.offline, pause=args.pause,
                     cache_path=args.out, refresh=args.refresh)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(registry["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())