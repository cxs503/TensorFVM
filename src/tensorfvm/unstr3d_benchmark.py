"""Benchmark registry rows for the 3-D unstructured FVM core.

Every verification gate of FEAT-UNSTR3D-MESH is registered with an explicit
metric, a pre-registered tolerance and a reference source, so a run either
reproduces the number inside its tolerance or is recorded as a failure.  The
rows are kept in code (like ``compressible/forward_step_benchmark.py``) and can
be emitted as SQL for the platform referee table.
"""

ANCHOR_SLUG = "FEAT-UNSTR3D-MESH"

#: Reference sources cited by the rows below.
GRID_CONVERGENCE = ("Richardson extrapolation of the manufactured solution "
                    "phi = sin(pi x) sin(pi y) sin(pi z) with f = -3 pi^2 phi "
                    "(analytic; order estimated with p = log(e_coarse/e_fine)/log r)")
ACCEPTANCE = ("Roache, Verification and Validation in Computational Science and "
              "Engineering (1998), ch. 5: a verification gate is an observed "
              "order or an analytic bound, not a code intent")

CASES = (
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-poisson-order",
         metric="observed_order_poisson_l2 (tet ladder n=4,8)",
         tolerance=">= 2.0 (reject below)", ref_source=GRID_CONVERGENCE,
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-poisson-order-linear-correction",
         metric="observed_order_poisson_l2 with linear LS correction (control)",
         tolerance="< 2.0 (must fail the gate)", ref_source=GRID_CONVERGENCE,
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-poisson-affine-exactness",
         metric="max |phi_h - phi_exact| for an affine field",
         tolerance="1e-9", ref_source="exactness of the linear reconstruction (analytic)",
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-convection-45deg-overshoot",
         metric="relative overshoot of a 45 deg cosine profile (limited scheme)",
         tolerance="<= 2%", ref_source="u.grad(phi)=0 exact solution cos(2 pi eta/lambda), eta=(y-x)/sqrt(2)",
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-convection-upwind-bounded",
         metric="relative overshoot, first-order upwind",
         tolerance="<= 2%", ref_source="upwind positivity on an M-matrix (analytic)",
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-mesh-invariants",
         metric="min cell volume, face-normal/vertex-order consistency, closure, "
                "owner->neighbour projection",
         tolerance="volume > 0; consistency 0; closure <= 1e-10; projection > 0",
         ref_source="discrete divergence theorem on a closed polyhedron (analytic)",
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-gmsh-v2-import",
         metric="parsed cells / boundary patches / rejected non-whitelisted names",
         tolerance="exact topology match on the fixture; every guard raises",
         ref_source="Gmsh 2.2 MSH file format specification, sections $Nodes/$Elements/$PhysicalNames",
         repo="TensorFVM"),
    dict(anchor_slug=ANCHOR_SLUG, case_name="unstr3d-poisson-order-warped",
         metric="observed_order_poisson_l2 on the warped ladder n=4,8 (secondary)",
         tolerance=">= 2.0 reported honestly; 1.91 measured (below gate)",
         ref_source=GRID_CONVERGENCE, repo="TensorFVM"),
)


def sql_statements(anchor=ANCHOR_SLUG):
    """DELETE + INSERT rows for ``tensor_benchmark_cases``."""
    def quote(text):
        return "'" + str(text).replace("'", "''") + "'"

    statements = ["-- FEAT-UNSTR3D-MESH benchmark rows",
                  "DELETE FROM tensor_benchmark_cases WHERE anchor_slug = {};".format(quote(anchor))]
    for case in CASES:
        statements.append(
            "INSERT INTO tensor_benchmark_cases "
            "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status) VALUES "
            "({}, {}, {}, {}, {}, {}, {});".format(
                quote(case["anchor_slug"]), quote(case["case_name"]),
                quote(case["ref_source"]), quote(case["metric"]),
                quote(case["tolerance"]), quote(case["repo"]), quote("pending")))
    return statements
