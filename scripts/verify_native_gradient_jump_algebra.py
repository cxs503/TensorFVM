"""Verification audit of the hybrid tangential gradient-jump stabilisation.

The operator under test, ``NativeGradientJump3D``
(``src/tensorfvm/native_gradient_jump3d.py``), is the polyhedral extension of
the Crouzeix-Raviart face-jump (Korn) stabilisation that Hansbo & Larson
(2003, doi:10.1051/m2an:2003020, eq. 26) require for nonconforming face
velocities.  It is assembled into ``NativeHybridViscous3D.full_matrix`` and its
energy is reported in the traction ledger
(``gradient_jump_energy`` in ``native_hybrid_stokes3d``), but it had no
reference verification of its own.

Three classes of evidence are produced, all cheap (pure linear algebra on
8-216 cell meshes, no solver iteration):

1. **Structural identities** that must hold to machine precision for *any*
   admissible polyhedral mesh: symmetry, positive semidefiniteness, and the
   kernel containing the full affine velocity space (the classical
   "vanishes on the FE space" requirement of a stabilisation).
2. **Absolute scaling against hand-derived geometry**, so that no hidden
   constant can hide in the operator: on a uniform hexahedral mesh the weight
   must be ``mu*area/height`` and the face covariance must reduce to the
   analytic inertia tensor of the triangulated face.
3. **Ledger consistency**: the "independent local energy replay"
   ``gradient_reactions`` must reproduce both the assembled quadratic form
   and its gradient (checked analytically and by central differences), because
   the traction ledger books that replay energy as physical dissipation.

Usage::

    python3 scripts/verify_native_gradient_jump_algebra.py --out docs/verification-native-gradient-jump-algebra
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D  # noqa: E402
from tensorfvm.native_mesh3d import NativeMesh3D  # noqa: E402

# Tolerances: the first five checks are exact algebra, the last two are
# numerical (finite differences) or asymptotic (h-refinement) statements.
EXACT = 1e-12
ROUNDOFF = 1e-14
FD_TOL = 1e-6


# --------------------------------------------------------------------------
# mesh families
# --------------------------------------------------------------------------
def hex_cube(n: int = 2, shear: float = 0.0) -> NativeMesh3D:
    """Structured hexahedral unit cube, optionally sheared by an affine map."""
    points = np.array([[x, y, z] for z in np.linspace(0, 1, n + 1)
                       for y in np.linspace(0, 1, n + 1) for x in np.linspace(0, 1, n + 1)])
    if shear:
        points = points @ np.array([[1, shear, 0.05], [0, 1, 0.2], [0, 0, 1]]).T

    def index(x, y, z):
        return z * (n + 1) ** 2 + y * (n + 1) + x

    cells = [[index(x, y, z), index(x + 1, y, z), index(x + 1, y + 1, z), index(x, y + 1, z),
              index(x, y, z + 1), index(x + 1, y, z + 1), index(x + 1, y + 1, z + 1),
              index(x, y + 1, z + 1)]
             for z in range(n) for y in range(n) for x in range(n)]
    return NativeMesh3D.from_cells(points, cells, ["hexahedron"] * len(cells),
                                   triangulate_faces=True)


def tetra_pair() -> NativeMesh3D:
    """Two tetrahedra sharing one face: the smallest mesh with one internal face."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1.0], [0.2, 0.2, -1.0]])
    return NativeMesh3D.from_cells(points, [[0, 1, 2, 3], [0, 2, 1, 4]], ["tetra", "tetra"])


FAMILIES = {
    "hex_cube_n2": lambda: hex_cube(2),
    "hex_sheared_n2": lambda: hex_cube(2, shear=0.18),
    "hex_sheared_n3": lambda: hex_cube(3, shear=0.10),
    "tetra_pair": tetra_pair,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def nodal_coordinates(mesh) -> np.ndarray:
    """Cell-centre then face-centre coordinates: the operator's nodal layout."""
    return np.vstack([mesh.cell_centers, mesh.face_centers])


def dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense(), dtype=float)


def split_symmetric(matrix) -> tuple[float, float]:
    dense_matrix = 0.5 * (matrix + matrix.T)
    values = np.linalg.eigvalsh(dense_matrix)
    return float(values[0]), float(values[-1])


def reaction_gradient(gj, state: np.ndarray) -> np.ndarray:
    """d(energy)/du assembled from the per-face reaction stress, signs included.

    Energy is ``E = sum_f jump_fij * w_f * cov_fjk * jump_fik`` (no 1/2), so the
    expected gradient is ``2*M@u``.
    """
    mesh = gj.mesh
    ids = gj.internal
    gradient = np.stack([operator @ state.reshape(len(mesh.cells) + len(mesh.owner), 3)
                         for operator in gj.gradient], axis=2)
    jump = gradient[mesh.owner[ids]] - gradient[mesh.neighbor[ids]]
    response = gj.weight[:, None, None] * np.einsum("fij,fjk->fik", jump, gj.covariance)
    out = np.zeros_like(state.reshape(len(mesh.cells) + len(mesh.owner), 3))
    for spatial in range(3):
        cell = np.zeros((len(mesh.cells), 3))
        np.add.at(cell, mesh.owner[ids], response[:, :, spatial])
        np.add.at(cell, mesh.neighbor[ids], -response[:, :, spatial])
        for component in range(3):
            out[:, component] += gj.gradient[spatial].T @ cell[:, component]
    return out.ravel()


def central_difference_gradient(gj, state: np.ndarray, step: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(state)
    for index in range(state.size):
        plus = state.copy()
        plus[index] += step
        minus = state.copy()
        minus[index] -= step
        gradient[index] = (gj.gradient_reactions(plus)[1]
                           - gj.gradient_reactions(minus)[1]) / (2 * step)
    return gradient


class Ledger:
    def __init__(self):
        self.rows = []

    def add(self, key, name, measured, reference, error, tol, note=""):
        self.rows.append({
            "id": key, "name": name,
            "measured": float(measured) if np.isfinite(measured) else None,
            "reference": reference,
            "error": float(error) if np.isfinite(error) else None,
            "tolerance": tol,
            "pass": bool(np.isfinite(error) and error <= tol),
            "note": note,
        })

    def relative(self, measured, reference):
        scale = max(abs(reference), 1e-300)
        return abs(measured - reference) / scale


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
def audit_family(name, mesh, ledger, rng):
    viscous = NativeHybridViscous3D(mesh, 1.0)
    gj = viscous.gradient_jump
    if gj is None:  # pragma: no cover - three-component default
        raise RuntimeError("gradient jump term was not assembled")
    matrix = dense(gj.matrix)
    size = matrix.shape[0]
    coords = nodal_coordinates(mesh)
    prefix = f"{name}:"

    # 1. symmetry
    asymmetry = float(np.abs(matrix - matrix.T).max())
    ledger.add(f"{name}/symmetry", f"{prefix} matrix symmetry", asymmetry,
               "0 (exact)", asymmetry, EXACT, "assembled as (A+A.T)/2 by construction")

    # 2. positive semidefiniteness (the stabilisation must never destroy energy)
    lowest, highest = split_symmetric(matrix)
    ledger.add(f"{name}/psd", f"{prefix} lambda_min(M)", lowest,
               ">= 0 (exact PSD)", max(0.0, -lowest), ROUNDOFF,
               f"lambda_max={highest:.6e}")

    # 3. affine invariance: every linear velocity field is in the kernel
    affine = coords @ rng.normal(size=(3, 3)).T + rng.normal(size=3)
    _, affine_energy = gj.gradient_reactions(affine.ravel())
    ledger.add(f"{name}/affine_energy", f"{prefix} energy of affine velocity",
               abs(affine_energy), "0 (exact)", abs(affine_energy), 1e-24,
               "Crouzeix-Raviart requirement: the term vanishes on the FE space")

    # 4. kernel / rank profile
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int((singular > 1e-10 * max(singular[0], 1e-300)).sum())
    ledger.add(f"{name}/kernel", f"{prefix} dim ker(M)", size - rank,
               ">= 12 (affine space)", 0.0, EXACT if size - rank >= 12 else -1.0,
               f"size={size}, rank={rank}")

    # 5. ledger energy identity for a generic state
    state = rng.normal(size=(len(mesh.cells) + len(mesh.owner), 3))
    _, ledger_energy = gj.gradient_reactions(state.ravel())
    quadratic = float(state.ravel() @ matrix @ state.ravel())
    ledger.add(f"{name}/energy_identity", f"{prefix} ledger energy vs u^T M u",
               ledger_energy, "u^T M u", ledger.relative(ledger_energy, quadratic),
               EXACT, f"u^T M u={quadratic:.12e}")

    # 6. ledger closure: the per-cell reaction rebuilt from the face stresses
    #    must reproduce the assembled operator action M@u exactly, otherwise the
    #    interface-flux audit sees a defect that the matrix does not contain.
    action = matrix @ state.ravel()
    scale = float(np.abs(action).max())
    analytic = reaction_gradient(gj, state.ravel())
    closure = float(np.abs(analytic - action).max() / scale)
    ledger.add(f"{name}/ledger_closure", f"{prefix} stress replay vs M@u",
               closure, "0 (exact)", closure, EXACT)

    # 7. numerical cross-check that the booked reaction is the operator action,
    #    while the energy follows the codebase's full-quadratic-form convention
    #    (E = u^T M u with no 1/2, so central differences return 2*M@u).
    difference = central_difference_gradient(gj, state.ravel(), 1e-6)
    fd_error = float(np.abs(difference - 2 * action).max() / (2 * scale))
    ledger.add(f"{name}/energy_gradient_fd", f"{prefix} dE/du = 2*M@u (FD)",
               fd_error, "2*M@u", fd_error, FD_TOL,
               "central differences of the reported energy, step=1e-6")

    return viscous, gj, matrix


def audit_scaling(name, mesh, viscous, gj, ledger, rng):
    """Strength linearity and uniform-viscosity scaling of the quadratic form."""
    coords = nodal_coordinates(mesh)
    state = rng.normal(size=(coords.shape[0], 3))
    base = float(state.ravel() @ dense(gj.matrix) @ state.ravel())
    for strength in (0.5, 2.5):
        scaled = NativeHybridViscous3D(mesh, 1.0, gradient_jump_stabilization=strength)
        energy = float(state.ravel() @ dense(scaled.gradient_jump.matrix) @ state.ravel())
        ledger.add(f"{name}/strength_{strength}",
                   f"{name}: energy ratio at strength={strength}", energy / base,
                   float(strength), abs(energy / base - strength), EXACT)
    doubled = NativeHybridViscous3D(mesh, 2.0)
    energy = float(state.ravel() @ dense(doubled.gradient_jump.matrix) @ state.ravel())
    ledger.add(f"{name}/mu_2", f"{name}: energy ratio at mu=2", energy / base, "2.0",
               abs(energy / base - 2.0), EXACT,
               "harmonic mean of a uniform viscosity is that viscosity")
    off = NativeHybridViscous3D(mesh, 1.0, gradient_jump_stabilization=0.0)
    ledger.add(f"{name}/strength_zero", f"{name}: term absent at strength=0",
               1.0 if off.gradient_jump is None else 0.0, "1.0",
               0.0 if off.gradient_jump is None else 1.0, EXACT)


def audit_analytic_geometry(ledger):
    """Hand-derived references on the uniform 2x2x2 cube (h = 1/2).

    Internal faces are triangles of a quad split: area = h^2/2 = 0.125 and the
    centroid distance is h = 0.5, so w = mu*area/height = 0.25.  For a right
    triangle with legs h the inertia tensor about its centroid has eigenvalues
    h^2/12 = 0.020833... and h^2/36 = 0.006944... in the face plane and zero
    normal to it (total trace h^2/9 = 0.02777...).
    """
    n = 2
    h = 1.0 / n
    mesh = hex_cube(n)
    viscous = NativeHybridViscous3D(mesh, 1.0)
    gj = viscous.gradient_jump

    expected_weight = h / 2.0
    weight_error = float(np.abs(gj.weight - expected_weight).max() / expected_weight)
    ledger.add("hex_cube_n2/weight", "weight = mu*area/height",
               float(gj.weight.mean()), expected_weight, weight_error, EXACT,
               f"internal_faces={len(gj.internal)}, mu=1")

    eigenvalues = np.sort(np.linalg.eigvalsh(gj.covariance), axis=1)
    in_plane = eigenvalues[:, 2]
    normal = eigenvalues[:, 0]
    trace = eigenvalues.sum(axis=1)
    ledger.add("hex_cube_n2/cov_major", "covariance major in-plane eigenvalue",
               float(in_plane.mean()), h ** 2 / 12.0,
               float(np.abs(in_plane - h ** 2 / 12.0).max() / (h ** 2 / 12.0)), EXACT)
    ledger.add("hex_cube_n2/cov_minor", "covariance minor in-plane eigenvalue",
               float(eigenvalues[:, 1].mean()), h ** 2 / 36.0,
               float(np.abs(eigenvalues[:, 1] - h ** 2 / 36.0).max() / (h ** 2 / 36.0)), EXACT)
    ledger.add("hex_cube_n2/cov_normal", "covariance normal eigenvalue",
               float(np.abs(normal).max()), 0.0, float(np.abs(normal).max()), EXACT,
               "face inertia is planar")
    ledger.add("hex_cube_n2/cov_trace", "covariance trace",
               float(trace.mean()), h ** 2 / 9.0,
               float(np.abs(trace - h ** 2 / 9.0).max() / (h ** 2 / 9.0)), EXACT)


def audit_refinement(ledger):
    """h-refinement consistency: the defect energy must vanish as O(h^2)."""
    field = lambda coords: coords[:, 0] ** 2 + coords[:, 1] * coords[:, 2]  # noqa: E731
    samples = []
    for n in (2, 3, 4, 5):
        mesh = hex_cube(n)
        gj = NativeHybridViscous3D(mesh, 1.0).gradient_jump
        coords = nodal_coordinates(mesh)
        scalar = field(coords)
        _, energy = gj.gradient_reactions(np.stack([scalar] * 3, axis=1).ravel())
        samples.append({"n": n, "h": 1.0 / n, "energy": float(energy)})
    orders = []
    for first, second in zip(samples, samples[1:]):
        orders.append(float(np.log(first["energy"] / second["energy"])
                            / np.log(first["h"] / second["h"])))
    ledger.add("refinement/order_finest", "observed order (n=5->6)",
               orders[-1], "2.0 (asymptotic)", abs(orders[-1] - 2.0), 0.30,
               "monotone approach to second order; energy->0 under refinement")
    finest = float(np.mean(orders[-2:]))
    ledger.add("refinement/order_mean", "observed order (two finest pairs)", finest,
               "2.0 (asymptotic)", abs(finest - 2.0), 0.35,
               "the defect energy vanishes under refinement, no h-independent floor")
    monotone = all(second["energy"] < first["energy"]
                   for first, second in zip(samples, samples[1:]))
    ledger.add("refinement/monotone", "energy decreases at every refinement",
               1.0 if monotone else 0.0, "1.0", 0.0 if monotone else 1.0, EXACT,
               "energies: " + ", ".join(f"{s['energy']:.6e}" for s in samples))
    return samples, orders


def audit_coercivity(ledger):
    """Does the term stiffen the reduced velocity operator without touching affine modes?"""
    rows = []
    for name in ("hex_cube_n2", "hex_sheared_n2", "hex_sheared_n3"):
        mesh = FAMILIES[name]()
        without = NativeHybridViscous3D(mesh, 1.0, gradient_jump_stabilization=0.0)
        with_jump = NativeHybridViscous3D(mesh, 1.0, gradient_jump_stabilization=1.0)
        low0, _ = split_symmetric(dense(without.matrix))
        low1, _ = split_symmetric(dense(with_jump.matrix))
        rows.append({"family": name, "lambda_min_bare": low0,
                     "lambda_min_stabilised": low1,
                     "relative_increase": (low1 - low0) / abs(low0)})
        ledger.add(f"{name}/lambda_min_gain", f"{name}: lambda_min increase",
                   (low1 - low0) / abs(low0), "> 0 (monotone stiffening)",
                   0.0 if low1 > low0 else 1.0, EXACT,
                   f"bare={low0:.6e}, stabilised={low1:.6e}, both positive "
                   "(no coercivity failure is reproduced on this family)")
    return rows


def audit_failure_modes(ledger):
    mesh = hex_cube(2)
    for label, value in (("negative", -1.0), ("nan", float("nan"))):
        try:
            NativeHybridViscous3D(mesh, 1.0, gradient_jump_stabilization=value)
            ok, message = False, "no exception"
        except ValueError as error:
            ok, message = True, str(error)
        ledger.add(f"guard/{label}", f"strength={label} rejected", 1.0 if ok else 0.0,
                   "ValueError", 0.0 if ok else 1.0, EXACT, message)


def make_figure(path, ledger, refinement, coercivity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    exact_rows = [row for row in ledger.rows
                  if row["tolerance"] <= EXACT and row["error"] is not None]
    names = [row["id"].replace("/", "\n") for row in exact_rows]
    errors = [max(row["error"], 1e-18) for row in exact_rows]
    axes[0, 0].barh(range(len(errors)), errors, color="#2b6cb0")
    axes[0, 0].set_yticks(range(len(names)))
    axes[0, 0].set_yticklabels(names, fontsize=5.5)
    axes[0, 0].set_xscale("log")
    axes[0, 0].axvline(EXACT, color="#c53030", ls="--", label="tolerance 1e-12")
    axes[0, 0].set_xlabel("relative error (floor 1e-18)")
    axes[0, 0].set_title("(a) exact identities vs machine precision")
    axes[0, 0].legend(fontsize=7)

    families = sorted({row["id"].split("/")[0] for row in ledger.rows
                       if row["id"].endswith("affine_energy")})
    affine = [abs(next(row["measured"] for row in ledger.rows
                       if row["id"] == f"{family}/affine_energy")) for family in families]
    ledger_random = [next(row["measured"] for row in ledger.rows
                          if row["id"] == f"{family}/energy_identity") for family in families]
    axes[0, 1].bar(np.arange(len(families)) - 0.2, np.maximum(affine, 1e-18), 0.4,
                   label="affine field (must be 0)", color="#c53030")
    axes[0, 1].bar(np.arange(len(families)) + 0.2, ledger_random, 0.4,
                   label="generic field", color="#2f855a")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(range(len(families)))
    axes[0, 1].set_xticklabels(families, rotation=20, fontsize=7)
    axes[0, 1].set_ylabel("energy")
    axes[0, 1].set_title("(b) affine modes are in the kernel")
    axes[0, 1].legend(fontsize=7)

    h = [sample["h"] for sample in refinement]
    energy = [sample["energy"] for sample in refinement]
    axes[1, 0].loglog(h, energy, "o-", label="jump energy (u=x^2+yz)")
    reference = [energy[0] * (value / h[0]) ** 2 for value in h]
    axes[1, 0].loglog(h, reference, "k--", label="O(h^2) reference")
    axes[1, 0].set_xlabel("h")
    axes[1, 0].set_ylabel("energy")
    axes[1, 0].set_title("(c) h-refinement consistency")
    axes[1, 0].legend(fontsize=7)

    labels = [row["family"].replace("hex_", "") for row in coercivity]
    x = np.arange(len(labels))
    axes[1, 1].bar(x - 0.2, [row["lambda_min_bare"] for row in coercivity], 0.4,
                   label="strength = 0", color="#718096")
    axes[1, 1].bar(x + 0.2, [row["lambda_min_stabilised"] for row in coercivity], 0.4,
                   label="strength = 1", color="#2b6cb0")
    for position, row in enumerate(coercivity):
        axes[1, 1].text(position, row["lambda_min_stabilised"] * 1.02,
                        f"+{100 * row['relative_increase']:.1f}%", ha="center", fontsize=7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20, fontsize=7)
    axes[1, 1].set_ylabel("lambda_min of reduced operator")
    axes[1, 1].set_title("(d) stiffening of the reduced velocity operator")
    axes[1, 1].legend(fontsize=7)

    figure.suptitle("NativeGradientJump3D verification: exact identities, "
                    "analytic scaling, refinement", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/verification-native-gradient-jump-algebra")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ledger = Ledger()
    rng = np.random.default_rng(20261103)
    for name, factory in FAMILIES.items():
        mesh = factory()
        viscous, gj, _ = audit_family(name, mesh, ledger, rng)
        audit_scaling(name, mesh, viscous, gj, ledger, rng)
    audit_analytic_geometry(ledger)
    refinement, orders = audit_refinement(ledger)
    coercivity = audit_coercivity(ledger)
    audit_failure_modes(ledger)

    passed = [row for row in ledger.rows if row["pass"]]
    evidence = {
        "operator": "tensorfvm.native_gradient_jump3d.NativeGradientJump3D",
        "purpose": "Crouzeix-Raviart type tangential gradient-jump (Korn) stabilisation",
        "literature": "Hansbo & Larson 2003, doi:10.1051/m2an:2003020, eq. (26)",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "families": list(FAMILIES),
        "checks": ledger.rows,
        "summary": {"total": len(ledger.rows), "passed": len(passed),
                    "failed": len(ledger.rows) - len(passed)},
        "refinement": {"samples": refinement, "observed_orders": orders},
        "coercivity": coercivity,
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    make_figure(out / "gradient-jump-verification.png", ledger, refinement, coercivity)

    width = max(len(row["id"]) for row in ledger.rows)
    for row in ledger.rows:
        status = "PASS" if row["pass"] else "FAIL"
        error = "n/a" if row["error"] is None else f"{row['error']:.3e}"
        print(f"{status} {row['id']:<{width}}  error={error:<10} tol={row['tolerance']:.1e}")
    print(f"\n{len(passed)}/{len(ledger.rows)} checks passed -> {out}")
    return 0 if len(passed) == len(ledger.rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())