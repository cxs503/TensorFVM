"""Production-operator legs for the three-grid log--log-slope order estimator.

Motivation (verification debt, declared by
``docs/verification-three-grid-loglog-cellcentre/REPORT.md`` §5.5)
--------------------------------------------------------------------
:mod:`tensorfvm.verification.loglog_order` verified the *estimator* and two
textbook 1-D periodic cell-centred operators against their formal orders.  Its
report explicitly withholds three things:

* the estimator has **never been applied to a production TensorFVM operator**
  (the 1-D ladders are stand-alone stencils, not the native mesh machinery);
* **boundary treatment is not covered** -- "a near-wall first-order boundary
  condition drags the global order down to 1" is quoted as a known trap but is
  not measured anywhere;
* three-dimensional, genuinely non-orthogonal meshes are not covered.

This module closes the first two and measures the third, by running the *same*
estimator on the production scalar-diffusion path
(:class:`tensorfvm.native_viscous3d.NativeViscous3D`, the operator that
``native_*`` production solvers assemble) on three grid levels.

Problem
-------
``div(mu grad u) = 0`` on the unit cube with Dirichlet data, where

    u(x, y, z) = exp(k (z - 1/2)) cos(k x),      k = 2,

which is harmonic (``lap u == 0`` identically, verified to round-off in
:func:`harmonic_residual`), so the Dirichlet problem's exact solution *is*
``u``.  The discrete solution is compared with

* the exact **cell average** ``(1/V) Int u`` (the admissible reference for a
  cell-centred scheme; closed form on the uniform cube mesh), and
* the exact **point value** ``u(x_c)`` (what the production verification
  studies in :mod:`tensorfvm.verification.native_order3d` use).

Predictions, declared *before* any run
--------------------------------------
Let ``h`` be the cell size.  The interior flux is a second-order face gradient
and the Dirichlet datum is imposed at the face centroid, so:

===============  ======================================  =====================
leg              setup                                   predicted order
===============  ======================================  =====================
``A``            exact face datum, cell-average ref, L2  2
``A-pt``         exact face datum, point-value ref, L2   2
``B``            datum placed at the owner cell centre   diagnostic (no
                 -- the two-point face flux then reads     formal order)
                 ``u_P - u_P = 0``, i.e. Dirichlet
                 silently becomes zero-flux
``B-int``        same, interior cells only, L2           2
``B-wall``       same, wall layer L-infinity            diagnostic (O(1)
                                                          error: no decay)
``C``            warped mesh (warp = 0.1), point value   2
===============  ======================================  =====================

The ``B`` legs are deliberately *not* given a formal order.  Substituting the
owner-cell value for the face datum does not approximate the face value to
first order: it makes the two-point boundary flux identically zero, so the
Dirichlet face behaves as an adiabatic wall with an ``O(1)`` flux error.  The
error is then confined to a wall layer of cell-fraction ``O(h)``, giving
``L2 = sqrt(h * 1) = h^0.5`` and no decay at all in ``L-inf`` -- which is why
they are reported as diagnostics (INFO) and the interior leg ``B-int`` carries
the gate.  All gates use the platform's 15 % band.

Environment note
----------------
The numpy 2.4.6 build in the authoring container inverts a batch of 3x3 matrices
through LAPACK at ~1.4 ms per matrix instead of ~1 us (measured: 3.7 s for
512 matrices), which makes production operator assembly 40 s at ``n = 8`` and
minutes at ``n >= 32`` -- a pure environment defect, not solver work.  The
harness therefore installs :func:`fast3x3_inverse` (closed-form adjugate,
verified against ``numpy.linalg.inv`` to 1.7e-16 by
``tests/test_loglog_order_bc3d.py``) for the duration of a run and restores the
original afterwards.  Nothing else in the production path is touched.

Command line
------------
``python -m tensorfvm.verification.loglog_order_bc3d [n0,n1,n2] [warp]``
writes ``summary.json``, ``fig_bc_order.png`` and ``REPORT.md`` into
``docs/verification-loglog-order-bc3d/``.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .loglog_order import cell_error_norms, loglog_order

__all__ = [
    "K",
    "ORDER_TOLERANCE",
    "PREDICTIONS",
    "fast3x3_inverse",
    "numpy_inverse_shim",
    "harmonic_scalar",
    "harmonic_residual",
    "cube_cell_averages",
    "boundary_data",
    "solve_leg",
    "ladder",
    "run",
    "figure",
    "main",
    "OUTPUT_DIR",
]

K = 2.0
ORDER_TOLERANCE = 0.15

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "docs" / "verification-loglog-order-bc3d"

#: Declared a priori (see module docstring); theory_order per leg.
PREDICTIONS: Tuple[Tuple[str, str, Optional[float]], ...] = (
    ("A", "uniform, exact face datum, cell-average ref, L2", 2.0),
    ("A-pt", "uniform, exact face datum, point-value ref, L2", 2.0),
    ("B", "uniform, datum at the owner cell centre (zero-flux collapse), L2", None),
    ("B-int", "uniform, same datum, interior-only L2", 2.0),
    ("B-wall", "uniform, same datum, wall-layer L-inf", None),
    ("C", "warped 0.1, exact face datum, point-value ref, L2", 2.0),
)


# --------------------------------------------------------------------------- #
# environment shim (closed-form 3x3 inverse)
# --------------------------------------------------------------------------- #
def fast3x3_inverse(a):
    """Closed-form batch inverse of 3x3 matrices; falls back for other shapes."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        return _ORIGINAL_INV(a)
    c00 = arr[:, 1, 1] * arr[:, 2, 2] - arr[:, 1, 2] * arr[:, 2, 1]
    c01 = arr[:, 1, 0] * arr[:, 2, 2] - arr[:, 1, 2] * arr[:, 2, 0]
    c02 = arr[:, 1, 0] * arr[:, 2, 1] - arr[:, 1, 1] * arr[:, 2, 0]
    det = arr[:, 0, 0] * c00 - arr[:, 0, 1] * c01 + arr[:, 0, 2] * c02
    out = np.empty_like(arr)
    out[:, 0, 0] = c00 / det
    out[:, 0, 1] = (arr[:, 0, 2] * arr[:, 2, 1] - arr[:, 0, 1] * arr[:, 2, 2]) / det
    out[:, 0, 2] = (arr[:, 0, 1] * arr[:, 1, 2] - arr[:, 0, 2] * arr[:, 1, 1]) / det
    out[:, 1, 0] = (arr[:, 1, 2] * arr[:, 2, 0] - arr[:, 1, 0] * arr[:, 2, 2]) / det
    out[:, 1, 1] = (arr[:, 0, 0] * arr[:, 2, 2] - arr[:, 0, 2] * arr[:, 2, 0]) / det
    out[:, 1, 2] = (arr[:, 0, 2] * arr[:, 1, 0] - arr[:, 0, 0] * arr[:, 1, 2]) / det
    out[:, 2, 0] = (arr[:, 1, 0] * arr[:, 2, 1] - arr[:, 1, 1] * arr[:, 2, 0]) / det
    out[:, 2, 1] = (arr[:, 0, 1] * arr[:, 2, 0] - arr[:, 0, 0] * arr[:, 2, 1]) / det
    out[:, 2, 2] = (arr[:, 0, 0] * arr[:, 1, 1] - arr[:, 0, 1] * arr[:, 1, 0]) / det
    return out


_ORIGINAL_INV = np.linalg.inv


@contextmanager
def numpy_inverse_shim(enabled: bool = True):
    """Temporarily route ``numpy.linalg.inv`` through :func:`fast3x3_inverse`."""
    if not enabled:
        yield False
        return
    np.linalg.inv = fast3x3_inverse
    try:
        yield True
    finally:
        np.linalg.inv = _ORIGINAL_INV


# --------------------------------------------------------------------------- #
# exact solution and references
# --------------------------------------------------------------------------- #
def harmonic_scalar(points, k: float = K) -> np.ndarray:
    """``u = exp(k (z - 1/2)) cos(k x)`` -- harmonic, ``lap u = 0`` identically."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    return np.exp(k * (p[:, 2] - 0.5)) * np.cos(k * p[:, 0])


def harmonic_residual(points, k: float = K) -> float:
    """Finite-difference residual of ``lap u`` normalised by ``k^2 |u|``.

    Not a proof -- :func:`harmonic_scalar` is analytic -- but a transcription
    guard: a sign slip in the closed form shows up here as ``O(1)``.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    eps = 1e-5
    lap = np.zeros(len(p))
    for axis in (0, 2):                      # u depends on x and z only
        e = np.zeros(3)
        e[axis] = eps
        lap += (harmonic_scalar(p + e, k) - 2.0 * harmonic_scalar(p, k)
                + harmonic_scalar(p - e, k)) / eps ** 2
    scale = k ** 2 * np.abs(harmonic_scalar(p, k))
    return float(np.max(np.abs(lap) / np.maximum(scale, 1e-30)))


def cube_cell_averages(centers, h: float, k: float = K) -> np.ndarray:
    """Exact cell average on an axis-aligned cube of side ``h`` (separable)."""
    c = np.atleast_2d(np.asarray(centers, dtype=float))
    x, z = c[:, 0], c[:, 2]
    avg_x = (np.sin(k * (x + h / 2)) - np.sin(k * (x - h / 2))) / (k * h)
    avg_z = (np.exp(k * (z + h / 2 - 0.5)) - np.exp(k * (z - h / 2 - 0.5))) / (k * h)
    return avg_x * avg_z


def boundary_data(op, faces: np.ndarray, mode: str, k: float = K) -> np.ndarray:
    """Full face array of Dirichlet data for the two boundary treatments.

    ``mode='face_value'``  exact value at the face centroid -- second-order;
    ``mode='cell_value'``  value at the owner cell centre -- the first-order
                           "use the cell value at the wall" convention.
    """
    if mode not in ("face_value", "cell_value"):
        raise ValueError("mode must be 'face_value' or 'cell_value'")
    mesh = op.mesh
    data = np.zeros(op.F)
    selected = mesh.face_centers[faces] if mode == "face_value" else mesh.cell_centers[mesh.owner[faces]]
    data[faces] = harmonic_scalar(selected, k)
    return data


# --------------------------------------------------------------------------- #
# one grid level
# --------------------------------------------------------------------------- #
def solve_leg(n: int, *, warp: float = 0.0, mode: str = "face_value",
              viscosity: float = 1.0, k: float = K) -> Dict[str, object]:
    """Assemble the production scalar operator and solve the Dirichlet problem."""
    from scipy.sparse.linalg import spsolve
    from tensorfvm.native_viscous3d import NativeViscous3D
    from tensorfvm.verification.native_order3d import hex_box

    t0 = time.perf_counter()
    # a warped quad face is non-planar: the mesh builder requires triangles there
    mesh = hex_box(int(n), float(warp), triangulate_faces=bool(warp))
    t_mesh = time.perf_counter()
    op = NativeViscous3D(mesh, mesh.neighbor < 0, viscosity, wall_closure="two_point")
    t_op = time.perf_counter()
    faces = op.dirichlet
    data = boundary_data(op, faces, mode, k)
    rhs = -(op.divergence_matrix @ op.scalar_flux_source(dirichlet_values=data))
    sol = spsolve(op.scalar_diffusion_matrix.tocsc(), rhs)
    t_solve = time.perf_counter()
    residual = float(np.linalg.norm(op.scalar_diffusion_matrix @ sol - rhs)
                     / max(np.linalg.norm(rhs), 1e-30))

    cc = mesh.cell_centers
    vol = np.asarray(mesh.cell_volumes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=int).reshape(-1, 8)
    vertices = np.asarray(mesh.vertices, dtype=float)[cells]
    exact_avg = cube_cell_averages(cc, 1.0 / n, k) if warp == 0.0 else None
    exact_point = harmonic_scalar(cc, k)
    layer = ((cc < 1.0 / n) | (cc > 1.0 - 1.0 / n)).any(axis=1)
    h = float(np.mean(vol) ** (1.0 / 3.0))
    row: Dict[str, object] = {
        "n": int(n), "h": h, "warp": float(warp), "mode": mode,
        "cells": int(len(mesh.cells)), "faces": int(len(mesh.owner)),
        "dirichlet_faces": int(faces.sum()),
        "residual": residual, "wall_fraction": float(layer.mean()),
        "vertex_span": float(np.abs(vertices).sum(axis=(0, 1)).min()),
        "t_mesh_s": t_mesh - t0, "t_operator_s": t_op - t_mesh,
        "t_solve_s": t_solve - t_op,
    }
    for name, ref in (("cell_average", exact_avg), ("point_value", exact_point)):
        if ref is None:
            continue
        err = sol - ref
        norms = cell_error_norms(err, vol)
        row[f"{name}_l2"] = norms["l2"]
        row[f"{name}_linf"] = norms["linf"]
        row[f"{name}_l2_interior"] = float(math.sqrt(np.mean(err[~layer] ** 2)))
        row[f"{name}_linf_wall"] = float(np.max(np.abs(err[layer])))
    return row


def ladder(n_cells: Sequence[int], *, warp: float = 0.0, mode: str = "face_value",
           reference: str = "cell_average", metric: str = "l2",
           theory: Optional[float] = 2.0, k: float = K) -> Dict[str, object]:
    """Run the production solve on each level and fit the log--log slope."""
    rows = [solve_leg(n, warp=warp, mode=mode, k=k) for n in n_cells]
    key = f"{reference}_{metric}"
    spacings = [float(r["h"]) for r in rows]
    errors = [float(r[key]) for r in rows]
    study = loglog_order(spacings, errors, theory_order=theory,
                         tolerance=ORDER_TOLERANCE)
    return dict(key=key, levels=rows, study=study.as_dict(),
                spacings=spacings, errors=errors)


# --------------------------------------------------------------------------- #
# the whole cross-validation
# --------------------------------------------------------------------------- #
def run(n_cells: Sequence[int] = (8, 16, 24), *, warp: float = 0.1,
        shim: bool = True) -> Dict[str, object]:
    """Every declared leg on the same three-grid ladder."""
    grid = tuple(int(n) for n in n_cells)
    if len(grid) < 3:
        raise ValueError("three grid levels are required")
    legs: Dict[str, Dict[str, object]] = {}
    with numpy_inverse_shim(shim) as used_shim:
        legs["A"] = ladder(grid, mode="face_value", reference="cell_average", theory=2.0)
        legs["A-pt"] = ladder(grid, mode="face_value", reference="point_value", theory=2.0)
        legs["B"] = ladder(grid, mode="cell_value", reference="cell_average", theory=None)
        legs["B-int"] = ladder(grid, mode="cell_value", reference="cell_average",
                               metric="l2_interior", theory=2.0)
        legs["B-wall"] = ladder(grid, mode="cell_value", reference="cell_average",
                                metric="linf_wall", theory=None)
        if warp > 0.0:
            legs["C"] = ladder(grid, warp=warp, mode="face_value",
                               reference="point_value", theory=2.0)
    criteria: List[Dict[str, object]] = []
    for name, description, theory in PREDICTIONS:
        if name not in legs:
            criteria.append(dict(
                id=name, description=description, theory_order=theory,
                observed_order=float("nan"), order_stderr=float("nan"),
                relative_deviation=float("nan"), verdict="NOT_RUN",
                asymptotic=False, withheld_reason="warp leg not requested (warp=0)",
                tolerance=ORDER_TOLERANCE))
            continue
        study = legs[name]["study"]
        criteria.append(dict(
            id=name, description=description, theory_order=theory,
            observed_order=study["order"], order_stderr=study["order_stderr"],
            relative_deviation=study["relative_deviation"],
            verdict=study["verdict"], asymptotic=study["asymptotic"],
            withheld_reason=study.get("withheld_reason", ""),
            tolerance=ORDER_TOLERANCE,
        ))
    # the two boundary legs must differ: measuring both is what separates
    # "the trap is quoted" from "the trap is quantified".
    sep = dict(
        id="B-separation",
        description="wall-layer L-inf order (B-wall) is below the interior order (B-int)",
        wall_order=legs["B-wall"]["study"]["order"],
        interior_order=legs["B-int"]["study"]["order"],
        verdict="PASS" if legs["B-wall"]["study"]["order"] < legs["B-int"]["study"]["order"] else "FAIL",
    )
    # reference convention: cell-average vs point-value error magnitude.
    ratio_rows = []
    for a, b in zip(legs["A"]["levels"], legs["A-pt"]["levels"]):
        ratio_rows.append(dict(n=a["n"], cell_average_l2=a["cell_average_l2"],
                               point_value_l2=b["point_value_l2"],
                               ratio=b["point_value_l2"] / a["cell_average_l2"]))
    passed = sum(1 for c in criteria if c["verdict"] == "PASS")
    info = sum(1 for c in criteria if c["verdict"] == "INFO")
    summary = dict(
        module="tensorfvm.verification.loglog_order_bc3d",
        grids=list(grid), warp=warp, k=K,
        tolerance=ORDER_TOLERANCE,
        environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                         platform=platform.platform(),
                         numpy_inverse_shim=bool(used_shim),
                         shim_note="closed-form 3x3 adjugate; equivalence guarded by tests"),
        harmonic_residual_max=harmonic_residual(np.array([[0.13, 0.42, 0.77],
                                                          [0.61, 0.05, 0.29]])),
        criteria=criteria, separation=sep, reference_ratio=ratio_rows,
        legs={k: dict(study=v["study"], levels=v["levels"], key=v["key"])
              for k, v in legs.items()},
        passed=int(passed), info=int(info),
        failed=int(sum(1 for c in criteria if c["verdict"] == "FAIL")),
    )
    return summary


def figure(summary: Dict[str, object], path: Path) -> None:
    """Two panels: error vs h (log--log) and observed vs predicted order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=140)
    ax = axes[0]
    styles = {"A": ("cell-average ref (L2)", "o"), "A-pt": ("point-value ref (L2)", "s"),
              "B": ("1st-order datum (L2)", "^"), "B-int": ("1st-order datum, interior", "v"),
              "B-wall": ("1st-order datum, wall L-inf", "D")}
    for key, (label, marker) in styles.items():
        leg = summary["legs"].get(key)
        if leg is None:
            continue
        ax.loglog(leg["spacings"], leg["errors"], marker + "-", label=label)
    hs = np.array(summary["legs"]["A"]["spacings"])
    ref = summary["legs"]["A"]["errors"][0]
    for slope, ls in ((1.0, ":"), (1.5, "--"), (2.0, "-")):
        ax.loglog(hs, ref * (hs / hs[0]) ** slope, ls, color="0.6", lw=0.8,
                  label=f"slope {slope:g}")
    ax.set_xlabel("h = (V/N)^(1/3)")
    ax.set_ylabel("error")
    ax.set_title("production scalar diffusion, error vs h")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    shown = [c for c in summary["criteria"] if c["verdict"] != "NOT_RUN"]
    ids = [str(c["id"]) for c in shown]
    obs = [float(c["observed_order"]) for c in shown]
    th = [np.nan if c["theory_order"] is None else float(c["theory_order"])
          for c in shown]
    x = np.arange(len(ids))
    ax.bar(x - 0.2, np.nan_to_num(th), 0.4, label="predicted")
    ax.bar(x + 0.2, obs, 0.4, label="observed (log--log fit)")
    for i, c in enumerate(shown):
        ax.text(i, max(0.0 if np.isnan(th[i]) else th[i], obs[i]) + 0.06, c["verdict"],
                ha="center", fontsize=7,
                color="green" if c["verdict"] == "PASS" else "red")
    ax.set_xticks(x, ids, fontsize=8)
    ax.set_ylabel("order p")
    ax.set_title("observed vs predicted order (bands: 15 %)")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# Three-grid log--log-slope order verification on the production scalar operator",
        "",
        f"grids = {summary['grids']}, k = {summary['k']}, tolerance = "
        f"{summary['tolerance']:.0%}, harmonic residual = "
        f"{summary['harmonic_residual_max']:.2e}, "
        f"numpy inverse shim = {summary['environment']['numpy_inverse_shim']}",
        "",
        "## Observed vs predicted order",
        "",
        "| leg | setup | predicted | observed | stderr | relative deviation | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in summary["criteria"]:
        lines.append(
            f"| {c['id']} | {c['description']} | "
            f"{'--' if c['theory_order'] is None else format(c['theory_order'], '.2f')} | "
            f"{float(c['observed_order']):.4f} | {float(c['order_stderr']):.4f} | "
            f"{float(c['relative_deviation']):.3%} | {c['verdict']} |")
    lines += ["", "## Error ladders", "",
              "| leg | " + " | ".join(f"e(n={r['n']})" for r in summary["legs"]["A"]["levels"]) + " |",
              "|---" * (1 + len(summary["legs"]["A"]["levels"])) + "|"]
    for c in summary["criteria"]:
        leg = summary["legs"].get(c["id"])
        if leg is None:
            lines.append(f"| {c['id']} | not run |")
            continue
        lines.append(f"| {c['id']} | " + " | ".join(f"{e:.6e}" for e in leg["errors"]) + " |")
    lines += ["", "## Reference convention (cell average vs point value)", "",
              "| n | cell-average L2 | point-value L2 | ratio |", "|---|---|---|---|"]
    for r in summary["reference_ratio"]:
        lines.append(f"| {r['n']} | {r['cell_average_l2']:.6e} | "
                     f"{r['point_value_l2']:.6e} | {r['ratio']:.4f} |")
    lines += ["", f"PASS {summary['passed']} / FAIL {summary['failed']} / "
                  f"INFO {summary.get('info', 0)}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    grid = tuple(int(v) for v in argv[0].split(",")) if argv else (8, 16, 24)
    warp = float(argv[1]) if len(argv) > 1 else 0.1
    summary = run(grid, warp=warp)
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure(summary, out / "fig_bc_order.png")
    (out / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
    for c in summary["criteria"]:
        pred = "--" if c["theory_order"] is None else format(c["theory_order"], ".2f")
        dev = "--" if c["theory_order"] is None else f"{float(c['relative_deviation']):.2%}"
        print(f"{c['id']:7s} p={float(c['observed_order']):.4f} pred={pred} dev={dev} "
              f"{c['verdict']} {c['withheld_reason']}")
    print(f"PASS {summary['passed']} / FAIL {summary['failed']} / "
          f"INFO {summary.get('info', 0)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())