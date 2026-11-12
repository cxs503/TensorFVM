"""Independent re-derivation of the m15458 (CASE-CAVITY-RE1000-FVM) evidence bundle.

Why this script exists
----------------------
Mission m15458 delivered a Re = 1000 lid-driven-cavity benchmark against Ghia,
Ghia & Shin (1982) Tables I/II.  A delivery audit must not trust the delivered
scorer: it re-extracts the centreline profiles **from the committed velocity
field** with an independent interpolation (piecewise linear, not the shipped
Hermite), recomputes every headline number, and records where the shipped metric
and the independent one disagree.

Inputs (frozen snapshot taken from the m15458 worktree, git commit 220c729c):
    artifacts/audit-m15458/summary.json
    artifacts/audit-m15458/fields/Re1000_n64.pt
    artifacts/audit-m15458/fields/Re1000_n32.pt

Outputs:
    artifacts/audit-m15458/audit_metrics.json     (all numbers below)
    artifacts/audit-m15458/fig_audit_profiles.png (field-derived profiles vs Ghia)

Run:
    PYTHONPATH=src python3 scripts/audit_cavity_re1000_m15458.py
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(HERE, "artifacts", "audit-m15458")
RE = 1000.0


def load_case(summary: dict, n: int) -> dict:
    return next(c for c in summary["cases"] if c["n"] == n)


def field_profiles(n: int, root: str = BUNDLE):
    """Cell-centred (u, v) from the staggered field; no solver code involved.

    ``u`` has shape (n+1, n): face x = i*h, cell-centre y = (j+0.5)*h.
    ``v`` has shape (n, n+1): cell-centre x = (i+0.5)*h, face y = j*h.
    The vertical centreline x = 0.5 is the face index n//2 (exact, no x
    interpolation); the horizontal centreline y = 0.5 is the v-face index n//2.
    """
    blob = torch.load(os.path.join(root, "fields", f"Re1000_n{n}.pt"),
                      map_location="cpu", weights_only=False)
    u = blob["u"].numpy()
    v = blob["v"].numpy()
    h = 1.0 / n
    y_cells = (np.arange(n) + 0.5) * h
    return u, v, h, y_cells, blob.get("steps")


def metrics_for(n: int, summary: dict, root: str = BUNDLE) -> dict:
    case = load_case(summary, n)
    u, v, h, y_cells, steps = field_profiles(n, root)

    # --- independent extraction of the two Ghia centrelines -----------------
    u_cl = u[n // 2, :]      # x = 0.5 exactly (staggered face)
    v_cl = v[:, n // 2]      # y = 0.5 exactly (staggered face)

    ghia_u = np.array(table(RE, "u"))
    ghia_v = np.array(table(RE, "v"))

    u_lin = np.interp(ghia_u[:, 0], y_cells, u_cl)
    v_lin = np.interp(ghia_v[:, 0], y_cells, v_cl)
    # endpoints sit on Dirichlet walls: the profile samples stop half a cell
    # short of them, so take the boundary condition, not an extrapolation
    u_lin[0], u_lin[-1] = 1.0, 0.0
    v_lin[0], v_lin[-1] = 0.0, 0.0

    shipped_u = np.array(case["u"]["profiles"]["computed"])
    shipped_v = np.array(case["v"]["profiles"]["computed"])
    ref_u = np.array(case["u"]["profiles"]["reference"])
    ref_v = np.array(case["v"]["profiles"]["reference"])

    du_lin = u_lin - ghia_u[:, 1]
    dv_lin = v_lin - ghia_v[:, 1]

    # --- incompressibility and streamfunction -------------------------------
    div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    psi = np.zeros((n, n))
    for j in range(1, n):
        psi[:, j] = psi[:, j - 1] + h * 0.5 * (uc[:, j - 1] + uc[:, j])
    iy, jx = np.unravel_index(np.argmin(psi), psi.shape)
    centre = ((iy + 0.5) * h, (jx + 0.5) * h)   # (x, y)
    gx, gy = VORTEX_CENTRE[RE]
    shipped_centre = tuple(case["vortex_centre"]["computed"])

    i_ref = int(np.argmin(ghia_u[:, 1]))
    j_ref = int(np.argmax(ghia_v[:, 1]))
    return {
        "n": n,
        "steps": steps,
        "reference_arrays_equal_repo_tables": {
            "u": bool(np.array_equal(ref_u, ghia_u[:, 1])),
            "v": bool(np.array_equal(ref_v, ghia_v[:, 1])),
        },
        "shipped_vs_field_profile_max_abs_diff": {
            "u": float(np.abs(shipped_u - u_lin).max()),
            "v": float(np.abs(shipped_v - v_lin).max()),
        },
        "shipped_metric": {
            "u_min": float(np.min(shipped_u)),
            "v_max": float(np.max(shipped_v)),
            "max_abs_du": float(np.abs(shipped_u - ref_u).max()),
            "max_abs_dv": float(np.abs(shipped_v - ref_v).max()),
        },
        "boundary_node_artefact": {
            "shipped_u_at_lid_(y=1)": float(shipped_u[0]),
            "shipped_u_at_bottom_(y=0)": float(shipped_u[-1]),
            "shipped_v_at_x=1": float(shipped_v[0]),
            "shipped_v_at_x=0": float(shipped_v[-1]),
            "true_bc": {"u_lid": 1.0, "u_bottom": 0.0, "v_x1": 0.0, "v_x0": 0.0},
            "shipped_max_abs_du_concentrated_at_bc_node": bool(
                np.abs(shipped_u - ref_u).argmax() == 0),
        },
        "independent_metric": {
            "interp": "piecewise-linear on staggered field, exact BC endpoints",
            "u_min_relative_error_at_ghia_y": float(
                abs(u_lin[i_ref] - ghia_u[i_ref, 1]) / abs(ghia_u[i_ref, 1])),
            "u_min_relative_error_full_profile": float(
                abs(u_cl.min() - ghia_u[i_ref, 1]) / abs(ghia_u[i_ref, 1])),
            "v_max_relative_error_at_ghia_x": float(
                abs(v_lin[j_ref] - ghia_v[j_ref, 1]) / ghia_v[j_ref, 1]),
            "v_max_relative_error_full_profile": float(
                abs(v_cl.max() - ghia_v[j_ref, 1]) / ghia_v[j_ref, 1]),
            "max_abs_du_all_points": float(np.abs(du_lin).max()),
            "max_abs_du_interior_points": float(np.abs(du_lin[1:-1]).max()),
            "max_abs_dv_all_points": float(np.abs(dv_lin).max()),
            "max_abs_dv_interior_points": float(np.abs(dv_lin[1:-1]).max()),
        },
        "vortex": {
            "independent_psi_min": float(psi.min()),
            "independent_centre_xy": list(centre),
            "shipped_centre_xy": list(shipped_centre),
            "ghia_centre_xy": [gx, gy],
            "independent_euclid_dist": float(np.hypot(centre[0] - gx, centre[1] - gy)),
            "independent_chebyshev_dist": float(max(abs(centre[0] - gx), abs(centre[1] - gy))),
            "shipped_euclid_dist": float(np.hypot(shipped_centre[0] - gx, shipped_centre[1] - gy)),
        },
        "incompressibility": {"max_abs_cell_divergence": float(np.abs(div).max())},
        "field_sha256": hashlib.sha256(
            open(os.path.join(root, "fields", f"Re1000_n{n}.pt"), "rb").read()).hexdigest(),
    }


def figure(summary: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ghia_u = np.array(table(RE, "u"))
    ghia_v = np.array(table(RE, "v"))
    for n, style in ((32, ":"), (64, "-")):
        u, v, h, y_cells, _ = field_profiles(n)
        axes[0].plot(u[n // 2, :], y_cells, style, lw=1.6, label=f"FVM n={n}")
        axes[1].plot((np.arange(n) + 0.5) * h, v[:, n // 2], style, lw=1.6, label=f"FVM n={n}")
    axes[0].plot(ghia_u[:, 1], ghia_u[:, 0], "ko", ms=3.5, label="Ghia et al. 1982")
    axes[1].plot(ghia_v[:, 0], ghia_v[:, 1], "ko", ms=3.5, label="Ghia et al. 1982")
    axes[0].set(xlabel="u/U  (x=0.5)", ylabel="y/L", title="Vertical centreline, Re=1000")
    axes[1].set(xlabel="x/L", ylabel="v/U  (y=0.5)", title="Horizontal centreline, Re=1000")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Audit re-extraction from the committed field (m15458)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)


def main() -> None:
    summary = json.load(open(os.path.join(BUNDLE, "summary.json")))
    out = {
        "audit_of": "m15458 / CASE-CAVITY-RE1000-FVM",
        "anchor": summary["anchor"],
        "reynolds": summary["reynolds"],
        "ledger_tolerance": summary["tolerance_fraction"],
        "cases_present_in_bundle": [c["n"] for c in summary["cases"]],
        "cases": [metrics_for(n, summary) for n in (c["n"] for c in summary["cases"])],
    }
    path = os.path.join(BUNDLE, "audit_metrics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    figure(summary, os.path.join(BUNDLE, "fig_audit_profiles.png"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()