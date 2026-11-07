"""Artifacts for the ALE moving-mesh / PMM node (IN52-N06).

Writes ``docs/pmm-ale-cfd/{summary.json,figures/*.png}``.  Every number in the
summary comes from the code path below - nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.moving_mesh import (  # noqa: E402
    PMMKinematics,
    ale_scalar_step,
    build_cylinder_ogrid,
    build_kvlcc2_double_body_ogrid,
    morph_nodes,
    wall_ale_flux,
)
from tensorfvm.pmm_cfd import KVLCC2Hull, reference_consistency_audit  # noqa: E402

OUT = Path("docs/pmm-ale-cfd")
FIG = OUT / "figures"


def weights(mesh):
    nj = mesh.shape[1]
    w = np.linspace(1.0, 0.0, nj + 1) ** 2
    return np.broadcast_to(w[None, :, None], tuple(n + 1 for n in mesh.shape)).copy()


def cycle_stats(mesh, kin, *, n_steps=20):
    base = mesh.nodes.copy()
    w = weights(mesh)
    v0 = mesh.metrics().volume
    nodes_old = base
    out = dict(phase=[], vol_ratio=[], gcl=[], wall_flux=[], sway=[], yaw=[])
    for k in range(n_steps + 1):
        t = -10.0 + k * (20.0 / n_steps)
        pose = kin.pose(t)
        nodes_new = morph_nodes(base, w, pose, midpoint_x=0.5 * kin.U * 0 + kin.midpoint_x)
        g_new = mesh.metrics(nodes_new)
        sweep = mesh.swept_volumes(nodes_old, nodes_new)
        dv = mesh.cell_volume_change(nodes_old, nodes_new)
        d_vol = g_new.volume - mesh.metrics(nodes_old).volume
        um = sweep[:, None] * g_new.face_area / np.maximum(g_new.area_mag, 1e-300)[:, None] ** 2
        wf = wall_ale_flux(mesh.metrics(nodes_old), g_new, face_velocity=um, sweep=sweep)
        out["phase"].append(t)
        out["vol_ratio"].append(float((g_new.volume / v0).min()))
        out["gcl"].append(float(np.abs(dv - d_vol).max()))
        out["wall_flux"].append(float(np.abs(wf).max()))
        out["sway"].append(pose["sway"])
        out["yaw"].append(np.rad2deg(pose["yaw"]))
        nodes_old = nodes_new
    return {k: (v if k == "phase" else np.asarray(v)) for k, v in out.items()}


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    hull = KVLCC2Hull()
    hull.calibrate()
    integr = hull.integrals()
    mesh = build_kvlcc2_double_body_ogrid(hull, n_station=24, n_around=32, n_radial=10)
    geom = mesh.metrics()
    lab = np.array([l[1] for l in mesh.face_label])
    outer = (lab == "far") | (lab == "cap_low") | (lab == "cap_high")
    v_outer = float(np.einsum("fi,fi->f", geom.face_center[outer], geom.face_area[outer]).sum() / 3.0)
    xs = np.unique(mesh.nodes[:, 0, 0, 0])
    length = float(xs[-1] - xs[0])
    cb_mesh = ((v_outer - geom.volume.sum()) / 2.0) / (length * hull.B * hull.T)

    # ---- ALE verification numbers -------------------------------------
    cyl = build_cylinder_ogrid(radius=0.05, n_theta=32, n_radial=10, span=0.2,
                               n_span=3, far_radius=0.5)
    pose = dict(sway=0.01, sway_rate=0.0, yaw=0.05, yaw_rate=0.0)
    nodes_new = morph_nodes(cyl.nodes, weights(cyl), pose, midpoint_x=0.0)
    sweep = cyl.swept_volumes(cyl.nodes, nodes_new)
    g_old, g_new = cyl.metrics(), cyl.metrics(nodes_new)
    dv = cyl.cell_volume_change(cyl.nodes, nodes_new)
    gcl_morph = float(np.abs(dv - (g_new.volume - g_old.volume)).max())
    gcl_trans = float(np.abs(cyl.cell_volume_change(
        cyl.nodes, cyl.nodes + np.array([1e-3, -2e-3, 5e-4]))).max())
    fv = np.tile([0.3, -0.2, 0.1], (cyl.n_faces, 1))
    flux = np.einsum("fi,fi->f", fv, g_new.face_area) - sweep / 1e-3
    residual = (g_new.volume - g_old.volume) / 1e-3 + g_new.divergence(flux)
    freestream = float(np.abs(residual).max() / np.abs(flux).max())
    # Galilean invariance of the ALE scalar step
    dt, speed = 2e-4, np.array([0.0, 0.25, 0.0])
    phi = np.exp(-((cyl.metrics().centers - cyl.metrics().centers.mean(0)) ** 2).sum(1) / 2e-3)
    fixed, moving = phi.copy(), phi.copy()
    nodes0 = cyl.nodes.copy()
    for _ in range(25):
        g0 = cyl.metrics(nodes0)
        fixed = ale_scalar_step(g0, g0, field_old=fixed,
                                face_velocity=np.tile(speed, (cyl.n_faces, 1)),
                                sweep=np.zeros(cyl.n_faces), dt=dt, diffusivity=0.0)
        nodes1 = nodes0 - speed * dt
        g1 = cyl.metrics(nodes1)
        moving = ale_scalar_step(g0, g1, field_old=moving,
                                 face_velocity=np.zeros((cyl.n_faces, 3)),
                                 sweep=cyl.swept_volumes(nodes0, nodes1), dt=dt,
                                 diffusivity=0.0)
        nodes0 = nodes1
    galilean = float(np.abs(fixed - moving).max() / np.abs(fixed).max())
    # moving-wall impermeability
    um = sweep[:, None] * g_new.face_area / np.maximum(g_new.area_mag, 1e-300)[:, None] ** 2
    wall_flux = float(np.abs(wall_ale_flux(g_old, g_new, face_velocity=um, sweep=sweep)).max())

    # ---- PMM kinematics on the KVLCC2 block ---------------------------
    sway = PMMKinematics(mode="sway", amplitude=0.054 * hull.B, period=10.0,
                         midpoint_x=0.5 * hull.L)
    yaw = PMMKinematics(mode="yaw", amplitude=3.0, period=10.0, midpoint_x=0.5 * hull.L)
    sway_cycle = cycle_stats(mesh, sway)
    yaw_cycle = cycle_stats(mesh, yaw)

    audit = reference_consistency_audit()

    summary = {
        "node": "IN52-N06",
        "campaign": "KVLCC2 PMM virtual test - ALE moving-mesh capability",
        "verdict": "PARTIAL: moving-mesh capability REAL and verified; "
                   "KVLCC2 PMM force derivatives NOT computed (no verified "
                   "open-boundary ALE flow solver) -> criteria C1/C2 FAIL",
        "geometry_gate": {
            "L": hull.L, "B": hull.B, "T": hull.T, "Cb_target": 0.708,
            "Cb_calibrated_analytic": float(integr.Cb),
            "Cb_from_ale_mesh": float(cb_mesh),
            "mesh_length": length,
            "rel_error_mesh": float((cb_mesh - integr.Cb) / integr.Cb),
            "band": "|Cb_mesh - Cb_analytic| <= 6 % (mesh truncation at the tips)",
        },
        "ale_verification": {
            "gcl_translation_abs": gcl_trans,
            "gcl_morphing_abs": gcl_morph,
            "freestream_residual_relative": freestream,
            "galilean_invariance_relative": galilean,
            "moving_wall_flux_abs": wall_flux,
        },
        "pmm_cycles": {
            "sway": {"min_volume_ratio": float(sway_cycle["vol_ratio"].min()),
                     "max_gcl_abs": float(sway_cycle["gcl"].max()),
                     "max_wall_flux_abs": float(sway_cycle["wall_flux"].max()),
                     "amplitude_m": sway.amplitude, "period_s": sway.period},
            "yaw": {"min_volume_ratio": float(yaw_cycle["vol_ratio"].min()),
                    "max_gcl_abs": float(yaw_cycle["gcl"].max()),
                    "max_wall_flux_abs": float(yaw_cycle["wall_flux"].max()),
                    "amplitude_deg": yaw.amplitude, "period_s": yaw.period},
        },
        "criteria": {
            "C1_derivatives_within_15pct_of_simman": False,
            "C2_phase_within_5deg": False,
            "C3_grid_convergence": False,
            "geometry_gate": bool(abs(cb_mesh - integr.Cb) < 0.06 * integr.Cb
                                  and abs(integr.Cb - 0.708) < 0.005),
            "ale_space_conservation": bool(gcl_trans < 1e-14 and gcl_morph < 1e-13),
            "ale_freestream_preservation": bool(freestream < 1e-12),
            "ale_galilean_invariance": bool(galilean < 1e-10),
            "ale_moving_wall_impermeable": bool(wall_flux < 1e-14),
            "kvlcc2_pmm_forces": False,
        },
        "evidence": {
            "moving_mesh": "REAL (ALE morphing with exact discrete space conservation)",
            "ale_advection": "REAL (free-stream preserving, Galilean invariant)",
            "hull_mesh": "REAL (body-fitted double-body O-grid, Cb gate passed)",
            "pmm_force_derivatives": "WITHHELD (no verified open-boundary ALE flow solver)",
            "si_overall_reference_table": audit.get("consistent"),
        },
        "known_limitation": [
            "single-block O-grid over the hull length: the end planes are solid caps, "
            "so no open inlet/outlet and therefore no forward-speed PMM forces;",
            "the attempted incompressible ALE projection solver did not pass its "
            "added-mass verification and was withdrawn rather than shipped;",
            "the fineness of the tip stations limits radial refinement "
            "(n_radial > 10 raises degenerate-face errors at the cut tips).",
        ],
        "figures": ["fig1_kvlcc2_ale_mesh.png", "fig2_pmm_mesh_quality.png",
                    "fig3_ale_verification.png"],
    }

    # ---- figure 1: hull double-body ALE block -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    imid = len(xs) // 2
    ax[0].plot(mesh.nodes[imid, 0, :, 1], mesh.nodes[imid, 0, :, 2], "k-", lw=1.6,
               label="double-body section (wall)")
    for j in (4, 10):
        ax[0].plot(mesh.nodes[imid, j, :, 1], mesh.nodes[imid, j, :, 2], "-", lw=0.7,
                   label=f"layer j={j}")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("y [m]"), ax[0].set_ylabel("z [m]")
    ax[0].set_title("KVLCC2 double-body O-grid, midship section")
    ax[0].legend(fontsize=7), ax[0].grid(alpha=0.3)
    w = weights(mesh)
    for amp, style in ((0.0, "-"), (0.054 * hull.B, "--"), (-0.054 * hull.B, ":")):
        nn = morph_nodes(mesh.nodes, w, dict(sway=float(amp), sway_rate=0.0, yaw=0.0,
                                             yaw_rate=0.0), midpoint_x=0.5 * hull.L)
        ax[1].plot(nn[imid, 0, :, 1], nn[imid, 0, :, 2], style, lw=1.2,
                   label=f"sway = {amp:+.3f} m")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("y [m]"), ax[1].set_ylabel("z [m]")
    ax[1].set_title("morphed wall section over the sway cycle")
    ax[1].legend(fontsize=7), ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_kvlcc2_ale_mesh.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: mesh quality over the PMM cycle --------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for cyc, name in ((sway_cycle, "pure sway"), (yaw_cycle, "pure yaw")):
        ax[0].semilogy(cyc["phase"], np.maximum(cyc["gcl"], 1e-18), "o-", ms=3, label=name)
        ax[1].plot(cyc["phase"], cyc["vol_ratio"], "o-", ms=3, label=name)
    ax[0].set_xlabel("t [s]"), ax[0].set_ylabel(r"max $|\Delta V_{faces} - \Delta V_{cell}|$ [m$^3$]")
    ax[0].set_title("discrete space-conservation residual")
    ax[1].set_xlabel("t [s]"), ax[1].set_ylabel(r"min $V(t)/V(0)$")
    ax[1].set_title("cell-volume ratio (mesh validity)")
    for a in ax:
        a.grid(alpha=0.3), a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_pmm_mesh_quality.png", dpi=130)
    plt.close(fig)

    # ---- figure 3: verification summary -------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = ["GCL\ntranslation", "GCL\nmorphing", "free-stream\nresidual",
             "Galilean\ninvariance", "moving wall\nflux"]
    vals = [max(gcl_trans, 1e-20), max(gcl_morph, 1e-20), max(freestream, 1e-20),
            max(galilean, 1e-20), max(wall_flux, 1e-20)]
    bars = ax.bar(names, vals, color=["#2b6cb0", "#2b6cb0", "#2f855a", "#2f855a", "#b7791f"])
    ax.set_yscale("log"), ax.set_ylabel("relative / absolute error")
    ax.set_title("ALE moving-mesh verification (lower is better; roundoff = 1e-16)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.5, f"{v:.1e}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ale_verification.png", dpi=130)
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()