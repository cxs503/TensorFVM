"""Generate the figures of the KVLCC2 PMM campaign report.

Usage: ``PYTHONPATH=src python3 scripts/pmm_report_figures.py``
Figures land in ``docs/pmm-cfd/figures/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.pmm_cfd import (
    KVLCC2Hull,
    apply_strip_model,
    run_pmm_campaign,
    section_added_mass_ellipse,
)

OUT = Path("docs/pmm-cfd/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_geometry(hull: KVLCC2Hull) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xi = np.linspace(0, 1, 201)
    wl = np.array([hull.wl_half_breadth_fraction(v) for v in xi])
    axes[0].plot(xi * hull.L, wl * hull.B / 2, lw=2)
    axes[0].fill_between(xi * hull.L, -wl * hull.B / 2, wl * hull.B / 2, alpha=0.15)
    axes[0].set_xlabel("x [m] (AP -> FP)")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title(f"Design waterline  (Cwp = {hull.integrals(nx=241, nz=121).cwp:.3f})")
    axes[0].grid(alpha=0.3)
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        yy, zz = hull.section_outline(frac * hull.L)
        axes[1].plot(yy, zz, lw=1.5, label=f"x/L = {frac}")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("y [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("Body sections (Cm from the section family)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig1_geometry.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_force_histories(hull: KVLCC2Hull, campaign: dict) -> Path:
    cond = campaign["conditions"]
    omega = cond["omega"]
    t = np.linspace(0, 2 * 2 * np.pi / omega, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for a in cond["amplitudes_over_L"]:
        sig = apply_strip_model(hull, t, mode="sway", amplitude=a * hull.L,
                                omega=omega, U=cond["U"], rho=cond["rho"])
        axes[0].plot(t, sig["Y"], lw=1.4, label=f"A/L = {a}")
        sig_y = apply_strip_model(hull, t, mode="yaw", amplitude=np.radians(3.0),
                                  omega=omega, U=cond["U"], rho=cond["rho"])
        axes[1].plot(t, sig_y["N"], lw=1.4, label=f"psi_a = 3 deg (A/L = {a} used as yaw amp.)")
        break
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("Y [N]")
    axes[0].set_title("Pure sway: lateral force")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("N [N m]")
    axes[1].set_title("Pure yaw: yaw moment")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig2_pmm_forces.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_derivative_comparison(campaign: dict) -> Path:
    dev = campaign["deviations"]
    keys = ["Yv", "Nv", "Yvdot", "Nvdot", "Yr", "Nr", "Yrdot", "Nrdot"]
    comp = [dev[k]["computed"] for k in keys]
    table = [dev[k]["reference_table"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    idx = np.arange(len(keys))
    ax.bar(idx - 0.2, comp, 0.4, label="this work (strip, analytic + drag model)")
    ax.bar(idx + 0.2, table, 0.4, label="in-repo SIMMAN table (audited: inconsistent)")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_xticks(idx, keys)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("derivative [-] (ITTC non-dimensionalisation)")
    ax.set_title("Manoeuvring derivatives: computed vs the in-repo reference table")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig3_derivative_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_amplitude_and_plausibility(hull: KVLCC2Hull, campaign: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    amp = campaign["conditions"]["amplitudes_over_L"]
    yv = [r["derivatives"]["Yv"] for r in campaign["runs"]
          if r["mode"] == "sway" and r["n_stations"] == 41]
    yvd = [r["derivatives"]["Yvdot"] for r in campaign["runs"]
           if r["mode"] == "sway" and r["n_stations"] == 41]
    axes[0].plot(amp, yv, "o-", label="Y_v (drag-dominated)")
    axes[0].plot(amp, yvd, "s--", label="Y_vdot (inertial, exact)")
    axes[0].set_xlabel("A/L [-]")
    axes[0].set_ylabel("derivative [-]")
    axes[0].set_title("Amplitude sensitivity of the extracted derivatives")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    audit = campaign["reference_audit"]
    labels = ["reference\n(implied)", "slender-body\n(2D exact)", "expected\nrange"]
    values = [audit["m22_reference_over_rho_nabla"], audit["m22_slender_over_rho_nabla"], 0.75]
    axes[1].bar(labels, values, color=["tab:red", "tab:orange", "tab:green"])
    axes[1].axhspan(0.5, 1.0, color="tab:green", alpha=0.15,
                    label="potential-theory range for a tanker")
    axes[1].set_ylabel(r"$m_{22}/(\rho\nabla)$ [-]")
    axes[1].set_title("Added-mass plausibility audit")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig4_amplitude_plausibility.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    hull = KVLCC2Hull()
    hull.calibrate()
    campaign = run_pmm_campaign(amplitudes_over_L=(0.005, 0.01, 0.02),
                                station_counts=(21, 41, 81), n_samples=400)
    paths = [fig_geometry(hull), fig_force_histories(hull, campaign),
             fig_derivative_comparison(campaign), fig_amplitude_and_plausibility(hull, campaign)]
    campaign["figures"] = [str(p) for p in paths]
    campaign["section_added_mass_unit"] = section_added_mass_ellipse(
        beam=hull.B, draught=hull.T)
    (OUT.parent / "pmm_derivatives_summary.json").write_text(
        json.dumps(campaign, indent=2, default=float))
    print(json.dumps({"figures": [str(p) for p in paths],
                      "criteria": campaign["criteria"]}, indent=1, default=float))


if __name__ == "__main__":
    main()