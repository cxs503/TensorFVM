"""Validation / reference-surface generator for :mod:`tensorfvm.planing_savitsky`.

Produces (inside ``docs/planing-savitsky-verification/``):

* ``figures/*.png``   -- lift chart, deadrise family, running attitude,
  resistance build-up, porpoising map;
* ``results.json``    -- every number quoted in the report, computed here
  (no hand-copied digits);
* stdout summary     -- the acceptance-criteria ledger used by the report.

Run: ``python scripts/planing_savitsky_validation.py`` (matplotlib Agg, ~2 s).
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfvm import planing_savitsky as ps  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                       "planing-savitsky-verification")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# --------------------------------------------------------------------------- #
# reference hull (representative 8 m / 2.46 t planing craft)
# --------------------------------------------------------------------------- #
BEAM = 2.0            # m
HULL_LENGTH = 7.0     # m
WEIGHT = 24125.0      # N  (~2.46 t)
BETA = 15.0           # deg
LCG_OVER_BEAM = 1.6   # CG ahead of the transom, in beams
RHO = ps.RHO_SEA_WATER
NU = ps.NU_SEA_WATER


def reference_operating_points():
    """10 lift-law operating points + an independent algebraic re-evaluation.

    ``C_L0 = tau^1.1 (0.0120 lam^0.5 + 0.0055 lam^2.5 / C_V^2)`` is the law
    fitted by Savitsky to his planing-plate data (the curves of his lift chart).
    The second column re-derives the same quantity from the *wetted-area*
    normalisation ``C_L0/lam = tau^1.1 (0.0120 lam^-0.5 + 0.0055 lam^1.5/C_V^2)``
    -- a different float path, so the residual is a transcription check of the
    implementation, not an independent physical validation (that is what the
    limit/negative-control suite and the equilibrium tests are for).
    """
    cases = [
        (1.0, 2.0, 4.0, 0.0),
        (2.0, 3.0, 6.0, 10.0),
        (2.0, 1.5, 8.0, 20.0),
        (3.0, 2.5, 5.0, 15.0),
        (4.0, 2.0, 4.0, 12.5),
        (4.0, 3.5, 10.0, 0.0),
        (6.0, 1.0, 6.0, 25.0),
        (6.0, 4.0, 3.0, 8.0),
        (8.0, 2.0, 12.0, 18.0),
        (10.0, 2.5, 7.0, 30.0),
    ]
    rows = []
    for C_V, lam, tau, beta in cases:
        cl = ps.lift_coefficient(C_V, lam, tau, beta)
        cl0 = ps.lift_coefficient_flat_plate(C_V, lam, tau)
        area_form = tau ** 1.1 * (0.0120 * lam ** -0.5 + 0.0055 * lam ** 1.5 / C_V ** 2)
        if beta > 0:
            area_form -= 0.0065 * beta * cl0 ** 0.6 / lam
        rel = abs(area_form * lam - cl) / cl
        rows.append({
            "C_V": C_V, "lam": lam, "tau_deg": tau, "beta_deg": beta,
            "C_L_flat_plate": cl0, "C_L_deadrise": cl,
            "C_L_from_wetted_area_form": area_form * lam,
            "rel_residual": rel,
            "in_envelope": ps.in_envelope(C_V, lam, tau, beta),
        })
    return rows


def limit_checks():
    """Negative controls / limits (machine-checkable, no literature needed)."""
    out = {}

    # (d1) beta = 0 must return the flat-plate law bit-for-bit
    worst = 0.0
    for C_V in (1.0, 2.0, 5.0):
        for lam in (1.0, 2.0, 4.0):
            for tau in (3.0, 8.0):
                worst = max(worst, abs(ps.lift_coefficient(C_V, lam, tau, 0.0)
                                       - ps.lift_coefficient_flat_plate(C_V, lam, tau)))
    out["beta0_flat_plate_max_abs_dev"] = worst

    # (d2) tau -> 0 and lam -> 0 make C_L vanish with the documented exponents
    out["tau_to_zero_C_L_at_0p01deg_beta0"] = ps.lift_coefficient(3.0, 2.0, 0.01, 0.0)
    out["tau_to_zero_C_L_at_0p01deg_beta15"] = ps.lift_coefficient(3.0, 2.0, 0.01, 15.0)
    out["deadrise_breakdown_C_L0_beta15"] = ps.deadrise_breakdown_lift(15.0)
    out["deadrise_breakdown_C_L0_beta30"] = ps.deadrise_breakdown_lift(30.0)
    out["C_L_at_tau_2deg_in_envelope_floor"] = ps.lift_coefficient(4.0, 2.3, 2.0, 15.0)
    out["lam_to_zero_C_L"] = ps.lift_coefficient(3.0, 1e-6, 6.0, 0.0)
    ratio = (ps.lift_coefficient_flat_plate(2.0, 2.0, 8.0)
             / ps.lift_coefficient_flat_plate(2.0, 2.0, 4.0))
    out["tau_exponent_from_two_points"] = math.log(ratio) / math.log(2.0)
    lam_exponent = (math.log(ps.lift_coefficient_flat_plate(2.0, 4.0, 6.0)
                             / ps.lift_coefficient_flat_plate(2.0, 1.0, 6.0))
                    / math.log(4.0))
    out["lam_effective_exponent_C_V2"] = lam_exponent
    asym = (ps.lift_coefficient_flat_plate(1.0e6, 4.0, 6.0)
            / (6.0 ** 1.1 * 0.0120 * 2.0))
    out["high_C_V_asymptote_ratio_to_sqrt_law"] = asym

    # (d3) C_V -> 0: the law diverges, and the V-independent remnant is exposed
    out["C_L_at_C_V_0p001"] = ps.lift_coefficient_flat_plate(1e-3, 2.0, 6.0)
    out["hydrostatic_residual_N"] = ps.hydrostatic_residual_force(0.05, BEAM, 2.0, 6.0)
    out["lift_force_at_C_V_0p01_N"] = ps.lift_force(0.01, BEAM, 2.0, 6.0)
    out["lift_force_dynamic_part_at_C_V_0p01_N"] = (
        ps.lift_force(0.01, BEAM, 2.0, 6.0) - ps.hydrostatic_residual_force(0.01, BEAM, 2.0, 6.0))
    out["lift_force_at_C_V_1_N"] = ps.lift_force(1.0, BEAM, 2.0, 6.0)

    # deadrise monotonicity and the documented 0.0065*beta*C_L0^0.6 law
    betas = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    cl0 = ps.lift_coefficient_flat_plate(2.0, 3.0, 6.0)
    out["deadrise_family"] = [
        {"beta_deg": b,
         "C_L": ps.lift_coefficient(2.0, 3.0, 6.0, b),
         "decrement": cl0 - ps.lift_coefficient(2.0, 3.0, 6.0, b),
         "law_0p0065_beta_CL0_0p6": 0.0065 * b * cl0 ** 0.6,
         "relative_reduction": 1.0 - ps.lift_coefficient(2.0, 3.0, 6.0, b) / cl0}
        for b in betas]

    # centre of pressure: asymptote and inverse round-trip
    out["lcp_over_beam_high_C_V_ratio_to_0p75lam"] = (
        ps.center_of_pressure_over_beam(1e6, 2.5) / (0.75 * 2.5))
    worst = 0.0
    for C_V in (0.8, 1.0, 2.0, 3.5, 5.0, 10.0):
        for lam in (0.5, 1.0, 2.0, 3.5, 5.0):
            lcp = ps.center_of_pressure_over_beam(C_V, lam)
            back = ps.mean_wetted_length_from_center_of_pressure(C_V, lcp)
            worst = max(worst, abs(back - lam) / lam)
    out["lcp_inverse_max_rel_error"] = worst

    # spray drag: prescribed law + b/L dependence
    out["spray_drag_family"] = [
        {"C_V": cv, "b_over_L": bl,
         "dC_D": ps.spray_drag_coefficient(cv, BEAM, BEAM / bl),
         "law_0p003_sqrt_bL_CV2": 0.003 * math.sqrt(bl) * cv ** 2}
        for cv in (1.0, 2.0, 4.0, 6.0, 8.0) for bl in (0.1, 0.25)]
    out["spray_share_of_resistance_pct"] = None  # filled from the sweep below

    # ITTC-57 reference point
    out["ittc57_Re_1e9"] = ps.ittc57_friction_coefficient(1e9)
    out["ittc57_Re_1e9_analytic_0p075_over_49"] = 0.075 / 49.0
    return out


def sweep():
    """Running-condition sweep over the reference hull."""
    speeds = [1.0 + 0.1 * i for i in range(int(round((7.0 - 1.0) / 0.1)) + 1)]
    states = ps.trim_curve(speeds, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH,
                           rho=RHO, nu=NU)
    return states


def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}

    results["reference_hull"] = {
        "beam_m": BEAM, "hull_length_m": HULL_LENGTH, "weight_N": WEIGHT,
        "weight_kg": WEIGHT / ps.G, "deadrise_deg": BETA,
        "lcg_over_beam": LCG_OVER_BEAM, "lcg_m": LCG_OVER_BEAM * BEAM,
        "rho": RHO, "nu": NU,
    }
    results["operating_points"] = reference_operating_points()
    results["limits"] = limit_checks()

    # ---------------- running condition sweep ---------------- #
    states = sweep()
    results["sweep"] = [s.as_dict() for s in states]
    in_env = [s for s in states if s.in_envelope]
    results["sweep_summary"] = {
        "n_points": len(states),
        "n_in_envelope": len(in_env),
        "C_V_range": [states[0].C_V, states[-1].C_V],
        "C_L_range": [min(s.C_L for s in states), max(s.C_L for s in states)],
        "C_L_range_in_envelope": ([min(s.C_L for s in in_env), max(s.C_L for s in in_env)]
                                 if in_env else None),
        "design_point": min(states, key=lambda s: abs(s.C_V - 2.0)).as_dict(),
        "max_lift_balance_residual": max(abs(s.C_L - s.C_L_required) for s in states),
        "max_moment_residual": max(
            abs(s.lcp_over_beam - LCG_OVER_BEAM) for s in states),
        "tau_range_deg": [min(s.tau_deg for s in states), max(s.tau_deg for s in states)],
        "lam_range": [min(s.lam for s in states), max(s.lam for s in states)],
        "LD_range_in_envelope": ([min(s.lift_to_drag for s in in_env),
                                  max(s.lift_to_drag for s in in_env)] if in_env else None),
        "LD_range_full_sweep": [min(s.lift_to_drag for s in states),
                                max(s.lift_to_drag for s in states)],
        "spray_force_at_C_V7_over_weight": [
            {"C_V": s.C_V, "spray_over_weight": s.spray_force / s.weight} for s in states[::10]],
    }
    results["limits"]["spray_share_of_resistance_pct"] = [
        {"C_V": s.C_V, "spray_pct": 100.0 * s.spray_force / s.resistance,
         "friction_pct": 100.0 * s.friction_force / s.resistance}
        for s in states[::5]]

    # ---------------- stability ---------------- #
    diag_cv = [1.0 + 0.1 * i for i in range(41)]
    diag = [ps.stability_diagnostics(cv, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH,
                                     rho=RHO, nu=NU) for cv in diag_cv]
    results["stability_beta15"] = [d.as_dict() for d in diag]
    results["criteria_agreement"] = {
        "n": len(diag),
        "n_stiffness_unstable": sum(1 for d in diag if d.quasi_static_unstable),
        "n_trim_slope_unstable": sum(1 for d in diag if not d.trim_slope_stable),
        "n_prone": sum(1 for d in diag if d.porpoising_prone),
        "n_dCL_dlam_negative": sum(1 for d in diag if not d.dCL_dlam_positive),
        "min_det": min(d.stiffness_det for d in diag),
        "max_det": max(d.stiffness_det for d in diag),
        "trim_slope_range": [min(d.trim_slope for d in diag),
                             max(d.trim_slope for d in diag)],
        "dM_dlam_range": [min(d.dM_dlam for d in diag), max(d.dM_dlam for d in diag)],
    }

    betas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    cvs = [0.8 + 0.1 * i for i in range(62)]
    grid = ps.porpoising_map(cvs, betas, BEAM, WEIGHT, LCG_OVER_BEAM, HULL_LENGTH,
                             rho=RHO, nu=NU)
    results["porpoising_boundaries"] = [
        ps.porpoising_boundary(cvs, b, BEAM, WEIGHT, LCG_OVER_BEAM, HULL_LENGTH,
                               rho=RHO, nu=NU) for b in betas]
    results["porpoising_grid"] = {
        "C_V": cvs, "beta_deg": betas,
        "prone_fraction": {
            f"{b:g}": sum(1 for d in grid if d.beta_deg == b and d.porpoising_prone)
            / len(cvs) for b in betas},
        "stiffness_unstable_fraction": {
            f"{b:g}": sum(1 for d in grid if d.beta_deg == b and d.quasi_static_unstable)
            / len(cvs) for b in betas},
    }

    # ---------------- figures ---------------- #
    # fig 1: Savitsky lift chart (C_L0/tau^1.1 vs lam for several C_V)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    lams = [0.4 + 0.02 * i for i in range(int((5.0 - 0.4) / 0.02) + 1)]
    for cv in (1.0, 2.0, 4.0, 6.0, 8.0):
        ys = [ps.lift_coefficient_flat_plate(cv, lam, 1.0) for lam in lams]
        ax.plot(lams, ys, label=f"$C_V$={cv:g}")
    ax.set_xlabel(r"mean wetted length / beam  $\lambda$")
    ax.set_ylabel(r"$C_{L0}/\tau^{1.1}$  ($\tau$ in deg)")
    ax.set_title("Savitsky (1964) flat-plate dynamic-lift law")
    ax.grid(True, ls=":", lw=0.6)
    ax.legend(title="speed coefficient")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_lift_chart.png"), dpi=140)
    plt.close(fig)

    # fig 2: deadrise family
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bplot = [0.0 + 0.25 * i for i in range(int(30 / 0.25) + 1)]
    for cv, tau in ((2.0, 4.0), (2.0, 8.0), (4.0, 4.0), (4.0, 8.0)):
        ys = [ps.lift_coefficient(cv, 3.0, tau, b) for b in bplot]
        ax.plot(bplot, ys, label=f"$C_V$={cv:g}, $\\tau$={tau:g}$^\\circ$")
    ax.set_xlabel(r"deadrise angle $\beta$ [deg]")
    ax.set_ylabel(r"$C_{L\beta}$  at $\lambda=3$")
    ax.set_title("Deadrise decrement $C_{L\\beta}=C_{L0}-0.0065\\beta C_{L0}^{0.6}$")
    ax.grid(True, ls=":", lw=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_deadrise.png"), dpi=140)
    plt.close(fig)

    # fig 3: running attitude
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for beta in (10.0, 15.0, 20.0):
        st = ps.trim_curve([0.8 + 0.1 * i for i in range(63)], BEAM, WEIGHT, beta,
                           LCG_OVER_BEAM, HULL_LENGTH, rho=RHO, nu=NU)
        axs[0].plot([s.C_V for s in st], [s.tau_deg for s in st], label=f"$\\beta$={beta:g}$^\\circ$")
        axs[1].plot([s.C_V for s in st], [s.lam for s in st], label=f"$\\beta$={beta:g}$^\\circ$")
    axs[0].axhline(ps.TAU_MIN_DEG, color="k", ls="--", lw=0.8,
                   label="envelope floor $\\tau_{min}$")
    axs[0].set_xlabel(r"$C_V$"); axs[0].set_ylabel(r"running trim $\tau$ [deg]")
    axs[0].set_title("Equilibrium trim")
    axs[1].set_xlabel(r"$C_V$"); axs[1].set_ylabel(r"$\lambda=L_m/b$")
    axs[1].set_title("Mean wetted length")
    for a in axs:
        a.grid(True, ls=":", lw=0.6); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_running_attitude.png"), dpi=140)
    plt.close(fig)

    # fig 4: resistance build-up and L/D
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))
    axs[0].plot([s.speed for s in states], [s.resistance for s in states], "k-", label="$R_T$")
    axs[0].plot([s.speed for s in states], [s.friction_force for s in states], "b--", label="$R_f$")
    axs[0].plot([s.speed for s in states], [s.spray_force for s in states], "r-.", label="$R_{spray}$")
    axs[0].plot([s.speed for s in states],
                [s.weight * math.tan(math.radians(s.tau_deg)) for s in states],
                "g:", label=r"$\Delta\tan\tau$")
    axs[0].set_xlabel("speed [m/s]"); axs[0].set_ylabel("resistance [N]")
    axs[0].set_title("Resistance build-up ($\\beta=15^\\circ$)")
    axs[0].grid(True, ls=":", lw=0.6); axs[0].legend(fontsize=8)
    axs[1].plot([s.speed for s in states], [s.lift_to_drag for s in states], "k-")
    axs[1].set_xlabel("speed [m/s]"); axs[1].set_ylabel("$\\Delta/R_T$")
    axs[1].set_title("Lift-to-drag ratio")
    axs[1].grid(True, ls=":", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_resistance.png"), dpi=140)
    plt.close(fig)

    # fig 5: porpoising map
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for b in betas:
        rows = [d for d in grid if d.beta_deg == b]
        ax.plot([d.C_V for d in rows], [d.trim_slope for d in rows], label=f"$\\beta$={b:g}$^\\circ$")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$C_V$"); ax.set_ylabel(r"$d\tau/dC_V$  [deg]")
    ax.set_title("Trim-slope porpoising indicator (positive = prone)")
    ax.grid(True, ls=":", lw=0.6); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_porpoising.png"), dpi=140)
    plt.close(fig)

    results["figures"] = sorted(os.listdir(FIG_DIR))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---------------- stdout ledger ---------------- #
    ops = results["operating_points"]
    print("== operating points (lift law transcription check) ==")
    print(f"{'C_V':>5} {'lam':>5} {'tau':>5} {'beta':>5} {'C_L0':>9} {'C_Lb':>9} {'rel_res':>10}")
    for r in ops:
        print(f"{r['C_V']:5.2f} {r['lam']:5.2f} {r['tau_deg']:5.1f} {r['beta_deg']:5.1f}"
              f" {r['C_L_flat_plate']:9.5f} {r['C_L_deadrise']:9.5f} {r['rel_residual']:10.2e}")
    print("max rel residual:", max(r["rel_residual"] for r in ops))
    lim = results["limits"]
    print("== limits ==")
    for k, v in lim.items():
        if isinstance(v, list):
            continue
        print(f"  {k} = {v}")
    print("== sweep summary ==")
    for k, v in results["sweep_summary"].items():
        print(f"  {k} = {v}")
    print("== stability ==")
    for k, v in results["criteria_agreement"].items():
        print(f"  {k} = {v}")
    print("== porpoising boundaries ==")
    for r in results["porpoising_boundaries"]:
        print("  ", r)
    print("== prone fraction by beta ==")
    print("  ", results["porpoising_grid"]["prone_fraction"])
    print("  ", results["porpoising_grid"]["stiffness_unstable_fraction"])
    print("figures:", results["figures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())