"""Runner for the CHT / Stefan / ice-steel verification set (P295W1R2-N01).

Every case writes a machine-readable ledger and the report figures are generated
from *those* ledgers (no number is transcribed by hand):

    python -m tensorfvm.cht.runner_phase --case all

Cases
-----
``interface``  C1: ice/steel interface continuity + flux conservation, clean run
               and three injected mutations (each must turn the criterion red).
``stefan``     C2: enthalpy-porosity front vs the Neumann similarity solution, time
               step and mushy-band refinement, plus the discrete energy ledger.
``ice_steel``  C3: steady ice/steel field vs the 1-D analytic series solution and a
               2-D laterally varying field for the figure.
``icing``      C4: W4-N08 cross-adjudication -- closure reproduction at the same
               icing condition, temperature-field L2 against an independently
               coded reference, and the conduction-aware regime/freezing fraction.

Outputs default to ``src/tensorfvm/verification/artifacts/cht-stefan`` (ledgers)
and ``docs/cht-stefan-ice-steel`` (figures + ``RESULTS.md``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ice_steel import (ICE, STEEL, LayeredWall, LayeredWall2D, SurfaceBC,
                        TwoDomainCHT, analytic_series_field, interface_audit)
from .icing import (IcingEnvironment, icing_state_adiabatic,
                    icing_state_with_conduction, load_w4n08_reference)
from .phase_change import (EnthalpySlab1D, NeumannSolution, PhaseChangeMaterial,
                           RadialBoundary)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "cht-stefan"
DOCS = ROOT / "docs" / "cht-stefan-ice-steel"

T_FREEZE = 271.35       # K  (-1.8 degC, sea-water freezing point)
T_BACK = 258.15         # K  (-15 degC cold hold; documented design assumption)
H_ICE = 0.0334          # m
H_STEEL = 0.010         # m
R_CONTACT = 1.0e-4      # m^2 K/W
LATENT = 334.0e3        # J/kg (campaign value)
DT_MUSH = 0.5           # K  (campaign value)


def _dump(out: Path, name: str, payload: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    return path


# ---------------------------------------------------------------------- C1
def case_interface(out: Path) -> dict:
    def panel(mutation=None, q_init=100.0):
        ice = LayeredWall([(H_ICE, ICE, 40)], [], SurfaceBC(t_ext=T_FREEZE, resistance=0.0),
                          SurfaceBC(t_ext=T_BACK, resistance=0.0))
        steel = LayeredWall([(H_STEEL, STEEL, 10)], [], SurfaceBC(t_ext=T_FREEZE, resistance=0.0),
                            SurfaceBC(t_ext=T_BACK, resistance=0.0))
        res = TwoDomainCHT(ice, steel, r_contact=R_CONTACT, q_init=q_init,
                           mutation=mutation).solve()
        audit = interface_audit(res["t_ice"], ice.dx[0], ice.k[0], res["t_steel"],
                                steel.dx[0], steel.k[0], R_CONTACT, T_FREEZE - T_BACK)
        audit["iterations"] = res["iterations"]
        audit["coupling_residual"] = res["residual"]
        audit["q"] = res["q"]
        return audit

    clean = panel()
    reference = analytic_series_field([H_ICE, H_STEEL], [ICE.k, STEEL.k],
                                      T_FREEZE, T_BACK, R_CONTACT)
    mutations = {m: panel(m) for m in ("single_sweep", "sign_flipped", "steel_decoupled")}
    for name, data in mutations.items():
        data["red"] = bool(data["flux_imbalance_rel"] > 1e-6
                           or data["temperature_continuity_rel"] > 1e-2)
    payload = {
        "case": "ice_steel_interface_ch1",
        "design": {"h_ice_m": H_ICE, "h_steel_m": H_STEEL, "r_contact_m2K_W": R_CONTACT,
                   "t_surface_K": T_FREEZE, "t_back_K": T_BACK,
                   "k_ice": ICE.k, "k_steel": STEEL.k},
        "clean": clean,
        "analytic": {"q": reference["q"], "r_total": reference["r_total"],
                     "contacts": reference["contacts"]},
        "mutations": mutations,
        "acceptance": {
            "temperature_continuity_limit": 1e-2,
            "flux_residual_limit": 1e-6,
            "temperature_continuity_rel": clean["temperature_continuity_rel"],
            "flux_imbalance_rel": clean["flux_imbalance_rel"],
            "all_mutations_red": all(v["red"] for v in mutations.values()),
            "q_vs_analytic_rel": abs(clean["q"] / reference["q"] - 1.0),
            "passed": bool(clean["temperature_continuity_rel"] <= 1e-2
                           and clean["flux_imbalance_rel"] <= 1e-6
                           and all(v["red"] for v in mutations.values())),
        },
    }
    _dump(out, "interface", payload)
    return payload


# ---------------------------------------------------------------------- C2
def stefan_run(nx: int, dt: float, dt_mush: float, t_end: float = 1200.0,
               length: float = 0.04, record=(300.0, 600.0, 1200.0)):
    material = PhaseChangeMaterial(k=ICE.k, rho=ICE.rho, c=ICE.c,
                                   latent=LATENT, dT_mush=dt_mush)
    t_surface = material.t_melt - 10.0
    slab = EnthalpySlab1D(length, nx, material, material.t_liquidus,
                          RadialBoundary(t_ext=t_surface, resistance=0.0))
    snaps = slab.run(t_end, dt, record=list(record))
    exact = NeumannSolution(material, t_surface)
    rows = []
    for snap in snaps:
        x_exact = exact.front(snap["time"])
        t_exact = exact.temperature(snap["x"], snap["time"])
        interior = snap["x"] < 0.8 * max(snap["front"], 1e-12)
        rows.append({
            "time_s": snap["time"],
            "front_m": snap["front_liquid_fraction"],
            "front_integral_m": snap["front"],
            "front_exact_m": x_exact,
            "front_rel_error": abs(snap["front_liquid_fraction"] / x_exact - 1.0),
            "front_integral_rel_error": abs(snap["front"] / x_exact - 1.0),
            "max_temperature_error_K": float(np.max(
                np.abs(snap["temperature"][interior] - t_exact[interior])))
            if np.any(interior) else 0.0,
        })
    ledger = slab.ledger()
    return material, t_surface, slab, snaps, exact, rows, ledger


def case_stefan(out: Path) -> dict:
    nominal = stefan_run(nx=400, dt=0.5, dt_mush=DT_MUSH)
    material, t_surface, slab, snaps, exact, rows, ledger = nominal
    sweeps = []
    for nx in (200, 400, 800):
        for dt in (1.0, 0.5):
            _, _, _, _, _, rr, led = stefan_run(nx=nx, dt=dt, dt_mush=DT_MUSH)
            sweeps.append({"nx": nx, "dt_s": dt, "dT_mush_K": DT_MUSH,
                           "max_front_rel_error": max(r["front_rel_error"] for r in rr),
                           "energy_imbalance_rel": led["imbalance_rel"]})
    for dt_mush in (0.5, 0.2, 0.05):
        _, _, _, _, _, rr, led = stefan_run(nx=400, dt=0.5, dt_mush=dt_mush)
        sweeps.append({"nx": 400, "dt_s": 0.5, "dT_mush_K": dt_mush,
                       "max_front_rel_error": max(r["front_rel_error"] for r in rr),
                       "energy_imbalance_rel": led["imbalance_rel"]})
    payload = {
        "case": "stefan_neumann",
        "design": {"material": {"k": material.k, "rho": material.rho, "c": material.c,
                                "latent_J_kg": material.latent, "t_melt_K": material.t_melt,
                                "dT_mush_K": material.dT_mush},
                   "t_surface_K": t_surface, "driving_dT_K": material.t_melt - t_surface,
                   "stefan_number": exact.stefan_number, "lambda": exact.lam,
                   "alpha_m2_s": material.alpha, "nx": 400, "dt_s": 0.5},
        "rows": rows,
        "sweeps": sweeps,
        "ledger": ledger,
        "snapshots": [{"time_s": s["time"], "x": s["x"].tolist(),
                       "temperature": s["temperature"].tolist()}
                      for s in snaps],
        "acceptance": {
            "limit_rel": 3e-2,
            "max_front_rel_error": max(r["front_rel_error"] for r in rows),
            "energy_imbalance_rel": ledger["imbalance_rel"],
            "passed": bool(max(r["front_rel_error"] for r in rows) <= 3e-2),
        },
    }
    _dump(out, "stefan", payload)
    return payload


# --------------------------------------------------------------------- C3
def case_ice_steel(out: Path) -> dict:
    reference = analytic_series_field([H_ICE, H_STEEL], [ICE.k, STEEL.k],
                                      T_FREEZE, T_BACK, R_CONTACT, n_cells=6000)
    refine = []
    for n_ice, n_steel in ((10, 4), (20, 5), (40, 10), (80, 20)):
        wall = LayeredWall2D([(H_ICE, ICE, n_ice), (H_STEEL, STEEL, n_steel)], [R_CONTACT],
                             t_bottom=SurfaceBC(t_ext=T_BACK, resistance=0.0),
                             top_bc=[SurfaceBC(t_ext=T_FREEZE, resistance=0.0)], ny=1)
        field = wall.solve()[0]
        t_ref = np.interp(wall.xc, reference["x"], reference["temperature"])
        refine.append({"n_ice": n_ice, "n_steel": n_steel,
                       "l2_rel": float(np.linalg.norm(field - t_ref)
                                       / np.linalg.norm(reference["temperature"] - T_BACK)),
                       "max_abs_error_K": float(np.max(np.abs(field - t_ref)))})

    # 2-D field: surface state varies laterally (wet film -> dry cold wall)
    ny = 25
    top = [SurfaceBC(t_ext=T_FREEZE - 10.0 * (j / (ny - 1)) ** 1.5, resistance=0.0)
           for j in range(ny)]
    wall2d = LayeredWall2D([(H_ICE, ICE, 40), (H_STEEL, STEEL, 10)], [R_CONTACT],
                           t_bottom=SurfaceBC(t_ext=T_BACK - 8.0, resistance=0.0),
                           top_bc=top, ny=ny)
    field2d = wall2d.solve()
    payload = {
        "case": "ice_steel_steady_field",
        "design": {"h_ice_m": H_ICE, "h_steel_m": H_STEEL, "r_contact": R_CONTACT,
                   "t_surface_K": T_FREEZE, "t_back_K": T_BACK,
                   "k_ice": ICE.k, "k_steel": STEEL.k},
        "analytic": {"q": reference["q"], "r_total": reference["r_total"],
                     "contacts": reference["contacts"]},
        "refinement": refine,
        "field_2d": {"ny": ny, "yc": wall2d.yc.tolist(), "xc": wall2d.xc.tolist(),
                     "temperature": field2d.tolist()},
        "acceptance": {
            "limit_rel": 5e-2,
            "l2_rel": refine[-1]["l2_rel"],
            "max_abs_error_K": refine[-1]["max_abs_error_K"],
            "passed": bool(refine[-1]["l2_rel"] <= 5e-2),
        },
    }
    _dump(out, "ice_steel", payload)
    return payload


# --------------------------------------------------------------------- C4
def _reference_environment():
    ref = load_w4n08_reference()
    e = ref["environment"]
    env = IcingEnvironment(t_air=e["t_air_degC"], t_sea=e["t_sea_degC"],
                           t_freeze=e["t_sea_degC"], wind_speed=e["wind_speed_m_s"],
                           char_length=e["char_length_m"],
                           relative_humidity=e["relative_humidity"],
                           spray_flux=e["spray_flux_kg_m2_s"], exposure=6.0 * 3600.0)
    return env, ref


def case_icing(out: Path) -> dict:
    env, ref = _reference_environment()
    mod = ref["module_output"]
    state = icing_state_adiabatic(env)
    keys = (("q_conv", "q_conv_W_m2"), ("q_evap", "q_evap_W_m2"),
            ("q_lw", "q_lw_W_m2"))
    compare = []
    for ours_key, theirs_key in keys:
        ours = float(getattr(state, ours_key))
        theirs = float(mod[theirs_key])
        compare.append({"quantity": ours_key, "ours_W_m2": ours,
                        "w4n08_W_m2": theirs,
                        "rel_error": abs(ours - theirs) / max(abs(theirs), 1.0)})
    compare.append({"quantity": "q_req", "ours_W_m2": float(state.q_req),
                    "w4n08_W_m2": float(mod["q_req_W_m2"]),
                    "rel_error": abs(state.q_req / mod["q_req_W_m2"] - 1.0)})
    compare.append({"quantity": "freezing_fraction",
                    "ours": float(state.freezing_fraction),
                    "w4n08": float(mod["freezing_fraction"]),
                    "rel_error": abs(state.freezing_fraction
                                     / mod["freezing_fraction"] - 1.0)})
    compare.append({"quantity": "icing_rate_mm_per_h",
                    "ours": float(state.icing_rate_mm_per_h),
                    "w4n08": float(mod["icing_rate_mm_per_h"]),
                    "rel_error": abs(state.icing_rate_mm_per_h
                                     / mod["icing_rate_mm_per_h"] - 1.0)})

    # temperature field at the same icing condition (module ice thickness)
    h_ice = mod["icing_rate_mm_per_h"] * env.exposure / 3600.0 / 1000.0
    t_s = 273.15 + mod["surface_temperature_degC"]
    wall = LayeredWall2D([(h_ice, ICE, 40), (H_STEEL, STEEL, 10)], [R_CONTACT],
                         t_bottom=SurfaceBC(t_ext=T_BACK, resistance=0.0),
                         top_bc=[SurfaceBC(t_ext=t_s, resistance=0.0)], ny=1)
    field = wall.solve()[0]
    reference = analytic_series_field([h_ice, H_STEEL], [ICE.k, STEEL.k],
                                      t_s, T_BACK, R_CONTACT, n_cells=6000)
    t_ref = np.interp(wall.xc, reference["x"], reference["temperature"])
    l2 = float(np.linalg.norm(field - t_ref)
               / np.linalg.norm(reference["temperature"] - T_BACK))

    # conduction-aware adjudication: sweep the back-face temperature
    r_wall = h_ice / ICE.k + R_CONTACT + H_STEEL / STEEL.k
    sweep = []
    for t_back_c in (-1.8, -3.0, -5.0, -10.0, -15.0, -20.0, -25.0):
        st = icing_state_with_conduction(env, r_wall, t_back_c)
        sweep.append({"t_back_degC": t_back_c, "regime": st.regime,
                      "t_surface_degC": float(st.t_surface),
                      "freezing_fraction": float(st.freezing_fraction),
                      "q_cond_W_m2": float(st.q_cond),
                      "icing_rate_mm_per_h": float(st.icing_rate_mm_per_h)})
    st_cold = icing_state_with_conduction(env, r_wall, T_BACK - 273.15)

    payload = {
        "case": "w4n08_cross_adjudication",
        "provenance": ref["_provenance"],
        "data_sha256": ref["_sha256"],
        "condition_string": ref["condition_string"],
        "environment": env.__dict__,
        "module_output": mod,
        "our_state": {"regime": state.regime, "t_surface_degC": float(state.t_surface),
                      "freezing_fraction": float(state.freezing_fraction),
                      "icing_rate_mm_per_h": float(state.icing_rate_mm_per_h),
                      "h_c": float(state.h_c), "terms_balance": float(state.terms_balance)},
        "closure_compare": compare,
        "field": {"h_ice_m": h_ice, "t_surface_K": t_s, "t_back_K": T_BACK,
                  "xc": wall.xc.tolist(), "temperature": field.tolist(),
                  "reference_temperature": t_ref.tolist(),
                  "l2_rel": l2,
                  "q_series_W_m2": float(reference["q"]),
                  "q_contact_jump_K": float(reference["q"] * R_CONTACT)},
        "conduction_aware": {"r_wall_m2K_W": r_wall, "sweep": sweep,
                             "at_cold_hold": {"t_back_degC": T_BACK - 273.15,
                                              "regime": st_cold.regime,
                                              "freezing_fraction": float(st_cold.freezing_fraction),
                                              "t_surface_degC": float(st_cold.t_surface),
                                              "icing_rate_mm_per_h": float(st_cold.icing_rate_mm_per_h)}},
        "acceptance": {
            "limit_rel": 5e-2,
            "field_l2_rel": l2,
            "max_closure_rel_error": max(c["rel_error"] for c in compare),
            "passed": bool(l2 <= 5e-2 and max(c["rel_error"] for c in compare) <= 1e-4),
        },
    }
    _dump(out, "w4n08_crosscheck", payload)
    return payload


# ------------------------------------------------------------------ figures
def _figures(interface, stefan, ice_steel, icing, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # fig1: Stefan front + profiles
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = stefan["rows"]
    times = np.array([r["time_s"] for r in rows])
    axes[0].plot(times, np.array([r["front_exact_m"] for r in rows]) * 1e3, "k-",
                 label="Neumann exact")
    axes[0].plot(times, np.array([r["front_m"] for r in rows]) * 1e3, "o",
                 label="enthalpy-porosity FV")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("front X [mm]")
    axes[0].set_title(f"C2 Stefan front (max err "
                      f"{100 * max(r['front_rel_error'] for r in rows):.2f} %)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    snap = stefan["snapshots"][-1]
    exact = NeumannSolution(PhaseChangeMaterial(k=ICE.k, rho=ICE.rho, c=ICE.c,
                                               latent=LATENT, dT_mush=DT_MUSH),
                            stefan["design"]["t_surface_K"])
    x = np.asarray(snap["x"])
    axes[1].plot(x * 1e3, np.asarray(snap["temperature"]) - 273.15, "o", ms=3,
                 label="FV (t = %.0f s)" % snap["time_s"])
    axes[1].plot(x * 1e3, exact.temperature(x, snap["time_s"]) - 273.15, "k-",
                 label="Neumann exact")
    axes[1].set_xlim(0, 30); axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("T [degC]")
    axes[1].set_title("C2 temperature profile"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); path = outdir / "fig1_stefan_neumann.png"; fig.savefig(path, dpi=130)
    plt.close(fig); figs.append(path)

    # fig2: 2-D ice/steel field + through-thickness profiles
    f2 = ice_steel["field_2d"]
    t2 = np.asarray(f2["temperature"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    im = axes[0].contourf(np.asarray(f2["yc"]) * 1000, np.asarray(f2["xc"]) * 1e3,
                          t2.T - 273.15, levels=20, cmap="coolwarm")
    axes[0].set_xlabel("lateral y [mm]"); axes[0].set_ylabel("through thickness x [mm]")
    axes[0].invert_yaxis(); axes[0].set_title("C3 ice/steel temperature field [degC]")
    fig.colorbar(im, ax=axes[0], label="T [degC]")
    for j in (0, len(f2["yc"]) - 1):
        axes[1].plot(np.asarray(f2["xc"]) * 1e3, t2[j] - 273.15,
                     label=f"column y = {1000 * f2['yc'][j]:.0f} mm")
    ref = analytic_series_field([H_ICE, H_STEEL], [ICE.k, STEEL.k], T_FREEZE, T_BACK,
                                R_CONTACT, n_cells=2000)
    axes[1].plot(ref["x"] * 1e3, ref["temperature"] - 273.15, "k--", label="1-D analytic")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("T [degC]")
    axes[1].set_title(f"C3 L2 = {100 * ice_steel['acceptance']['l2_rel']:.3f} %")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); path = outdir / "fig2_ice_steel_field.png"; fig.savefig(path, dpi=130)
    plt.close(fig); figs.append(path)

    # fig3: interface audit + mutations
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["clean"] + list(interface["mutations"])
    flux = [interface["clean"]["flux_imbalance_rel"]] + \
        [interface["mutations"][m]["flux_imbalance_rel"] for m in interface["mutations"]]
    cont = [interface["clean"]["temperature_continuity_rel"]] + \
        [interface["mutations"][m]["temperature_continuity_rel"] for m in interface["mutations"]]
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, np.maximum(flux, 1e-18), width=0.4, label="flux imbalance (limit 1e-6)")
    ax.bar(pos + 0.2, np.maximum(cont, 1e-18), width=0.4,
           label="temperature continuity (limit 1e-2)")
    ax.axhline(1e-6, ls=":", c="C0"); ax.axhline(1e-2, ls=":", c="C1")
    ax.set_yscale("log"); ax.set_xticks(pos); ax.set_xticklabels(labels)
    ax.set_ylabel("relative residual")
    ax.set_title("C1 interface audit: clean vs injected mutations")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); path = outdir / "fig3_interface_mutations.png"; fig.savefig(path, dpi=130)
    plt.close(fig); figs.append(path)

    # fig4: W4-N08 cross-check
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    comp = [c for c in icing["closure_compare"] if "w4n08_W_m2" in c]
    labels = [c["quantity"] for c in comp]
    ours = [c["ours_W_m2"] for c in comp]
    theirs = [c["w4n08_W_m2"] for c in comp]
    pos = np.arange(len(labels))
    axes[0].bar(pos - 0.2, ours, width=0.4, label="TensorFVM closure")
    axes[0].bar(pos + 0.2, theirs, width=0.4, label="W4-N08 artifact")
    axes[0].set_xticks(pos); axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("flux [W/m$^2$]"); axes[0].legend(fontsize=8)
    axes[0].set_title("C4 surface heat balance at the same icing condition")
    axes[0].grid(alpha=0.3, axis="y")
    sweep = icing["conduction_aware"]["sweep"]
    tb = [s["t_back_degC"] for s in sweep]
    axes[1].plot(tb, [s["freezing_fraction"] for s in sweep], "o-", label="freezing fraction")
    axes[1].axhline(icing["module_output"]["freezing_fraction"], ls="--",
                    label="W4-N08 (adiabatic substrate)")
    axes[1].set_xlabel("structure back-face temperature [degC]")
    axes[1].set_ylabel("freezing fraction f [-]")
    axes[1].set_title("C4 conduction-aware adjudication"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); path = outdir / "fig4_w4n08_crosscheck.png"; fig.savefig(path, dpi=130)
    plt.close(fig); figs.append(path)
    return figs


# ----------------------------------------------------------------- RESULTS
def _results_markdown(interface, stefan, ice_steel, icing, figs) -> str:
    def fmt(v, nd=3):
        return f"{v:.{nd}f}"

    lines = []
    lines.append("# CHT / Stefan / ice-steel verification results (auto-generated)\n")
    lines.append("> Generated by `python -m tensorfvm.cht.runner_phase --case all` from the "
                 "JSON ledgers in `src/tensorfvm/verification/artifacts/cht-stefan/`. "
                 "Do not edit by hand.\n")
    a = interface["acceptance"]
    lines.append("## C1 ice/steel interface\n")
    lines.append("| quantity | value | limit | pass |\n|---|---|---|---|")
    lines.append(f"| temperature continuity (rel. to dT drive) | {a['temperature_continuity_rel']:.3e} "
                 f"| 1e-2 | {a['temperature_continuity_rel'] <= 1e-2} |")
    lines.append(f"| interface heat-flux residual | {a['flux_imbalance_rel']:.3e} | 1e-6 "
                 f"| {a['flux_imbalance_rel'] <= 1e-6} |")
    lines.append(f"| q vs analytic series | {a['q_vs_analytic_rel']:.3e} | - | - |")
    lines.append(f"| all mutations detected | {a['all_mutations_red']} | true | "
                 f"{a['all_mutations_red']} |\n")
    lines.append("| mutation | flux imbalance | temperature continuity | red |\n|---|---|---|---|")
    for m, data in interface["mutations"].items():
        lines.append(f"| {m} | {data['flux_imbalance_rel']:.3e} "
                     f"| {data['temperature_continuity_rel']:.3e} | {data['red']} |")
    lines.append("")
    a = stefan["acceptance"]
    lines.append("## C2 Stefan phase change vs Neumann\n")
    lines.append("| t [s] | front FV [mm] | front exact [mm] | rel. error | max T error [K] |"
                 "\n|---|---|---|---|---|")
    for r in stefan["rows"]:
        lines.append(f"| {r['time_s']:.0f} | {1000 * r['front_m']:.3f} "
                     f"| {1000 * r['front_exact_m']:.3f} | {100 * r['front_rel_error']:.2f} % "
                     f"| {r['max_temperature_error_K']:.3f} |")
    lines.append(f"\nEnergy ledger imbalance: {stefan['ledger']['imbalance_rel']:.3e} "
                 f"(limit 1e-9), Stefan number {stefan['design']['stefan_number']:.5f}, "
                 f"lambda = {stefan['design']['lambda']:.5f}.\n")
    lines.append("| nx | dt [s] | dT_mush [K] | max front error | energy imbalance |"
                 "\n|---|---|---|---|---|")
    for s in stefan["sweeps"]:
        lines.append(f"| {s['nx']} | {s['dt_s']} | {s['dT_mush_K']} "
                     f"| {100 * s['max_front_rel_error']:.2f} % "
                     f"| {s['energy_imbalance_rel']:.2e} |")
    lines.append("")
    a = ice_steel["acceptance"]
    lines.append("## C3 ice/steel steady field\n")
    lines.append("| n_ice | n_steel | L2 rel. error | max abs error [K] |\n|---|---|---|---|")
    for r in ice_steel["refinement"]:
        lines.append(f"| {r['n_ice']} | {r['n_steel']} | {100 * r['l2_rel']:.4f} % "
                     f"| {r['max_abs_error_K']:.2e} |")
    lines.append(f"\nAcceptance L2 = {100 * a['l2_rel']:.4f} % (limit 5 %), "
                 f"series flux q = {ice_steel['analytic']['q']:.3f} W/m$^2$, "
                 f"contact jump = {ice_steel['analytic']['contacts'][0]['t_left'] - ice_steel['analytic']['contacts'][0]['t_right']:.4f} K.\n")
    a = icing["acceptance"]
    lines.append("## C4 cross-adjudication with W4-N08 icing\n")
    lines.append(f"Condition (from the W4-N08 artifact): `{icing['condition_string']}`; "
                 f"artifact sha256 `{icing['data_sha256'][:16]}…`.\n")
    lines.append("| quantity | TensorFVM | W4-N08 | rel. error |\n|---|---|---|---|")
    for c in icing["closure_compare"]:
        if "ours_W_m2" in c:
            lines.append(f"| {c['quantity']} [W/m2] | {c['ours_W_m2']:.4f} "
                         f"| {c['w4n08_W_m2']:.4f} | {c['rel_error']:.2e} |")
        else:
            lines.append(f"| {c['quantity']} | {c['ours']:.6f} | {c['w4n08']:.6f} "
                         f"| {c['rel_error']:.2e} |")
    f = icing["field"]
    lines.append(f"\n* temperature field L2 (CHT vs independent reference, same ice "
                 f"{1000 * f['h_ice_m']:.2f} mm): **{100 * f['l2_rel']:.4f} %** (limit 5 %)")
    ca = icing["conduction_aware"]
    lines.append(f"* wall resistance surface->back: {ca['r_wall_m2K_W']:.6f} m$^2$K/W, "
                 f"contact jump {f['q_contact_jump_K']:.4f} K, series q = "
                 f"{f['q_series_W_m2']:.2f} W/m$^2$")
    lines.append(f"* conduction-aware adjudication at the cold hold "
                 f"({ca['at_cold_hold']['t_back_degC']:.1f} degC): regime "
                 f"{icing['module_output']['regime']} -> {ca['at_cold_hold']['regime']}, "
                 f"f = {icing['module_output']['freezing_fraction']:.3f} -> "
                 f"{ca['at_cold_hold']['freezing_fraction']:.3f}, icing rate "
                 f"{icing['module_output']['icing_rate_mm_per_h']:.2f} -> "
                 f"{ca['at_cold_hold']['icing_rate_mm_per_h']:.2f} mm/h")
    lines.append("")
    lines.append("| back-face T [degC] | regime | f | surface T [degC] | q_cond [W/m2] "
                 "| icing rate [mm/h] |\n|---|---|---|---|---|---|")
    for s in ca["sweep"]:
        lines.append(f"| {s['t_back_degC']:.1f} | {s['regime']} | {s['freezing_fraction']:.3f} "
                     f"| {s['t_surface_degC']:.2f} | {s['q_cond_W_m2']:.1f} "
                     f"| {s['icing_rate_mm_per_h']:.2f} |")
    lines.append("")
    lines.append("## Figures\n")
    for path in figs:
        lines.append(f"![{path.stem}](figures/{path.name})\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    choices=["all", "interface", "stefan", "ice_steel", "icing"])
    ap.add_argument("--out", default=str(ARTIFACTS))
    ap.add_argument("--docs", default=str(DOCS))
    args = ap.parse_args()
    out = Path(args.out)
    cases = {}
    if args.case in ("all", "interface"):
        cases["interface"] = case_interface(out)
    if args.case in ("all", "stefan"):
        cases["stefan"] = case_stefan(out)
    if args.case in ("all", "ice_steel"):
        cases["ice_steel"] = case_ice_steel(out)
    if args.case in ("all", "icing"):
        cases["icing"] = case_icing(out)
    if args.case == "all":
        figs = _figures(cases["interface"], cases["stefan"], cases["ice_steel"],
                        cases["icing"], Path(args.docs) / "figures")
        (Path(args.docs)).mkdir(parents=True, exist_ok=True)
        (Path(args.docs) / "RESULTS.md").write_text(
            _results_markdown(cases["interface"], cases["stefan"], cases["ice_steel"],
                              cases["icing"], figs))
        summary = {k: v["acceptance"] for k, v in cases.items()}
        summary["all_passed"] = all(v["passed"] for v in summary.values())
        _dump(out, "acceptance", summary)
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(cases[args.case]["acceptance"], indent=2))


if __name__ == "__main__":
    main()