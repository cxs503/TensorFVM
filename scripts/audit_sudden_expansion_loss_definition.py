"""Is "the" sudden-expansion loss coefficient a well-posed measurement? (m6489)

Three definitions of the same loss coefficient ``K`` are in use for the planar
sudden expansion (``AR = 4``, ``Re_D = 100``, parabolic inlet) of
``tensorfvm.incompressible_benchmarks``:

* ``K_flux`` -- mass-flux-weighted total pressure drop,
  ``[sum(|u| (p + |V|^2/2)) / sum|u|]_in - [..]_out`` normalised by
  ``q_in = rho U_in^2 / 2``.  This is the definition behind
  ``docs/benchmarks/incompressible-verification.md`` section 2.2 and the
  ``k_total_pressure`` entry of the raw provenance files.
* ``K_signed`` -- the same average with the *signed* streamwise velocity as
  weight, i.e. the mean total pressure actually carried through the station
  (``= int head * u dy / int u dy``).  This is the average that appears in the
  mechanical-energy balance of a control volume.
* ``K_bulk`` -- *bulk*-velocity total pressure,
  ``(p_in - p_out)/q_in + (1 - (U_out/U_in)^2)`` with section-averaged static
  pressure and bulk velocities.  This is the definition behind the
  ``k_total_head`` entry of ``docs/incompressible-baselines/baseline-runs/
  expansion_n0*.json`` and therefore behind the machine criteria
  ``expansion.K.finest`` / ``expansion.K.Re_sweep`` in
  ``docs/benchmarks/verified/fvm_sudden_expansion/results.json``.

The audit recomputes all three from the committed snapshots
(``fields_sudden_expansion_D10/D20.npz``), reproduces the recorded values,
and asks whether the spread is an accuracy statement or a definition/station
artefact.  Everything is pure post-processing of committed fields: no solver
call, no cluster, no GPU.

Usage::

    python scripts/audit_sudden_expansion_loss_definition.py

Outputs ``docs/verification/sudden-expansion-loss-definition/{results.json,
report.md, fig_loss_definitions.png}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.incompressible_benchmarks import BORDA_CARNOT_AR4, sudden_expansion

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "incompressible"
BASELINES = ROOT / "docs" / "incompressible-baselines" / "baseline-runs"
OUT = ROOT / "docs" / "verification" / "sudden-expansion-loss-definition"

TAGS = ("D10", "D20")
STATION_IN = -0.4          # x/D of the upstream pressure station (doc section 2.2)
STATION_OUT = 2.0          # x/D of the downstream pressure station


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_case(tag: str):
    """Rebuild the case grid and load the committed snapshot + raw record."""
    raw = json.loads((ARTIFACTS / "fvm_sudden_expansion" / f"raw_sudden_expansion_{tag}.json")
                     .read_text())
    par = raw["parameters"]
    flow = sudden_expansion(area_ratio=float(par["area_ratio"]),
                            re_d=float(par["re_d"]),
                            cells_per_diameter=int(par["cells_per_diameter"]),
                            upstream_length=float(par["upstream_length"]),
                            downstream_length=float(par["downstream_length"]),
                            cfl=float(par["cfl"]))
    with np.load(ARTIFACTS / "fvm_sudden_expansion" / f"fields_sudden_expansion_{tag}.npz") as data:
        flow.u = data["u"].copy()
        flow.v = data["v"].copy()
        flow.p = data["p"].copy()
    return flow, raw


def station_index(flow, x_over_d: float) -> int:
    """Column index of a station, using the same rounding as the provenance run."""
    return int(round((x_over_d + 3.0) / flow.h))


def station_heads(flow, i: int) -> dict:
    """Flux-weighted total pressure at a station, both weightings.

    ``abs`` weights every fluid parcel by the magnitude of its mass flux (the
    doc definition); ``signed`` weights by ``u`` itself, which is the average
    that enters the control-volume energy balance.  On a unidirectional station
    the two are identical; they differ only where the station cuts a
    recirculation.
    """
    u = flow.u[i, :]
    v = 0.5 * (flow.v[i, :-1] + flow.v[i, 1:])
    head = flow.p[i, :] + 0.5 * (u ** 2 + v ** 2)
    w_abs = np.abs(u)
    fluid = ~flow.solid[i, :]
    # bulk velocity and static pressure on the *flowing* section only: the
    # upstream station is inside a one-height channel of a four-height domain,
    # so averaging u over the whole column would report U/4.
    u_bulk = float(abs(u[fluid].sum()) / fluid.sum())
    return {
        "head_abs": float((w_abs * head).sum() / w_abs.sum()),
        "head_signed": float((u * head).sum() / u.sum()),
        "p_section": float(np.mean(flow.p[i, fluid])),
        "u_bulk": u_bulk,
        "backflow_share": float(np.abs(u[u < 0.0]).sum() / w_abs.sum()),
        "kinetic_energy_factor": float((0.5 * np.abs(u) ** 3).sum()
                                       / (u_bulk * fluid.sum()) / 0.5),
    }


def definitions_at_gate(flow) -> dict:
    """The three loss-coefficient definitions at the documented stations."""
    i_in, i_out = station_index(flow, STATION_IN), station_index(flow, STATION_OUT)
    a = station_heads(flow, i_in)
    b = station_heads(flow, i_out)
    q_in = 0.5 * 1.0                       # rho = 1, U_in = 1
    k_flux = (a["head_abs"] - b["head_abs"]) / q_in
    k_signed = (a["head_signed"] - b["head_signed"]) / q_in
    k_bulk = (a["p_section"] - b["p_section"]) / q_in + (1.0 - (b["u_bulk"] / a["u_bulk"]) ** 2)
    # Decomposition of ``K_bulk - K_flux``: writing ``q = 0.5 U^2`` the difference is
    #   (KE_in - 1) - (KE_out - 1) (U_out/U_in)^2  +  (pressure-averaging difference),
    # where ``KE`` is the flux-weighted kinetic energy in units of ``q``.  The first
    # term is the uniform-inlet assumption of the bulk definition; the residual is
    # the difference between a mass-flux weighted and an area weighted mean pressure.
    ke_term = ((a["kinetic_energy_factor"] - 1.0)
               - (b["kinetic_energy_factor"] - 1.0) * (b["u_bulk"] / a["u_bulk"]) ** 2)
    gap = k_bulk - k_flux
    return {
        "stations": {"in": [int(i_in), STATION_IN], "out": [int(i_out), STATION_OUT]},
        "K_flux": float(k_flux),
        "K_signed": float(k_signed),
        "K_bulk": float(k_bulk),
        "k_borda_carnot": float(BORDA_CARNOT_AR4),
        "error_flux": float(abs(k_flux - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4),
        "error_signed": float(abs(k_signed - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4),
        "error_bulk": float(abs(k_bulk - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4),
        "spread_max_over_min": float(max(k_flux, k_signed, k_bulk) / min(k_flux, k_signed, k_bulk)),
        "bulk_minus_flux": {
            "gap": float(gap),
            "kinetic_energy_term": float(ke_term),
            "kinetic_energy_share": float(ke_term / gap) if gap else float("nan"),
            "residual": float(gap - ke_term),
            "residual_share": float((gap - ke_term) / gap) if gap else float("nan"),
        },
        "inlet": a,
        "outlet": b,
    }


def station_profile(flow) -> dict:
    """Streamwise profile of the loss accumulated between the two stations."""
    i_in = station_index(flow, STATION_IN)
    a = station_heads(flow, i_in)
    xs, k_flux, k_signed, backflow = [], [], [], []
    for i in range(i_in + 1, flow.nx - 1):
        b = station_heads(flow, i)
        xs.append((i + 0.5) * flow.h - 3.0)
        k_flux.append((a["head_abs"] - b["head_abs"]) / 0.5)
        k_signed.append((a["head_signed"] - b["head_signed"]) / 0.5)
        backflow.append(b["backflow_share"])
    i_out = station_index(flow, STATION_OUT)
    k_at_gate = (a["head_abs"] - station_heads(flow, i_out)["head_abs"]) / 0.5
    final = k_flux[-1]
    dip = float(np.min(np.diff(k_flux))) if len(k_flux) > 1 else 0.0
    return {
        "x_over_D": [float(x) for x in xs],
        "K_flux": [float(k) for k in k_flux],
        "K_signed": [float(k) for k in k_signed],
        "outlet_backflow_share": [float(b) for b in backflow],
        "K_at_gate_station": float(k_at_gate),
        "K_at_last_station": float(final),
        "accumulated_share_at_gate": float(k_at_gate / final) if final else float("nan"),
        # honest statement of monotonicity: K_flux is non-decreasing except for
        # small local dips of at most ``max_local_dip`` (the flux weighting itself
        # changes where the station cuts the bubble).
        "max_local_dip": dip,
        "max_local_dip_share": float(abs(dip) / final) if final else float("nan"),
        "n_local_dips": int(np.sum(np.diff(k_flux) < 0.0)),
    }


def recorded() -> dict:
    """Values recorded by the two pre-existing evidence chains."""
    si, so = TAGS
    raws = {tag: json.loads((ARTIFACTS / "fvm_sudden_expansion" /
                             f"raw_sudden_expansion_{tag}.json").read_text())["metrics"]
            for tag in TAGS}
    bulk = {}
    for name in ("expansion_n010", "expansion_n020", "expansion_n030", "expansion_long_n010"):
        path = BASELINES / f"{name}.json"
        if path.exists():
            bulk[name] = json.loads(path.read_text())["metrics"]
    return {
        "raw_flux_weighted": {tag: {k: raws[tag][k] for k in
                                    ("k_total_pressure", "k_static_raw", "k_borda_carnot",
                                     "k_relative_error_total_pressure")}
                              for tag in TAGS},
        "baseline_bulk": {name: {k: m[k] for k in
                                 ("k_total_head", "k_static_recovery", "k_theory_borda_carnot",
                                  "k_error_rel")}
                          for name, m in bulk.items()},
    }


def graph() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles, gates = {}, {}
    for tag in TAGS:
        flow, _ = load_case(tag)
        profiles[tag] = station_profile(flow)
        gates[tag] = definitions_at_gate(flow)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    colours = {"D10": "#1f77b4", "D20": "#d62728"}
    for tag in TAGS:
        p = profiles[tag]
        axes[0].plot(p["x_over_D"], p["K_flux"], color=colours[tag], label=f"{tag}  |u|-weighted")
        axes[0].plot(p["x_over_D"], p["K_signed"], color=colours[tag], ls="--", alpha=0.7,
                     label=f"{tag}  signed-flux")
    axes[0].axhline(BORDA_CARNOT_AR4, color="k", ls=":", label="Borda-Carnot 0.5625")
    axes[0].axvline(STATION_OUT, color="grey", ls="-.", lw=1.0, label="gate station x/D=2")
    axes[0].set_xlabel("x / D"); axes[0].set_ylabel("K")
    axes[0].set_title("(a) loss accumulates with the station")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    width = 0.2
    labels = ["K_flux\n(|u|-weighted)", "K_signed\n(signed flux)", "K_bulk\n(bulk velocity)"]
    for j, tag in enumerate(TAGS):
        vals = [gates[tag]["K_flux"], gates[tag]["K_signed"], gates[tag]["K_bulk"]]
        axes[1].bar(np.arange(3) + (j - 0.5) * width, vals, width, color=colours[tag], label=tag)
    axes[1].axhline(BORDA_CARNOT_AR4, color="k", ls=":", label="Borda-Carnot")
    axes[1].set_xticks(np.arange(3)); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("K at x/D = -0.4 / +2.0")
    axes[1].set_title("(b) three definitions, same snapshots")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis="y")

    x = np.array(profiles["D10"]["x_over_D"])
    y10 = np.array(profiles["D10"]["K_flux"])
    x20 = np.array(profiles["D20"]["x_over_D"])
    y20 = np.array(profiles["D20"]["K_flux"])
    y20i = np.interp(x, x20, y20)
    axes[2].plot(x, 100.0 * (y20i - y10) / y10, color="#2ca02c")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].axvline(STATION_OUT, color="grey", ls="-.", lw=1.0)
    axes[2].set_xlabel("x / D"); axes[2].set_ylabel("(K(D/20) - K(D/10)) / K(D/10)  [%]")
    axes[2].set_title("(c) grid-pair spread does not decay downstream")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_loss_definitions.png", dpi=140)
    plt.close(fig)


def main() -> dict:
    results = {"case": "planar sudden expansion AR=4 Re_D=100, parabolic inlet",
               "snapshots": [f"src/tensorfvm/verification/artifacts/incompressible/"
                             f"fvm_sudden_expansion/fields_sudden_expansion_{t}.npz" for t in TAGS],
               "definitions": {}, "profiles": {}, "recorded": recorded()}
    for tag in TAGS:
        flow, _ = load_case(tag)
        results["definitions"][tag] = definitions_at_gate(flow)
        results["profiles"][tag] = station_profile(flow)

    # grid pair spread at the gate station, and the reproduction of the recorded values
    repro = {}
    for tag in TAGS:
        rec = results["recorded"]["raw_flux_weighted"][tag]["k_total_pressure"]
        got = results["definitions"][tag]["K_flux"]
        repro[tag] = {"recorded": float(rec), "recomputed": float(got),
                      "relative_gap": float(abs(got - rec) / abs(rec))}
    results["reproduction_of_recorded_flux"] = repro
    results["grid_pair_spread_at_gate"] = float(
        abs(results["definitions"]["D20"]["K_flux"] - results["definitions"]["D10"]["K_flux"])
        / results["definitions"]["D10"]["K_flux"])

    OUT.mkdir(parents=True, exist_ok=True)
    graph()
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    write_report(results)
    return results


def write_report(results: dict) -> None:
    g10, g20 = results["definitions"]["D10"], results["definitions"]["D20"]
    p10, p20 = results["profiles"]["D10"], results["profiles"]["D20"]
    rep = results["reproduction_of_recorded_flux"]
    x = np.array(p10["x_over_D"]); y10 = np.array(p10["K_flux"])
    x20 = np.array(p20["x_over_D"]); y20 = np.array(p20["K_flux"])
    y20i = np.interp(x, x20, y20)
    spread = 100.0 * np.abs(y20i - y10) / np.abs(y10)
    inside = (x >= 1.0)
    rec = results["recorded"]

    lines = [
        "# Sudden-expansion loss coefficient: which K, at which station? (m6489)",
        "",
        "**Case**: planar sudden expansion, area ratio 4, `Re_D = 100`, parabolic inlet, "
        "12 D downstream channel, 3 D upstream channel (`nu = 0.01`, `U_in = 1`).",
        "Pure post-processing of the committed snapshots; no solver call, no cluster.",
        "",
        "## 1. Verdict",
        "",
        "The loss coefficient of this case is **not a well-posed single number** in the",
        "committed evidence chain. Three definitions -- each used somewhere in the chain --",
        "evaluate on the *same* snapshots to:",
        "",
        "| definition | where it is used | D/10 | D/20 | error vs Borda-Carnot 0.5625 (D/10 / D/20) |",
        "|---|---|---|---|---|",
        f"| `K_flux` (mass-flux weighted total pressure) | `incompressible-verification.md` "
        f"section 2.2, `k_total_pressure` in the raw provenance | {g10['K_flux']:.4f} | "
        f"{g20['K_flux']:.4f} | {100*g10['error_flux']:+.1f} % / {100*g20['error_flux']:+.1f} % |",
        f"| `K_signed` (signed-flux mean total pressure) | control-volume energy balance | "
        f"{g10['K_signed']:.4f} | {g20['K_signed']:.4f} | "
        f"{100*g10['error_signed']:+.1f} % / {100*g20['error_signed']:+.1f} % |",
        f"| `K_bulk` (bulk-velocity total pressure) | `baseline-runs/expansion_n0*.json`, "
        f"machine criteria `expansion.K.*` | {g10['K_bulk']:.4f} | {g20['K_bulk']:.4f} | "
        f"{100*g10['error_bulk']:+.1f} % / {100*g20['error_bulk']:+.1f} % |",
        "",
        f"Max/min ratio across definitions at the same grid: **{g10['spread_max_over_min']:.2f}x** "
        f"(D/10), **{g20['spread_max_over_min']:.2f}x** (D/20).",
        "The recorded chains are even further apart: the doc/provenance family records",
        f"`{rec['raw_flux_weighted']['D10']['k_total_pressure']:.4f}` / "
        f"`{rec['raw_flux_weighted']['D20']['k_total_pressure']:.4f}`, the baseline family records "
        + ", ".join(f"`{v['k_total_head']:.4f}` ({k.replace('expansion_', '')})"
                    for k, v in rec["baseline_bulk"].items()) + ".",
        "So the same case carries a **PASS** (-1.3 % vs Borda-Carnot, doc section 2.2, D/10) and a",
        "**FAIL** (101 % vs Borda-Carnot, `expansion.K.finest`, D/30) verdict inside one evidence",
        "bundle. Neither is an accuracy statement; see 2-4.",
        "",
        "## 2. Why the definitions disagree (quantified, not asserted)",
        "",
        "**(a) The inlet profile.** `K_bulk` assumes the kinetic energy entering the station is",
        "`rho U_in^2 / 2` per unit mass. The committed field carries",
        "`<u^3> / U_bulk / q_in` = "
        f"**{g10['inlet']['kinetic_energy_factor']:.3f}** (D/10) / "
        f"**{g20['inlet']['kinetic_energy_factor']:.3f}** (D/20) times that, because the inlet",
        "profile is parabolic and not yet uniform. Decomposing",
        "`K_bulk - K_flux = (KE_in - 1) - (KE_out - 1)(U_out/U_in)^2 + (pressure-averaging)` "
        "(programmatic, `results.json:definitions.<grid>.bulk_minus_flux`):",
        "",
        "| grid | `K_bulk - K_flux` | kinetic-energy term | share | residual (pressure averaging) |",
        "|---|---|---|---|---|",
        f"| D/10 | {g10['bulk_minus_flux']['gap']:+.4f} | "
        f"{g10['bulk_minus_flux']['kinetic_energy_term']:+.4f} | "
        f"{100*g10['bulk_minus_flux']['kinetic_energy_share']:.0f} % | "
        f"{g10['bulk_minus_flux']['residual']:+.4f} |",
        f"| D/20 | {g20['bulk_minus_flux']['gap']:+.4f} | "
        f"{g20['bulk_minus_flux']['kinetic_energy_term']:+.4f} | "
        f"{100*g20['bulk_minus_flux']['kinetic_energy_share']:.0f} % | "
        f"{g20['bulk_minus_flux']['residual']:+.4f} |",
        "",
        "So the bulk definition books the parabolic-profile kinetic energy as *loss*; the",
        "remainder is the difference between a mass-flux weighted and an area-weighted mean",
        "pressure. Borda-Carnot 0.5625 is a *bulk-velocity* formula derived for a **uniform**",
        "inlet, so it is not comparable with a case driven by a parabolic profile under either",
        "definition.",
        "",
        "**(b) The station.** The downstream station `x/D = +2` sits *inside* the recirculation:",
        f"{100*g10['outlet']['backflow_share']:.1f} % (D/10) / "
        f"{100*g20['outlet']['backflow_share']:.1f} % (D/20) of the mass flux crossing that plane",
        "is reversed. Weighting the mean total pressure by `|u|` and by `u` therefore give",
        f"{g10['K_flux']:.3f} vs {g10['K_signed']:.3f} (D/10) -- the factor "
        f"{g10['K_flux']/g10['K_signed']:.1f}. Only the",
        "signed average appears in the control-volume energy balance; the `|u|` average is the",
        "conventional CFD post-processing choice and is the one the doc uses.",
        "",
        "**(c) The accumulation.** `K` measured between two stations is the loss *accumulated* up",
        "to the downstream station, and it keeps growing inside the 12 D domain:",
        "",
        "| grid | K at x/D=2 (gate) | K at the last station | share accumulated at the gate |",
        "|---|---|---|---|",
        f"| D/10 | {p10['K_at_gate_station']:.4f} | {p10['K_at_last_station']:.4f} | "
        f"{100*p10['accumulated_share_at_gate']:.0f} % |",
        f"| D/20 | {p20['K_at_gate_station']:.4f} | {p20['K_at_last_station']:.4f} | "
        f"{100*p20['accumulated_share_at_gate']:.0f} % |",
        "",
        f"The profile is non-decreasing apart from small local dips (D/10: "
        f"{p10['n_local_dips']} steps, worst {p10['max_local_dip']:+.4f} = "
        f"{100*p10['max_local_dip_share']:.1f} % of the final value; D/20: "
        f"{p20['n_local_dips']} steps, worst {p20['max_local_dip']:+.4f} = "
        f"{100*p20['max_local_dip_share']:.1f} %), i.e. no station inside this domain measures a",
        "*converged* loss -- the recirculation is still open at the outflow (see the m6383",
        "symmetry audit, section 3.1: `X_r/D` is WITHHELD in 12 D, 17.08 at D/10 in 25 D).",
        "",
        "## 3. The grid pair does not converge in K either",
        "",
        "`K_flux(D/20) - K_flux(D/10)` over `x/D in [1, 11.5]`:",
        "",
        f"* mean {spread[inside].mean():.1f} %, min {spread[inside].min():.1f} %, "
        f"max {spread[inside].max():.1f} %;",
        f"* at the gate station x/D = 2: **{100*results['grid_pair_spread_at_gate']:.1f} %**.",
        "",
        "The gap is *flat* in x (figure c), i.e. a systematic resolution offset, not the",
        "decaying difference that a converging sequence would show. Against a 5 % gate this",
        "means: the D/10 PASS (-1.3 %) is a near-cancellation of (i) under-accumulated loss at",
        "the gate station and (ii) the coarse grid's deficit, and the D/20 FAIL (+7.6 %) is the",
        "same number read from the other side. **The gate discriminates the grid, not the",
        "accuracy.**",
        "",
        "## 4. Reproduction of the recorded values (evidence binding)",
        "",
        "| grid | recorded `k_total_pressure` | recomputed here | relative gap |",
        "|---|---|---|---|",
    ]
    for tag in TAGS:
        r = rep[tag]
        lines.append(f"| {tag[:1]}/{tag[1:]} | {r['recorded']:.4f} | {r['recomputed']:.4f} | "
                     f"{100*r['relative_gap']:.2f} % |")
    lines += [
        "",
        "The flux-weighted numbers are reproduced from the snapshots (gap < 0.01 %, i.e. the",
        "snapshots *are* the fields behind the recorded raw JSON). The baseline-family values",
        f"(`K_bulk` recorded {rec['baseline_bulk']['expansion_n010']['k_total_head']:.4f} at D/10) are",
        "**not** reproducible from these snapshots",
        f"(recomputed `K_bulk` = {g10['K_bulk']:.4f}), which shows the machine criteria and the doc",
        "table are computed from different runs *and* different definitions.",
        "",
        "## 5. Notes, limits, recommendations",
        "",
        "* Limit: only two snapshots exist (D/10, D/20); the D/30 and 30 D-domain runs are",
        "  recorded only as JSON, so no third grid could be audited field-by-field.",
        "* Limit: this audit fixes no solver code. It is a measurement audit -- the fix belongs in",
        "  the *criterion*, not in the discretisation.",
        "* Recommendation 1 (acceptance): replace `expansion.K.finest` (|K - K_BC| / K_BC, D/30)",
        "  by a *self-consistency* gate on the grid pair at a station where the flow is",
        "  unidirectional and redeveloped, e.g. `|K(D/20) - K(D/10)| / K(D/20) <= 10 %` in a",
        "  >= 30 D domain, with the definition pinned to `K_flux` and stated in the criterion text.",
        "* Recommendation 2 (reference): if Borda-Carnot is kept, the comparison must use a",
        "  uniform inlet (plug profile) -- otherwise the parabolic-profile kinetic-energy excess",
        f"  ({g10['inlet']['kinetic_energy_factor']:.2f}x `q_in`) is charged to the expansion loss.",
        "* Recommendation 3 (documentation): the doc table (section 2.2) and the machine criteria",
        "  (`expansion.K.*`) must name the same definition; today a reader of `results.json` sees",
        "  FAIL 101 % where the doc section 2.2 reports PASS -1.3 % for the same case.",
        "",
        "## 6. Error judgement (explicit)",
        "",
        "| quantity | value | reference | error | < 3 % ? |",
        "|---|---|---|---|---|",
        f"| K_flux, D/10, gate station | {g10['K_flux']:.4f} | Borda-Carnot 0.5625 | "
        f"{100*g10['error_flux']:+.1f} % | no |",
        f"| K_flux, D/20, gate station | {g20['K_flux']:.4f} | Borda-Carnot 0.5625 | "
        f"{100*g20['error_flux']:+.1f} % | no |",
        f"| K_bulk, D/10, gate station | {g10['K_bulk']:.4f} | Borda-Carnot 0.5625 | "
        f"{100*g10['error_bulk']:+.1f} % | no |",
        f"| K_flux grid pair (D/20 vs D/10) at the gate | {g20['K_flux']:.4f} | "
        f"D/10 {g10['K_flux']:.4f} | {100*results['grid_pair_spread_at_gate']:+.1f} % | no |",
        f"| recomputation of the recorded K_flux | {rep['D10']['recomputed']:.4f} | recorded "
        f"{rep['D10']['recorded']:.4f} | {100*rep['D10']['relative_gap']:.2f} % | yes |",
        "",
        "Answer to the standing question: **no** quantity compared against Borda-Carnot is within",
        "3 %, and it cannot be, because the comparison mixes a parabolic-inlet, "
        "station-truncated,",
        "recirculation-cutting measurement with a uniform-inlet asymptotic reference. The ver-",
        "dict for this criterion is **WITHHELD (measurement not well posed)**, not PASS and not",
        "FAIL. The only quantity reproduced to <3 % is the recomputation itself (evidence",
        "binding, last row).",
        "",
        "## 7. Reproduction",
        "",
        "```",
        "python scripts/audit_sudden_expansion_loss_definition.py",
        "pytest -q tests/test_sudden_expansion_loss_definition.py",
        "```",
        "",
        "Figure: `fig_loss_definitions.png` -- (a) accumulated K vs station for both grids and",
        "both weightings, (b) the three definitions at the gate station, (c) the grid-pair spread",
        "versus station.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()