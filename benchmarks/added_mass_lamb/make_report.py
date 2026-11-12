#!/usr/bin/env python3
"""Build (and optionally post) the paper-style report for this benchmark.

The markdown body and the figures are generated *from the artefacts* in this
directory, so the report cannot disagree with `convergence.json` /
`ellipsoid_scan.csv` and no number is retyped.

Usage
-----
    PYTHONPATH=src python benchmarks/added_mass_lamb/make_report.py            # writes report.json
    PYTHONPATH=src python benchmarks/added_mass_lamb/make_report.py --title-suffix "(W24)"

Posting is done by the caller with the platform credentials, e.g.::

    curl -s -X POST $TENSOR_API_BASE/tensor/reports -H "Authorization: Bearer $TOKEN" \\
         -H 'Content-Type: application/json' -d @report.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def b64(path: pathlib.Path) -> dict:
    return {"name": path.stem,
            "caption": CAPTIONS.get(path.stem, path.stem),
            "png_base64": base64.b64encode(path.read_bytes()).decode()}


CAPTIONS = {
    "convergence": "Added-mass error of the worst axis for every body vs the representative "
                   "panel size h = 1/sqrt(N). Dashed/dotted guides are slopes 1 and 2.",
    "deviation_finest": "Deviation k_panel/k_anchor - 1 at the finest mesh for every case and "
                        "axis; the dashed lines are the +-5 % acceptance band.",
    "quadrature_study": "Same body, same mesh, two surface-integration rules: the centroid rule "
                        "carries the self-panel (singular) bias, the Gauss rule removes it.",
    "anchor_limits": "Anchor limits: slender-prolate k_axial -> 0 / k_transverse -> 1, and flat "
                     "oblate M_33/(8/3 rho a^3) -> 1 (k_3 itself diverges).",
}


def scan_rows() -> list[dict]:
    with (HERE / "ellipsoid_scan.csv").open() as fh:
        return list(csv.DictReader(fh))


def markdown(anchors: dict, conv: dict, rows: list[dict]) -> str:
    worst = conv["worst"]
    crit = {c["id"]: c for c in conv["criteria"]}
    L: list[str] = []
    add = L.append
    add("## 1. Case and physical background")
    add("")
    add("Potential-flow added mass of an ellipsoid family in an unbounded ideal fluid. The")
    add("question is not whether one body is close to a number, but whether the source-panel")
    add("solver used by the manoeuvring model reproduces the whole *family* predicted by Lamb")
    add("(1932, sec. 108-114) as the mesh is refined, where its worst case sits, and what the")
    add("observed order of convergence is.")
    add("")
    add("`M_ij = -rho * closed_integral_S phi_i n_j dS`, `laplace(phi_i) = 0`,")
    add("`dphi_i/dn = -n_i`, `phi_i -> 0` at infinity; for an ellipsoid `M_ii = k_i rho V`,")
    add("`V = (4/3) pi a b c`, `k_1 = alpha0/(2 - alpha0)` with")
    add("`alpha0 = a b c * integral_0^inf dl / ((a^2+l) sqrt((a^2+l)(b^2+l)(c^2+l)))`.")
    add("")
    add("| case | family | a | b | c | k1 | k2 | k3 |")
    add("|---|---|---|---|---|---|---|---|")
    for cse in anchors["cases"]:
        ax = cse["semi_axes"]
        k = cse["k"]
        add(f"| `{cse['label']}` | {cse['family']} | {ax[0]:g} | {ax[1]:g} | {ax[2]:g} | "
            f"{k[0]:.6f} | {k[1]:.6f} | {k[2]:.6f} |")
    add("")
    add("## 2. Method and toolchain")
    add("")
    add("`TensorFVM` `tensorfvm.panel_potential` (constant-strength source panels, 3x3 Gauss")
    add("quadrature per panel, `+1/2` single-layer jump on the matrix diagonal) driving")
    add("`tensorfvm.panel_added_mass.added_mass_matrix`; latitude/longitude quad meshes of the")
    add("exact ellipsoid at five levels per body; anchors from the two in-repo Lamb routes")
    add("(closed form for sphere/prolate/oblate, log-mapped defining integral otherwise).")
    add("Driver: `benchmarks/added_mass_lamb/run_benchmark.py`; figures:")
    add("`make_figures.py`. Ladder: " + ", ".join(conv["levels"]) + " (N_theta x N_phi).")
    add("")
    add("Three singular/near-singular terms are handled explicitly: (i) self-panel normal")
    add("velocity = 0 principal value + 1/2 jump; (ii) self-panel single-layer potential")
    add("`integral dS/|x-r'|` (weakly singular, integrable) evaluated by the same 3x3 Gauss")
    add("rule with a `r^2 < 1e-20` near-field cutoff; (iii) the added-mass surface integral")
    add("evaluated with the solver's own Gauss nodes (`quadrature='gauss'`) instead of a")
    add("per-panel midpoint rule. A controlled comparison of (iii) is included below.")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Anchor limits (criterion 1)")
    add("")
    add("| limit | quantity | value | reference | deviation | verdict |")
    add("|---|---|---|---|---|---|")
    for r in anchors["limit_checks"]:
        if r["limit"] == "slender prolate":
            val = "; ".join(f"b/a={e['b_over_a']:g}: {e['k_axial']:.5f}/{e['k_transverse']:.5f}"
                            for e in r["value"])
            dev = f"k_ax {r['final_gap'][0]:.2e}, 1-k_tr {r['final_gap'][1]:.2e}"
        elif r["limit"] == "flat oblate disc":
            val = "; ".join(f"c/a={e['c_over_a']:g}: {e['M_normal_over_disc']:.6f}"
                            for e in r["value"])
            dev = f"M33 gap {r['final_gap'][0]:.2e}"
        else:
            val = ", ".join(f"{v:.10f}" for v in r["value"])
            dev = f"{r['worst_deviation_pct']:.2e} %"
        add(f"| {r['limit']} ({r['quantity']}) | {r['parameter']} | {val} | {r['reference']} | "
            f"{dev} | {r['verdict']} |")
    add("")
    add("### 3.2 Per-case deviations and criteria (criteria 2 and 5)")
    add("")
    add("| case | level | N | k1 (dev %) | k2 (dev %) | k3 (dev %) | verdict |")
    add("|---|---|---|---|---|---|---|")
    by: dict[str, dict[str, dict[int, dict]]] = {}
    for row in rows:
        by.setdefault(row["case"], {}).setdefault(row["level"], {})[int(row["axis"])] = row
    for case_label, levels in by.items():
        for level, axes in levels.items():
            cells, verdict = [], "PASS"
            for axis in (1, 2, 3):
                r = axes[axis]
                cells.append(f"{float(r['k_panel']):.5f} ({float(r['dev_pct']):+.2f})")
                verdict = "FAIL" if r["pass_5pct"] != "PASS" else verdict
            add(f"| `{case_label}` | {level} | {axes[1]['n_panels']} | {cells[0]} | {cells[1]} | "
                f"{cells[2]} | {verdict} |")
    add("")
    add("### 3.3 Order of convergence, Richardson and GCI (criterion 3)")
    add("")
    add("| case | axis | |dev| % finest | p_all | p_coarse | k_Richardson (dev %) | GCI % |")
    add("|---|---|---|---|---|---|---|")
    for case in conv["cases"]:
        for comp in case["components"]:
            add(f"| `{case['label']}` | {comp['axis']} | {abs(comp['finest_dev_pct']):.3f} | "
                f"{comp['p_all_fit']:.3f} | {comp['p_coarse_fit']:.3f} | "
                f"{comp['k_richardson']:.6f} ({comp['richardson_dev_pct']:+.3f}) | "
                f"{comp['gci_pct']:.3f} |")
    add("")
    add("### 3.4 Worst case (criterion 4)")
    add("")
    add(f"`{worst['case']}` axis {worst['axis']} ({worst['family']}, semi-axes "
        f"{worst['semi_axes']}): raw deviation {worst['dev_pct']:+.3f} %, extrapolated "
        f"{worst['richardson_dev_pct']:+.3f} %, GCI {worst['gci_pct']:.3f} %; error per level "
        + ", ".join(f"{e * 100:+.2f} %" for e in worst["rel_error_by_level"]) + ".")
    add("")
    add("## 4. Figures")
    add("")
    add("Figures 1-4 (attached): mesh convergence per body; deviation of every component at the")
    add("finest mesh against the +-5 % band; the centroid-vs-Gauss self-panel comparison; the two")
    add("anchor limit ladders.")
    add("")
    add("## 5. Notes, traps and limits")
    add("")
    add("* The observed orders are *not* the textbook 2: flat-quad faceting, the numerically")
    add("  evaluated weakly singular self-term and the anisotropic pole-crowded mesh all")
    add("  contribute. Richardson extrapolation still recovers most components to well inside")
    add("  the band, which is why it is reported next to the raw values.")
    add("* The latitude/longitude mesh has degenerate pole quads (two coincident corners), so")
    add("  mesh shape is measured as `max_edge/sqrt(area)`; an edge-length ratio is unbounded")
    add("  there and would be a meaningless diagnostic.")
    add("* The inscribed polyhedron under-predicts both the volume and the added mass; the two")
    add("  deficits correlate across all reported rows, which is the quantitative basis for")
    add("  attributing the residual to faceting rather than to the singular self-term.")
    add("* Scope: unbounded ideal fluid only (no free surface, no forward speed, no forward")
    add("  speed/frequency dependence). Agreement here is a property of the solver on the")
    add("  ellipsoid family, not of a hull model.")
    add("")
    add("## 6. Error verdict against the anchors")
    add("")
    comps = [c for case in conv["cases"] for c in case["components"]]
    rich = [abs(c["richardson_dev_pct"]) for c in comps]
    raw = [abs(c["finest_dev_pct"]) for c in comps]
    add(f"* Raw finest-mesh values: worst {max(raw):.2f} %, {sum(1 for v in raw if v <= 3.0)}")
    add(f"  of {len(raw)} components already inside 3 %.")
    add(f"* Richardson-extrapolated values: worst {max(rich):.2f} %, "
        f"{sum(1 for v in rich if v <= 3.0)} of {len(rich)} inside 3 %, "
        f"{sum(1 for v in rich if v <= 5.0)} of {len(rich)} inside 5 %.")
    add(f"* Anchor limits: criterion 1 {crit[1]['verdict']}; per-component +-5 % at the finest "
        f"mesh: criterion 2 {crit[2]['verdict']}; GCI < 5 %: criterion 3 {crit[3]['verdict']}.")
    add("* Is the key quantity below 3 %? For every component after Richardson extrapolation: "
        + ("**yes**" if max(rich) <= 3.0 else f"**no** - worst {max(rich):.2f} %")
        + "; for the raw finest mesh: "
        + ("**yes**" if max(raw) <= 3.0 else f"**no** - worst {max(raw):.2f} %")
        + ". Raw failures are reported as FAIL in the table above, not smoothed over.")
    add("")
    add("Artefacts: `anchors.json`, `ellipsoid_scan.csv`, `convergence.json`, `README.md`,")
    add("`figures/`.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "report.json"))
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--mission-id", default="15976")
    ap.add_argument("--scenario-key", default="paper-exp-added-mass-lamb-ellipsoid-1")
    args = ap.parse_args(argv)

    anchors = load("anchors.json")
    conv = load("convergence.json")
    rows = scan_rows()
    figs = [b64(p) for p in sorted((HERE / "figures").glob("*.png"))]
    worst = conv["worst"]
    summary = (f"Source-panel added mass vs Lamb (1932) ellipsoid anchors: {len(conv['cases'])} "
               f"bodies x {len(conv['levels'])} meshes; worst raw deviation "
               f"{worst['dev_pct']:+.2f} % ({worst['case']} axis {worst['axis']}), "
               f"extrapolated {worst['richardson_dev_pct']:+.2f} %; criteria "
               + "/".join(f"C{c['id']}={c['verdict']}" for c in conv["criteria"]))
    payload = {
        "title": "Source-panel added mass on the Lamb ellipsoid family: anchors, "
                 "axis-ratio scan, convergence order and GCI" + args.title_suffix,
        "summary": summary,
        "scenario_key": args.scenario_key,
        "mission_id": args.mission_id,
        "tags": ["TensorFVM", "added-mass", "potential-flow", "panel-method",
                 "Lamb-1932", "verification", "GCI", "AN-LAMB-ELLIPSOID"],
        "markdown": markdown(anchors, conv, rows),
        "figures": figs,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload))
    print(f"[report] {args.out} ({len(payload['markdown'])} chars, {len(figs)} figures)")
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())