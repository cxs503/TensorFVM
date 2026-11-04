"""Inject the machine-generated tables of the Ghia ladder into the note.

Reads ``docs/verification-lid-cavity-ladder/summary.json`` (falling back to
``summary.partial.json`` while the ladder is still running) and rewrites
``docs/verification-lid-cavity-ladder/results-tables.md`` plus the four
``<!--MARKER-->`` blocks of ``report.md``.  No number in the note is typed by
hand: every cell below comes from the JSON the runner wrote.

Usage::

    PYTHONPATH=src python3 scripts/summarize_lid_cavity_ladder.py
"""

from __future__ import annotations

import json
import os

from tensorfvm.lid_cavity_reference import VORTEX_CENTRE

OUT = "docs/verification-lid-cavity-ladder"


def load() -> dict:
    for name in ("summary.json", "summary.partial.json"):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            with open(path) as handle:
                data = json.load(handle)
            data["_source"] = name
            return data
    raise SystemExit("no summary in " + OUT)


def relative(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


def tables(summary: dict) -> dict[str, str]:
    meta = summary["meta"]
    cases = sorted(summary["cases"], key=lambda c: (c["reynolds"], c["n"]))

    # ---- Table 1: the ladder itself
    lines = [
        "### 3.1 Every row, as written by the runner "
        f"(`{summary['_source']}`)",
        "",
        "| Re | n | steps | converged | residual | max\\|Δu\\| | rms\\|Δu\\| | max\\|Δv\\| "
        "| u_min (Ghia) | v_max (Ghia) | vortex (Ghia) | max\\|div\\| | wall [s] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        s = c["solver"]
        u, v = c["u"], c["v"]
        vs = c["vortex_centre"]["computed"]
        ref_vs = VORTEX_CENTRE[c["reynolds"]]
        lines.append(
            f"| {c['reynolds']:g} | {c['n']} | {s['steps']} | "
            f"{'yes' if s['converged'] else 'NO'} | {s['residual_rate']:.2e} | "
            f"{u['max_abs_error']:.4f} | {u['rms_error']:.4f} | {v['max_abs_error']:.4f} | "
            f"{u['extremum']['min_computed'][1]:.5f} ({u['extremum']['min_reference'][1]:.5f}) | "
            f"{v['extremum']['max_computed'][1]:.5f} ({v['extremum']['max_reference'][1]:.5f}) | "
            f"({vs[0]:.3f},{vs[1]:.3f}) ({ref_vs[0]:.3f},{ref_vs[1]:.3f}) | "
            f"{c['max_divergence']:.1e} | {s['wall_seconds']:.0f} |")
    t1 = "\n".join(lines) + "\n"

    # ---- Table 2: key-quantity relative errors, with the 3 % verdict per quantity
    lines = [
        "### 3.2 Key quantities against the Ghia tables (relative error, per quantity)",
        "",
        "| Re | n | u_min rel | v_max rel | ω at (0.5,0.5) rel | vortex centre \\|Δ\\|/L | "
        "max\\|Δu\\| | <3 % (key quantities) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        u, v = c["u"], c["v"]
        u_rel = relative(u["extremum"]["min_computed"][1], u["extremum"]["min_reference"][1])
        v_rel = relative(v["extremum"]["max_computed"][1], v["extremum"]["max_reference"][1])
        vs = c["vortex_centre"]["computed"]
        ref_vs = VORTEX_CENTRE[c["reynolds"]]
        dvs = ((vs[0] - ref_vs[0]) ** 2 + (vs[1] - ref_vs[1]) ** 2) ** 0.5
        verdict = "yes" if (u_rel < 0.03 and v_rel < 0.03 and c["u"]["max_abs_error"] < 0.03) else "no"
        lines.append(f"| {c['reynolds']:g} | {c['n']} | {100 * u_rel:.2f} % | {100 * v_rel:.2f} % | "
                     f"n/a | {dvs:.3f} | {c['u']['max_abs_error']:.4f} | {verdict} |")
    t2 = "\n".join(lines) + "\n"

    # ---- Table 3: grid convergence of the two extrema
    conv = summary.get("convergence", {})
    lines = ["### 3.3 Grid convergence of the primary extrema (bisection sequence)", "",
             "| Re | quantity | series | order p | Richardson | finest | finest vs Richardson |",
             "|---|---|---|---|---|---|---|"]
    for rey in sorted(conv, key=float):
        for key, label in (("u_min", "u_min"), ("v_max", "v_max")):
            block = conv[rey][key]
            series = conv[rey]["u_min_series" if key == "u_min" else "v_max_series"]
            shown = ", ".join(f"n={n}: {q:.5f}" for n, q in series)
            if not block or block.get("degenerate"):
                lines.append(f"| {rey} | {label} | {shown} | — (needs 3 bisected grids) | — | — | — |")
            else:
                lines.append(f"| {rey} | {label} | {shown} | {block['order']:.2f} | "
                             f"{block['richardson']:.5f} | {block['finest']:.5f} | "
                             f"{100 * block['relative_vs_richardson']:.2f} % |")
    t3 = "\n".join(lines) + "\n"

    # ---- Table 4: cost model, steps to reach the steady-state gate
    lines = ["### 4.1 Step budget actually required (`max|Δu| ≤ "
             f"{meta['tolerance']:g}`, cfl = {meta['cfl']})", "",
             "| Re | n = 32 | n = 64 | n = 128 | steps / (n = 32) | steps ∝ Re^? |",
             "|---|---|---|---|---|---|"]
    by = {(c["reynolds"], c["n"]): c for c in cases}
    grids = sorted({c["n"] for c in cases})
    reys = sorted({c["reynolds"] for c in cases})
    for rey in reys:
        cells = []
        for n in (32, 64, 128):
            c = by.get((rey, n))
            cells.append(f"{c['solver']['steps']}" if c else "not run")
        base = by.get((rey, 32))
        ratio = "—"
        if base:
            fine = by.get((rey, 128)) or by.get((rey, 64))
            if fine:
                ratio = f"×{fine['solver']['steps'] / base['solver']['steps']:.1f} (n = {fine['n']})"
        lines.append(f"| {rey:g} | " + " | ".join(cells) + f" | {ratio} | |")
    t4 = "\n".join(lines) + "\n"
    return {"RESULTS": t1 + "\n" + t2 + "\n" + t3, "COST": t4}


def main() -> None:
    summary = load()
    blocks = tables(summary)
    with open(os.path.join(OUT, "results-tables.md"), "w") as handle:
        handle.write("# Machine-generated tables — Ghia ladder\n\n"
                     f"source: `{summary['_source']}`\n\n" +
                     "\n".join(blocks.values()))
    path = os.path.join(OUT, "report.md")
    text = open(path).read()
    for marker, body in blocks.items():
        token = f"<!--{marker}-->"
        if token in text:
            text = text.replace(token, body.strip())
    open(path, "w").write(text)
    print("injected from", summary["_source"], "->", path)


if __name__ == "__main__":
    main()