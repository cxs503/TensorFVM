#!/usr/bin/env python3
"""Render and score the Ghia Re = 400 v-centreline *proposal* (m3240 reissue).

What this script is for
-----------------------
The committed ladder bundle (``docs/verification-lid-cavity-ladder/summary.json``)
leaves one point of the Re = 400 v centreline failing: at ``x = 0.9063`` the error
grows under refinement (0.1146 / 0.1424 / 0.1493 at n = 32/64/128).  The
attribution audit (``scripts/audit_ghia_re400_v_attribution.py``, doc
``docs/GHIA_RE400_V_ATTRIBUTION.md``) proved the extraction and the solver are
innocent and registered the printed entry as a *dispute*.

That audit stops at "registered".  This script adds the one thing the register
cannot say by itself: **what the ladder would read if the entry were corrected**.
It scores every committed case twice -- against the transcription exactly as
printed, and against the same table with the proposal substituted (opt-in via
``tensorfvm.lid_cavity_reference.corrected_table``) -- and writes the comparison,
the figures and a report into ``docs/verification-ghia-re400-v-centreline/``.

The transcription itself is never edited here.  The proposal stays
``status="proposed", applied=False``; this artefact exists so a human adjudicating
the dispute against the printed Table II has both columns in front of them.

Usage::

    PYTHONPATH=src python scripts/render_ghia_re400_v_proposal.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorfvm.lid_cavity_reference import (  # noqa: E402
    DISPUTES, REFERENCE, corrected_table, proposed_corrections, table,
)

BUNDLE = os.path.join(REPO, "docs", "verification-lid-cavity-ladder", "summary.json")
DEFAULT_OUT = os.path.join(REPO, "docs", "verification-ghia-re400-v-centreline")
GATE = 0.03           # platform acceptance band on a pointwise centreline error
DISPUTED = (400.0, "v", 0.9063)


# --------------------------------------------------------------------- scoring
def _max_abs(profile: dict, reference: list[float]) -> float:
    """``max|computed - reference|`` of one committed centreline block."""
    return max(abs(c - r) for c, r in zip(profile["computed"], reference))


def rescore(bundle: dict) -> list[dict]:
    """Score every committed case against the printed and the proposed table.

    Pure function of the bundle plus the reference module: no solver run, so the
    report can be regenerated (and checked) anywhere the bundle exists.
    """
    rows = []
    for case in bundle["cases"]:
        rey, n = float(case["reynolds"]), int(case["n"])
        row: dict = {"reynolds": rey, "n": n}
        for which in ("u", "v"):
            block = case[which]
            printed = list(block["profiles"]["reference"])
            proposed = list(printed)
            for (pr, pw, px), record in DISPUTES.items():
                if pr == rey and pw == which:
                    index = [round(c, 4) for c in block["profiles"]["coord"]].index(round(px, 4))
                    proposed[index] = record["richardson_limit"]
            row[which] = {
                "printed_max_abs_error": _max_abs(block["profiles"], printed),
                "proposed_max_abs_error": _max_abs(block["profiles"], proposed),
                "printed_rms_error": block["rms_error"],
                "gate": GATE,
                "pass_printed": _max_abs(block["profiles"], printed) <= GATE,
                "pass_proposed": _max_abs(block["profiles"], proposed) <= GATE,
            }
        # the disputed point itself, at this resolution
        if (rey, "v") == DISPUTED[:2]:
            block = case["v"]
            index = [round(c, 4) for c in block["profiles"]["coord"]].index(round(DISPUTED[2], 4))
            row["disputed_point"] = {
                "x": DISPUTED[2],
                "computed": block["profiles"]["computed"][index],
                "published": block["profiles"]["reference"][index],
                "proposed": DISPUTES[DISPUTED]["richardson_limit"],
                "delta_printed": block["profiles"]["delta"][index],
            }
        row["max_divergence"] = case["max_divergence"]
        rows.append(row)
    return rows


def summarise(rows: list[dict], bundle: dict) -> dict:
    """Everything the report quotes, in one machine-readable dict."""
    by_key = {(r["reynolds"], r["n"]): r for r in rows}
    disputed_rows = [by_key[(400.0, n)] for n in (32, 64, 128) if (400.0, n) in by_key]
    record = DISPUTES[DISPUTED]
    return {
        "reference": REFERENCE,
        "bundle_reference_self_check_ok": bool(bundle["meta"]["reference_self_check"]["ok"]),
        "gate": GATE,
        "proposals": list(proposed_corrections()),
        "disputed": {
            "key": f"{DISPUTED[0]:g}/{DISPUTED[1]}/{DISPUTED[2]:.4f}",
            "reynolds": DISPUTED[0], "which": DISPUTED[1], "x": DISPUTED[2],
            "published": record["published"],
            # list of {n, value}: JSON round-trips integer dict keys as strings
            "computed": [{"n": n, "value": by_key[(400.0, n)]["disputed_point"]["computed"]}
                         for n in (32, 64, 128) if (400.0, n) in by_key],
            "richardson_limit": record["richardson_limit"],
            "observed_order": record["observed_order"],
            "limit_relative_error": record["limit_relative_error"],
            "interpolation_error_bound_n128": record["metric_interpolation_error_bound_n128"],
        },
        "cases": rows,
        "max_printed_error_overall": max(
            max(r["u"]["printed_max_abs_error"], r["v"]["printed_max_abs_error"]) for r in rows),
        "max_proposed_error_overall": max(
            max(r["u"]["proposed_max_abs_error"], r["v"]["proposed_max_abs_error"]) for r in rows),
        "failed_rows_printed": [f"{r['reynolds']:g}/n{r['n']}"
                                for r in rows
                                if not (r["u"]["pass_printed"] and r["v"]["pass_printed"])],
        "failed_rows_proposed": [f"{r['reynolds']:g}/n{r['n']}"
                                 for r in rows
                                 if not (r["u"]["pass_proposed"] and r["v"]["pass_proposed"])],
        "disputed_rows": [{"n": r["n"],
                           "printed": r["v"]["printed_max_abs_error"],
                           "proposed": r["v"]["proposed_max_abs_error"]}
                          for r in disputed_rows],
    }


# --------------------------------------------------------------------- figures
def render_figures(metrics: dict, out_dir: str, bundle: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    plt.rcParams.update({"figure.dpi": 110, "font.size": 8, "axes.grid": True,
                         "grid.alpha": 0.3, "lines.markersize": 3.5})
    cases = {(float(c["reynolds"]), int(c["n"])): c for c in bundle["cases"]}

    # (1) the disputed point: printed entry, solver sequence, proposal
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ref = [p for p in table(400.0, "v") if 0.75 <= p[0] <= 1.0]
    ax.plot([p[1] for p in ref], [p[0] for p in ref], "k+-", lw=1.0,
            label="Ghia (1982) Table II, as printed")
    prop = {p[0]: p[1] for p in corrected_table(400.0, "v")}
    ax.plot([prop[p[0]] for p in ref], [p[0] for p in ref], ls="none", marker="s",
            mfc="none", mec="tab:red", ms=6, label="same table with the proposal")
    for n, style in ((32, ":"), (64, "--"), (128, "-")):
        if (400.0, n) not in cases:
            continue
        block = cases[(400.0, n)]["v"]["profiles"]
        idx = [i for i, x in enumerate(block["coord"]) if 0.75 <= x <= 1.0]
        ax.plot([block["computed"][i] for i in idx], [block["coord"][i] for i in idx],
                style, lw=1.0, label=f"solver, n = {n}")
    ax.annotate("disputed entry\n-0.23827", xy=(-0.23827, 0.9063), xytext=(-0.33, 0.80),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=7)
    ax.annotate("Richardson limit\n-0.38989", xy=(prop[0.9063], 0.9063), xytext=(-0.60, 0.79),
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "tab:red"},
                fontsize=7, color="tab:red")
    ax.set_xlabel("v / U  (y = 0.5)")
    ax.set_ylabel("x / L")
    ax.set_title("Re = 400: the one disputed entry, and what stands there")
    ax.legend(fontsize=6, loc="lower left")
    path = os.path.join(out_dir, "re400_v_disputed_point.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(path)

    # (2) error vs resolution, as printed vs under the proposal
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, side, title in ((axes[0], "printed", "as printed (gate red at Re = 400)"),
                            (axes[1], "proposed", "proposal applied (opt-in)")):
        for rey, marker in ((100.0, "o"), (400.0, "s"), (1000.0, "^")):
            pts = sorted((r["n"], r["v"][f"{side}_max_abs_error"]) for r in metrics["cases"]
                         if r["reynolds"] == rey)
            if not pts:
                continue
            ax.semilogy([p[0] for p in pts], [max(p[1], 1e-6) for p in pts], marker + "-",
                        lw=1.0, label=f"Re = {rey:g}")
        ax.axhline(GATE, color="k", ls="--", lw=0.9)
        ax.text(33, GATE * 1.12, "3 % gate", fontsize=6)
        ax.set_xlabel("n (cavity cells per side)")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("max |v_computed - v_ref|  / U")
    axes[0].legend(fontsize=6)
    path = os.path.join(out_dir, "v_error_vs_resolution.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(path)

    # (3) all six centrelines at the finest committed resolution
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 6.4), sharex="col")
    finest = 128
    for row, rey in enumerate((100.0, 400.0, 1000.0)):
        ns = sorted(n for (r_, n) in cases if r_ == rey)
        n = finest if (rey, finest) in cases else ns[-1]
        case = cases[(rey, n)]
        for col, which in enumerate(("u", "v")):
            ax = axes[row][col]
            ref = table(rey, which)
            ax.plot([p[1] for p in ref], [p[0] for p in ref], "k+", ms=5,
                    label="Ghia (1982), as printed")
            block = case[which]["profiles"]
            ax.plot(block["computed"], block["coord"], "-", lw=1.0, label=f"solver n = {n}")
            for (pr, pw, px), record in DISPUTES.items():
                if pr == rey and pw == which:
                    ax.plot([record["richardson_limit"]], [px], "s", mfc="none", mec="tab:red",
                            label="proposal")
            if row == 0:
                ax.set_title(f"{which} centreline", fontsize=8)
            if col == 0:
                ax.set_ylabel("y / L" if which == "u" else "x / L")
            ax.legend(fontsize=5, loc="best")
        axes[row][0].annotate(f"Re = {rey:g}", xy=(0.03, 0.82), xycoords="axes fraction",
                              fontsize=8)
    axes[-1][0].set_xlabel("u / U  (x = 0.5)")
    axes[-1][1].set_xlabel("v / U  (y = 0.5)")
    path = os.path.join(out_dir, "all_centrelines_n128.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(path)
    return made


# ---------------------------------------------------------------------- report
_TEMPLATE = """# Ghia (1982) Re = 400 v-centreline: the proposal, scored both ways

Mission m3240 (reissued as m4004 after PR #82 was closed for being developed off the
current baseline). Status: **proposal registered, transcription untouched**.
Inputs: `docs/verification-lid-cavity-ladder/summary.json` (committed ladder),
`src/tensorfvm/lid_cavity_reference.py` ({reference}),
`docs/GHIA_RE400_V_ATTRIBUTION.md` (the attribution this builds on).
Artefacts: `scripts/render_ghia_re400_v_proposal.py`, `metrics.json` and the three
figures in this directory; gate in `tests/test_ghia_re400_v_proposal.py`.

## 1. Case

Square cavity `[0, L]^2`, no-slip on three walls, lid `u = U` at `y = L`, `L = U = 1`.
Staggered MAC finite volume (`tensorfvm.lid_cavity.LidCavitySolver`), Re = U L / nu in
{{100, 400, 1000}}, grids n in {{32, 64, 128}}, CFL {cfl}, steady-state tolerance
{tolerance:g} with a {max_steps} step cap. Reference: {reference}. Scoring interpolates
the converged mid-plane profile to the tabulated coordinates with a Catmull-Rom cubic.

## 2. Method

1. The attribution audit proved the extraction innocent: at the disputed point the
   solver value converges at clean second order ({order:.2f}) to a Richardson limit of
   {limit:.5f}, the interpolation error of the metric path at n = 128 is bounded by
   {interp:.1e} (two orders below the gap), and the printed entry is the only corner in
   all six tables.
2. This script therefore scores every committed ladder row **twice**, from the stored
   profiles: once against `table(Re, comp)` (as printed, {n_cases} rows) and once against
   `corrected_table(Re, comp)`, which substitutes the registered proposal
   ({key}: {published:.5f} -> {limit:.5f}) and nothing else.
3. The proposal is `status="proposed"`, `applied=False`. It is reachable only through the
   opt-in `corrected=True` path; no verification artefact, gate or committed number in
   this repository reads it.

## 3. Results

The disputed point, per resolution (`x = {x:.4f}`, `y = 0.5`):

| n | solver v | printed v | delta vs printed | proposal | delta vs proposal |
|---|----------|-----------|------------------|----------|-------------------|
{point_rows}

Re = 400 v-centreline error, the only row in the ladder that moves:

| n | max abs err, as printed | <3 %? | max abs err, proposal applied | <3 %? |
|---|------------------------|-------|-------------------------------|-------|
{disputed_rows}

All seven committed rows, both scorings (errors in units of U):

| Re | n | u err | v err (printed) | v err (proposal) | gate printed | gate proposal | max div |
|----|---|-------|-----------------|------------------|--------------|---------------|---------|
{all_rows}

Worst error over the whole ladder: **{worst_printed:.4f}** as printed,
**{worst_proposed:.4f}** with the proposal applied. Rows failing the {gate:.0%} pointwise
gate: printed **{failed_printed}**, proposal **{failed_proposed}**.

Figures: `re400_v_disputed_point.png` (the entry, the solver sequence and the proposal),
`v_error_vs_resolution.png` (error vs n, both scorings), `all_centrelines_n128.png`
(all six centrelines at the finest committed grid).

## 4. Notes and limits

* Nothing is corrected. `table(Re, comp)` returns the transcription byte for byte;
  `self_check()` now also proves that the proposal layer touches exactly the registered
  entry, moves no extremum, and applies nothing by default (`self_check()["proposals"]`).
* The proposal is the solver's Richardson limit, not an independent measurement. That is
  a hypothesis about a printed number, and it is labelled as one: adjudicating it needs
  the printed Table II or an independent DNS at `(x = {x:.4f}, y = 0.5)`.
* The proposal does not move the extrema, so the relative-error criteria of
  `ghia_acceptance` are unaffected; only the pointwise v criterion changes.
* `ghia_acceptance` still fails by default and the strict xfail in
  `tests/test_lid_cavity_re1000_evidence.py` stays red. A pass is available only through
  `allow_registered_disputes=True`, which reports every ignored point back.

## 5. Error assessment against the reference

| quantity | baseline | value | <3 %? |
|----------|----------|-------|-------|
{error_table}

Answer, stated plainly: **as printed, the Re = 400 v centreline is NOT inside 3 % at any
committed resolution ({vdisc_printed:.4f} worst, and the error grows with refinement), so no
amount of extra grid closes it.** With the proposal applied the same measurements are inside
3 % at every resolution that resolves the lid shear layer, n >= 64 (worst {vdisc_proposed:.4f});
the n = 32 row still misses on the proposal ({n32_proposed:.4f}) but that row already misses on
u as printed ({n32_u:.4f}), i.e. it is a resolution limit and not a reference question. Both
columns are printed side by side on purpose: the second one is conditional on a correction
nobody has approved, and quoting it alone is exactly the failure mode this artefact exists to
prevent.

## Reproduce

```bash
PYTHONPATH=src python scripts/render_ghia_re400_v_proposal.py     # regenerates this file
PYTHONPATH=src python -m pytest tests/test_ghia_re400_v_proposal.py -q
```
"""


def _error_table(metrics: dict) -> str:
    rows = []
    for case in metrics["cases"]:
        rey, n = case["reynolds"], case["n"]
        for which in ("u", "v"):
            printed = case[which]["printed_max_abs_error"]
            proposed = case[which]["proposed_max_abs_error"]
            if printed == proposed:
                verdict = "YES" if printed <= GATE else "NO"
                rows.append(f"| Re = {rey:g} {which} (n = {n}) | Ghia Table "
                            f"{'I' if which == 'u' else 'II'}, as printed | "
                            f"max abs err {printed:.4f} | {verdict} |")
            else:
                rows.append(f"| Re = {rey:g} {which} (n = {n}) | Ghia Table II, as printed | "
                            f"max abs err {printed:.4f} | "
                            f"{'YES' if printed <= GATE else 'NO'} |")
                rows.append(f"| Re = {rey:g} {which} (n = {n}) | Ghia Table II, proposal applied | "
                            f"max abs err {proposed:.4f} | "
                            f"{'YES' if proposed <= GATE else 'NO'} |")
        rows.append(f"| Re = {rey:g} divergence (n = {n}) | exact (div u = 0) | "
                    f"{case['max_divergence']:.1e} | YES |")
    return "\n".join(rows)


def write_report(metrics: dict, bundle: dict, out_dir: str) -> str:
    disputed = metrics["disputed"]
    point_rows = "\n".join(
        f"| {n} | {value:.5f} | {disputed['published']:.5f} | "
        f"{value - disputed['published']:+.5f} | {disputed['richardson_limit']:.5f} | "
        f"{value - disputed['richardson_limit']:+.5f} |"
        for n, value in ((item["n"], item["value"]) for item in disputed["computed"]))
    dis_rows = "\n".join(
        f"| {r['n']} | {r['printed']:.4f} | {'YES' if r['printed'] <= GATE else 'NO'} | "
        f"{r['proposed']:.4f} | {'YES' if r['proposed'] <= GATE else 'NO'} |"
        for r in metrics["disputed_rows"])
    all_rows = "\n".join(
        f"| {c['reynolds']:g} | {c['n']} | {c['u']['printed_max_abs_error']:.4f} | "
        f"{c['v']['printed_max_abs_error']:.4f} | {c['v']['proposed_max_abs_error']:.4f} | "
        f"{'pass' if c['u']['pass_printed'] and c['v']['pass_printed'] else 'FAIL'} | "
        f"{'pass' if c['u']['pass_proposed'] and c['v']['pass_proposed'] else 'FAIL'} | "
        f"{c['max_divergence']:.1e} |"
        for c in metrics["cases"])
    meta = bundle["meta"]
    text = _TEMPLATE.format(
        reference=metrics["reference"],
        cfl=meta["cfl"], tolerance=meta["tolerance"], max_steps=meta["max_steps"],
        order=disputed["observed_order"], limit=disputed["richardson_limit"],
        interp=disputed["interpolation_error_bound_n128"],
        n_cases=len(metrics["cases"]), key=disputed["key"],
        published=disputed["published"], x=DISPUTED[2],
        point_rows=point_rows, disputed_rows=dis_rows, all_rows=all_rows,
        worst_printed=metrics["max_printed_error_overall"],
        worst_proposed=metrics["max_proposed_error_overall"],
        gate=GATE,
        failed_printed=", ".join(metrics["failed_rows_printed"]) or "none",
        failed_proposed=", ".join(metrics["failed_rows_proposed"]) or "none",
        error_table=_error_table(metrics),
        vdisc_printed=max(r["printed"] for r in metrics["disputed_rows"]),
        vdisc_proposed=max(r["proposed"] for r in metrics["disputed_rows"] if r["n"] >= 64),
        n32_proposed=next((r["proposed"] for r in metrics["disputed_rows"] if r["n"] == 32),
                          float("nan")),
        n32_u=next(c["u"]["printed_max_abs_error"] for c in metrics["cases"]
                   if c["reynolds"] == 400.0 and c["n"] == 32),
    )
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as handle:
        handle.write(text)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=BUNDLE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    with open(args.bundle) as handle:
        bundle = json.load(handle)
    os.makedirs(args.out, exist_ok=True)
    rows = rescore(bundle)
    metrics = summarise(rows, bundle)
    with open(os.path.join(args.out, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=1)
    if not args.no_figures:
        render_figures(metrics, args.out, bundle)
    write_report(metrics, bundle, args.out)
    print(json.dumps({k: metrics[k] for k in
                      ("max_printed_error_overall", "max_proposed_error_overall",
                       "failed_rows_printed", "failed_rows_proposed", "disputed_rows")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())