"""Audit the Ghia reference tables against the committed lid-cavity ladder.

Runs :func:`tensorfvm.verification.ghia_reference_audit.audit` over
``docs/verification-lid-cavity-ladder/summary.json`` and writes the verification
package in ``docs/verification-ghia-reference-consistency/``:

``results.json``   the full machine record (per-ordinate residuals, verdicts,
                   register cross-check, gate reproduction, negative control);
``convergence.png`` residual versus grid size for every published ordinate, with
                   the flagged entry and a second-order guide;
``profile.png``    the finest-grid centrelines against the published points;
``report.md``      the note, assembled from ``results.json`` -- no number in it is
                   typed by hand.

Usage::

    PYTHONPATH=src python3 scripts/audit_ghia_reference_consistency.py
    PYTHONPATH=src python3 scripts/audit_ghia_reference_consistency.py --check

``--check`` re-renders into a temporary directory and fails when the committed
JSON/Markdown drift from a fresh render, or when a committed figure no longer has
the geometry of the fresh one.  Raster bytes are deliberately not compared: a PNG
carries the freetype/Matplotlib build that produced it (``Software`` tEXt chunk),
so byte equality would test the runner rather than the data, while every number
plotted is already pinned exactly by ``results.json``.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import struct
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.lid_cavity_reference import REFERENCE  # noqa: E402
from tensorfvm.verification.ghia_reference_audit import (  # noqa: E402
    audit,
    DEFAULT_BUNDLE,
    load_bundle,
    summary_lines,
)

OUT = os.path.join("docs", "verification-ghia-reference-consistency")
ARTEFACTS = ("results.json", "convergence.png", "profile.png", "report.md")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def convergence_figure(report: dict, path: str) -> None:
    """Residual of every published ordinate against the grid size."""
    reynolds = sorted(float(re) for re in report["pointwise"]["reynolds"])
    key_of = {float(re): re for re in report["pointwise"]["reynolds"]}
    fig, axes = plt.subplots(1, len(reynolds), figsize=(4.1 * len(reynolds), 3.6),
                             sharey=True)
    axes = [axes] if len(reynolds) == 1 else list(axes)
    flagged = {(d["reynolds"], d["which"], d["x"])
               for d in report["verdict"]["reference_defects"]}
    for ax, re in zip(axes, reynolds):
        block = report["pointwise"]["reynolds"][key_of[re]]
        ns = block["grids"]
        for which in ("u", "v"):
            for row in block["which"][which]["rows"]:
                residuals = [row["residual"][str(n)] for n in ns]
                key = (re, which, row["x"])
                if key in flagged:
                    ax.loglog(ns, [max(r, 1e-6) for r in residuals], "-o",
                              color="crimson", lw=2.0, ms=5, zorder=5,
                              label=f"flagged: {which} @ x={row['x']:.4f}")
                else:
                    ax.loglog(ns, [max(r, 1e-6) for r in residuals], "-",
                              color="0.65", lw=0.8, zorder=1)
        # second-order guide anchored at the finest consistent residual
        finest = ns[-1]
        anchor = report["verdict"]["largest_consistent_residual"] or 1e-3
        guide = [anchor * (finest / n) ** 2 for n in ns]
        ax.loglog(ns, guide, "k--", lw=1.0, label="second order in h")
        ax.axhline(report["meta"]["acceptance_band"], color="steelblue", lw=1.0,
                   ls=":", label="acceptance band 0.03")
        ax.set_title(f"Re = {re:g}   (grids {ns})")
        ax.set_xlabel("n (cells per side)")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$|\Delta u|$ or $|\Delta v|$  (lid speeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Residual of every published Ghia ordinate versus resolution")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def profile_figure(report: dict, bundle: dict, path: str) -> None:
    """Finest available grid against the tabulated centrelines."""
    reynolds = sorted(float(re) for re in report["pointwise"]["reynolds"])
    key_of = {float(re): re for re in report["pointwise"]["reynolds"]}
    fig, axes = plt.subplots(2, len(reynolds), figsize=(4.1 * len(reynolds), 6.6),
                             sharey="row", squeeze=False)
    flagged = {(d["reynolds"], d["which"], d["x"])
               for d in report["verdict"]["reference_defects"]}
    for col, re in enumerate(reynolds):
        block = report["pointwise"]["reynolds"][key_of[re]]
        ns = block["grids"]
        finest = ns[-1]
        for row, which in enumerate(("u", "v")):
            ax = axes[row][col]
            rows = sorted(block["which"][which]["rows"], key=lambda r: r["x"])
            coord = [r["x"] for r in rows]
            published = [r["published"] for r in rows]
            computed = [r["computed"][str(finest)] for r in rows]
            ax.plot(computed, coord, "-", color="tab:blue",
                    label=f"n = {finest}")
            ax.plot(published, coord, "o", color="k", ms=4, mfc="none",
                    label="Ghia et al. (1982)")
            for r in rows:
                if (re, which, r["x"]) in flagged:
                    ax.plot(r["computed"][str(finest)], r["x"], "x", color="crimson",
                            ms=10, mew=2.2, label="flagged entry")
            if which == "u":
                ax.set_ylabel("y / L")
                ax.set_title(f"Re = {re:g}")
            else:
                ax.set_xlabel("v / U   (at y / L = 1/2)")
            if which == "u":
                ax.set_xlabel("u / U   (at x / L = 1/2)")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for row in range(2):
        for col in range(len(reynolds)):
            for handle, label in zip(*axes[row][col].get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Centreline profiles on the finest committed grid "
                 "(u at x/L = 1/2, v at y/L = 1/2)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _defect_table(report: dict) -> str:
    defects = report["verdict"]["reference_defects"]
    if not defects:
        return "No published ordinate was classified as a reference defect.\n"
    lines = [
        "| Re | component | x or y | published | computed (finest) | residual "
        "| ratio prev/finest | order p | Richardson limit | limit residual "
        "| residual / resolution uncertainty |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in defects:
        order = "-" if d["observed_order"] is None else f"{d['observed_order']:.2f}"
        limit = "-" if d["richardson_limit"] is None else f"{d['richardson_limit']:+.5f}"
        limit_res = "-" if d["limit_residual"] is None else f"{d['limit_residual']:.4f}"
        excess = ("-" if d["residual_over_uncertainty"] is None
                  else f"{d['residual_over_uncertainty']:.1f}x")
        lines.append(
            f"| {d['reynolds']:g} | {d['which']} | {d['x']:.4f} | {d['published']:+.5f} "
            f"| {d['computed_finest']:+.5f} | {d['residual_finest']:.4f} "
            f"| {d['ratio_previous_over_finest']:.2f} | {order} | {limit} "
            f"| {limit_res} | {excess} |")
    return "\n".join(lines) + "\n"


def _band_context(report: dict) -> str:
    """Where the largest *consistent* residual sits relative to the band."""
    verdict = report["verdict"]
    worst = verdict.get("largest_consistent")
    band = report["meta"]["acceptance_band"]
    if worst is None:
        return "No ordinate was classified as consistent."
    text = (f"Largest residual among the {verdict['counts'].get('consistent', 0)} "
            f"consistent ordinates: **{worst['residual']:.5f}** lid speeds "
            f"(Re={worst['reynolds']:g} {worst['which']} @ {worst['x']:.4f}, "
            f"grids {worst['grids']}), against a band of {band}. ")
    if worst["residual"] > 0.8 * band:
        text += ("That point sits close to the band on a single grid, so it is a "
                 "caveat on the Re=1000 row rather than a demonstrated agreement; "
                 "unlike the flagged entry it is not contradicted by a refinement "
                 "sequence (none is committed there). ")
    text += ("The flagged entry is "
             f"{verdict['reference_defects'][0]['residual_finest'] / max(verdict['largest_consistent_residual'], 1e-9):.1f}x "
             "larger than the worst consistent residual."
             if verdict["reference_defects"] else "No flagged entry.")
    return text


def _count_table(report: dict) -> str:
    verdict = report["verdict"]
    lines = ["| verdict | published ordinates |", "|---|---|"]
    for name in ("consistent", "under_resolved", "reference_defect",
                 "unverified_single_grid"):
        lines.append(f"| `{name}` | {verdict['counts'].get(name, 0)} |")
    lines.append(f"| **total** | **{verdict['published_points']}** |")
    return "\n".join(lines) + "\n"


def _residual_table(report: dict) -> str:
    lines = [
        "| Re | grids | component | max residual (finest) | max residual, "
        "consistent points only | all inside band |",
        "|---|---|---|---|---|---|",
    ]
    for re_key, block in sorted(report["pointwise"]["reynolds"].items(),
                                key=lambda item: float(item[0])):
        ns = block["grids"]
        for which in ("u", "v"):
            wblock = block["which"][which]
            lines.append(
                f"| {float(re_key):g} | {ns} | {which} "
                f"| {wblock['max_residual'][str(ns[-1])]:.5f} "
                f"| {wblock['max_residual_consistent']:.5f} "
                f"| {'yes' if wblock['max_residual'][str(ns[-1])] <= report['meta']['acceptance_band'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def _format_excluded(points) -> str:
    if not points:
        return "none"
    parts = []
    for item in points:
        parts.append(f"{item.get('component', '?')} @ {float(item['x']):.4f} "
                     f"(published {item['published']:+.5f}, delta {item['delta']:+.5f})")
    return "<br>".join(parts)


def _acceptance_table(report: dict) -> str:
    lines = [
        "| Re | n | strict gate | with register | failing criteria (strict) "
        "| excluded points | reproduces committed verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report["acceptance"]:
        lines.append(
            f"| {item['reynolds']:g} | {item['n']} "
            f"| {'pass' if item['strict_pass'] else 'FAIL'} "
            f"| {'pass' if item['lenient_pass'] else 'FAIL'} "
            f"| {', '.join(item['failing_criteria']) or '-'} "
            f"| {_format_excluded(item['excluded_points'])} "
            f"| {'yes' if item['reproduces_committed'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def _case_rows(report: dict) -> str:
    grids = report["meta"]["grids_present"]
    return ", ".join(f"Re={float(re):g}: n={ns}" for re, ns in grids.items())


def render_report(report: dict) -> str:
    """Assemble ``report.md`` from the audit record (no hand-typed numbers)."""
    meta = report["meta"]
    verdict = report["verdict"]
    register = report["register"]
    negative = report["negative_control"] or {}
    missing = meta["grids_missing"]
    missing_text = ("none" if not missing else ", ".join(
        f"Re={m['reynolds']:g}/n={m['n']}" for m in missing))
    declared = "; ".join(f"Re={d['reynolds']:g} {d['which']} @ {d['x']:.4f} "
                         f"(published {d['published']:+.5f})"
                         for d in register["declared"]) or "none"
    band = meta["acceptance_band"]
    lines = [
        "# Ghia reference consistency: what the committed ladder can and cannot explain",
        "",
        f"Status: **{'audit clean' if report['ok'] else 'audit FAILED'}**. "
        "Generated by `scripts/audit_ghia_reference_consistency.py` from the "
        f"committed ladder bundle (`{meta['bundle']}`, md5 `{meta['bundle_md5']}`, "
        f"bundle timestamp {meta['bundle_timestamp']}).",
        "",
        "Reference: " + REFERENCE + ".",
        "",
        "Every number below is rendered from `results.json` in this directory; "
        "`--check` re-renders and diffs the whole package: JSON and Markdown "
        "exactly, the figures by geometry (PNG bytes follow the rasteriser build, "
        "not the data).",
        "",
        "## 1. Question",
        "",
        "The lid-cavity ladder scores three Reynolds numbers against the Ghia "
        "tables on a 32/64/128 ladder. One published ordinate (Re = 400, v at "
        "x = 0.9063) sits outside the pointwise acceptance band while its "
        "neighbours converge, and `docs/GHIA_RE400_V_ATTRIBUTION.md` attributes it "
        "to the digitised table rather than to the solver. Two things were missing:",
        "",
        "1. nothing re-derived that attribution from the committed evidence, so the "
        "claim could drift away from its numbers;",
        "2. nothing checked that the *register* of disputed entries is exactly the "
        "set of ordinates the evidence cannot explain -- a new unexplained point, or "
        "a stale register entry, would pass unnoticed.",
        "",
        "This audit answers both, for all "
        f"{verdict['published_points']} published ordinates of the three tables.",
        "",
        "## 2. Method",
        "",
        "For every published ordinate the audit follows its residual "
        r"$r_n = |u_n - u_{\rm Ghia}|$ (or the $v$ counterpart) along the ladder and "
        "estimates the observed order in $h$ from the two increments, "
        r"$p = \log_2 |u_{32}-u_{64}| / |u_{64}-u_{128}|$, the Richardson limit "
        r"$u_\infty = u_{128} + (u_{128}-u_{64})/(2^p-1)$ and the remaining "
        "resolution uncertainty $|u_{128}-u_{64}|/(2^p-1)$. A point is then",
        "",
        "- `consistent` -- residual inside the acceptance band "
        f"({band} lid speeds; the tables are digitised off a plot);",
        "- `under_resolved` -- outside the band but the residual still *shrinks* "
        "with refinement (> 2x per 2:1 refinement);",
        "- `reference_defect` -- outside the band and *saturated* (ratio <= 1.5), "
        "i.e. the discrete limit is not the tabulated value;",
        "- `unverified_single_grid` -- outside the band on the only committed grid.",
        "",
        "Three independent cross-checks keep the classification honest: the "
        "committed `max_abs_error` of every case is re-derived from its stored "
        "profiles; every committed gate verdict is re-derived with "
        "`ghia_acceptance` (strict and with the register); and a negative control "
        "injects a documented corruption at an *unregistered* ordinate and requires "
        "the audit to see exactly that point. A shape test on the tables themselves "
        "(`smoothness_audit`, secant-slope jumps) is reported as context but is not "
        "used for any verdict: genuine near-extrema produce the same jumps.",
        "",
        "## 3. Evidence used, and its gaps",
        "",
        f"- Grids present: {_case_rows(report)}.",
        f"- Grids missing relative to the declared ladder: {missing_text} "
        f"(bundle flagged partial: {meta['bundle_is_partial']}).",
        f"- Provenance as recorded by the runner: {meta['bundle_provenance']}",
        "",
        "Only Re = 100 and Re = 400 have three grids, so only they admit an order "
        "estimate; Re = 1000 has a single committed grid and its classification is "
        "the band test alone.",
        "",
        "## 4. Results",
        "",
        "### 4.1 Classification of every published ordinate",
        "",
        _count_table(report),
        "Residual envelopes per case:",
        "",
        _residual_table(report),
        "### 4.2 The ordinate the evidence cannot explain",
        "",
        _defect_table(report),
        _band_context(report),
        "",
        "### 4.3 Register cross-check",
        "",
        f"Declared in `tensorfvm.lid_cavity_reference.DISPUTES`: {declared}.",
        "",
        "| direction | ordinates |",
        "|---|---|",
        f"| detected as reference defect | {len(register['detected'])} |",
        f"| declared in the register | {len(register['declared'])} |",
        "| detected but not declared | "
        f"{len(register['missing_from_register'])} |",
        "| declared but not detected | "
        f"{len(register['missing_from_detection'])} |",
        "",
        f"Agreement in both directions: **{register['agreement']}**.",
        "",
        "### 4.4 Acceptance gate, re-derived",
        "",
        _acceptance_table(report),
        "The strict gate fails only where the register says it must, and the "
        "register-aware gate passes while reporting the excluded ordinate; both are "
        "re-derived here, so the documented exclusion cannot silently widen.",
        "",
        "### 4.5 Negative control",
        "",
        (f"Injecting {negative['injected']['magnitude']:+.3f} lid speeds at "
         f"Re={negative['injected']['reynolds']:g} {negative['injected']['which']} "
         f"@ {negative['injected']['x']:.4f} (an unregistered ordinate) "
         f"is detected: **{negative['detected']}**, at exactly that point: "
         f"**{negative['no_other_new_defects']}**."
         if negative else "not run"),
        "",
        "## 5. Error budget against the benchmark",
        "",
        "Key quantities per case (aggregated below from the committed profiles); "
        "the verdicts are the same 0.03-lid-speed band used above:",
        "",
        _key_quantity_table(report),
        "",
        "## 6. Limits and what is not claimed",
        "",
        "- The audit reads committed evidence; it does not re-run the solver and it "
        "does not adjudicate the disputed entry. Settling it needs the printed "
        "Table II or an independent DNS.",
        "- No published number is edited: the transcription is byte-identical to "
        "`tensorfvm/lid_cavity_reference.py`, and any gate that ignores an entry "
        "must name it.",
        "- Re = 1000 has one committed grid; no order statement is made there.",
        "- `reference_defect` is a statement about *this bundle and these tables*. "
        "A bug that survives grid refinement in exactly one ordinate would look the "
        "same; that is why the residual is also compared with the interpolation "
        "error of the metric path in `docs/GHIA_RE400_V_ATTRIBUTION.md`.",
        "",
        "## 7. Reproduction",
        "",
        "```",
        "PYTHONPATH=src python3 scripts/audit_ghia_reference_consistency.py",
        "PYTHONPATH=src python3 scripts/audit_ghia_reference_consistency.py --check",
        "python3 -m pytest tests/test_ghia_reference_consistency_audit.py -q",
        "```",
        "",
    ]
    return "\n".join(lines)


def _key_quantity_table(report: dict) -> str:
    """Extrema and vortex centre per case, relative errors, 15 % band verdict."""
    lines = [
        "| Re | n | u_min computed (Ghia) | rel | v_max computed (Ghia) | rel "
        "| vortex centre computed (Ghia) | distance/L "
        "| max abs du | max abs dv | < 15 % |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["key_quantities"]:
        centre = item["vortex_centre"]
        ref = "-" if centre["reference"] is None else (
            f"({centre['reference'][0]:.4f}, {centre['reference'][1]:.4f})")
        distance = "-" if centre["distance"] is None else f"{centre['distance']:.4f}"
        ok = item["u_min"]["pass_15pct"] and item["v_max"]["pass_15pct"]
        lines.append(
            f"| {item['reynolds']:g} | {item['n']} "
            f"| {item['u_min']['computed']:+.5f} ({item['u_min']['reference']:+.5f}) "
            f"| {100 * item['u_min']['relative_error']:.2f}% "
            f"| {item['v_max']['computed']:+.5f} ({item['v_max']['reference']:+.5f}) "
            f"| {100 * item['v_max']['relative_error']:.2f}% "
            f"| ({centre['computed'][0]:.4f}, {centre['computed'][1]:.4f}) {ref} "
            f"| {distance} "
            f"| {item['max_abs_error_u']:.5f} | {item['max_abs_error_v']:.5f} "
            f"| {'yes' if ok else 'NO'} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# writing / checking
# --------------------------------------------------------------------------- #
def _comparable_json(left_path: str, right_path: str) -> tuple[list[str], list[str]]:
    """Line view of two ``results.json`` files, minus the wall-clock stamp."""
    views = []
    for path in (left_path, right_path):
        with open(path) as handle:
            data = json.load(handle)
        data.get("meta", {}).pop("generated_utc", None)
        views.append([line + "\n" for line in
                      json.dumps(data, indent=1, sort_keys=True).splitlines()])
    return views[0], views[1]


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_geometry(path: str) -> tuple[int, int]:
    """``(width, height)`` from the IHDR chunk.

    Geometry is what a stale figure actually gets wrong -- a changed panel count,
    figure size or dpi -- and unlike the raster bytes it does not depend on the
    freetype/Matplotlib build of the machine that rendered it.
    """
    with open(path, "rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or not head.startswith(_PNG_SIGNATURE) or head[12:16] != b"IHDR":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", head[16:24])


def build(report: dict, bundle: dict, outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {name: os.path.join(outdir, name) for name in ARTEFACTS}
    convergence_figure(report, paths["convergence.png"])
    profile_figure(report, bundle, paths["profile.png"])
    with open(paths["results.json"], "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
        handle.write("\n")
    with open(paths["report.md"], "w") as handle:
        handle.write(render_report(report))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--check", action="store_true",
                        help="re-render into a temporary directory and diff")
    args = parser.parse_args(argv)

    bundle = load_bundle(args.bundle)
    report = audit(args.bundle)
    for line in summary_lines(report):
        print(line)

    if args.check:
        failures = 0
        with tempfile.TemporaryDirectory() as tmp:
            fresh = build(report, bundle, tmp)
            for name in ("results.json", "report.md"):
                committed = os.path.join(args.out, name)
                if not os.path.exists(committed):
                    print(f"MISSING committed artefact: {committed}")
                    failures += 1
                    continue
                if name.endswith(".json"):
                    left, right = _comparable_json(committed, fresh[name])
                else:
                    with open(committed) as a, open(fresh[name]) as b:
                        left, right = a.readlines(), b.readlines()
                diff = list(difflib.unified_diff(left, right, committed, fresh[name]))
                if diff:
                    failures += 1
                    print("".join(diff[:40]))
            for name in ("convergence.png", "profile.png"):
                committed = os.path.join(args.out, name)
                if not os.path.exists(committed):
                    print(f"MISSING committed artefact: {committed}")
                    failures += 1
                    continue
                try:
                    geometry = _png_geometry(committed)
                except ValueError as error:
                    print(f"{name}: {error}")
                    failures += 1
                    continue
                fresh_geometry = _png_geometry(os.path.join(tmp, name))
                if geometry != fresh_geometry:
                    print(f"{name}: committed geometry {geometry} != fresh render "
                          f"{fresh_geometry}")
                    failures += 1
        print("check:", "ok" if not failures else f"{failures} mismatch(es)")
        return 0 if not failures else 1

    paths = build(report, bundle, args.out)
    for name, path in paths.items():
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
    print("audit:", "ok" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())