#!/usr/bin/env python3
"""Two-phase VOF verification benchmark (P400W2-N01) — one-command reproduction.

Usage
-----
    python docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py --threads 4
    python docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py --quick          # smoke run
    python docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py --only-zalesak   # one leg

Outputs (default ``docs/P400W2-N01-two-phase-vof-verification/artifacts/``):

* ``twophase_verification.json`` — every number of every leg plus the gate
  verdicts; the report is rendered *from this file*, so no number is hand-copied;
* ``figures/fig1_zalesak_fields.png`` — initial vs one-revolution fields;
* ``figures/fig2_zalesak_metrics.png`` — area/perimeter error vs resolution;
* ``figures/fig3_dam_front.png`` — measured front vs the Ritter lines;
* ``figures/fig4_dam_mass.png`` — mass ledger;
* ``figures/fig5_dam_profile.png`` — depth profile at t = 0.3 s vs analytic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfvm.twophase_bench.campaign import DEFAULTS, evaluate_gates, run_campaign  # noqa: E402
from tensorfvm.twophase_bench.metrics2d import column_depth  # noqa: E402
from tensorfvm.twophase_bench.ritter import (  # noqa: E402
    DamBreakGeometry,
    ritter_depth,
    ritter_front,
    ritter_front_at_depth,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _figures(results: dict, figdir: str, geo: DamBreakGeometry) -> list:
    os.makedirs(figdir, exist_ok=True)
    made = []
    if results.get("zalesak"):
        made += _figures_zalesak(results, figdir)
    if results.get("dam_break"):
        made += _figures_dam(results, figdir, geo)
    return made


def _figures_zalesak(results: dict, figdir: str) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []

    # --- fig1: Zalesak fields -------------------------------------------------
    z = [r for r in results["zalesak"] if r["cells"] == max(
        r["cells"] for r in results["zalesak"])]
    z_sorted = sorted(z, key=lambda r: ("upwind", "hric", "cicsam", "plic").index(
        r["scheme"]) if r["scheme"] in ("upwind", "hric", "cicsam", "plic") else 9)
    fig, axes = plt.subplots(1, len(z_sorted) + 1, figsize=(3.4 * (len(z_sorted) + 1), 3.4),
                             squeeze=False)
    ax = axes[0][0]
    ax.imshow(z_sorted[0]["_exact"].numpy(), origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"exact (initial), N={z_sorted[0]['cells']}")
    for ax, row in zip(axes[0][1:], z_sorted):
        im = ax.imshow(row["_field"].numpy(), origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{row['scheme']}: area err {row['area_error_contour']:.2%}\n"
                     f"perim err {row['perimeter_error']:.2%}")
        ax.set_xticks([])
        ax.set_yticks([])
    for a in axes[0]:
        a.set_xticks([] if a is not axes[0][0] else a.get_xticks())
    fig.suptitle("Zalesak slotted disk after one rigid revolution", y=1.02)
    path = os.path.join(figdir, "fig1_zalesak_fields.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append(path)

    # --- fig2: Zalesak metrics vs resolution ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    schemes = sorted({r["scheme"] for r in results["zalesak"]},
                     key=lambda s: ("upwind", "hric", "cicsam", "plic").index(s)
                     if s in ("upwind", "hric", "cicsam", "plic") else 9)
    for scheme in schemes:
        rows = sorted([r for r in results["zalesak"] if r["scheme"] == scheme],
                      key=lambda r: r["cells"])
        n = [r["cells"] for r in rows]
        axes[0].loglog(n, [r["area_error_contour"] for r in rows], "o-", label=scheme)
        axes[1].loglog(n, [r["perimeter_error"] for r in rows], "o-", label=scheme)
    for ax, title, lim in ((axes[0], "contour-area error vs analytic", 0.01),
                           (axes[1], "perimeter error vs initial contour", 0.03)):
        ax.axhline(lim, color="k", ls="--", lw=0.8, label=f"criterion {lim:.0%}")
        ax.set_title(title)
        ax.set_xlabel("cells per direction")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Zalesak shape fidelity after one revolution (criterion C2 at N=128)")
    path = os.path.join(figdir, "fig2_zalesak_metrics.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append(path)

    return made


def _figures_dam(results: dict, figdir: str, geo: DamBreakGeometry) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    # --- fig3: dam-break front vs Ritter -------------------------------------
    legs = [l for l in results["dam_break"] if l["scheme"] in ("cicsam", "upwind")
            and l.get("samples")]
    if not legs:
        legs = [l for l in results["dam_break"] if l.get("samples")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    t = [s["t"] for s in legs[0]["samples"]]
    axes[0].plot(t, [ritter_front(x, geo) for x in t], "k-", lw=1.6,
                 label="Ritter tip $W+2c_0t$")
    t_dense = [0.02 + 0.002 * i for i in range(int((0.30 - 0.02) / 0.002) + 1)]
    axes[0].plot(t_dense, [ritter_front_at_depth(x, legs[0]["h_ref_m"], geo) for x in t_dense],
                 "k--", lw=1.1, label="Ritter at $h=h_{ref}$ (matched depth)")
    styles = {28: "o", 56: "s"}
    for leg in legs:
        axes[0].plot(t, [s["front_depth_m"] for s in leg["samples"]], styles.get(
            leg["cells_per_height"], "o") + "-", lw=1.0,
            label=f"{leg['scheme']} H/{leg['cells_per_height']} (at $h_{{ref}}$)")
        axes[1].plot(t, [100 * s["front_error_at_depth"] for s in leg["samples"]],
                     styles.get(leg["cells_per_height"], "o") + "-", lw=1.0,
                     label=f"{leg['scheme']} H/{leg['cells_per_height']}")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("$x_{front}$ [m]")
    axes[0].set_title("surge front vs Ritter (1892)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].axhline(10.0, color="k", ls="--", lw=0.8, label="criterion 10%")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("relative front error [%]")
    axes[1].set_title("front error against the matched-depth Ritter position")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    path = os.path.join(figdir, "fig3_dam_front.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append(path)

    # --- fig4: mass ledger ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for leg in results["dam_break"]:
        ax.semilogy(leg["mass_times"], [abs(d) + 1e-20 for d in leg["mass_relative_series"]],
                    lw=1.0, label=f"dam {leg['scheme']} H/{leg['cells_per_height']}")
    for row in results["zalesak"]:
        ax.semilogy(row["drift_series_time"],
                    [abs(d) + 1e-20 for d in row["drift_series_relative"]], ls=":", lw=0.9,
                    label=f"zalesak {row['scheme']} N={row['cells']}")
    ax.axhline(1e-3, color="k", ls="--", lw=0.8, label="criterion 1e-3 (0.1%)")
    ax.set_xlabel("t (non-dimensional for Zalesak)")
    ax.set_ylabel("|M(t)-M(0)|/M(0)")
    ax.set_title("global mass ledger (flux-form transport)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    path = os.path.join(figdir, "fig4_dam_mass.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append(path)

    # --- fig5: depth profile at t_end ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for leg in results["dam_break"]:
        if not leg.get("_profile"):
            continue
        if leg["cells_per_height"] != max(l["cells_per_height"] for l in results["dam_break"]):
            continue
        prof = leg["_profile"]
        x = prof["x"]
        axes[0].plot(x, prof["numeric"], lw=1.2, label=f"{leg['scheme']} (numeric)")
    prof0 = [l for l in results["dam_break"] if l.get("_profile")][0]
    x = prof0["_profile"]["x"]
    t_end = prof0["t_end"]
    axes[0].plot(x, ritter_depth(np.asarray(x) - geo.width, t_end, geo), "k--", lw=1.2,
                 label="Ritter analytic")
    axes[0].set_xlim(geo.x_left + 0.18, geo.x_right - 0.1)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("depth h [m]")
    axes[0].set_title(f"depth profile at t = {t_end:.2f} s")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    with_field = [l for l in results["dam_break"] if l.get("_field_depth")]
    field_leg = ([l for l in with_field if l["scheme"] == "cicsam"] or with_field)[-1]
    im = axes[1].imshow(np.asarray(field_leg["_field_depth"]).reshape(1, -1), aspect="auto",
                        extent=[geo.x_left, geo.x_right, 0, 1], cmap="Blues", vmin=0, vmax=1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("x [m]")
    axes[1].set_title("liquid (blue) at t = %.2f s — CICSAM H/%d" % (t_end, field_leg["cells_per_height"]))
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02, label="alpha (column-mean)")
    path = os.path.join(figdir, "fig5_dam_profile.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append(path)
    return made


def _strip(results: dict) -> dict:
    out = json.loads(json.dumps(results, default=_default))
    return out


def _merge(paths, out: str) -> int:
    os.makedirs(out, exist_ok=True)
    legs = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            legs.append(json.load(fh))
    zalesak, dam, figures = [], [], []
    seen = set()
    for payload in legs:
        for row in payload.get("zalesak", []):
            key = ("z", row["scheme"], row["cells"])
            if key not in seen:
                seen.add(key)
                zalesak.append(row)
        for row in payload.get("dam_break", []):
            key = ("d", row["scheme"], row["cells_per_height"])
            if key not in seen:
                seen.add(key)
                dam.append(row)
        figures += payload.get("figures", [])
    if not zalesak or not dam:
        _log(f"merge needs both legs: zalesak={len(zalesak)} dam={len(dam)}")
        return 2
    cfg = dict(DEFAULTS)
    cfg["declared_zalesak_cells"] = max(r["cells"] for r in zalesak)
    cfg["dam_cells_per_height"] = tuple(sorted({d["cells_per_height"] for d in dam}))
    cfg["zalesak_resolutions"] = tuple(sorted({r["cells"] for r in zalesak}))
    cfg["dam_schemes"] = tuple(sorted({d["scheme"] for d in dam}))
    hill = []
    audit = {}
    seen_hill = set()
    for payload in legs:
        for row in payload.get("hill_diagnostic", []):
            key = (row["scheme"], row["cells"])
            if key not in seen_hill:
                seen_hill.add(key)
                hill.append(row)
        cand = payload.get("limiter_audit") or {}
        finite = cand and all(v.get("max_alpha") == v.get("max_alpha")
                              for v in cand.values())
        if finite:
            audit = cand
        elif not audit:
            audit = cand
    results = {"config": legs[0].get("config", {}), "zalesak": zalesak,
               "dam_break": dam, "hill": hill, "limiter_audit": audit}
    gates = evaluate_gates(results, cfg)
    payload = {
        "node": "P400W2-N01",
        "title": "FVM two-phase VOF verification face: dam-break wave front, "
                 "Zalesak slotted disk, mass ledger",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": legs[0].get("environment", {}),
        "config": legs[0].get("config", {}),
        "gate_config": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
        "gates": gates,
        "zalesak": zalesak,
        "dam_break": dam,
        "hill_diagnostic": hill,
        "limiter_audit": audit,
        "figures": sorted(set(figures)),
        "legs": [os.path.basename(os.path.dirname(os.path.abspath(p))) or p
                 for p in paths],
    }
    # collect the figures of every leg into one directory so the report has a
    # single stable artefact location (the paths inside the JSON are rewritten)
    import shutil

    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)
    collected = []
    seen = {}
    for rel in sorted(set(figures)):
        src = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.exists(src):
            _log(f"figure missing: {src}")
            continue
        base = os.path.basename(src)
        stem, ext = os.path.splitext(base)
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:  # several legs drew the same panel: keep both, tagged
            base = f"{stem}-leg{seen[base]}{ext}"
        dst = os.path.join(figdir, base)
        shutil.copyfile(src, dst)
        collected.append(os.path.relpath(dst, ROOT))
    # keep panels already present in the output directory: the leg directories
    # are intermediates and may have been cleaned up after an earlier merge
    import glob

    for path in sorted(glob.glob(os.path.join(figdir, "*.png"))):
        rel = os.path.relpath(path, ROOT)
        if rel not in collected:
            collected.append(rel)
    payload["figures"] = sorted(collected)
    path = os.path.join(out, "twophase_verification.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    _log(f"merged {len(paths)} legs -> {path} ({len(collected)} figures)")
    for gate, value in gates.items():
        if gate == "all_passed":
            continue
        _log(f"{gate}: {'PASS' if value['passed'] else 'FAIL'} — {value['criterion']}")
    _log(f"all passed: {gates['all_passed']}")
    return 0 if gates["all_passed"] else 1


def _default(obj):
    if isinstance(obj, torch.Tensor):
        return None
    raise TypeError(type(obj))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--zalesak-res", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--zalesak-schemes", nargs="+", default=["upwind", "cicsam", "plic"])
    ap.add_argument("--dam-cells", type=int, nargs="+", default=[28, 56])
    ap.add_argument("--dam-schemes", nargs="+", default=["upwind", "cicsam", "plic"])
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--only-zalesak", action="store_true")
    ap.add_argument("--only-dam", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (not the gate)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="combine leg JSONs produced by separate runs, re-evaluate "
                         "the gates and write the final artefacts (no new compute)")
    args = ap.parse_args(argv)

    if args.merge:
        return _merge(args.merge, args.out)

    if args.quick:
        args.zalesak_res = [32]
        args.zalesak_schemes = ["upwind", "cicsam"]
        args.dam_cells = [14]
        args.dam_schemes = ["upwind", "cicsam"]
    if args.only_zalesak:
        args.dam_cells = []
    if args.only_dam:
        args.zalesak_res = []

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    geo = DamBreakGeometry(upstream=0.26, downstream=0.60, tank_height=0.07)
    t0 = time.time()
    _log(f"campaign start: zalesak={args.zalesak_res} x {args.zalesak_schemes}; "
         f"dam={args.dam_cells} x {args.dam_schemes}; cfl={args.cfl}; "
         f"threads={args.threads}")

    results = run_campaign(
        zalesak_resolutions=tuple(args.zalesak_res) if args.zalesak_res else (),
        zalesak_schemes=tuple(args.zalesak_schemes),
        dam_cells_per_height=tuple(args.dam_cells) if args.dam_cells else (),
        dam_schemes=tuple(args.dam_schemes),
        cfl=args.cfl,
        threads=args.threads,
        progress=_log,
        keep_fields=not args.no_figures,
    )
    cfg = dict(DEFAULTS)
    if args.zalesak_res:
        cfg["declared_zalesak_cells"] = max(args.zalesak_res)
    if args.dam_cells:
        cfg["dam_cells_per_height"] = tuple(args.dam_cells)
    gates = evaluate_gates(results, cfg)

    # figures need the fields: recompute the two display cases lazily is avoided
    # by keeping the fields in the result tree until the figures are drawn.
    made = [] if args.no_figures else _figures(results, os.path.join(args.out, "figures"), geo)

    payload = {
        "node": "P400W2-N01",
        "title": "FVM two-phase VOF verification face: dam-break wave front, "
                 "Zalesak slotted disk, mass ledger",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": time.time() - t0,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "threads": args.threads,
            "platform": platform.platform(),
        },
        "config": results["config"],
        "gate_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "gates": gates,
        "dam_break": [{k: v for k, v in leg.items() if not k.startswith("_")}
                      for leg in results["dam_break"]],
        "dam_break_profiles": [{k: v for k, v in leg.items() if k.startswith("_profile")}
                               for leg in results["dam_break"]],
        "zalesak": [{k: v for k, v in row.items() if not k.startswith("_")}
                    for row in results["zalesak"]],
        "hill_diagnostic": results.get("hill", []),
        "limiter_audit": results.get("limiter_audit", {}),
        "figures": [os.path.relpath(p, ROOT) for p in made],
    }
    path = os.path.join(args.out, "twophase_verification.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    _log(f"wrote {path} ({os.path.getsize(path)} B), wall {payload['wall_seconds']:.1f} s")
    for gate, value in gates.items():
        if gate == "all_passed":
            continue
        _log(f"{gate}: {'PASS' if value['passed'] else 'FAIL'} — {value['criterion']}")
    _log(f"all passed: {gates['all_passed']}")
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())