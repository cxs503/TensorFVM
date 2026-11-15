#!/usr/bin/env python3
"""High-order spectral ladder (N = 48 ... 128) for the lid-driven cavity, Re = 1000.

What this closes
----------------
``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL.md`` §4 registers one open item: the
dense-LU Newton path could not go past ``N = 64`` (the Jacobian is ``(2m)**2``
doubles, i.e. ``O(N**4)`` memory), so the spectral limit had to be extrapolated
from **two** orders with a decay ratio borrowed from the wall-slip defect.  A
two-point extrapolation cannot detect its own failure mode.

The sparse-Jacobian Krylov path added in :mod:`tensorfvm.cavity_spectral`
(``linear_solver="krylov"``) makes ``N >= 96`` affordable.  With four orders the
successive-difference decay ratio becomes *observable*, which turns the
extrapolation from an assumption into a measurement, and lets the two-order
statement be tested instead of trusted.

Everything printed here comes from the JSON artefacts under
``docs/CASE-FVM-CAVITY-RE1000/runs/spectral/``; nothing is transcribed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm import lid_cavity_reference as ref

CASE = Path("docs/CASE-FVM-CAVITY-RE1000")
SPEC = CASE / "runs" / "spectral"
FIGS = CASE / "figures"
GHIA = {1000.0: {"u_min": -0.38289, "v_max": 0.37095,
                 "vortex": (0.5313, 0.5625)}}


def load_ladder(reynolds: float, tags: tuple[str, ...]) -> list[dict]:
    """Merge every stored record for ``reynolds``; later tags win per order."""
    tag_suffix = "re1000" if abs(reynolds - 1000.0) < 1e-9 else f"re{int(reynolds)}"
    by_order: dict[int, dict] = {}
    for tag in tags:
        for f in sorted(SPEC.glob(f"cavity_spectral_N*_{tag}.json")):
            rec = json.loads(f.read_text())
            if abs(rec.get("reynolds", -1) - reynolds) > 1e-9:
                continue
            if not rec.get("metrics"):
                continue
            rec["_file"] = f.name
            by_order[int(rec["order"])] = rec
    return [by_order[n] for n in sorted(by_order)]


def geometric_limit(orders: list[int], values: list[float]) -> dict:
    """Successive-difference decay of a spectrally converging sequence.

    For a geometrically converging sequence the differences between successive
    orders decay with a constant ratio ``r``, so the remaining error of the
    finest level is ``d_last / (r - 1)``.  ``r`` is now *measured* on
    consecutive pairs; agreement between the measured ratios is the check that
    the geometric model actually holds (a two-point extrapolation cannot do
    this).
    """
    d = np.abs(np.diff(values))
    ratios = [float(d[i] / d[i + 1]) for i in range(len(d) - 1) if d[i + 1] > 0]
    # only the finest pairs carry the asymptotic ratio; the coarse ones are
    # reported (they show whether the geometric model holds) but not used
    tail = ratios[-2:] if len(ratios) >= 2 else ratios
    r = float(np.mean(tail)) if tail else float("nan")
    increment = float(d[-1] / (r - 1.0)) if r and r > 1.0 else float("finite")
    limit = float(values[-1] + increment)
    return {
        "orders": orders,
        "values": [float(v) for v in values],
        "successive_differences": [float(x) for x in d],
        "decay_ratios": ratios,
        "decay_ratio_used": r,
        "extrapolation_increment": increment,
        "limit": limit,
        "finest_value": float(values[-1]),
        "finest_delta": float(d[-1]),
        "uncertainty": abs(increment),
    }


def two_order_limit(orders: list[int], values: list[float], ratio: float) -> dict:
    """The *previous* two-point extrapolation, reproduced for comparison."""
    lim = values[-1] + (values[-1] - values[-2]) / (ratio - 1.0)
    return {"orders": orders[-2:], "decay_ratio": ratio, "limit": float(lim),
            "delta_vs_four_order": None}


def fvm_richardson() -> dict:
    """Richardson limits of the production FVM ladder (recomputed, not copied)."""
    out = {}
    for key, field, which in (("u_min", "u", "min_computed"),
                              ("v_max", "v", "max_computed")):
        vals = []
        for n in (32, 64, 128):
            d = json.loads((CASE / "runs" / f"lid_cavity_Re1000_n{n}.json").read_text())
            vals.append(float(d[field]["extremum"][which][1]))
        d1, d2 = vals[1] - vals[0], vals[2] - vals[1]
        p = float(np.log(abs(d1 / d2)) / np.log(2.0))
        lim = vals[2] + d2 / (2.0 ** p - 1.0)
        out[key] = {"ladder": vals, "p": p, "limit": float(lim)}
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    tags = ("re1000", "re1000hi96", "re1000hi128")
    recs = load_ladder(1000.0, tags)
    if not recs:
        raise SystemExit("no spectral records found")
    # N = 24 (Re = 1000) is the documented under-resolved control row
    # (lid_slip_defect 9.5e-2); it must not enter a convergence estimate.
    recs = [r for r in recs if r["order"] >= 32]
    orders = [r["order"] for r in recs]
    print("orders available:", orders)

    tables, verdict = {}, {}
    for key, sub in (("u_min", None), ("v_max", None), ("psi_min", None),
                     ("vortex_x", 0), ("vortex_y", 1)):
        vals = [r["metrics"][key] if sub is None
                else r["metrics"]["vortex_centre"][sub] for r in recs]
        tables[key] = geometric_limit(orders, vals)

    fvm = fvm_richardson()
    gh = GHIA[1000.0]

    # the previously published two-order estimate (SPECTRAL.md §3.2), rebuilt
    # from the same 48/64 records so that the comparison is not a hand copy
    prev = {}
    if 48 in orders and 64 in orders:
        i48, i64 = orders.index(48), orders.index(64)
        slip = [recs[i48]["metrics"]["lid_slip_defect"],
                recs[i64]["metrics"]["lid_slip_defect"]]
        ratio = slip[0] / slip[1]
        for key in ("u_min", "v_max", "psi_min"):
            vals = [recs[i48]["metrics"][key], recs[i64]["metrics"][key]]
            prev[key] = two_order_limit([48, 64], vals, ratio)
            prev[key]["delta_vs_four_order"] = (
                tables[key]["limit"] - prev[key]["limit"])

    for key, gkey in (("u_min", "u_min"), ("v_max", "v_max")):
        sp = tables[key]
        verdict[key] = {
            "spectral_limit_four_order": sp["limit"],
            "spectral_uncertainty": sp["uncertainty"],
            "ghia": gh[gkey],
            "spectral_vs_ghia_rel": abs(sp["limit"] - gh[gkey]) / abs(gh[gkey]),
            "spectral_vs_fvm_rel": abs(sp["limit"] - fvm[key]["limit"])
            / abs(fvm[key]["limit"]),
            "fvm_vs_ghia_rel": abs(fvm[key]["limit"] - gh[gkey]) / abs(gh[gkey]),
            "two_order_vs_four_order_shift": (
                None if key not in prev else
                abs(prev[key]["delta_vs_four_order"])),
        }
    vx, vy = tables["vortex_x"]["limit"], tables["vortex_y"]["limit"]
    verdict["vortex"] = {
        "spectral_limit": [vx, vy],
        "ghia": list(gh["vortex"]),
        "rel_err": [abs(vx - gh["vortex"][0]) / gh["vortex"][0],
                    abs(vy - gh["vortex"][1]) / gh["vortex"][1]],
    }

    # ---- figures ---------------------------------------------------------
    ok = recs
    ns = [r["order"] for r in ok]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key in ((axes[0], "u_min"), (axes[1], "v_max")):
        vals = [r["metrics"][key] for r in ok]
        ax.plot(ns, vals, "o-", label="spectral (Krylov Newton)")
        ax.axhline(tables[key]["limit"], ls="--", c="C2",
                   label=f"four-order limit {tables[key]['limit']:+.6f}")
        if key in prev:
            ax.axhline(prev[key]["limit"], ls=(0, (1, 1)), c="C4",
                       label=f"old two-order {prev[key]['limit']:+.6f}")
        ax.axhline(fvm[key]["limit"], ls=":", c="C3",
                   label=f"FVM Richardson {fvm[key]['limit']:+.6f}")
        ax.axhline(gh[key], ls="-.", c="k", label=f"Ghia {gh[key]:+.5f}")
        ax.set(xlabel="polynomial order N", ylabel=key,
               title=f"{key} at Re = 1000, N = {min(ns)}...{max(ns)}")
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "spectral_hiorder_convergence.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for key, mk in (("u_min", "o"), ("v_max", "s"), ("psi_min", "^")):
        ax.semilogy(ns, [r["metrics"][key] * 0 + 1 for r in ok], " ")  # keep x range
        d = tables[key]["successive_differences"]
        ax.semilogy(ns[1:], d, mk + "-", label=f"|Δ{key}| between orders")
    ax.set(xlabel="polynomial order N (upper of the pair)",
           ylabel="successive difference",
           title="geometric error decay at Re = 1000 (measured ratios)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "spectral_hiorder_decay.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    gt = np.array(ref.U_CENTRELINE[1000.0], dtype=float)
    gv = np.array(ref.V_CENTRELINE[1000.0], dtype=float)
    for r in ok[-2:]:
        p = r["profiles"]["u_centreline"]
        axes[0].plot(p["u"], p["y"], lw=1.2, label=f"N={r['order']}")
        p = r["profiles"]["v_centreline"]
        axes[1].plot(p["x"], p["v"], lw=1.2, label=f"N={r['order']}")
    axes[0].plot(gt[:, 1], gt[:, 0], "o", ms=3.5, label="Ghia (1982) Tbl I")
    axes[1].plot(gv[:, 0], gv[:, 1], "o", ms=3.5, label="Ghia (1982) Tbl II")
    axes[0].set(xlabel="u / U", ylabel="y / L", title="Re = 1000, u(y) at x = L/2")
    axes[1].set(xlabel="x / L", ylabel="v / U", title="Re = 1000, v(x) at y = L/2")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "spectral_hiorder_centrelines.png", dpi=130)
    plt.close(fig)

    report = {
        "audit_unit": "high-order spectral ladder (sparse-Jacobian Krylov Newton)",
        "reynolds": 1000.0,
        "tags": list(tags),
        "orders": orders,
        "ladder": [{"order": r["order"],
                    "nodes_per_side": r["nodes_per_side"],
                    "linear_solver": r["method"].get("linear_solver", "direct"),
                    "metrics": r["metrics"],
                    "ghia": r.get("ghia"),
                    "wall_seconds": r["wall_seconds"],
                    "gmres_iterations": [h.get("gmres_iterations")
                                         for h in r["newton"].get("history", [])],
                    "file": r["_file"]} for r in ok],
        "extrapolation": tables,
        "two_order_extrapolation": prev,
        "fvm": fvm,
        "verdict": verdict,
        "figures": ["figures/spectral_hiorder_convergence.png",
                    "figures/spectral_hiorder_decay.png",
                    "figures/spectral_hiorder_centrelines.png"],
    }
    (SPEC / "spectral_hiorder_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    print(json.dumps({"verdict": verdict,
                      "extrapolation": {k: {kk: vv for kk, vv in v.items()
                                            if kk not in ("orders", "values")}
                                        for k, v in tables.items()},
                      "fvm": fvm}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())