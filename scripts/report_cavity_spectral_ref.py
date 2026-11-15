#!/usr/bin/env python3
"""Build the tables and figures of ``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL.md``.

Every number in the report is injected from the JSON artefacts in
``docs/CASE-FVM-CAVITY-RE1000/runs/``; nothing is transcribed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm import lid_cavity_reference as ref
from tensorfvm.cavity_spectral import SpectralCavity
from scripts.run_cavity_spectral_ref import march

CASE = Path("docs/CASE-FVM-CAVITY-RE1000")
SPEC = CASE / "runs" / "spectral"
FIGS = CASE / "figures"
GHIA = {1000.0: (-0.38289, 0.37095), 100.0: (-0.21090, 0.17527),
        400.0: (-0.32726, 0.30203)}


def load(tag: str) -> list[dict]:
    files = sorted(SPEC.glob(f"cavity_spectral_N*_{tag}.json"))
    return [json.loads(f.read_text()) for f in files]


def richardson2(f1: float, f2: float, f3: float, r: float = 2.0) -> tuple[float, float]:
    """Observed order and Richardson limit from three grid levels (ratio ``r``)."""
    d1, d2 = f2 - f1, f3 - f2
    if d1 == 0 or d2 == 0 or d1 / d2 <= 0:
        return float("nan"), float("nan")
    p = np.log(abs(d1 / d2)) / np.log(r)
    return float(p), float(f3 + d2 / (r ** p - 1.0))


def fvm_limits() -> dict:
    """Richardson limits of the shipped FVM ladder, recomputed from its JSONs.

    ``lid_cavity_Re1000_n{32,64,128}.json`` store the extrema of the production
    solution; the observed order and the limit are recomputed here from those
    three numbers (uniform ratio r = 2), so the comparison carries no hand copy.
    """
    out = {}
    for q, field, which in (("u_min", "u", "min_computed"),
                            ("v_max", "v", "max_computed")):
        vals = []
        for n in (32, 64, 128):
            d = json.loads((CASE / "runs" / f"lid_cavity_Re1000_n{n}.json").read_text())
            vals.append(float(d[field]["extremum"][which][1]))
        p, lim = richardson2(*vals)
        out[q] = {"ladder": vals, "p": p, "limit": lim}
    return out


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    re1000, re100 = load("re1000"), load("re100")

    # ---- spectral limit: geometric extrapolation from the two finest orders
    def limit_of(recs: list[dict], key: str, sub: str | None = None) -> dict:
        ok = [r for r in recs if r.get("metrics")]
        ok.sort(key=lambda r: r["order"])
        vals = [(r["order"], r["metrics"][key] if sub is None
                 else r["metrics"][sub]) for r in ok]
        (n1, f1), (n2, f2) = vals[-2], vals[-1]
        slip = [(r["order"], r["metrics"]["lid_slip_defect"]) for r in ok]
        ratio = slip[-2][1] / slip[-1][1]          # observed error decay per grid
        lim = f2 + (f2 - f1) / (ratio - 1.0)
        return {"pairs": vals, "decay_ratio": ratio, "limit": lim,
                "finest_delta": abs(f2 - f1)}

    tabs = {}
    for key in ("u_min", "v_max", "psi_min"):
        tabs[key] = limit_of(re1000, key)
    cx = [(r["order"], r["metrics"]["vortex_centre"][0]) for r in re1000 if r.get("metrics")]
    cy = [(r["order"], r["metrics"]["vortex_centre"][1]) for r in re1000 if r.get("metrics")]
    tabs["vortex_x"] = {"pairs": cx, "finest": cx[-1][1]}
    tabs["vortex_y"] = {"pairs": cy, "finest": cy[-1][1]}

    fvm = fvm_limits()
    verdict = {}
    for key, gidx in (("u_min", 0), ("v_max", 1)):
        sp = tabs[key]["limit"]
        ghia = GHIA[1000.0][gidx]
        fl = fvm[key]["limit"]
        verdict[key] = {
            "spectral_limit": sp,
            "fvm_richardson_limit": fl,
            "spectral_vs_fvm_rel": abs(sp - fl) / abs(fl),
            "ghia": ghia,
            "spectral_vs_ghia_rel": abs(sp - ghia) / abs(ghia),
            "fvm_vs_ghia_rel": abs(fl - ghia) / abs(ghia),
        }

    # ---- figures ---------------------------------------------------------
    ok = sorted([r for r in re1000 if r.get("metrics")], key=lambda r: r["order"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    gt = np.array(ref.U_CENTRELINE[1000.0], dtype=float)
    gv = np.array(ref.V_CENTRELINE[1000.0], dtype=float)
    for r in ok[-2:]:
        p = r["profiles"]["u_centreline"]
        axes[0].plot(p["u"], p["y"], lw=1.2, label=f"spectral N={r['order']}")
        p = r["profiles"]["v_centreline"]
        axes[1].plot(p["x"], p["v"], lw=1.2, label=f"spectral N={r['order']}")
    axes[0].plot(gt[:, 1], gt[:, 0], "o", ms=3.5, label="Ghia (1982) Table I")
    axes[1].plot(gv[:, 0], gv[:, 1], "o", ms=3.5, label="Ghia (1982) Table II")
    axes[0].set(xlabel="u / U", ylabel="y / L", title="Re = 1000, vertical centreline")
    axes[1].set(xlabel="x / L", ylabel="v / U", title="Re = 1000, horizontal centreline")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "spectral_centrelines.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = [r["order"] for r in ok]
    for ax, key, idx in ((axes[0], "u_min", 0), (axes[1], "v_max", 1)):
        vals = [r["metrics"][key] for r in ok]
        ax.semilogx(ns, vals, "o-", label="spectral")
        ax.axhline(tabs[key]["limit"], ls="--", c="C2",
                   label=f"spectral extrapolation {tabs[key]['limit']:+.5f}")
        ax.axhline(fvm[key]["limit"], ls=":", c="C3",
                   label=f"FVM Richardson {fvm[key]['limit']:+.5f}")
        ax.axhline(GHIA[1000.0][idx], ls="-.", c="k",
                   label=f"Ghia {GHIA[1000.0][idx]:+.5f}")
        ax.set(xlabel="polynomial order N", ylabel=key,
               title=f"{key}: grid convergence at Re = 1000")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "spectral_convergence.png", dpi=130)
    plt.close(fig)

    # field contours (recomputed once; N = 48 is the smallest resolved order)
    solver = SpectralCavity(48)
    sol = march(solver, 1000.0)
    X = solver.X
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    cs = axes[0].contourf(X, X, sol.psi, 40, cmap="RdBu_r")
    axes[0].contour(X, X, sol.psi, 12, colors="k", linewidths=0.4)
    fig.colorbar(cs, ax=axes[0], label="psi / (U L)")
    axes[0].set(title="streamfunction, Re = 1000 (N = 48)", xlabel="x / L", ylabel="y / L")
    cv = axes[1].contourf(X, X, sol.omega, 40, cmap="viridis")
    fig.colorbar(cv, ax=axes[1], label="omega L / U")
    axes[1].set(title="vorticity, Re = 1000 (N = 48)", xlabel="x / L", ylabel="y / L")
    fig.tight_layout()
    fig.savefig(FIGS / "spectral_fields.png", dpi=130)
    plt.close(fig)

    report = {
        "audit_unit": "spectral reference solver (Chebyshev collocation, Newton)",
        "reynolds": 1000.0,
        "ladder_re1000": [{"order": r["order"],
                           "status": r.get("status", "ok"),
                           "metrics": r.get("metrics"),
                           "ghia": r.get("ghia"),
                           "wall_seconds": r["wall_seconds"]} for r in re1000],
        "ladder_re100": [{"order": r["order"],
                          "status": r.get("status", "ok"),
                          "metrics": r.get("metrics")} for r in re100],
        "extrapolation": tabs,
        "fvm": fvm,
        "verdict": verdict,
        "field_check_n48": {"psi_min": sol.metrics["psi_min"],
                            "lid_slip_defect": sol.metrics["lid_slip_defect"],
                            "poisson_residual_max": sol.metrics["poisson_residual_max"]},
        "figures": ["figures/spectral_centrelines.png",
                    "figures/spectral_convergence.png",
                    "figures/spectral_fields.png"],
    }
    (SPEC / "spectral_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"verdict": verdict, "extrapolation":
                      {k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                       for k, v in tabs.items()},
                      "fvm": fvm}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())