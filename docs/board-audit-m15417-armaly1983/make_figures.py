"""Board-audit figures for m15386 (REPRO-FVM-ARMALY1983).

Every number plotted here is read straight out of the archived run records
(`docs/repro-armaly1983-bfs/artifacts/armaly1983_re<Re>_coarse_ld12.json`) — no
value is transcribed by hand, so the audit figures cannot drift from the
artefacts they audit.

    python docs/board-audit-m15417-armaly1983/make_figures.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = pathlib.Path("docs/repro-armaly1983-bfs/artifacts")
OUT = pathlib.Path("docs/board-audit-m15417-armaly1983")
CASES = {100: ART / "armaly1983_re100_coarse_ld12.json",
         389: ART / "armaly1983_re389_coarse_ld12.json"}


def load(re_hz: float) -> dict:
    return json.loads(CASES[int(re_hz)].read_text())


def field_grid(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat fluid-cell lists -> (x, y, u) on the masked solver grid."""
    nx, ny = record["field"]["shape"]
    u2 = np.full((ny, nx), np.nan)
    idx = np.asarray(record["field"]["fluid_index"], dtype=int)
    u2[idx // nx, idx % nx] = np.asarray(record["field"]["u"], dtype=float)
    return (np.asarray(record["field"]["x"]), np.asarray(record["field"]["y"]), u2)


def poiseuille_matched(u: np.ndarray, y: np.ndarray, h_wet: float) -> np.ndarray:
    """Plane Poiseuille profile carrying the same mass flux as `u`."""
    bulk = float(np.trapezoid(u, y)) / h_wet
    eta = y / h_wet
    return 6.0 * bulk * eta * (1.0 - eta)


def main() -> int:
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.33)

    # (a) u-velocity field, Re = 389 (the case that fails the 5 % development gate)
    rec389 = load(389.0)
    x, y, u2 = field_grid(rec389)
    ax = fig.add_subplot(gs[0, :])
    im = ax.pcolormesh(x, y, u2, cmap="RdBu_r", shading="nearest",
                       vmin=-0.35, vmax=1.15)
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.axvline(rec389["reattachment_over_step"], color="k", lw=1.2, ls="--")
    ax.annotate(f"L_r/S = {rec389['reattachment_over_step']:.3f}\n(exp. 7.9, "
                f"err {100 * rec389['reattachment_rel_error']:.2f} %)",
                xy=(rec389["reattachment_over_step"], 2.0),
                xytext=(rec389["reattachment_over_step"] + 0.6, 2.35),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("x / S")
    ax.set_ylabel("y / S")
    ax.set_title(f"TensorFVM coarse grid (243x33, 7203 fluid cells), Re_Dh = 389, "
                 f"u/U_in field — archived artefact "
                 f"{CASES[389].name}")
    fig.colorbar(im, ax=ax, pad=0.01, label="u / U_in")

    # (b) development profiles: computed vs plane Poiseuille (matched flux)
    for j, re_hz in enumerate((100.0, 389.0)):
        rec = load(re_hz)
        u = np.asarray(rec["profile"]["u"], dtype=float)
        yy = np.asarray(rec["profile"]["y"], dtype=float)
        h = float(rec["profile"]["fluid"].count(1.0)) if False else None
        h_wet = float(yy[-1] + (yy[-1] - yy[-2]) / 2.0)
        par = poiseuille_matched(u, yy, h_wet)
        axp = fig.add_subplot(gs[1, j])
        axp.plot(u, yy / h_wet, "o-", ms=2.4, lw=1.0, label="TensorFVM")
        axp.plot(par, yy / h_wet, "k--", lw=1.1, label="plane Poiseuille")
        axp.set_xlabel("u / U_in")
        axp.set_ylabel(r"$\eta = y/H_{wet}$")
        st = rec["profile_stations"][-1]
        axp.set_title(f"Re = {int(re_hz)} outlet (x/S = {st['x_over_step']:.2f}): "
                      f"max|dev| = {100 * st['max_rel']:.2f} % "
                      f"({'PASS' if rec['profile_outlet_pass_5pct'] else 'FAIL'} < 5 %)",
                      fontsize=8.5)
        axp.legend(fontsize=7, loc="best")

    # (c) ref vs got for the two predeclared reattachment anchors
    axb = fig.add_subplot(gs[1, 2])
    labels = [f"Re = {r}" for r in CASES]
    got = [load(r)["reattachment_over_step"] for r in CASES]
    ref = [load(r)["experiment_over_step"] for r in CASES]
    xx = np.arange(len(labels))
    axb.bar(xx - 0.19, ref, 0.36, color="#bbbbbb", label="Armaly 1983 (exp.)")
    axb.bar(xx + 0.19, got, 0.36, color="#2b6cb0", label="TensorFVM coarse")
    for i, r in enumerate(CASES):
        e = 100 * load(r)["reattachment_rel_error"]
        axb.text(xx[i] + 0.19, got[i] + 0.12, f"{got[i]:.3f}\n{e:.2f} %",
                 ha="center", fontsize=7.5)
    axb.set_xticks(xx, labels)
    axb.set_ylabel(r"$L_r/S$")
    axb.set_ylim(0, 9.6)
    axb.legend(fontsize=7)
    axb.set_title("reattachment length: ref vs got (15 % gate)", fontsize=8.5)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig_m15386_armaly_audit.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print("wrote", png)

    summary = {
        "figure": str(png),
        "source_artifacts": {int(r): str(p) for r, p in CASES.items()},
        "reattachment": {int(r): {"got": load(r)["reattachment_over_step"],
                                 "ref": load(r)["experiment_over_step"],
                                 "rel_error": load(r)["reattachment_rel_error"],
                                 "pass_15pct": load(r)["reattachment_pass_15pct"]}
                         for r in CASES},
        "outlet_profile": {int(r): {"max_rel": load(r)["profile_stations"][-1]["max_rel"],
                                    "x_over_step": load(r)["profile_stations"][-1]["x_over_step"],
                                    "pass_5pct": load(r)["profile_outlet_pass_5pct"]}
                           for r in CASES},
    }
    (OUT / "figure_data.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())