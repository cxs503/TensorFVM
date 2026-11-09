"""Re-grade the planar sudden-expansion loss coefficient against Borda--Carnot.

Why
---
The archived benchmark ``docs/benchmarks/verified/fvm_sudden_expansion/`` reports
``K = 1.131`` at D/30 versus ``K_BC = 0.5625`` and declares **FAIL (101 %)**,
while a later audit of the same case claims **+1.3 % PASS** using
``k_from_static_recovery`` at D/10.  Both numbers are read off the *same*
nominal case, so at most one of them can be the loss coefficient.  This script
re-derives every published route from the stored evidence with one code path
(:mod:`tensorfvm.expansion_loss`), adds the step-plane extrapolation that the
MAC implementation uses, and lets the numbers decide.

Modes
-----
``run``    solve one 12 D ``Re_D = 100`` case (D/10) with the incompressible
           baseline driver and store ``expansion_n010.json`` + fields under
           ``docs/sudden-expansion-loss-verdict/runs``.
``report`` post-process that run plus the archived artifacts into
           ``docs/sudden-expansion-loss-verdict/results.json`` and a figure.

Everything in ``report`` is a pure function of the committed artifacts, so the
output is reproducible and no number is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from tensorfvm.expansion_loss import (  # noqa: E402
    borda_carnot,
    kinetic_energy_factor,
    linear_slope,
    relative_error,
    section_average,
    station_metrics,
    step_plane_value,
    verdict,
)

K_BC = borda_carnot(4.0)
OUT = REPO / "docs" / "sudden-expansion-loss-verdict"
RUNS = OUT / "runs"
ARCHIVE = REPO / "docs" / "incompressible-baselines" / "baseline-runs"


# --------------------------------------------------------------------------- #
# run mode
# --------------------------------------------------------------------------- #
def run_case(level: int = 10, tag: str = "expansion_n010") -> dict:
    import run_fvm_incompressible_baselines as driver

    RUNS.mkdir(parents=True, exist_ok=True)
    job = dict(case="sudden_expansion", level=level, tag=tag,
               budget=dict(max_iter=20000), init_from=None)
    record = driver.run_one(job, str(RUNS))
    print(f"[run] {tag}: k_total_head={record['metrics']['k_total_head']:.6f} "
          f"iterations={record['metrics']['iterations']} "
          f"wall={record['wall_seconds']}s", flush=True)
    return record


# --------------------------------------------------------------------------- #
# report mode
# --------------------------------------------------------------------------- #
def _load_run(tag: str):
    with open(RUNS / f"{tag}.json") as fh:
        record = json.load(fh)
    with np.load(RUNS / f"{tag}_fields.npz") as data:
        fields = {k: data[k] for k in data.files}
    return record, fields


def _columns(u, p, nx, ny, *, window_start_over_D, outlet_station):
    """Per-column area- and flux-weighted static/total pressure profiles."""
    from tensorfvm.expansion_loss import _as_flow_field

    u = _as_flow_field(u, nx, ny, "u")
    p = _as_flow_field(p, nx, ny, "p")
    xs, p_col, pt_bulk, pt_flux = [], [], [], []
    for i in range(window_start_over_D, outlet_station + 1):
        uc = np.clip(u[i, :], 0.0, None)
        pc = p[i, :]
        xs.append(i)
        p_col.append(float(pc.mean()))
        pt_bulk.append(float(pc.mean()) + 0.5 * float(uc.mean()) ** 2)
        pt_flux.append(float((uc * (pc + 0.5 * uc ** 2)).sum() / uc.sum()))
    return np.asarray(xs, dtype=float), np.asarray(p_col), np.asarray(pt_bulk), np.asarray(pt_flux)


def build_report() -> dict:
    record, fields = _load_run("expansion_n010")
    meta = record["meta"]
    nx, ny = int(meta["nx"]), int(meta["ny"])
    hx, D = float(meta["hx"]), float(meta["D"])
    y0, y1 = meta["inlet_window"]
    rows = list(range(int(round(y0 / meta["hy"])), int(round(y1 / meta["hy"]))))
    i_out = nx - max(3, int(round(2.0 * D / hx)))
    u = np.asarray(fields["u"], dtype=float)
    p = np.asarray(fields["p"], dtype=float)

    station = station_metrics(u, p, rows, i_out, area_ratio=meta["area_ratio"])

    # developed-window profiles: last two diameters of the domain
    start = max(i_out - int(round(2.0 * D / hx)), 1)
    xs, p_col, pt_bulk, pt_flux = _columns(u, p, nx, ny,
                                           window_start_over_D=start,
                                           outlet_station=i_out)
    x_length = xs * hx                     # streamwise distance from x = 0
    q_in = station["q_in"]
    slope_p = linear_slope(x_length, p_col)                  # dp/dx
    slope_pt_bulk = linear_slope(x_length, pt_bulk)          # d(pt)/dx (area avg)
    slope_pt_flux = linear_slope(x_length, pt_flux)          # d(pt)/dx (flux avg)
    x_out = float(i_out * hx)

    cp_step = step_plane_value(station["cp"], slope_p / q_in, x_out)
    k_step_static = (1.0 - 1.0 / meta["area_ratio"] ** 2) - cp_step
    k_step_bulk = step_plane_value(station["k_station_bulk"],
                                  slope_pt_bulk / q_in, x_out)
    k_step_flux = step_plane_value(station["k_station_flux"],
                                  slope_pt_flux / q_in, x_out)
    # fit quality: residual of the linear model over the window
    def _r2(y):
        y = np.asarray(y, dtype=float)
        ss = float(((y - y.mean()) ** 2).sum())
        return float("nan") if ss == 0.0 else 1.0 - float(((y - np.poly1d(
            np.polyfit(x_length, y, 1))(x_length)) ** 2).sum()) / ss

    # implied wall shear from the measured static slope versus the fully
    # developed plane-Poiseuille value dp/dx = -12 mu U / H**2 (H = D = 1).
    mu = float(meta["nu"])
    u_out = station["u_out"]
    dpdx_poiseuille = -12.0 * mu * u_out / D ** 2
    development_ratio = float(slope_p / dpdx_poiseuille) if dpdx_poiseuille else float("nan")

    # ---- archived cross-checks (read, never transcribed) -------------------
    def _arch(name):
        with open(ARCHIVE / f"{name}.json") as fh:
            return json.load(fh)["metrics"]

    arch = {tag: _arch(tag) for tag in
            ("expansion_n010", "expansion_n020", "expansion_n030",
             "expansion_long_n010")}
    loss_def_path = REPO / "docs" / "verification" / "sudden-expansion-loss-definition" / "results.json"
    loss_def = json.load(open(loss_def_path)) if loss_def_path.exists() else {}
    sym_path = REPO / "docs" / "verification-expansion-symmetry" / "summary.json"
    mac = json.load(open(sym_path)) if sym_path.exists() else {}

    routes = {
        "station_bulk_D10": dict(value=station["k_station_bulk"],
                                 reference=K_BC, kind="bulk station (archived QoI)"),
        "station_flux_D10": dict(value=station["k_station_flux"],
                                 reference=K_BC, kind="mass-flux station"),
        "static_station_D10": dict(value=station["k_static_recovery"],
                                   reference=K_BC, kind="static recovery at station"),
        "static_step_D10": dict(value=k_step_static, reference=K_BC,
                                kind="static recovery extrapolated to step plane"),
        "bulk_step_D10": dict(value=k_step_bulk, reference=K_BC,
                              kind="bulk total pressure extrapolated"),
        "flux_step_D10": dict(value=k_step_flux, reference=K_BC,
                              kind="flux total pressure extrapolated"),
    }
    for key, row in routes.items():
        row["error_rel"] = relative_error(row["value"], K_BC)
        row["verdict"] = verdict(row["value"], K_BC)

    payload = dict(
        generated_by="scripts/audit_expansion_loss_verdict.py",
        k_borda_carnot=K_BC,
        new_run=dict(tag="expansion_n010", meta=meta,
                     iterations=record["metrics"]["iterations"],
                     wall_seconds=record["wall_seconds"],
                     outlet_column=i_out, developed_window=(int(xs[0]), int(xs[-1]))),
        station=station,
        slopes=dict(dp_dx=slope_p, dpt_dx_bulk=slope_pt_bulk,
                    dpt_dx_flux=slope_pt_flux,
                    dp_dx_fully_developed=dpdx_poiseuille,
                    development_ratio=development_ratio,
                    r2_pressure=_r2(p_col), r2_total_bulk=_r2(pt_bulk),
                    r2_total_flux=_r2(pt_flux),
                    window_length_over_D=float(x_length[-1] - x_length[0])),
        routes=routes,
        archived=dict(
            k_total_head={k: arch[k]["k_total_head"] for k in arch},
            k_static_recovery={k: arch[k]["k_static_recovery"] for k in arch},
            iterations={k: arch[k]["iterations"] for k in arch},
        ),
        mac_implementation=dict(
            summary=mac,
            note="torch MAC companion implementation, docs/verification-expansion-symmetry/",
        ),
        other_evidence=dict(
            flux_weighted_station=loss_def.get("recorded", {}).get("raw_flux_weighted"),
            bulk_minus_flux=loss_def.get("definitions", {}),
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return payload


def figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels, values, kinds = [], [], []
    for key, row in payload["routes"].items():
        labels.append(key)
        values.append(row["value"])
        kinds.append(row["verdict"])
    colors = ["tab:green" if k == "PASS" else "tab:red" for k in kinds]
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(payload["k_borda_carnot"], color="k", ls="--", lw=1.2,
               label=f"Borda-Carnot {payload['k_borda_carnot']:.4f}")
    ax.axhspan(payload["k_borda_carnot"] * 0.97, payload["k_borda_carnot"] * 1.03,
               color="grey", alpha=0.2, label="3 % band")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_D10", "") for l in labels], rotation=25, ha="right",
                       fontsize=8)
    ax.set_ylabel("K / (0.5 rho U_in^2)")
    ax.set_title("Planar sudden expansion, AR=4, Re_D=100, D/10:\nloss routes vs Borda-Carnot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT / "K_routes_D10.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[report] figure -> {png}")


def summarize(payload: dict) -> str:
    lines = ["| route | value | error vs 0.5625 | verdict |", "|---|---|---|---|"]
    for key, row in payload["routes"].items():
        lines.append(f"| {key} | {row['value']:.4f} | {100 * row['error_rel']:.2f} % | {row['verdict']} |")
    s = payload["slopes"]
    lines.append("")
    lines.append(f"dp/dx measured {s['dp_dx']:.4f} vs fully developed "
                 f"{s['dp_dx_fully_developed']:.4f} "
                 f"(development ratio {s['development_ratio']:.2f}); "
                 f"R2 of the linear window fit: pressure {s['r2_pressure']:.3f}, "
                 f"total (bulk) {s['r2_total_bulk']:.3f}.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "report", "both"), default="report")
    ap.add_argument("--level", type=int, default=10)
    args = ap.parse_args(argv)
    if args.mode in ("run", "both"):
        run_case(level=args.level)
    if args.mode in ("report", "both"):
        payload = build_report()
        figure(payload)
        print(summarize(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())