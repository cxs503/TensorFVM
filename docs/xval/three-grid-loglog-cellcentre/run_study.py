"""Driver: three-grid log-log-slope order verification of the cell-centred FV scheme.

Anchor: ``XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR``.

Runs, on the cell-centred compressible FV solver of this repository
(``EulerSolver``: collocated FV, MUSCL-Hancock + HLLC), the periodic entropy wave
whose exact solution is ``rho = 1 + A sin(k(x-ut))`` with a closed-form **cell
average**.  Reports the observed order read three ways -- log-log least-squares
slope, pairwise slopes, exact three-grid Richardson solve -- against the
cell-average metric (correct for a FV cell vector) and the centroid-point metric
(sampling biased), plus two controls:

* the **sampling floor**: a synthetic cell field of true order ``p`` is read
  through the centroid metric, which cannot report more than 2;
* the **non-uniform-ratio trap**: the textbook ``|ln(e21/e32)|/ln r`` formula on a
  ladder with ``r21 != r32`` versus the exact three-grid solve.

Runtime ~1 min on one core; deterministic (fixed grids, CFL, no randomness).

    PYTHONPATH=src python docs/xval/three-grid-loglog-cellcentre/run_study.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification import cellcentre_loglog as cc  # noqa: E402
from tensorfvm.verification.richardson import richardson_gci  # noqa: E402

# ---------------------------------------------------------------------------
# predeclared study definition (fixed before any number was inspected)
# ---------------------------------------------------------------------------
LADDERS = {
    # name: (grids, scheme order, cfl)
    "order2_uniform": ((40, 80, 160), 2, 0.4),
    "order1_uniform": ((40, 80, 160), 1, 0.4),
    "order2_fine": ((80, 160, 320), 2, 0.4),
    "order2_nonuniform": ((50, 100, 175), 2, 0.4),
    "order1_nonuniform": ((50, 100, 175), 1, 0.4),
    "order2_cfl010": ((40, 80, 160), 2, 0.1),
    "order2_coarse": ((10, 20, 40), 2, 0.4),
    "order2_mid": ((20, 40, 80), 2, 0.4),
}

#: Overlapping-triplet pairs used as the asymptotic-range gate.
CONSISTENCY_PAIRS = (
    ("order2_uniform", "order2_fine"),
    ("order2_coarse", "order2_mid"),
)

#: Synthetic order-recovery controls: (truth order, coefficient) pairs.
FLOOR_CASES = ((2, 1.0), (2, 10.0), (3, 1.0), (3, 10.0), (4, 1.0), (4, 10.0))
FLOOR_LADDERS = ((40, 80, 160), (80, 160, 320))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------
def synthetic_field(nx: int, true_order: float, coefficient: float, *,
                    t: float, amplitude: float = cc.AMPLITUDE,
                    velocity: float = cc.VELOCITY,
                    wavenumber: float = cc.WAVENUMBER) -> dict:
    """A cell field whose error against the exact cell average is ``c h^p``.

    ``u_h = avg(u) + c h^p sin(k(x-ut))`` is not a solver output; it is the
    negative control that isolates the *metric*: the cell-average metric must
    return ``p`` exactly, and the centroid metric must be floored at 2.
    """
    h = 1.0 / nx
    x = (np.arange(nx) + 0.5) * h
    average = cc.exact_cell_average(nx, t, amplitude=amplitude, velocity=velocity)
    point = cc.exact_point(nx, t, amplitude=amplitude, velocity=velocity)
    field = average + coefficient * h ** true_order * np.sin(
        wavenumber * (x - velocity * t))
    widths = np.full(nx, h)
    return {
        "nx": int(nx),
        "h": h,
        "cell_average": cc.error_norms(field, average, widths),
        "point": cc.error_norms(field, point, widths),
    }


def floor_control() -> list:
    out = []
    for grids in FLOOR_LADDERS:
        for true_order, coefficient in FLOOR_CASES:
            rows = [synthetic_field(nx, true_order, coefficient, t=0.2)
                    for nx in grids]
            entry = {
                "grids": list(grids),
                "true_order": true_order,
                "coefficient": coefficient,
                "l1": {
                    metric: [row[metric]["l1"] for row in rows]
                    for metric in ("cell_average", "point")
                },
            }
            for metric in ("cell_average", "point"):
                entry.setdefault("slope", {})[metric] = cc.loglog_slope(
                    cc.spacings(grids), entry["l1"][metric])
            entry["slope_dev"] = {
                metric: abs(entry["slope"][metric] - true_order) / true_order
                for metric in ("cell_average", "point")
            }
            out.append(entry)
            log(f"[floor] grids={list(grids)} p={true_order} c={coefficient}: "
                f"cellavg={entry['slope']['cell_average']:.4f} "
                f"point={entry['slope']['point']:.4f}")
    return out


def run_ladders() -> dict:
    out = {}
    for name, (grids, order, cfl) in LADDERS.items():
        t0 = time.time()
        study = cc.study(grids, order=order, cfl=cfl)
        study["seconds"] = round(time.time() - t0, 2)
        out[name] = study
        row = next(r for r in study["readouts"]
                   if r["metric"] == "cell_average" and r["norm"] == "l1")
        log(f"[{name}] order={order} cfl={cfl} grids={list(grids)} "
            f"p_loglog={row['p_loglog']:.4f} p_richardson={row['p_richardson']:.4f} "
            f"dev={100*row['loglog_rel_dev']:.2f}% ({study['seconds']}s)")
    return out


def l1_errors(study: dict, metric: str) -> list:
    return [float(row[metric]["l1"]) for row in study["rows"]]


def ratio_traps(studies: dict) -> list:
    out = []
    for name in ("order2_uniform", "order2_fine", "order2_nonuniform",
                 "order1_uniform", "order1_nonuniform"):
        study = studies[name]
        grids = study["grids"]
        entry = {"ladder": name, "grids": grids,
                 "theory_order": study["theory_order"]}
        for metric in ("cell_average", "point"):
            entry[metric] = cc.uniform_ratio_bias(
                grids, l1_errors(study, metric),
                theory_order=study["theory_order"])
        out.append(entry)
        log(f"[ratio] {name}: uniform_ladder={entry['cell_average']['uniform_ladder']} "
            f"r21={entry['cell_average']['r21']:.4f} r32={entry['cell_average']['r32']:.4f} "
            f"p_true={entry['cell_average']['p_true']:.4f} "
            f"p_uniform_form={entry['cell_average']['p_uniform_form']:.4f}")
    return out


def consistency_gates(studies: dict) -> list:
    out = []
    for a, b in CONSISTENCY_PAIRS:
        for metric in ("cell_average", "point"):
            for norm in ("l1", "l2", "linf"):
                read = lambda name: next(  # noqa: E731
                    r for r in studies[name]["readouts"]
                    if r["metric"] == metric and r["norm"] == norm)
                gate = cc.consistency(read(a), read(b))
                gate.update({"first": a, "second": b, "metric": metric,
                             "norm": norm,
                             "first_rel_dev": read(a)["loglog_rel_dev"],
                             "second_rel_dev": read(b)["loglog_rel_dev"]})
                out.append(gate)
                log(f"[gate] {a}/{b} {metric}/{norm}: p={gate['p_first']:.4f}/"
                    f"{gate['p_second']:.4f} rel_diff={100*gate['rel_diff']:.2f}% "
                    f"pass15={gate['pass_15']} devs="
                    f"{100*gate['first_rel_dev']:.2f}%/{100*gate['second_rel_dev']:.2f}%")
    return out


def gci_reference(studies: dict) -> dict:
    """Richardson/GCI reading of the finest uniform ladder (cross-check face)."""
    study = studies["order2_fine"]
    e = l1_errors(study, "cell_average")
    h = cc.spacings(study["grids"])
    # errors/ spacings are ordered coarse -> fine in the study, the three-grid
    # routine wants fine -> coarse.
    gci = richardson_gci(e[-1], e[-2], e[-3], list(reversed(h)), theory_order=2.0)
    return {
        "ladder": "order2_fine",
        "metric": "cell_average/l1",
        "grids": study["grids"],
        "errors": e,
        "order": gci.order,
        "order_used": gci.order_used,
        "convergence": gci.convergence,
        "asymptotic": gci.asymptotic,
        "gci": gci.gci,
        "band": list(gci.band),
        # the metric is an error norm whose limit is zero, so the extrapolated
        # value and its *relative* band are reported for completeness only.
        "extrapolated": gci.extrapolated,
    }


def summarize(studies: dict) -> dict:
    """Anchor acceptance sentences, all derived from the ladder readouts."""
    def read(ladder, metric, norm="l1"):
        return next(r for r in studies[ladder]["readouts"]
                    if r["metric"] == metric and r["norm"] == norm)

    design = {}
    for ladder in ("order2_uniform", "order1_uniform", "order2_fine",
                   "order2_nonuniform", "order1_nonuniform", "order2_cfl010"):
        row = read(ladder, "cell_average")
        design[ladder] = {
            "theory_order": row["p_theory"],
            "p_loglog": row["p_loglog"],
            "p_richardson": row["p_richardson"],
            "rel_dev": row["loglog_rel_dev"],
            "pass_3": row["loglog_pass_3"],
            "pass_15": row["loglog_pass_15"],
        }
    norms = {}
    for norm in ("l1", "l2", "linf"):
        row = read("order2_uniform", "cell_average", norm)
        norms[norm] = {"p_loglog": row["p_loglog"],
                       "p_richardson": row["p_richardson"],
                       "rel_dev": row["loglog_rel_dev"]}
    return {
        "design_order_readouts": design,
        "norm_sensitivity_order2_uniform": norms,
        "all_design_readouts_within_15pct": all(v["pass_15"] for v in design.values()),
        "all_design_readouts_within_3pct": all(v["pass_3"] for v in design.values()),
        "l1_within_3pct": all(v["pass_3"] for k, v in design.items()),
    }


def provenance() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=ROOT, text=True).strip())
    except Exception:  # pragma: no cover - git absent
        sha, dirty = "unknown", None
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "solver": "tensorfvm.compressible.solver.EulerSolver (cell-centred FV, "
                  "MUSCL-Hancock + HLLC, MC limiter)",
        "case": "periodic entropy wave rho=1+A sin(k(x-ut)), A=0.2, u=0.5, t=0.2",
        "thresholds": {"acceptable": cc.THRESHOLD_ACCEPTABLE,
                       "tight": cc.THRESHOLD_TIGHT},
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(payload: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    studies = payload

    # 1. log-log error plot, both metrics, order-2 and order-1 ladders
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ladder in zip(axes, ("order2_uniform", "order1_uniform")):
        study = studies["ladders"][ladder]
        h = np.array(cc.spacings(study["grids"]))
        for metric, style in (("cell_average", "o-"), ("point", "s--")):
            e = np.array([row[metric]["l1"] for row in study["rows"]])
            read = next(r for r in study["readouts"]
                        if r["metric"] == metric and r["norm"] == "l1")
            ax.loglog(h, e, style,
                      label=f"{metric}: slope {read['p_loglog']:.3f}")
        e_cell = np.array([row["cell_average"]["l1"] for row in study["rows"]])
        ax.loglog(h, e_cell[0] * (h / h[0]) ** study["theory_order"], "k:",
                  label=f"design slope {study['theory_order']:.0f}")
        ax.set_xlabel("h = 1/nx")
        ax.set_ylabel(r"$L_1(\rho)$")
        ax.set_title(f"{ladder} (declared order {study['theory_order']})")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_loglog_orders.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    # 2. sampling floor
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    floor = studies["floor_control"]
    labels, cell, point = [], [], []
    for entry in floor:
        labels.append(f"p={entry['true_order']}\nc={entry['coefficient']:g}\n"
                      f"n{entry['grids'][0]}-{entry['grids'][-1]}")
        cell.append(entry["slope"]["cell_average"])
        point.append(entry["slope"]["point"])
    xs = np.arange(len(labels))
    ax.plot(xs, cell, "o-", label="cell-average metric (correct)")
    ax.plot(xs, point, "s--", label="centroid-point metric")
    ax.axhline(2.0, color="k", ls=":", lw=1, label="slope 2 floor")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("observed log-log slope")
    ax.set_title("synthetic field of true order p: metric decides the reading")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    p = outdir / "fig_sampling_floor.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)

    # 3. uniform-ratio trap
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    traps = studies["ratio_traps"]
    names = [t["ladder"] for t in traps if not t["cell_average"]["uniform_ladder"]]
    true_p = [t["cell_average"]["p_true"] for t in traps
              if not t["cell_average"]["uniform_ladder"]]
    uni_p = [t["cell_average"]["p_uniform_form"] for t in traps
             if not t["cell_average"]["uniform_ladder"]]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, true_p, 0.4, label="exact three-grid solve")
    ax.bar(xs + 0.2, uni_p, 0.4, label="textbook uniform-r formula")
    for i, (a, b) in enumerate(zip(true_p, uni_p)):
        ax.text(i - 0.2, a + .05, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + .05, f"{b:.3f}", ha="center", fontsize=8)
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("observed order p")
    ax.set_title("non-uniform ladder (r21=1.75, r32=2.0): assumption cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = outdir / "fig_uniform_ratio_trap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p)
    return made


def main() -> int:
    outdir = HERE / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    studies = run_ladders()
    payload = {
        "anchor": "XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR",
        "provenance": provenance(),
        "ladders": studies,
        "floor_control": floor_control(),
        "ratio_traps": ratio_traps(studies),
        "consistency_gates": consistency_gates(studies),
        "gci_reference": gci_reference(studies),
        "summary": summarize(studies),
        "wall_seconds": round(time.time() - t0, 1),
    }
    path = outdir / "study.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    payload["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    log(f"[write] {path} sha256={payload['artifact_sha256']}")
    figs = figures(payload, HERE / "figures")
    for f in figs:
        log(f"[figure] {f}")
    log(f"[done] {payload['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())