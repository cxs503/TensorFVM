"""Driver: local order at a smooth crest -- where a MUSCL limiter eats the order.

Anchor: ``XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR`` (continuation of
the delivered global study, debt item "limiter degenerates to first order").

The global three-grid log--log slope of the MC-limited cell-centred scheme on the
periodic entropy wave is 2.06 (anchor).  This driver keeps the same solver, the
same exact cell average and the same estimator, and changes one thing: the norm
is taken over a **fixed number of cells** (5) around the crest, so an ``O(h)``
local error cannot be averaged away.  Controls: the unlimited centred slope
(``limiter='none'``), the max-slope cell (limiter inactive), all four TVD
limiters, and a phase shift that moves the crest off the cell face.

    PYTHONPATH=src python docs/xval/limiter-extremum-order/run_study.py

Writes ``artifacts/study.json`` and ``fig_crest_degradation.png`` next to this
file; ``make_report.py`` renders ``REPORT.md`` from the JSON (no number is typed
twice).
"""
from __future__ import annotations

import hashlib
import json
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

from tensorfvm.verification import limiter_extremum_order as lx  # noqa: E402

ARTIFACTS = HERE / "artifacts"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance() -> dict:
    sources = ["src/tensorfvm/verification/limiter_extremum_order.py",
               "src/tensorfvm/verification/cellcentre_loglog.py",
               "src/tensorfvm/verification/sod1978.py",
               "src/tensorfvm/verification/richardson.py",
               "src/tensorfvm/compressible/solver.py",
               "src/tensorfvm/compressible/reconstruct.py"]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    import torch
    return {"git_head": head, "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(),
            "source_sha256": {p: sha(ROOT / p) for p in sources}}


def build_ladders() -> tuple[dict, dict, float]:
    """Run every ladder once; return (limiters, phase control, seconds)."""
    t0 = time.time()
    ladders = lx.ladder(lx.LIMITERS, lx.GRIDS, phase=0.0)
    phase_ladder = lx.ladder(("mc",), lx.GRIDS, phase=lx.MIDCELL_PHASE)
    return ladders, phase_ladder, time.time() - t0


def summarize(ladders: dict, phase_ladder: dict) -> dict:
    studies = {name: lx.study(rows) for name, rows in ladders.items()}
    phase_study = lx.study(phase_ladder["mc"])
    ratios = lx.crest_magnitude_ratio(studies)
    gate = lx.criteria_gate(studies, ratios, phase_study=phase_study)
    return {"studies": studies, "phase_study": phase_study,
            "ratios": ratios, "gate": gate}


def figure(ladders: dict, out: Path) -> dict:
    """Two panels: log-log error ladders, and the error field at nx=160."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colours = {"mc": "C3", "none": "C0", "minmod": "C1", "vanleer": "C2",
               "superbee": "C4"}
    for name, rows in ladders.items():
        h = [r["h"] for r in rows]
        crest = [r["crest_linf"] for r in rows]
        ax.loglog(h, crest, "o-", color=colours.get(name), label=f"crest $L_\\infty$, {name}")
        if name == "mc":
            glob = [r["global_l1"] for r in rows]
            ax.loglog(h, glob, "s--", color="k", label="global $L_1$ (cell-average), mc")
    href = np.array([1 / 320, 1 / 40])
    anchor = ladders["mc"][-1]["crest_linf"]
    for p, style in ((1.0, ":"), (2.0, "-.")):
        ax.loglog(href, anchor * (href / href[0]) ** p, style, color="0.5",
                  label=f"$h^{{{int(p)}}}$ reference")
    ax.set_xlabel("cell size $h$")
    ax.set_ylabel("error of the cell average")
    ax.set_title("local vs global order, periodic entropy wave")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    rows_mc = ladders["mc"]
    row = next(r for r in rows_mc if r["nx"] == 160)
    row_none = next(r for r in ladders["none"] if r["nx"] == 160)
    x = (np.arange(160) + 0.5) / 160
    ax2.plot(x, np.array(row["error"]), color=colours["mc"],
             label="mc (TVD, clipped at the crest)")
    ax2.plot(x, np.array(row_none["error"]), color=colours["none"],
             label="none (unlimited centred slope)")
    crest = row["crest_x"]
    ax2.axvspan(crest - lx.WINDOW_HALF / 160, crest + lx.WINDOW_HALF / 160,
                color="0.85", zorder=0, label=f"crest window ({2*lx.WINDOW_HALF+1} cells)")
    ax2.axvline(crest, color="0.4", lw=0.8, ls="--")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"$\rho_h - \overline{\rho}$")
    ax2.set_title("error field, $n_x = 160$, $t = %.2f$" % row["t"])
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {"path": str(out.relative_to(ROOT)), "sha256": sha(out),
            "crest_x_nx160": crest,
            "crest_linf_mc_nx160": row["crest_linf"],
            "crest_linf_none_nx160": row_none["crest_linf"]}


def main() -> int:
    ladders, phase_ladder, seconds = build_ladders()
    for name, rows in ladders.items():
        log(f"[{name}] crest_linf: " +
            " ".join(f"{r['nx']}:{r['crest_linf']:.3e}" for r in rows))
    payload = summarize(ladders, phase_ladder)
    fig = figure(ladders, HERE / "fig_crest_degradation.png")
    study = {
        "anchor": "XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 2),
        "criteria_declared": lx.CRITERIA,
        "layout": {"grids": list(lx.GRIDS), "triplets": [list(t) for t in lx.TRIPLETS],
                   "window_half": lx.WINDOW_HALF,
                   "limiters": list(lx.LIMITERS),
                   "midcell_phase": lx.MIDCELL_PHASE,
                   "order": 2, "cfl": 0.4, "t_end": 0.2,
                   "amplitude": 0.2, "velocity": 0.5, "wavenumber": 2 * np.pi},
        "ladders": {name: {"rows": [{k: v for k, v in r.items() if k != "error"}
                                    for r in rows]}
                    for name, rows in ladders.items()},
        "studies": payload["studies"],
        "phase_study": payload["phase_study"],
        "crest_magnitude_ratio": payload["ratios"],
        "gate": payload["gate"],
        "figure": fig,
        "provenance": provenance(),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "study.json"
    path.write_text(json.dumps(study, indent=2, allow_nan=False, default=float) + "\n")
    log(json.dumps({"sha256": sha(path), "path": str(path.relative_to(ROOT)),
                    "seconds": study["seconds"]}, indent=2))
    for rec in study["gate"]:
        log(f"[gate] {rec['criterion']}: {'PASS' if rec['passed'] else 'FAIL'} "
            f"{json.dumps(rec['measured'], default=float)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())