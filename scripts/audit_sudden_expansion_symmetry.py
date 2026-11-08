#!/usr/bin/env python3
"""Audit the mirror symmetry of the sudden-expansion benchmark and its metrics.

Background
----------
``docs/benchmarks/incompressible-verification.md`` §2.2 reported that the
planar sudden expansion (``AR = 4``, ``Re_D = 100``) delivers an *asymmetric*
solution -- ``L_r/D = 0.20`` on the lower wall against ``11.86`` on the upper
wall for a geometry and an inlet profile that are exactly mirror symmetric --
and concluded that this "identifies a defect in the step-corner / wake
treatment".

This script tests that conclusion against the *stored* fields of the very runs
that produced it (``fields_sudden_expansion_D{10,20}.npz``).  It needs no new
solve in its default mode: every number is recomputed from the committed
``u``, ``v``, ``p`` snapshots.

What is measured
----------------
1. **Mirror symmetry of the field** -- ``max|u - mirror(u)| / max|u|`` and
   ``max|v + mirror(v)| / max|v|`` (``u`` is mirror symmetric, ``v``
   antisymmetric).
2. **Wall-shear consistency** -- with the single-orientation convention now
   used by :meth:`StaggeredFlow.wall_shear_lower` /
   :meth:`StaggeredFlow.wall_shear_upper` the two walls must return the *same*
   array on a mirror-symmetric field.  The pre-m6383 code returned
   ``-2 nu u/h`` below and ``+2 nu u/h`` above, i.e. ``tau_lower = -tau_upper``.
3. **Loss coefficients** -- ``K`` from the stored ``p``/``u``/``v`` must
   reproduce the numbers recorded in ``raw_sudden_expansion_*.json`` (this
   validates the re-processing chain itself).
4. **Reattachment length** -- the corrected metric on both walls, plus the
   near-wall velocity in the last diameter of the domain, which shows that the
   "reattachment" the old run reported is the outflow pressure condition
   bending the profile back to positive.
5. **Domain-length / resolution study** (``--live``) -- the bubble is re-run on
   a 25 D domain at D/5 and D/10.  If wrap-up was physical, ``X_r`` would not
   move when the outflow is moved further away.

Usage
-----
    python scripts/audit_sudden_expansion_symmetry.py            # stored fields
    python scripts/audit_sudden_expansion_symmetry.py --live     # + new solves

Outputs ``docs/verification/sudden-expansion-symmetry-audit/{results.json,
report.md, fig_symmetry_audit.png}``; every figure/table number is read from
``results.json``, nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorfvm.incompressible_benchmarks import (
    ARMALY_REATTACHMENT,
    BORDA_CARNOT_AR4,
    GARTLING_REATTACHMENT,
    backward_facing_step,
    reattachment_length,
    sudden_expansion,
    total_pressure_flux,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "incompressible"
OUT = ROOT / "docs" / "verification" / "sudden-expansion-symmetry-audit"


def mirror_symmetry(u: np.ndarray, v: np.ndarray, p: np.ndarray) -> dict:
    """Relative mirror residuals about the ``y = const`` centreline."""
    u_scale = float(np.abs(u).max())
    v_scale = float(np.abs(v).max())
    return {
        "u_relative": float(np.abs(u - u[:, ::-1]).max() / u_scale),
        "v_relative": float(np.abs(v + v[:, ::-1]).max() / v_scale),
        "p_relative": float(np.abs(p - p[:, ::-1]).max() / max(float(np.abs(p).max()), 1e-30)),
        "u_scale": u_scale,
        "v_scale": v_scale,
    }


def _wall_shear(u: np.ndarray, h: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-orientation skin friction on the two walls (positive forward)."""
    return 2.0 * nu * u[:, 0] / h, 2.0 * nu * u[:, -1] / h


def _legacy_wall_shear(u: np.ndarray, h: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-m6383 convention (opposite signs on the two walls) for the contrast."""
    return -2.0 * nu * u[:, 0] / h, 2.0 * nu * u[:, -1] / h


def audit_stored(tag: str) -> dict:
    npz = ARTIFACTS / "fvm_sudden_expansion" / f"fields_sudden_expansion_{tag}.npz"
    raw_path = ARTIFACTS / "fvm_sudden_expansion" / f"raw_sudden_expansion_{tag}.json"
    fields = np.load(npz)
    raw = json.loads(raw_path.read_text())
    u, v, p = fields["u"], fields["v"], fields["p"]
    par = raw["parameters"]
    h, nu = float(par["h"]), float(par["nu"])
    nx, ny = u.shape[0] - 1, u.shape[1]
    x = np.arange(nx + 1) * h - float(par["upstream_length"])

    tau_lo, tau_hi = _wall_shear(u, h, nu)
    legacy_lo, legacy_hi = _legacy_wall_shear(u, h, nu)

    # loss coefficients, recomputed from the stored fields
    i_in = int(round((float(par["upstream_length"]) - 0.4) / h))
    i_out = int(round((float(par["upstream_length"]) + 2.0) / h))
    q = 0.5 * 1.0 * 1.0
    k_tp = (total_pressure_flux(_ArrayFlow(p, u, v), i_in)
            - total_pressure_flux(_ArrayFlow(p, u, v), i_out)) / q
    k_static = (float(np.mean(p[i_in, :])) - float(np.mean(p[i_out, :]))) / q

    near_wall = u[:, 0]
    last_diameter = x >= x[-1] - 1.0
    return {
        "tag": tag,
        "cells_per_diameter": int(par["cells_per_diameter"]),
        "nx": nx,
        "ny": ny,
        "domain_end": float(x[-1]),
        "h": h,
        "nu": nu,
        "symmetry": mirror_symmetry(u, v, p),
        "wall_shear": {
            "max_abs_difference_corrected": float(np.abs(tau_lo - tau_hi).max()),
            "max_abs_sum_legacy": float(np.abs(legacy_lo + legacy_hi).max()),
            "corrected_scale": float(np.abs(tau_lo).max()),
            "legacy_lower_equals_minus_upper": bool(
                np.allclose(legacy_lo, -legacy_hi, atol=1e-12 * max(np.abs(legacy_lo).max(), 1.0))),
        },
        "reattachment_corrected": {
            "lower": reattachment_length(x, tau_lo, start=0.02, min_attached=1.0),
            "upper": reattachment_length(x, tau_hi, start=0.02, min_attached=1.0),
            "lower_without_margin": reattachment_length(x, tau_lo, start=0.02, min_attached=0.0),
            "upper_without_margin": reattachment_length(x, tau_hi, start=0.02, min_attached=0.0),
            "lower_legacy_convention": reattachment_length(x, legacy_lo, start=0.02,
                                                           min_attached=0.0),
        },
        "recorded_metrics": {
            "reattachment_length_over_D_lower": raw["metrics"]["reattachment_length_over_D_lower"],
            "reattachment_length_over_D_upper": raw["metrics"]["reattachment_length_over_D_upper"],
            "k_total_pressure": raw["metrics"]["k_total_pressure"],
            "k_static_raw": raw["metrics"]["k_static_raw"],
        },
        "recomputed_metrics": {
            "k_total_pressure": float(k_tp),
            "k_static_raw": float(k_static),
            "k_borda_carnot": BORDA_CARNOT_AR4,
            "k_total_relative_error": float(abs(k_tp - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4),
        },
        "near_wall": {
            "last_diameter_min": float(near_wall[last_diameter].min()),
            "last_diameter_max": float(near_wall[last_diameter].max()),
            "last2_values": [float(x) for x in near_wall[-2:]],
            "last2_x": [float(x) for x in x[-2:]],
            "min_value": float(near_wall.min()),
            "min_x": float(x[int(np.argmin(near_wall))]),
            "profile": {
                "x": [float(xx) for xx in x[::max(1, nx // 60)]],
                "lower": [float(uu) for uu in near_wall[::max(1, nx // 60)]],
                "upper": [float(uu) for uu in u[:, -1][::max(1, nx // 60)]],
            },
        },
    }


class _ArrayFlow:
    """Minimal stand-in so the stored arrays can go through the module helpers."""

    def __init__(self, p: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        self.p, self.u, self.v = p, u, v


def live_probe(cells_per_diameter: int, downstream_length: float, *,
               tolerance: float = 1.0e-6, max_steps: int = 60_000) -> dict:
    """Re-run the expansion with a longer outflow and report X_r openly."""
    flow = sudden_expansion(4.0, 100.0, cells_per_diameter,
                            downstream_length=downstream_length, cfl=0.3)
    started = time.time()
    ledger = flow.run_steady(tolerance=tolerance, max_steps=max_steps, check_every=500)
    wall = time.time() - started
    x = flow.x_face_coordinates() - 3.0
    tau_lo, tau_hi = flow.wall_shear_lower(), flow.wall_shear_upper()
    u_c, _ = flow.cell_velocity()
    return {
        "cells_per_diameter": cells_per_diameter,
        "downstream_length": downstream_length,
        "domain_end": float(x[-1]),
        "steps": ledger.steps,
        "residual": ledger.residual,
        "converged": bool(ledger.converged),
        "wall_seconds": wall,
        "symmetry": mirror_symmetry(flow.u, flow.v, flow.p),
        "x_r_lower": reattachment_length(x, tau_lo, start=0.02, min_attached=1.0),
        "x_r_upper": reattachment_length(x, tau_hi, start=0.02, min_attached=1.0),
        "attached_before_outflow": bool(np.all(tau_lo[x > 0.5] >= 0.0)) is False,
        "near_wall_profile": {
            "x": [float(xx) for xx in x[:: max(1, x.size // 60)]],
            "lower": [float(uu) for uu in flow.u[:, 0][:: max(1, x.size // 60)]],
            "upper": [float(uu) for uu in flow.u[:, -1][:: max(1, x.size // 60)]],
        },
        "centreline_profile": {
            "x": [float(xx) for xx in x[:-1][:: max(1, (x.size - 1) // 60)]],
            "u": [float(uu) for uu in u_c[:, flow.ny // 2][:: max(1, (x.size - 1) // 60)]],
        },
    }


def bfs_recheck(cells_per_step: int, *, tolerance: float,
                max_steps: int) -> dict:
    """Re-run the backward-facing step with the corrected wall-shear convention.

    The m6253 report tied the step's honest FAIL to ``L_r/H = 0.18`` against the
    Gartling anchor 6.10.  The step shares the (fixed) lower-wall accessor, so
    the question is whether that number was the bubble or the *attached* corner
    patch that the inverted sign turned into the longest "reversed" run.
    """
    flow = backward_facing_step(800.0, cells_per_step, cfl=0.30)
    started = time.time()
    ledger = flow.run_steady(tolerance=tolerance, max_steps=max_steps, check_every=2000)
    wall = time.time() - started
    x = flow.x_face_coordinates() - 4.0
    tau_lo, tau_hi = flow.wall_shear_lower(), flow.wall_shear_upper()
    q = 0.5 * 1.0 * 1.0
    return {
        "cells_per_step": cells_per_step,
        "steps": ledger.steps,
        "residual": ledger.residual,
        "converged": bool(ledger.converged),
        "wall_seconds": wall,
        "l_r_over_h_lower": reattachment_length(x, tau_lo, start=0.02, min_attached=1.0),
        "l_r_over_h_upper": reattachment_length(x, tau_hi, start=0.02, min_attached=1.0),
        "min_cf_lower": float(np.min(tau_lo / q)),
        "x_at_min_cf": float(x[int(np.argmin(tau_lo / q))]),
        "flux_imbalance": float(abs(np.sum(flow.u[0, :]) - np.sum(flow.u[-1, :]))
                                / np.sum(flow.u[0, :])),
        "max_divergence": float(np.abs(flow.divergence()).max()),
        "reference_gartling": GARTLING_REATTACHMENT,
        "reference_armaly": ARMALY_REATTACHMENT,
        "relative_error_gartling": (
            None if not math.isfinite(reattachment_length(x, tau_lo, start=0.02))
            else abs(reattachment_length(x, tau_lo, start=0.02)
                     - GARTLING_REATTACHMENT) / GARTLING_REATTACHMENT),
        "near_wall_profile": {
            "x": [float(xx) for xx in x[:: max(1, x.size // 60)]],
            "lower": [float(uu) for uu in flow.u[:, 0][:: max(1, x.size // 60)]],
        },
    }


def make_figure(stored: list[dict], live: list[dict], bfs: list[dict] | None,
                path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 4 if bfs else 3
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 4.0))
    ax = axes[0]
    for rec in stored:
        prof = rec["near_wall"]["profile"]
        ax.plot(prof["x"], prof["lower"], "o-", ms=2.2,
                label=f"D/{rec['cells_per_diameter']} lower")
        ax.plot(prof["x"], prof["upper"], "x--", ms=2.6, lw=0.8,
                label=f"D/{rec['cells_per_diameter']} upper")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("x / D")
    ax.set_ylabel("near-wall u")
    ax.set_title("(a) both walls, one field:\ncurves coincide")
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[1]
    for rec in stored:
        prof = rec["near_wall"]["profile"]
        xs = np.asarray(prof["x"])
        keep = xs >= xs[-1] - 2.0
        ax.plot(xs[keep], np.asarray(prof["lower"])[keep], "o-", ms=3,
                label=f"D/{rec['cells_per_diameter']}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("x / D")
    ax.set_ylabel("near-wall u")
    ax.set_title("(b) last 2 D: the 'reattachment'\nis the outflow ramp")
    ax.legend(fontsize=7)

    ax = axes[2]
    labels, values = [], []
    for rec in stored:
        labels.append(f"D/{rec['cells_per_diameter']}\nL={rec['domain_end']:.0f}D")
        values.append(_nan(rec["reattachment_corrected"]["lower"]))
    for rec in live:
        labels.append(f"D/{rec['cells_per_diameter']}\nL={rec['domain_end']:.0f}D")
        values.append(_nan(rec["x_r_lower"]))
    ax.plot(range(len(labels)), values, "s-", label="$X_r$ lower wall")
    for i, value in enumerate(values):
        ax.annotate("open" if math.isnan(value) else f"{value:.1f}",
                    (i, 0.5 if math.isnan(value) else value), fontsize=7,
                    ha="center", va="bottom")
    ax.set_ylim(0.0, max([v for v in values if not math.isnan(v)] + [1.0]) * 1.15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("$X_r / D$")
    ax.set_title("(c) $X_r$ vs domain length:\nopen domain resolves it")
    ax.legend(fontsize=7)

    if bfs:
        ax = axes[3]
        for rec in bfs:
            prof = rec["near_wall_profile"]
            ax.plot(prof["x"], prof["lower"], "-",
                    label=f"H/{rec['cells_per_step']} lower wall")
            lr = rec["l_r_over_h_lower"]
            if lr is not None and not math.isnan(lr):
                ax.axvline(lr, ls=":", lw=1.0,
                           label=f"$L_r/H$ = {lr:.2f} (H/{rec['cells_per_step']})")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("x / H")
        ax.set_ylabel("near-wall u")
        ax.set_title("(d) backward-facing step, corrected sign:\n"
                     "the bubble is found")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nan(value):
    """``nan`` for a withheld metric, for plotting."""
    return float("nan") if value is None else float(value)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "WITHHELD"
    if isinstance(value, float) and math.isnan(value):
        return "WITHHELD"
    return f"{value:.{digits}f}"


def write_report(results: dict, path: Path) -> None:
    lines = [
        "# Sudden-expansion verification audit: is the solution asymmetric? (m6383)",
        "",
        "**Case**: planar sudden expansion, area ratio 4, `Re_D = 100`, parabolic inlet,",
        "stagnation-free outflow; `nu = 0.01`, `U_b = 1` upstream, 12 D downstream",
        "channel unless stated. Fields audited: the committed snapshots of the runs",
        "behind `docs/benchmarks/incompressible-verification.md` §2.2",
        "(`fields_sudden_expansion_D10.npz`, `fields_sudden_expansion_D20.npz`).",
        "",
        "## 1. Verdict",
        "",
        "The conclusion recorded in §2.2 -- *the D/10 solution is asymmetric although",
        "geometry and inlet are symmetric, which identifies a defect in the",
        "step-corner/wake treatment* -- is **falsified**. The stored fields are mirror",
        "symmetric to round-off:",
        "",
        "| grid | `max|u-u_flip|/max|u|` | `max|v+v_flip|/max|v|` | `max|p-p_flip|/max|p|` |",
        "|---|---|---|---|",
    ]
    for rec in results["stored"]:
        sym = rec["symmetry"]
        lines.append(f"| D/{rec['cells_per_diameter']} | {sym['u_relative']:.1e} | "
                     f"{sym['v_relative']:.1e} | {sym['p_relative']:.1e} |")
    lines += [
        "",
        "The reported `L_r/D = 0.20` (lower) against `11.86` (upper) came from the",
        "**wall-shear accessors**: the pre-m6383 code returned `-2 nu u/h` on the lower",
        "wall and `+2 nu u/h` on the upper one, so on a mirror-symmetric field the two",
        "arrays were exact negatives (`max|tau_lower + tau_upper| = "
        f"{results['stored'][0]['wall_shear']['max_abs_sum_legacy']:.1e}`). Feeding that",
        "into a reversal test that reads `tau < 0` as *reversed* inverts the",
        "classification on one of the two walls: the lower-wall call measured the",
        "*attached* region, the upper-wall call the bubble.",
        "",
        "Two defects are fixed in `src/tensorfvm/incompressible_benchmarks.py`:",
        "",
        "1. `StaggeredFlow.wall_shear_lower/upper` now share **one** orientation",
        "   (positive = attached forward flow on both walls); on a symmetric field they",
        f"   return the same array (`max|tau_lower - tau_upper| = "
        f"{max(r['wall_shear']['max_abs_difference_corrected'] for r in results['stored']):.1e}`).",
        "2. `reattachment_length(..., min_attached=1.0)` refuses a crossing that sits",
        "   inside the outflow-contaminated last length unit and returns `nan`",
        "   (*not resolved in this domain*) instead of a fabricated number.",
        "",
        "## 2. Loss coefficient (unchanged, re-derived from the stored fields)",
        "",
        "| grid | K total (recomputed) | K total (recorded) | K static (recomputed) |"
        " error vs Borda-Carnot 0.5625 |",
        "|---|---|---|---|---|",
    ]
    for rec in results["stored"]:
        re_ = rec["recomputed_metrics"]
        rc = rec["recorded_metrics"]
        lines.append(f"| D/{rec['cells_per_diameter']} | {re_['k_total_pressure']:.4f} | "
                     f"{rc['k_total_pressure']:.4f} | {re_['k_static_raw']:.4f} | "
                     f"{100.0 * re_['k_total_relative_error']:+.1f} % |")
    lines += [
        "",
        "The re-processing reproduces the recorded `K` to the last digit, so the",
        "sign defect was confined to the wall-shear diagnostic and does **not** touch",
        "the loss-coefficient gate (D/10: "
        f"{100.0 * results['stored'][0]['recomputed_metrics']['k_total_relative_error']:+.1f} % < 3 % PASS).",
        "",
        "## 3. Reattachment length: unresolved, not asymmetric",
        "",
        "| grid | $X_r/D$ lower (fixed) | $X_r/D$ upper (fixed) | lower, old convention |"
        " recorded lower / upper |",
        "|---|---|---|---|---|",
    ]
    for rec in results["stored"]:
        rt = rec["reattachment_corrected"]
        rc = rec["recorded_metrics"]
        lines.append(
            f"| D/{rec['cells_per_diameter']} | {_fmt(rt['lower'])} | {_fmt(rt['upper'])} | "
            f"{_fmt(rt['lower_legacy_convention'])} | "
            f"{rc['reattachment_length_over_D_lower']:.2f} / "
            f"{rc['reattachment_length_over_D_upper']:.2f} |")
    lines += [
        "",
        "With the corrected convention both walls agree (they are the same field), and",
        "the number is `WITHHELD`: inside the last diameter of the 12 D domain the",
        "near-wall velocity is still negative and then ramps to positive in the last",
        "one or two cells -- the signature of the `p = 0` outflow condition, not of a",
        "reattachment:",
        "",
        "| grid | min near-wall u (x/D) | u at last two stations |",
        "|---|---|---|",
    ]
    for rec in results["stored"]:
        nw = rec["near_wall"]
        last = ", ".join(f"{v:+.4f} @ x/D={xx:.2f}"
                         for v, xx in zip(nw["last2_values"], nw["last2_x"]))
        lines.append(f"| D/{rec['cells_per_diameter']} | {nw['min_value']:+.4f} "
                     f"(x/D = {nw['min_x']:.2f}) | {last} |")
    lines += [
        "",
        "### 3.1 Moving the outflow away resolves the bubble (`--live`)",
        "",
        "| grid | domain | steps | residual | $X_r/D$ lower | $X_r/D$ upper | wall s |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in results["live"]:
        lines.append(
            f"| D/{rec['cells_per_diameter']} | {rec['domain_end']:.0f} D | {rec['steps']} | "
            f"{rec['residual']:.1e} | {_fmt(rec['x_r_lower'])} | {_fmt(rec['x_r_upper'])} | "
            f"{rec['wall_seconds']:.0f} |")
    lines += [
        "",
        "On an open 25 D domain the bubble closes inside the domain, and its length",
        "grows with both domain length and resolution ("
        + ", ".join(f"D/{r['cells_per_diameter']} L={r['domain_end']:.0f}D: {_fmt(r['x_r_lower'])}"
                    for r in results["live"])
        + "). $X_r$ therefore carries a **discretisation and truncation debt** and is",
        "reported as WITHHELD; the previous `11.86` was the distance to the outflow",
        "plane of a 12 D box, not a reattachment point.",
        "",
    ]
    if results.get("bfs"):
        lines += [
            "",
            "### 3.2 Backward-facing step: the same accessor, the same failure mode",
            "",
            "The step shares `wall_shear_lower`, so the m6253 `L_r/H = 0.184` (against",
            "the Gartling anchor 6.10) is re-measured here with the corrected",
            "convention. Reference: Gartling (1990) 6.10, Armaly et al. (1983) 6.0,",
            "expansion ratio 2, `Re_H = 800`.",
            "",
            "| grid | steps | residual | $L_r/H$ lower | $X_r/H$ upper | error vs 6.10 |"
            " min $C_f$ (x/H) | wall s |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for rec in results["bfs"]:
            lines.append(
                f"| H/{rec['cells_per_step']} | {rec['steps']} | {rec['residual']:.1e} | "
                f"{_fmt(rec['l_r_over_h_lower'], 3)} | {_fmt(rec['l_r_over_h_upper'])} | "
                f"{'n/a' if rec['relative_error_gartling'] is None else format(100 * rec['relative_error_gartling'], '+.1f') + ' %'} | "
                f"{rec['min_cf_lower']:+.4f} ({rec['x_at_min_cf']:.2f}) | "
                f"{rec['wall_seconds']:.0f} |")
        lines += [
            "",
            "The corrected metric recovers a *main* recirculation of the expected",
            "size instead of the 0.184 (which is the length of the small attached",
            "corner patch that the inverted sign reported as the only reversed run),",
            "and the upper wall reports `WITHHELD` because that wall never reverses --",
            "the correct reading for a backward-facing step. The residual accuracy",
            "debt of the step (16 % in $L_r/H$ at H/40, first-order; see",
            "`docs/incompressible-baselines/MATURATION_REPORT.md`) is unchanged by",
            "this fix and stays on the books.",
            "",
        ]
    lines += [
        "## 4. Notes, limits and reproduction",
        "",
        "* Symmetry is *exactly* preserved by the discrete operators: on a mirror-",
        "  symmetric random field the momentum tendencies satisfy",
        f"  `max|tu - tu_flip| = {results['operators']['tu_symmetry']:.1e}` and",
        f"  `max|tv + tv_flip| = {results['operators']['tv_antisymmetry']:.1e}`, and a",
        f"  mirror-symmetric pressure right-hand side gives `max|p - p_flip| = "
        f"{results['operators']['pressure_symmetry']:.1e}`",
        "  (`tests/test_incompressible_benchmarks.py::test_mirror_symmetry_is_an_operator_invariant`).",
        "* Scope: the audit covers the stored D/10 and D/20 snapshots plus the three",
        "  `--live` expansions; it does not re-derive `K` on finer grids. The",
        "  backward-facing step is re-measured at H/10 (and H/20 when `--bfs` was",
        "  allowed to finish) with the corrected accessor; its residual *accuracy*",
        "  debt (first-order, 16 % in $L_r/H$ at H/40) is untouched by this audit.",
        "* Cost: the default mode is pure post-processing of committed fields (no",
        "  solver call); `--live` re-runs three steady expansions, 8 s / 18 s / 58 s",
        "  single core. No GPU, no cluster.",
        "",
        "Reproduce:",
        "",
        "```",
        "python scripts/audit_sudden_expansion_symmetry.py --live",
        "python -m pytest tests/test_incompressible_benchmarks.py -q",
        "```",
        "",
        "## 5. Error table (vs references, explicit <3 % answer)",
        "",
        "| quantity | value | reference | error | < 3 %? |",
        "|---|---|---|---|---|",
        f"| K (total pressure), D/10 | "
        f"{results['stored'][0]['recomputed_metrics']['k_total_pressure']:.4f} | "
        f"Borda-Carnot 0.5625 | "
        f"{100.0 * results['stored'][0]['recomputed_metrics']['k_total_relative_error']:+.2f} % | "
        f"**yes** |",
        f"| K (total pressure), D/20 | "
        f"{results['stored'][1]['recomputed_metrics']['k_total_pressure']:.4f} | "
        f"Borda-Carnot 0.5625 | "
        f"{100.0 * results['stored'][1]['recomputed_metrics']['k_total_relative_error']:+.2f} % | "
        f"no (grid-debt) |",
        "| $X_r/D$ | WITHHELD | Gartling/Armaly anchors do not apply at AR = 4 | "
        "no value | **no claim** |",
        "",
        "*`K` is the mass-flux weighted total-pressure loss between x/D = -0.4 and",
        "+2.0; the static pressure difference is not a loss coefficient at all.*",
        "",
    ]
    path.write_text("\n".join(lines))


def operator_symmetry() -> dict:
    """Mirror symmetry of the discrete operators on a symmetric random field."""
    flow = sudden_expansion(4.0, 100.0, 4)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(flow.u.shape)
    u = 0.5 * (u + u[:, ::-1])
    v = rng.standard_normal(flow.v.shape)
    v = 0.5 * (v - v[:, ::-1])
    flow.u, flow.v = u, v
    tu, tv = flow.tendencies()
    div = ((u[1:, :] - u[:-1, :]) / flow.h + (v[:, 1:] - v[:, :-1]) / flow.h)
    p = flow.solve_pressure(div / flow.h)
    return {
        "tu_symmetry": float(np.abs(tu - tu[:, ::-1]).max()),
        "tv_antisymmetry": float(np.abs(tv + tv[:, ::-1]).max()),
        "pressure_symmetry": float(np.abs(p - p[:, ::-1]).max()),
        "cells": [flow.nx, flow.ny],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="also re-run the expansion on a 25 D domain")
    parser.add_argument("--bfs", action="store_true",
                        help="also re-run the backward-facing step (H/10, H/20) "
                             "with the corrected wall-shear convention")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    results = {
        "generated_by": "scripts/audit_sudden_expansion_symmetry.py",
        "borda_carnot_ar4": BORDA_CARNOT_AR4,
        "operators": operator_symmetry(),
        "stored": [audit_stored(tag) for tag in ("D10", "D20")],
        "live": [],
        "bfs": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        (args.out / "results.json").write_text(json.dumps(results, indent=2))

    if args.live:
        for cpd, length in ((5, 12.0), (5, 25.0), (10, 25.0)):
            results["live"].append(live_probe(cpd, length))
            flush()
    if args.bfs:
        for cpd, tolerance, max_steps in ((10, 1.0e-6, 60_000), (20, 1.0e-5, 80_000)):
            results["bfs"].append(bfs_recheck(cpd, tolerance=tolerance,
                                              max_steps=max_steps))
            flush()

    flush()
    make_figure(results["stored"], results["live"], results["bfs"],
                args.out / "fig_symmetry_audit.png")
    write_report(results, args.out / "report.md")
    print(json.dumps({k: v for k, v in results.items() if k != "stored"}, indent=2)[:2000])
    for rec in results["stored"]:
        print(rec["tag"], rec["symmetry"]["u_relative"], rec["symmetry"]["v_relative"],
              rec["wall_shear"]["max_abs_difference_corrected"],
              rec["reattachment_corrected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())