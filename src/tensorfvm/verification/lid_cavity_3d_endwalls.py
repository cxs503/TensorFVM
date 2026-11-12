"""Verification driver: 3-D lid-driven cavity with spanwise **end walls**.

Closes the open half of ``docs/verification-catalog.md`` §10.1 item 1 (the
periodic-span 3-D entry punts on the end-wall geometry: *"no reference table in
this repository and no anchor here"*).  Seven pre-declared criteria, in the
order they can falsify the implementation:

C1 exact 2-D reduction with **free-slip** end walls (the span must be inert);
C2 negative control: **no-slip** end walls must change the answer (a BC that
   does nothing is not applied);
C3 mid-plane symmetry of the discrete solution (catches index/branch bugs --
   it is what exposed the inherited off-by-one cell in the wall MUSCL branch);
C4 divergence-free projection + end-wall faces closed;
C5 span family ``L_z = 1, 2, 4`` at Re = 100: the mid-plane centreline error
   against Ghia et al. (1982) must fall monotonically and extrapolate
   (``1/L_z -> 0``) into the 2-D band;
C6 grid refinement at ``L_z = 1``: the end-wall deficit is physical, not a
   discretisation artefact (it must be grid-stable);
C7 a span-modulated lid excites ``w != 0`` while keeping the mid-plane
   antisymmetry and the projection divergence-free.

The absent reference is stated instead of faked: there is **no** Ku/Hirsh/Taylor
table in this repository, so absolute accuracy of the end-wall geometry is
WITHHELD.  C5 is the strongest quantitative statement available offline.

Usage::

    PYTHONPATH=src python -m tensorfvm.verification.lid_cavity_3d_endwalls
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..lid_cavity import LidCavityConfig, LidCavitySolver
from ..lid_cavity_3d_endwalls import (
    LidCavity3DEndWallConfig,
    LidCavity3DEndWallSolver,
    score_endwall_against_ghia,
)

__all__ = ["CRITERIA", "run"]

REYNOLDS = 100.0

CRITERIA = {
    "C1_free_slip_reduction": {"tolerance": 1e-12, "unit": "max|du|,|dv| (lid speeds)"},
    "C2_no_slip_active": {"threshold": 1e-2, "unit": "max|du| vs 2-D"},
    "C3_midplane_symmetry": {"tolerance": 1e-12, "unit": "relative"},
    "C4_divergence_free": {"tolerance": 1e-12, "unit": "max|div u| h/U"},
    "C5_span_family_ghia": {"tolerance": 3e-2, "unit": "max|du|/U vs Ghia at L_z=4"},
    "C6_grid_stability": {"tolerance": 5e-2, "unit": "relative change of deficit n=24->32"},
    "C7_modulated_lid": {"threshold": 1e-3, "tolerance": 1e-10, "unit": "max|w|, asymmetry"},
}


def _steady(n_x: int, n_z: int, span_bc: str = "no_slip", **kw) -> LidCavity3DEndWallSolver:
    cfg = LidCavity3DEndWallConfig(n_x=n_x, n_z=n_z, reynolds=REYNOLDS,
                                   span_bc=span_bc, tolerance=1e-6,
                                   check_every=250, **kw)
    solver = LidCavity3DEndWallSolver(cfg)
    solver.run()
    return solver


def _deficit(solver: LidCavity3DEndWallSolver, two_d) -> float:
    """Relative end-wall deficit of the mid-span centreline ``u`` peak."""
    _, u3 = solver.centreline_u(mid_span=True)
    _, u2 = two_d.centreline_u()
    peak2 = float(u2.min())
    peak3 = float(u3.min())
    return float((peak3 - peak2) / peak2), peak3, peak2


def run(out_dir: str, n_span: int = 24, n_grid: tuple[int, ...] = (16, 24, 32),
        quick: bool = False) -> dict:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    t0 = time.time()
    summary: dict = {"reynolds": REYNOLDS, "criteria": {}, "runs": {}}

    # ---- C1/C2: exact reduction (free-slip) and negative control (no-slip) ----
    n_red, nz_red, dt, steps = 16, 4, 0.005, 200
    two_d = LidCavitySolver(LidCavityConfig(n=n_red, reynolds=REYNOLDS, time_step=dt))
    for _ in range(steps):
        two_d.step(dt)
    reductions = {}
    for bc in ("free_slip", "no_slip"):
        s = LidCavity3DEndWallSolver(LidCavity3DEndWallConfig(
            n_x=n_red, n_z=nz_red, reynolds=REYNOLDS, time_step=dt, span_bc=bc))
        for _ in range(steps):
            s.step(dt)
        reductions[bc] = {
            "du_vs_2d": float((s.u[:, :, 0] - two_d.u).abs().max()),
            "dv_vs_2d": float((s.v[:, :, 0] - two_d.v).abs().max()),
            "dp_vs_2d": float((s.p[:, :, 0] - two_d.p).abs().max()),
            "endwall_speed": s.endwall_speed(),
            "divergence": float(s.divergence().abs().max()),
            "wall_face_leak": s.wall_face_leak(),
        }
    summary["runs"]["reduction"] = {"n_x": n_red, "n_z": nz_red, "dt": dt, "steps": steps,
                                    "by_span_bc": reductions}
    free = reductions["free_slip"]
    c1 = max(free["du_vs_2d"], free["dv_vs_2d"])
    summary["criteria"]["C1_free_slip_reduction"] = {
        "value": c1, "tolerance": 1e-12, "pass": c1 <= 1e-12,
        "w_is_zero": free["endwall_speed"] <= 1e-14,
        "p_delta": free["dp_vs_2d"],
    }
    c2 = reductions["no_slip"]["du_vs_2d"]
    summary["criteria"]["C2_no_slip_active"] = {
        "value": c2, "threshold": 1e-2, "pass": c2 > 1e-2,
        "endwall_speed": reductions["no_slip"]["endwall_speed"],
    }

    # ---- C3/C4: steady no-slip cube state, symmetry and projection ----
    cube = _steady(16, 16)
    asym = cube.midplane_asymmetry()
    div = float(cube.divergence().abs().max())
    c3 = max(asym["u"], asym["v"])
    summary["runs"]["cube"] = {
        "n_x": cube.n_x, "n_z": cube.n_z, "span_length": cube.span_length,
        "steps": cube.steps, "residual": cube.residual,
        "endwall_speed": cube.endwall_speed(), "wall_face_leak": cube.wall_face_leak(),
        "midplane": asym, "divergence": div,
    }
    summary["criteria"]["C3_midplane_symmetry"] = {
        "value": c3, "tolerance": 1e-12, "tolerance_w": 1e-14,
        "pass": c3 <= 1e-12 and asym["w_midplane"] <= 1e-14}
    summary["criteria"]["C4_divergence_free"] = {
        "value": div, "tolerance": 1e-12,
        "pass": div <= 1e-12 and cube.wall_face_leak() == 0.0}

    # ---- C5: span family -> 2-D limit (Ghia band) ----
    spans = (1, 2) if quick else (1, 2, 4)
    family = []
    for span in spans:
        s = _steady(n_span, span * n_span)
        block = score_endwall_against_ghia(s, reynolds=REYNOLDS, mid_span=True)
        family.append({
            "span_length": s.span_length, "n_z": s.n_z, "steps": s.steps,
            "residual": s.residual, "endwall_speed": s.endwall_speed(),
            "err_u": block["u"]["max_abs_error_consistent"],
            "rms_u": block["u"]["rms_error_consistent"],
            "err_v": block["v"]["max_abs_error_consistent"],
            "rms_v": block["v"]["rms_error_consistent"],
            "divergence": block["max_divergence"],
            "centreline": {"y": block["u"]["profiles"]["coord"],
                           "computed": block["u"]["profiles"]["computed"],
                           "reference": block["u"]["profiles"]["reference"]},
        })
        print(f"  span L_z={s.span_length:4.1f}  u max err {block['u']['max_abs_error_consistent']:.4f}"
              f"  v max err {block['v']['max_abs_error_consistent']:.4f}", flush=True)
    summary["runs"]["span_family"] = family
    errs = [r["err_u"] for r in family]
    monotone = all(b <= a + 1e-12 for a, b in zip(errs, errs[1:]))
    if len(family) >= 3:
        # linear extrapolation of the error in 1/L_z to 1/L_z = 0
        inv = [1.0 / r["span_length"] for r in family]
        extrap = errs[-1] + (errs[-1] - errs[-2]) * (0.0 - inv[-1]) / (inv[-1] - inv[-2])
    else:
        extrap = errs[-1]
    summary["criteria"]["C5_span_family_ghia"] = {
        "values": errs, "monotone": monotone, "extrapolated_u_error": extrap,
        "tolerance": 3e-2,
        "pass": monotone and extrap <= 3e-2,
    }

    # ---- C6: grid stability of the end-wall deficit at L_z = 1 ----
    grids = n_grid[:2] if quick else n_grid
    deficits = []
    for n in grids:
        s = _steady(n, n)
        two = LidCavitySolver(LidCavityConfig(n=n, reynolds=REYNOLDS, tolerance=1e-6,
                                              check_every=250))
        two.run()
        d, peak3, peak2 = _deficit(s, two)
        deficits.append({"n": n, "deficit": d, "peak_3d": peak3, "peak_2d": peak2,
                         "steps": s.steps, "residual": s.residual})
        print(f"  grid n={n:3d}  deficit {d*100:6.2f}%  (u_min {peak3:+.4f} vs 2-D {peak2:+.4f})",
              flush=True)
    summary["runs"]["grid_family"] = deficits
    if len(deficits) >= 2:
        rel_change = abs(deficits[-1]["deficit"] - deficits[-2]["deficit"]) / abs(deficits[-2]["deficit"])
    else:
        rel_change = float("nan")
    summary["criteria"]["C6_grid_stability"] = {
        "value": rel_change, "tolerance": 5e-2, "pass": rel_change <= 5e-2,
        "deficits": [d["deficit"] for d in deficits],
    }

    # ---- C7: modulated lid, w != 0 with antisymmetry ----
    mod = _steady(12, 12, lid_spanwise_amplitude=0.2, lid_spanwise_wavenumber=1)
    asym_m = mod.midplane_asymmetry()
    c7_asym = max(asym_m["u"], asym_m["v"])
    summary["runs"]["modulated"] = {
        "n_x": mod.n_x, "n_z": mod.n_z, "steps": mod.steps, "residual": mod.residual,
        "endwall_speed": mod.endwall_speed(), "midplane": asym_m,
        "divergence": float(mod.divergence().abs().max()),
        "wall_face_leak": mod.wall_face_leak(),
    }
    summary["criteria"]["C7_modulated_lid"] = {
        "endwall_speed": mod.endwall_speed(), "asymmetry": c7_asym,
        "divergence": float(mod.divergence().abs().max()),
        "pass": (mod.endwall_speed() > 1e-3 and c7_asym <= 1e-10
                 and float(mod.divergence().abs().max()) <= 1e-12),
    }

    summary["withheld"] = [
        "absolute accuracy of the end-wall geometry against Ku/Hirsh/Taylor (1989): "
        "no reference table exists in this repository",
        "turbulent / high-Re 3-D states (explicit steady marching envelope)",
        "span aspect ratios other than 1, 2, 4",
    ]
    summary["wall_clock_seconds"] = time.time() - t0
    summary["all_pass"] = all(c["pass"] for c in summary["criteria"].values())
    summary["criteria_pass"] = f"{sum(c['pass'] for c in summary['criteria'].values())}/{len(summary['criteria'])}"

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    _figure(summary, out_dir)
    return summary


def _figure(summary: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                      # pragma: no cover - optional dep
        print(f"figure skipped: {exc}")
        return
    fam = summary["runs"]["span_family"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for row in fam:
        ax.plot(row["centreline"]["computed"], row["centreline"]["y"],
                marker="o", ms=3, lw=1.2, label=f"end walls L_z={row['span_length']:.0f}")
    ref = fam[-1]["centreline"]
    ax.plot(ref["reference"], ref["y"], "k--", lw=1.4, label="Ghia et al. (1982), Re=100")
    ax.set_xlabel("u / U  (x = L/2, span mid-plane)")
    ax.set_ylabel("y / L")
    ax.set_title("End-wall cavity -> 2-D limit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    ax2.plot([r["span_length"] for r in fam], [r["err_u"] for r in fam], "o-", label="max |du|/U")
    ax2.plot([r["span_length"] for r in fam], [r["err_v"] for r in fam], "s-", label="max |dv|/U")
    ax2.axhline(0.03, color="r", ls="--", lw=1, label="3 % band")
    ax2.set_xlabel("span length L_z / L")
    ax2.set_ylabel("error vs Ghia (span mid-plane)")
    ax2.set_title("Span-extrapolated 2-D limit")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_endwall_cavity.png"), dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/verification-lid-cavity-3d-endwalls")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.out, quick=args.quick)
    for name, block in summary["criteria"].items():
        print(f"{name:26s} pass={block['pass']}  { {k: v for k, v in block.items() if k != 'pass'} }")
    print("all_pass =", summary["all_pass"], f"({summary['criteria_pass']})",
          f"in {summary['wall_clock_seconds']:.1f} s")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":                  # pragma: no cover
    raise SystemExit(main())