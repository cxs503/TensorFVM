"""Regenerate the attribution audit for ``CASE-FVM-CAVITY-RE1000``.

Reads only stored artefacts (``runs/uncertainty.json``, ``runs/spectral/*.json``,
``runs/gate_registered.json``, the Ghia tables in
:mod:`tensorfvm.lid_cavity_reference`) and writes

* ``runs/attribution_audit.json`` - every number quoted in
  ``ATTRIBUTION_AUDIT.md``, so the document is injected, never hand-copied;
* ``figures/fig_attribution_audit.png`` - three panels (GCI bands vs the table,
  the spectral contraction ratios, the reference-side extremum floor).

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/audit_attribution.py

No solver is run: this audit answers *interpretation* questions about existing
numbers, so it must not be able to move them.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.cavity_attribution import (  # noqa: E402
    attribution_verdict,
    contraction_ratios,
    extremum_floor,
    finest_grid_disagreement,
    gci,
)
from tensorfvm.lid_cavity_reference import U_CENTRELINE, V_CENTRELINE  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGS = HERE / "figures"


def build() -> dict:
    unc = json.loads((RUNS / "uncertainty.json").read_text())
    rep = json.loads((RUNS / "spectral" / "spectral_report.json").read_text())
    gate = json.loads((RUNS / "gate_registered.json").read_text())

    refs = {"u_min": -0.38289, "v_max": 0.37095}

    # --- test 1: the solver's own discretisation uncertainty -----------------
    verdicts = {}
    for q in ("u_min", "v_max"):
        vals = unc["quantities"][q]["values"]
        verdicts[q] = attribution_verdict(
            label=q,
            coarse=vals[0],
            medium=vals[1],
            fine=vals[2],
            reference=refs[q],
        )

    # --- test 2: reference-side extremum floor, from the table alone --------
    floor = {}
    for q, tab, mode in (
        ("u_min", U_CENTRELINE[1000.0], "min"),
        ("v_max", V_CENTRELINE[1000.0], "max"),
    ):
        floor[q] = extremum_floor(tab, mode=mode)
    floor_re100 = {
        "u_min": extremum_floor(U_CENTRELINE[100.0], mode="min"),
        "v_max": extremum_floor(V_CENTRELINE[100.0], mode="max"),
    }
    floor_re400 = {
        "u_min": extremum_floor(U_CENTRELINE[400.0], mode="min"),
        "v_max": V_CENTRELINE[400.0],
    }
    floor_re400 = {
        "u_min": extremum_floor(U_CENTRELINE[400.0], mode="min"),
        "v_max": extremum_floor(V_CENTRELINE[400.0], mode="max"),
    }

    # --- test 3: raw vs extrapolated cross-code agreement --------------------
    pairs = rep["extrapolation"]["u_min"]["pairs"]
    spectral_ladder = [p[1] for p in pairs]
    obs_rho = contraction_ratios(spectral_ladder)
    raw = {}
    for q in ("u_min", "v_max"):
        pairs_q = rep["extrapolation"][q]["pairs"]
        raw[q] = {
            "spectral_fine": pairs_q[-1][1],
            "spectral_limit": rep["extrapolation"][q]["limit"],
            "fvm_fine": unc["quantities"][q]["values"][2],
            "fvm_richardson": rep["fvm"][q]["limit"],
            "raw_disagreement_fine_grids": finest_grid_disagreement(
                unc["quantities"][q]["values"][2], pairs_q[-1][1]
            ),
            "quoted_extrapolated_disagreement": rep["verdict"][q]["spectral_vs_fvm_rel"],
            "assumed_contraction_ratio": 1.0 / rep["extrapolation"][q]["decay_ratio"],
        }

    # --- Re = 100 control (converged spectral levels) -----------------------
    def metrics(n: int, re: int) -> dict:
        p = RUNS / "spectral" / f"cavity_spectral_N{n:03d}_re{re}.json"
        return json.loads(p.read_text())["metrics"]

    control = {
        "N48_u_min": metrics(48, 100)["u_min"],
        "N64_u_min": metrics(64, 100)["u_min"],
        "N48_minus_N64": abs(metrics(64, 100)["u_min"] - metrics(48, 100)["u_min"]),
        "ghia_re100_u_min": -0.21090,
        "residual_re100": metrics(64, 100)["u_min"] / -0.21090 - 1.0,
        "residual_re1000_fvm": unc["quantities"]["u_min"]["values"][2] / -0.38289 - 1.0,
    }
    control["residual_vs_re"] = {
        "re100": abs(control["residual_re100"]),
        "re1000": abs(control["residual_re1000_fvm"]),
    }

    # --- admissibility of the spectral solution as an arbitrating reference --
    admissibility = {}
    for n in (24, 32, 48, 64):
        m = metrics(n, 1000)
        admissibility[n] = {
            "lid_velocity_max": m["lid_velocity_max"],
            "wall_slip_defect": m["wall_slip_defect"],
            "u_min": m["u_min"],
        }

    return {
        "case": "CASE-FVM-CAVITY-RE1000",
        "reynolds": 1000.0,
        "reference_table": "Ghia, Ghia & Shin (1982) Tables I/II",
        "verdicts": verdicts,
        "reference_floor_re1000": floor,
        "reference_floor_re400": floor_re400,
        "reference_floor_re100": floor_re100,
        "cross_code": raw,
        "spectral_contraction_ratios": obs_rho,
        "spectral_ladder_orders": [p[0] for p in pairs],
        "re100_control": control,
        "spectral_admissibility": admissibility,
        "registered_gate_status": {
            "judge_grid": gate.get("judge_grid"),
            "registered_tolerance": gate.get("registered_tolerance"),
            "n_criteria": len(gate.get("criteria", [])),
        },
    }


def figure(audit: dict) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # panel 1: GCI bands vs the tabulated reference
    ax = axes[0]
    labels, colors = [], []
    for i, (q, v) in enumerate(audit["verdicts"].items()):
        lo, hi = v["gci"]["interval"]
        fine = v["finest"]
        ax.errorbar(
            i,
            fine,
            yerr=[[fine - lo], [hi - fine]],
            fmt="o",
            capsize=7,
            color="tab:blue",
            label="FVM n=128 ± GCI" if i == 0 else None,
            markersize=8,
        )
        ax.axhline(v["reference"], color="tab:red", linestyle="--", linewidth=1.2,
                   label="Ghia (1982)" if i == 0 else None)
        ax.annotate(f"pos {v['reference_position_in_band']:.2f}", (i, fine),
                    textcoords="offset points", xytext=(14, -14), fontsize=9)
        labels.append(f"{q}\n({v['verdict'].replace('_', ' ')})")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("value")
    ax.set_title("(a) reference inside the solver's own GCI band")
    ax.legend(fontsize=8, loc="best")

    # panel 2: spectral contraction ratios, observed vs assumed
    ax = axes[1]
    orders = audit["spectral_ladder_orders"]
    rhos = audit["spectral_contraction_ratios"]
    ax.plot(orders[1:-1], rhos, "o-", color="tab:green", label="observed $(v_{i+1}-v_i)/(v_i-v_{i-1})$")
    assumed = audit["cross_code"]["u_min"]["assumed_contraction_ratio"]
    ax.axhline(assumed, color="tab:purple", linestyle=":",
               label=f"assumed by stored extrapolation ({assumed:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("Chebyshev order N")
    ax.set_ylabel("contraction ratio")
    ax.set_title("(b) the extrapolation assumed a ratio the ladder refutes")
    ax.legend(fontsize=8)

    # panel 3: reference-side extremum floor
    ax = axes[2]
    x = np.arange(3)
    w = 0.35
    for k, (q, key) in enumerate((("u_min", "u_min"), ("v_max", "v_max"))):
        vals = [
            100 * audit["reference_floor_re100"][key]["floor_upper"],
            100 * audit["reference_floor_re400"][key]["floor_upper"],
            100 * audit["reference_floor_re1000"][key]["floor_upper"],
        ]
        ax.bar(x + (k - 0.5) * w, vals, w, label=f"{q} floor (upper)")
    ax.set_xticks(x, ["Re = 100", "Re = 400", "Re = 1000"])
    ax.set_ylabel("relative extremum floor  [%]")
    ax.set_title("(c) how well the table pins its own extremum")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGS / "fig_attribution_audit.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    audit = build()
    (RUNS / "attribution_audit.json").write_text(json.dumps(audit, indent=1))
    png = figure(audit)

    print("== verdicts ==")
    for q, v in audit["verdicts"].items():
        lo, hi = v["gci"]["interval"]
        print(f"{q:6s} {v['verdict']:24s} band=[{lo:+.6f},{hi:+.6f}] ref={v['reference']:+.5f} "
              f"pos={v['reference_position_in_band']:.2f} p={v['gci']['order']:.3f} "
              f"resid={v['residual_relative']:.3%}")
    print("== reference-side floor (table alone) ==")
    for re_key, name in (("reference_floor_re1000", "Re=1000"),):
        for q in ("u_min", "v_max"):
            f = audit[re_key][q]
            print(f"{name} {q:6s} tab={f['tabulated']:+.5f} floor lo/mid/up = "
                  f"{f['floor_lower']:.4%} / {f['floor_mid']:.4%} / {f['floor_upper']:.4%}")
    print("== raw vs extrapolated cross-code agreement ==")
    for q, d in audit["cross_code"].items():
        print(f"{q:6s} raw={d['raw_disagreement_fine_grids']:.3%} "
              f"quoted={d['quoted_extrapolated_disagreement']:.3%} "
              f"assumed_rho={d['assumed_contraction_ratio']:.3f}")
    print(f"== spectral observed contraction ratios: {audit['spectral_contraction_ratios']}")
    print(f"== Re=100 control: N48-N64 diff = {audit['re100_control']['N48_minus_N64']:.2e}, "
          f"residual(u_min,converged) = {audit['re100_control']['residual_re100']:.3%}")
    print(f"wrote {RUNS / 'attribution_audit.json'} and {png}")


if __name__ == "__main__":
    main()