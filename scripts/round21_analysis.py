#!/usr/bin/env python3
"""Round 21 analysis: the closure-free sister runs landed -- what do they say?

Inputs (all already in the repository, nothing is re-run here)

* ``docs/case-periodic-hills/round20_laminar_Re{700,1400,2800}_96x48.{json,npz}``
  -- the round-20 sister cases, fetched from cluster jobs ``cj3663``/``cj3664``/
  ``cj3665``;
* ``docs/case-periodic-hills/round18_ladder.json`` -- the Re=2800 SST ladder;
* ``docs/case-periodic-hills/round21_re1400_probe*.json`` -- the divergence
  probe of :mod:`scripts.round21_re1400_probe`.

Outputs

* ``round21_table.json`` -- every number used in ``ROUND21.md`` (zero hand copy);
* ``fig_round21_ladder.png`` -- reattachment ladder with both definitions;
* ``fig_round21_re1400.png`` -- the Re=1400 blow-up and its dt dependence.

The script never scores a diverged case, never interpolates a reference, and
prints the two definitions side by side instead of choosing one for the reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_field_metrics import (
    DEFINITION_ENSEMBLE, DEFINITION_MEAN_FIELD, backflow_extent_rows,
    definition_spread,
)
from tensorfvm.verification.periodic_hill_lower_re import (
    KB_REATTACHMENT_BY_RE, KB_SOURCE, attribution, attribution_limits, sister_table,
)
from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
WINDOW = (1.9, 7.1)


def load_case(path: Path) -> dict:
    payload = json.loads(path.read_text())
    npz = np.load(path.with_suffix(".npz"))
    x_over_h = npz["x_over_h"]
    certified = separation_reattachment(npz["mean_tau_b"], x_over_h, window=WINDOW)
    field = backflow_extent_rows(npz["mean_u"], x_over_h)
    series = [v for v in payload["series"]["x_r_over_H"] if v is not None]
    series = [v for v in series if np.isfinite(v)]
    row = {
        "artefact": path.name,
        "reynolds": float(payload["config"]["reynolds"]),
        "closure": payload.get("closure"),
        "diverged_at_step": payload["config"].get("diverged_at_step"),
        "reference_x_r_over_H": payload["lower_re"].get("reference_x_r_over_H"),
        "x_r_over_H_reported": payload["lower_re"].get("x_r_over_H"),
        "x_r_over_H_recertified": certified.get("x_r_over_H"),
        "x_s_over_H_recertified": certified.get("x_s_over_H"),
        "x_r_status": certified.get("status"),
        "backflow_extent_row0": field.get(0, {}).get("x_end"),
        "backflow_extent_rows": {str(k): v.get("x_end") for k, v in field.items()},
        "ensemble_n": len(series),
        "x_r_over_H_ensemble": float(np.median(series)) if series else None,
        "fluctuation_max_abs_u_minus_mean_u": float(
            np.max(np.abs(npz["u"] - npz["mean_u"]))),
        "rms_u": (float(np.sqrt(np.mean(npz["u"] ** 2)))
                  if np.all(np.isfinite(npz["u"])) else None),
        "lower_re": payload["lower_re"],
    }
    spread = definition_spread(row["x_r_over_H_reported"], row["x_r_over_H_ensemble"])
    row["definitions"] = spread
    ref = row["reference_x_r_over_H"]
    row["error_mean_field_rel"] = (None if ref in (None, 0) or
                                   row["x_r_over_H_reported"] is None
                                   else (row["x_r_over_H_reported"] - ref) / ref)
    row["error_ensemble_rel"] = (None if ref in (None, 0) or
                                 row["x_r_over_H_ensemble"] is None
                                 else (row["x_r_over_H_ensemble"] - ref) / ref)
    if row["backflow_extent_row0"] is not None and row["x_r_over_H_recertified"]:
        row["shear_vs_field_rel"] = ((row["backflow_extent_row0"]
                                     - row["x_r_over_H_recertified"])
                                    / row["x_r_over_H_recertified"])
    return row


def load_closure_rows() -> list[dict]:
    path = OUT / "round18_ladder.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for lvl in (data.get("levels") or data.get("rows") or []):
        if lvl.get("diverged_at_step") is not None or lvl.get("x_r_over_H") is None:
            continue
        x_r = float(lvl["x_r_over_H"])
        ref = KB_REATTACHMENT_BY_RE[2800.0]
        err = (x_r - ref) / ref
        rows.append({"reynolds": 2800.0, "closure": "sst", "x_r_over_H": x_r,
                     "tag": lvl.get("tag"), "error": err,
                     "verdict": "PASS" if abs(err) <= 0.10 else "FAIL"})
    return rows


def main() -> int:
    cases = [load_case(p) for p in sorted(OUT.glob("round20_laminar_Re*_96x48.json"))]
    closure_rows = load_closure_rows()
    measured = [c for c in cases if c["diverged_at_step"] is None]

    table = sister_table({f"r21_{c['reynolds']:g}_{c['closure']}": {
        "reynolds": c["reynolds"], "closure": c["closure"],
        "x_r_over_H": c["x_r_over_H_reported"],
        "reference_x_r_over_H": c["reference_x_r_over_H"],
        "diverged_at_step": c["diverged_at_step"]} for c in cases})
    table.update({
        "cases": cases,
        "closure_rows_2800": closure_rows,
        "source": KB_SOURCE,
        "definitions": {"mean_field": DEFINITION_MEAN_FIELD,
                        "ensemble": DEFINITION_ENSEMBLE},
        "attribution": attribution([c["lower_re"] for c in cases], closure_rows),
        "attribution_rows_used": [{"reynolds": c["reynolds"],
                                   "verdict": c["lower_re"].get("verdict"),
                                   "error": c["lower_re"].get("error")} for c in cases],
        "attribution_limits": attribution_limits(),
    })
    for probe_name in ("round21_re1400_probe.json", "round21_re1400_probe_dt001.json"):
        p = OUT / probe_name
        if p.exists():
            probe = json.loads(p.read_text())
            table[probe_name] = {"reynolds": probe["reynolds"], "dt": probe["dt"],
                                 "diverged_at_step": probe["diverged_at_step"],
                                 "diagnosis": probe["diagnosis"],
                                 "wall_clock_s": probe["wall_clock_s"]}
    (OUT / "round21_table.json").write_text(json.dumps(table, indent=1))

    # ---- figure 1: ladder, both definitions -------------------------------- #
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    res = np.array(sorted(KB_REATTACHMENT_BY_RE))
    xr = np.array([KB_REATTACHMENT_BY_RE[r] for r in res])
    ax.plot(res, xr, "k-o", lw=1.6, ms=6, label="Breuer 2009 / ERCOFTAC KB (reference)")
    ax.fill_between(res, xr * 0.9, xr * 1.1, color="0.85", zorder=0,
                    label="+/-10 % acceptance band")
    if closure_rows:
        ax.plot([2800.0] * len(closure_rows), [c["x_r_over_H"] for c in closure_rows],
                "s", ms=8, mfc="tab:red", mec="k",
                label="Re=2800 SST k-w (r18 ladder, closure on)")
    for c in cases:
        if c["diverged_at_step"] is not None:
            ax.annotate("DIVERGED\n(step %d)" % c["diverged_at_step"],
                        (c["reynolds"], 5.2), textcoords="offset points",
                        xytext=(0, -26), ha="center", fontsize=8, color="tab:purple")
            ax.plot(c["reynolds"], 5.19, "X", ms=10, mfc="tab:purple", mec="k")
            continue
        ax.errorbar(c["reynolds"], c["x_r_over_H_reported"],
                    yerr=[[0], [0]], fmt="D", ms=8, mfc="tab:blue", mec="k",
                    label="closure-free, crossing of time-mean wall shear")
        ax.plot(c["reynolds"], c["x_r_over_H_ensemble"], "v", ms=8, mfc="none",
                mec="tab:green", mew=1.8,
                label="closure-free, ensemble of instantaneous reattachment")
        ax.plot([c["reynolds"]] * 2, [c["x_r_over_H_reported"], c["x_r_over_H_ensemble"]],
                "-", color="tab:green", lw=0.9, alpha=0.7)
        ax.annotate(f"{c['x_r_over_H_reported']:.2f} / {c['x_r_over_H_ensemble']:.2f}",
                    (c["reynolds"], c["x_r_over_H_reported"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(res)
    ax.set_xticklabels([f"{int(r)}" for r in res])
    ax.set_xlabel("Re = U_b H / nu")
    ax.set_ylabel("x_r/H")
    ax.set_title("Periodic hills, closure-free branch: two definitions, measured", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_round21_ladder.png", dpi=150)

    # ---- figure 2: the Re=1400 blow-up ------------------------------------- #
    cur = json.loads((OUT / "round21_re1400_probe.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    tr = cur["trace"]
    ax.semilogy(tr["t"], tr["max_abs_u"], "o-", ms=2.2, lw=1.0,
                label=f"dt = {cur['dt']} (diverges at step {cur['diverged_at_step']})")
    alt = OUT / "round21_re1400_probe_dt001.json"
    if alt.exists():
        d1 = json.loads(alt.read_text())
        ax.semilogy(d1["trace"]["t"], d1["trace"]["max_abs_u"], "s-", ms=2.2, lw=1.0,
                    color="tab:green",
                    label=f"dt = {d1['dt']} (no divergence in {d1['steps_requested']} steps)")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("t = U_b t / H")
    ax.set_ylabel(r"$\max|u|/U_b$")
    ax.set_title("Closure-free Re = 1400: abrupt blow-up, dt-dependent", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_round21_re1400.png", dpi=150)

    print(json.dumps({"cases": [{k: c[k] for k in
                                 ("reynolds", "x_r_over_H_reported", "x_r_over_H_ensemble",
                                  "x_r_over_H_recertified", "backflow_extent_row0",
                                  "error_mean_field_rel", "diverged_at_step")}
                                for c in cases],
                      "attribution": table["attribution"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())