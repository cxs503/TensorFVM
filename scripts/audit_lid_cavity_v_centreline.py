"""Attribute the Re = 400 lid-cavity ``v``-centreline discrepancy.

Two independent blocks are produced:

``analytic``
    the extraction path (``centreline_v`` + ``hermite_interpolate``) is fed a
    staggered field of known closed form, so the measured error belongs to the
    extraction alone; a half-cell index-shift control proves the measurement can
    see the defect class the strict xfail hypothesises.
``live``
    a cached Re = 400 lid-cavity ladder (``--ladder``) is scored against Ghia et
    al. (1982) table II, both at the nearest stored node and at the published
    ordinate, and the point that exceeds the published peak magnitude in
    absolute value is localised.

Usage::

    python scripts/audit_lid_cavity_v_centreline.py --ladder path/to/ladder.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tensorfvm.lid_cavity_reference import table  # noqa: E402
from tensorfvm.verification.lid_cavity_extraction import (  # noqa: E402
    GHIA_V_ABS_MAX,
    GHIA_V_X,
    OBSERVED_RELATIVE_DEFECT,
    attribution,
    extraction_geometry,
    index_shift_control,
    interpolation_study,
    observed_orders,
)

OUT = ROOT / "src/tensorfvm/verification/artifacts/lid-cavity-extraction"
NS = (32, 64, 128)


def analytic_block() -> dict:
    geometry = [extraction_geometry(n) for n in NS]
    study = interpolation_study(NS)
    controls = index_shift_control((32, 64), shifts=(0.5, 1.0))
    verdict = attribution(study[0]["max_relative_error"])
    return {
        "geometry": geometry,
        "interpolation": study,
        "observed_order": observed_orders(study),
        "index_shift_control": controls,
        "attribution": verdict.as_dict(),
        "ghia_v_abs_max": GHIA_V_ABS_MAX,
        "observed_relative_defect": OBSERVED_RELATIVE_DEFECT,
        "ghia_ordinates": list(GHIA_V_X),
    }


def live_block(ladder: dict) -> dict:
    """Score a cached ladder against Ghia table II and localise the outlier."""
    reference = np.asarray([(x, v) for x, v in table(400.0, "v")])
    rows = []
    for key in sorted(ladder, key=lambda k: int(ladder[k]["n"])):
        case = ladder[key]
        x = np.asarray(case["x_grid"], dtype=float)
        v = np.asarray(case["v"], dtype=float)
        # nearest stored node, no interpolation: isolates the stored field
        nearest = np.abs(x[:, None] - reference[:, 0][None, :]).argmin(axis=0)
        node_v = v[nearest]
        error = np.abs(node_v - reference[:, 1])
        scale = GHIA_V_ABS_MAX
        worst = int(np.argmax(error))
        stored_peak = float(np.abs(v).max())
        peak_index = int(np.argmax(np.abs(v)))
        exceed = x[np.abs(v) > GHIA_V_ABS_MAX]
        at_defect = float(np.abs(node_v - reference[:, 1])[np.abs(reference[:, 0] - 0.9063).argmin()]
                          / abs(reference[np.abs(reference[:, 0] - 0.9063).argmin(), 1]))
        rows.append({
            "n": case["n"],
            "steps": case["steps"],
            "seconds": case["secs"],
            "converged": case["converged"],
            "max_divergence": case["max_divergence"],
            "max_abs_error": float(error.max()),
            "max_relative_error": float(error.max() / scale),
            "worst_ordinate": float(reference[worst, 0]),
            "worst_computed": float(node_v[worst]),
            "worst_reference": float(reference[worst, 1]),
            "relative_error_at_0p9063": float(abs(at_defect)),
            "stored_peak_abs_v": stored_peak,
            "stored_peak_exceeds_ghia_peak": bool(stored_peak > GHIA_V_ABS_MAX),
            "ordinates_exceeding_ghia_peak": [float(value) for value in exceed],
            "peak_ordinate": float(x[peak_index]),
            "x_grid": x.tolist(),
            "v_profile": v.tolist(),
        })
    return {"rows": rows, "ghia_abs_v_max": GHIA_V_ABS_MAX,
            "reference_table_ii": reference.tolist()}


def figure(analytic: dict, live: dict | None) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is a benchmark extra
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    h = [row["h"] for row in analytic["interpolation"]]
    err = [max(row["max_abs_error"], 1e-16) for row in analytic["interpolation"]]
    axes[0].loglog(h, err, "o-", label="extraction path (analytic field)")
    slope = analytic["observed_order"]
    h0, e0 = h[0], err[0]
    axes[0].loglog(h, [e0 * (x / h0) ** slope for x in h], "--",
                   label=f"observed order p = {slope:.2f}")
    worst = analytic["observed_relative_defect"] * analytic["ghia_v_abs_max"]
    axes[0].axhline(worst, color="r", label="recorded defect (0.149 x peak)")
    axes[0].set_xlabel("h = 1/n")
    axes[0].set_ylabel("max abs error at Ghia ordinates")
    axes[0].set_title("Extraction path vs the recorded defect")
    axes[0].legend(fontsize=7)
    if live and live["rows"]:
        reference = np.asarray(live["reference_table_ii"])
        case = max(live["rows"], key=lambda row: row["n"])
        x = np.asarray([r[0] for r in reference])
        axes[1].plot(x, reference[:, 1], "ks", ms=4, label="Ghia et al. (1982) Re=400")
        axes[1].plot(case["x_grid"], case["v_profile"], "-", lw=1.2,
                     color="C0", label=f"FVM n = {case['n']}")
        axes[1].axhline(GHIA_V_ABS_MAX, color="r", ls=":", lw=1, label="published |v| peak")
        axes[1].axhline(-GHIA_V_ABS_MAX, color="r", ls=":", lw=1)
        axes[1].set_xlabel("x/L")
        axes[1].set_ylabel("v/U on y/L = 1/2")
        axes[1].set_title(f"Live n = {case['n']}: node-wise max rel. error "
                          f"{case['max_relative_error']:.3f}")
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "extraction-attribution.png", dpi=130)
    plt.close(fig)
    return True


def report_markdown(payload: dict) -> str:
    analytic = payload["analytic"]
    lines = [
        "# Re = 400 lid-cavity v-centreline: extraction path vs solution field",
        "",
        "The strict xfail in `tests/test_lid_cavity_re1000_evidence.py` records a "
        f"v-centreline error that grows under refinement ({OBSERVED_RELATIVE_DEFECT:.3f} "
        "at n = 128) and hypothesises a *wall-adjacent extraction defect*. "
        "This audit measures the extraction path on fields of known closed form.",
        "",
        "## 1. Where the samples actually sit",
        "",
        "| n | h | v mid-plane y | exact | min node-to-Ghia offset (cells) |",
        "|---|---|---|---|---|",
    ]
    for row in analytic["geometry"]:
        lines.append(f"| {row['n']} | {row['h']:.6f} | {row['v_mid_plane_y']:.6f} | "
                     f"{row['v_mid_plane_is_exact']} | {row['worst_point_offset_cells']:.3f} |")
    lines += [
        "",
        "`v` is staggered as `v[x_cell, y_face]` with faces at `j*h`, so the horizontal "
        "mid-plane is the stored face `j = n/2`: no half-cell offset, no interpolation "
        "of the ordinate. The same holds for `u` at `x = L/2`.",
        "",
        "## 2. Extraction error on an analytic field",
        "",
        "| n | max abs error | max relative (to |v| peak) | <= 1e-2 ? |",
        "|---|---|---|---|",
    ]
    for row in analytic["interpolation"]:
        lines.append(f"| {row['n']} | {row['max_abs_error']:.3e} | "
                     f"{row['max_relative_error']:.3e} | "
                     f"{row['max_relative_error'] < 1e-2} |")
    lines += [
        "",
        f"Observed order in h: **{analytic['observed_order']:.2f}**.",
        "",
        "## 3. Positive control (does the measurement have teeth?)",
        "",
        "| n | sample shift (cells) | max abs error | clean error | ratio |",
        "|---|---|---|---|---|",
    ]
    for row in analytic["index_shift_control"]:
        lines.append(f"| {row['n']} | {row['shift_cells']} | {row['max_abs_error']:.3e} | "
                     f"{row['clean_max_abs_error']:.3e} | {row['detection_ratio']:.0f}x |")
    lines += ["", "## 4. Attribution", "",
              f"Extraction budget at n = 32: "
              f"{analytic['attribution']['extraction_relative_error']:.3e} relative; "
              f"recorded defect {analytic['attribution']['observed_relative_defect']:.3f}; "
              f"ratio **{analytic['attribution']['ratio']:.0f}x**.", "",
              f"**Verdict.** {analytic['attribution']['verdict']}.", ""]
    live = payload.get("live")
    if live and live["rows"]:
        lines += ["## 5. Live ladder (cached run, re-scored here)", "",
                  "| n | steps | converged | max rel. error | worst x/L | stored |v| peak | exceeds Ghia peak |",
                  "|---|---|---|---|---|---|---|"]
        for row in live["rows"]:
            lines.append(f"| {row['n']} | {row['steps']} | {row['converged']} | "
                         f"{row['max_relative_error']:.3f} | {row['worst_ordinate']:.4f} | "
                         f"{row['stored_peak_abs_v']:.4f} | "
                         f"{row['stored_peak_exceeds_ghia_peak']} |")
        lines += ["", "|v| exceeding the published peak is stored at x/L = "
                  + ", ".join(f"{row['ordinates_exceeding_ghia_peak']}" for row in live["rows"]), ""]
    lines += ["## 6. Scope", "",
              "This audit attributes the discrepancy; it does not fix the solution-field "
              "defect. The analytic block is exact (roundoff on nodes), so any residual "
              "attribution to the extraction path is bounded below 1 % of the profile "
              "scale, i.e. two orders below the recorded 14.9 %.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, default=None,
                        help="cached lid-cavity ladder JSON (n -> x_grid/v)")
    args = parser.parse_args()
    payload = {"analytic": analytic_block()}
    if args.ladder and args.ladder.exists():
        payload["live"] = live_block(json.loads(args.ladder.read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    payload["figure"] = figure(payload["analytic"], payload.get("live"))
    (OUT / "results.json").write_text(json.dumps(payload, indent=1))
    (OUT / "report.md").write_text(report_markdown(payload))
    print(json.dumps({"observed_order": payload["analytic"]["observed_order"],
                      "extraction_relative_error":
                          payload["analytic"]["attribution"]["extraction_relative_error"],
                      "ratio": payload["analytic"]["attribution"]["ratio"],
                      "live": [{"n": r["n"], "max_rel": round(r["max_relative_error"], 4),
                                "worst": r["worst_ordinate"],
                                "peak": round(r["stored_peak_abs_v"], 4),
                                "exceeds": r["stored_peak_exceeds_ghia_peak"],
                                "at_9063": round(r["relative_error_at_0p9063"], 4)}
                               for r in payload.get("live", {}).get("rows", [])]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())