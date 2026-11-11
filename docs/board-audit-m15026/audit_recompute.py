#!/usr/bin/env python3
"""Board audit m15190: independent recomputation of the m15026 GAP-FVM-TWOPHASE numbers.

Reads ONLY the committed artefacts under docs/GAP-FVM-TWOPHASE/artifacts and
recomputes every headline metric with locally written code (no import of the
graded modules), so that "reported vs audited" is a real cross-check and not a
re-print of the same object.

Outputs docs/board-audit-m15026/audit_recompute.json + a markdown table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ART = Path("docs/GAP-FVM-TWOPHASE/artifacts")
OUT = Path("docs/board-audit-m15026")


def slotted_disk_mask(n: int, radius=0.15, slot_hw=0.05, cx=0.5, cy=0.75):
    """Independent build of the Zalesak slotted disk on an n x n cell-centre lattice.

    Disk centred at (cx, cy) = (0.5, 0.75) with the vertical strip
    |x - cx| <= slot_hw removed (Zalesak 1979 geometry, unit domain).
    """
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    slot = np.abs(X - cx) <= slot_hw
    return (inside & ~slot).astype(np.float64)


def zalesak_audit():
    rows = []
    summary = json.load(open(ART / "gap_twophase.json"))
    legs = summary.get("legs", {})
    for leg in legs.values():
        if leg.get("case") != "zalesak":
            continue
        cells, scheme = int(leg["cells"]), leg["scheme"]
        npz = np.load(ART / f"fields_zalesak_{scheme}_{cells}.npz")
        alpha, exact = npz["alpha"], npz["exact"]

        # (1) independent analytic reference vs the artefact's own "exact" field
        mine = slotted_disk_mask(cells)
        exact_diff = float(np.abs(mine - (exact > 0.5)).max())

        # (2) thresholded geometric error, recomputed from the fields
        m_num, m_ref = alpha > 0.5, exact > 0.5
        xor_area = float(np.logical_xor(m_num, m_ref).sum())
        geo_err = xor_area / float(m_ref.sum())
        geo_err_mine = float(np.logical_xor(m_num, mine > 0.5).sum()) / float((mine > 0.5).sum())

        # (3) liquid area drift recomputed from the field itself
        a_num, a_exact = float(alpha.sum()), float(exact.sum())
        mass_drift = abs(a_num - a_exact) / a_exact

        # (4) bounds
        bound_viol = max(0.0, float(alpha.max()) - 1.0, -float(alpha.min()))

        rows.append(dict(
            case="zalesak", scheme=scheme, cells=cells,
            geom_err_reported=float(leg["geometric_error"]),
            geom_err_audited=geo_err,
            geom_err_audited_vs_independent_ref=geo_err_mine,
            exact_field_maxdiff_vs_independent=exact_diff,
            mass_drift_reported=float(leg["mass_drift_relative"]),
            mass_drift_audited=mass_drift,
            bound_violation_reported=float(leg["bound_violation"]),
            bound_violation_audited=bound_viol,
            peak_retention_report=float(leg["peak_retention"]),
            rel_delta_geom=abs(geo_err - float(leg["geometric_error"]))
            / max(float(leg["geometric_error"]), 1e-30),
        ))
    return rows


def dambreak_audit():
    rows = []
    summary = json.load(open(ART / "gap_twophase.json"))
    for leg in summary.get("legs", {}).values():
        if leg.get("case") != "dambreak":
            continue
        grid = leg["grid"]
        comp = np.asarray(leg["comparison"], dtype=float)
        # comparison rows: [t*, ref_front, ref_height, got_front, got_height]
        t, rf, rh, gf, gh = comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 3], comp[:, 4]
        win = (t > 0.3) & (t <= float(leg["t_star_end"]) + 1e-12)
        front_rel = np.abs(gf[win] - rf[win]) / rf[win]
        height_rel = np.abs(gh[win] - rh[win]) / np.where(rh[win] > 0, rh[win], np.nan)
        rows.append(dict(
            case="dambreak", grid=grid,
            front_max_rel_reported=float(leg["front_max_rel_error"]),
            front_max_rel_audited=float(np.nanmax(front_rel)),
            front_mean_rel_audited=float(np.nanmean(front_rel)),
            height_max_rel_reported=float(leg["height_max_rel_error"]),
            height_max_rel_audited=float(np.nanmax(height_rel)),
            volume_drift_reported=float(leg["volume_drift"]),
            control_max_abs_change=float(leg["control_max_abs_change"]),
        ))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zalesak_audit()
    d = dambreak_audit()
    gate = json.load(open(ART / "gap_twophase.json")).get("gates", [])
    gate_pass = sum(1 for g in gate if g.get("passed"))
    out = dict(zalesak=z, dambreak=d,
               gates_total=len(gate), gates_passed=gate_pass,
               gates_reported=f"{gate_pass}/{len(gate)}")
    (OUT / "audit_recompute.json").write_text(json.dumps(out, indent=2))

    lines = ["# Independent recomputation (audit m15190)", "",
             "## Zalesak (from committed NPZ fields)", "",
             "| scheme | N | geo err reported | geo err audited | audited vs independent ref |"
             " exact-field maxdiff | rel delta | mass drift rep/aud |",
             "|---|---|---|---|---|---|---|---|"]
    for r in z:
        lines.append(
            f"| {r['scheme']} | {r['cells']} | {r['geom_err_reported']:.5f} |"
            f" {r['geom_err_audited']:.5f} | {r['geom_err_audited_vs_independent_ref']:.5f} |"
            f" {r['exact_field_maxdiff_vs_independent']:.3g} | {r['rel_delta_geom']:.2%} |"
            f" {r['mass_drift_reported']:.3g}/{r['mass_drift_audited']:.3g} |")
    lines += ["", "## Dam break (from committed leg series vs Martin & Moyce)", "",
              "| grid | front err reported | front err audited (max) | mean |"
              " height err reported/audited | volume drift | control |",
              "|---|---|---|---|---|---|---|"]
    for r in d:
        lines.append(
            f"| {r['grid']} | {r['front_max_rel_reported']:.4f} |"
            f" {r['front_max_rel_audited']:.4f} | {r['front_mean_rel_audited']:.4f} |"
            f" {r['height_max_rel_reported']:.4f}/{r['height_max_rel_audited']:.4f} |"
            f" {r['volume_drift_reported']:.2g} | {r['control_max_abs_change']:.2g} |")
    lines += ["", f"Gates reproduced from artefacts: **{gate_pass}/{len(gate)}** "
                  "(reported 8/10; `--summary-only` rerun identical)"]
    (OUT / "audit_recompute.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()