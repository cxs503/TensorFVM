#!/usr/bin/env python3
"""Dam-break part of the audit: recompute the reported errors from the committed
leg series with the *producer's documented row layout*, and test whether the
error is a physics deficiency or a time-normalisation mismatch.

Row layout (src/tensorfvm/twophase_gap/dambreak_gate.py:104-107):
    [t*, got_front_half_cell, got_front_leading_edge, ref_front, got_height]

The Martin & Moyce tables are re-typed here from the cited source digitisation
(the anchor module's arrays) so the reference values are not read from the
graded module at run time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ART = Path("docs/GAP-FVM-TWOPHASE/artifacts")
OUT = Path("docs/board-audit-m15026")

# Martin & Moyce (1952) Fig. 3 digitisation, as used by the anchor
# (src/tensorfvm/vof_phase0/anchors.py:58-78) -- re-typed for audit independence.
MM_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                 6.0, 6.5, 7.0, 7.5, 8.0])
MM_X = np.array([0.0, 0.75, 1.40, 1.95, 2.45, 2.90, 3.30, 3.65, 3.98, 4.28,
                 4.55, 4.80, 5.03, 5.25, 5.45, 5.64, 5.82])
MM_HT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
MM_H = np.array([1.0, 0.72, 0.52, 0.40, 0.32, 0.27, 0.23, 0.20, 0.18, 0.16, 0.15])


def audit_leg(name, leg):
    comp = np.asarray(leg["comparison"], dtype=float)
    t = comp[:, 0]
    got_half, got_lead, ref, got_h = comp[:, 1], comp[:, 2], comp[:, 3], comp[:, 4]
    ref_h = np.interp(t, MM_HT, MM_H)
    rel_f = np.abs(got_half - ref) / np.maximum(np.abs(ref), 1e-6)
    rel_h = np.abs(got_h - ref_h) / np.maximum(np.abs(ref_h), 1e-6)
    # Best-fit *time* rescaling got_front(t* / alpha) vs ref(t*): does a pure
    # time-normalisation offset explain the error?
    best = None
    for alpha in np.linspace(0.2, 3.0, 561):
        g = np.interp(t / alpha, np.r_[0.0, t], np.r_[got_half[0], got_half])
        e = float(np.max(np.abs(g - ref) / np.maximum(np.abs(ref), 1e-6)))
        if best is None or e < best[1]:
            best = (float(alpha), e)
    # front-speed check: d(got)/dt vs d(ref)/dt over the linear part
    return dict(
        leg=name, grid=leg["grid"], n_rows=int(len(t)),
        t_window=[float(t.min()), float(t.max())],
        front_reported=float(leg["front_max_rel_error"]),
        front_audited_half=float(np.max(rel_f)),
        front_audited_leading_edge=float(
            np.max(np.abs(got_lead - ref) / np.maximum(np.abs(ref), 1e-6))),
        front_reported_leading_edge=float(leg["front_max_rel_error_leading_edge"]),
        height_reported=float(leg["height_max_rel_error"]),
        height_audited=float(np.max(rel_h)),
        got_front_at_t01=float(got_half[0]), ref_front_at_t01=float(ref[0]),
        got_height_at_t01=float(got_h[0]), ref_height_at_t01=float(ref_h[0]),
        best_time_scale=best[0], front_err_after_time_rescale=best[1],
        volume_drift=float(leg["volume_drift"]),
        control_max_abs_change=float(leg["control_max_abs_change"]),
        sample=[[float(a), float(b), float(c), float(d), float(e)]
                for a, b, c, d, e in zip(t, got_half, ref, got_h, ref_h)][::max(1, len(t) // 12)],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.load(open(ART / "gap_twophase.json"))
    rows = [audit_leg(k, v) for k, v in s["legs"].items() if v.get("case") == "dambreak"]
    (OUT / "audit_dambreak.json").write_text(json.dumps(rows, indent=2))
    hdr = ("| grid | front rep | front audited | lead rep/aud | height rep | height audited |"
           " got/ref front @t0 | best time scale | err after rescale |")
    lines = [hdr, "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['grid']} | {r['front_reported']:.4f} | {r['front_audited_half']:.4f} |"
            f" {r['front_reported_leading_edge']:.4f}/{r['front_audited_leading_edge']:.4f} |"
            f" {r['height_reported']:.4f} | {r['height_audited']:.4f} |"
            f" {r['got_front_at_t01']:.4f}/{r['ref_front_at_t01']:.4f} |"
            f" {r['best_time_scale']:.2f} | {r['front_err_after_time_rescale']:.4f} |")
    print("\n".join(lines))
    for r in rows:
        print("\n", r["leg"], "sample [t*, got_half, ref, got_h, ref_h]:")
        for row in r["sample"]:
            print("   ", " ".join(f"{v:8.4f}" for v in row))


if __name__ == "__main__":
    main()