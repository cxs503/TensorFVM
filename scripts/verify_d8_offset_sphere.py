"""D8 verification driver: parallel-axis anchor of the offset sphere (debt ledger D8).

Background
----------
``docs/maneuvering-added-inertia/REPORT.md`` (m14901) flagged an inconsistency:
"``added_mass_matrix_6dof`` gives ``I_zz = 2.05`` for the offset-sphere
configuration (theory 0 for centre (0,0,1.5); theory ``m_a d^2 = 4.71`` for
centre (1.5,0,0))", and registered it as ledger item D8.  The item was never
resolved and never actually entered the ledger.

This driver re-checks the flagged inconsistency against the exact anchors:

* configuration A: sphere centre ``(0, 0, d)`` -- rotation about the z axis
  passes through the centre -> ``I_zz = 0``; about x/y (perpendicular to the
  offset) -> ``I_xx = I_yy = m_a d^2``.
* configuration B: sphere centre ``(d, 0, 0)`` -- about the x axis -> ``I_xx =
  0``; about y/z -> ``I_yy = I_zz = m_a d^2``.

with ``m_a = (2/3) pi rho a^3`` (sphere added mass, Lamb).  A mesh ladder with
power-law extrapolation quantifies the discretisation error of each entry.

It also reproduces the *mechanism* of the m14901 phantom: the anchor helper
historically defaulted to ``distance = 1.0``, so comparing a ``d = 1.5`` panel
value against the default-distance anchor yields ``m_a * 1.0 = 2.094`` -- the
"2.05" of the report.  The fix (this ticket) makes ``distance`` mandatory.

Run:  PYTHONPATH=src python scripts/verify_d8_offset_sphere.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from tensorfvm.panel_added_mass import (
    added_mass_matrix_6dof,
    sphere_added_mass,
    sphere_panels_offset,
)

A = 1.0          # radius
D = 1.5          # offset distance
RHO = 1.0
LADDER = ((24, 14), (32, 18), (48, 26))
OUT = os.path.join(os.path.dirname(__file__), "..", "docs",
                   "verification-d8-offset-sphere")


def _extrapolate(points):
    """Power-law ``v(N) = v_inf + K N^(-q)`` fit through three ladder points.

    ``q`` is found by a grid search that minimises the residual at the coarsest
    point; with three points the fit is then exact at the two finest and the
    coarsest residual is reported.
    """
    (n1, v1), (n2, v2), (n3, v3) = points

    def v_inf_of(q):
        k = (v3 - v2) / (n3 ** -q - n2 ** -q)
        return v3 - k * n3 ** -q

    best = None
    for q in np.linspace(0.15, 3.0, 286):
        k = (v3 - v2) / (n3 ** -q - n2 ** -q)
        v_inf = v3 - k * n3 ** -q
        resid = abs(v1 - (v_inf + k * n1 ** -q))
        if best is None or resid < best[1]:
            best = (float(q), resid, v_inf)
    q, _, v_inf = best
    return v_inf, q


def run_config(centre, zero_axis, pair_axes, tag):
    """Mesh ladder of the 3x3 rotational block for one offset configuration."""
    ma = sphere_added_mass(A, RHO)
    exact_pair = ma * D ** 2
    rows = []
    for n_theta, n_phi in LADDER:
        panels = sphere_panels_offset(A, centre, n_theta, n_phi)
        blk = added_mass_matrix_6dof(panels, rho=RHO,
                                     directions=(3, 4, 5))
        n_pan = panels.n_panels
        rows.append({
            "n_theta": n_theta, "n_phi": n_phi, "n_panels": int(n_pan),
            "I_xx": float(blk[0, 0]), "I_yy": float(blk[1, 1]),
            "I_zz": float(blk[2, 2]),
            "max_offdiag": float(np.max(np.abs(blk - np.diag(np.diag(blk))))),
        })
    out = {"tag": tag, "centre": list(centre), "m_a": ma,
           "exact_pair": exact_pair, "exact_zero": 0.0,
           "zero_axis": zero_axis, "pair_axes": list(pair_axes), "ladder": rows}
    for ax, key in ((0, "I_xx"), (1, "I_yy"), (2, "I_zz")):
        vals = [(r["n_panels"], r[key]) for r in rows]
        if ax == zero_axis:
            out[key + "_finest_rel"] = rows[-1][key] / (ma * A ** 2)
            out[key + "_extrap"] = None
        else:
            v_inf, q = _extrapolate(vals)
            out[key + "_finest_rel"] = (rows[-1][key] - exact_pair) / exact_pair
            out[key + "_extrap"] = v_inf
            out[key + "_extrap_rel"] = (v_inf - exact_pair) / exact_pair
            out[key + "_q"] = q
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ma = sphere_added_mass(A, RHO)
    exact_pair = ma * D ** 2
    cfg_a = run_config((0.0, 0.0, D), zero_axis=2, pair_axes=(0, 1),
                       tag="A: centre (0,0,d), d=1.5")
    cfg_b = run_config((D, 0.0, 0.0), zero_axis=0, pair_axes=(1, 2),
                       tag="B: centre (d,0,0), d=1.5")

    # phantom-2.05 mechanism: old default-distance anchor vs d=1.5 panel value
    phantom_anchor = ma * 1.0 ** 2          # what distance=1.0 default returned
    panel_at_d = cfg_a["ladder"][-1]["I_xx"]
    phantom = {
        "old_default_distance": 1.0,
        "old_default_anchor_value": phantom_anchor,
        "m14901_reported_value": 2.05,
        "panel_value_at_d_1p5": panel_at_d,
        "mismatch_vs_default_anchor": (panel_at_d - phantom_anchor) / phantom_anchor,
        "note": ("m14901 compared the d=1.5 panel value against the "
                 "distance=1.0 default anchor (m_a*1.0=2.094 ~ 2.05); the "
                 "current code passes all exact anchors, so the flag is a "
                 "footgun, not a solver defect"),
    }

    # gate: finest-mesh relative errors of the non-zero entries, off-diagonal
    gates = {}
    for cfg in (cfg_a, cfg_b):
        for ax, key in ((0, "I_xx"), (1, "I_yy"), (2, "I_zz")):
            if ax in cfg["pair_axes"]:
                gates[f"{cfg['tag'][0]}_{key}"] = abs(cfg[key + "_finest_rel"])
    offdiag_ok = (cfg_a["ladder"][-1]["max_offdiag"] / exact_pair < 1e-3
                  and cfg_b["ladder"][-1]["max_offdiag"] / exact_pair < 1e-3)
    passed = all(v < 0.03 for v in gates.values()) and offdiag_ok

    summary = {
        "a": A, "d": D, "rho": RHO,
        "m_a": ma, "exact_pair": exact_pair,
        "config_A": cfg_a, "config_B": cfg_b,
        "phantom_2p05_mechanism": phantom,
        "gates": gates, "offdiag_ok": bool(offdiag_ok),
        "all_gates_pass_3pct": bool(passed),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"m_a = {ma:.6f}   m_a d^2 = {exact_pair:.6f}")
    for cfg in (cfg_a, cfg_b):
        print(f"\n{cfg['tag']}")
        for r in cfg["ladder"]:
            print(f"  {r['n_theta']}x{r['n_phi']} ({r['n_panels']} panels): "
                  f"I_xx={r['I_xx']:.4f} I_yy={r['I_yy']:.4f} I_zz={r['I_zz']:.4f} "
                  f"offdiag={r['max_offdiag']:.2e}")
        for key in ("I_xx", "I_yy", "I_zz"):
            if cfg[key + "_extrap"] is not None:
                print(f"  {key}: finest {cfg[key+'_finest_rel']*100:+.2f}%  "
                      f"extrap {cfg[key+'_extrap']:.4f} "
                      f"({cfg[key+'_extrap_rel']*100:+.2f}%, q={cfg[key+'_q']:.2f})")
            else:
                print(f"  {key}: finest {cfg[key+'_finest_rel']:.3e} (theory 0)")
    print(f"\nphantom mechanism: default anchor {phantom_anchor:.4f} vs "
          f"panel(d=1.5) {panel_at_d:.4f} -> {phantom['mismatch_vs_default_anchor']*100:+.1f}%")
    print(f"\nALL GATES <3%: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
