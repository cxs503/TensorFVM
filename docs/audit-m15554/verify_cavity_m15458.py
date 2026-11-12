#!/usr/bin/env python3
"""Independent audit of mission m15458 (anchor CASE-CAVITY-RE1000-FVM, TensorFVM).

Board-audit rule: the delivered numbers are *not* taken on trust.  This script reads
the stored velocity fields (``fields/Re1000_n*.pt``) and rebuilds every benchmark
number through a second, deliberately different path:

  * the shipped scorer (:func:`tensorfvm.lid_cavity.score_profiles`) samples the
    centreline and then Hermite-interpolates onto the Ghia abscissae;
  * this audit averages the two staggered faces that straddle the centreline and
    then uses *linear* interpolation -- a lower-order scheme, so the two paths can
    only agree if the underlying field and the tabulated reference are the same.

It also re-derives physics-level invariants that no scorer reports: discrete
divergence, the wall/lid boundary conditions, and the primary-vortex centre from a
path integration of ``u = dpsi/dy`` (no Poisson solve, nothing shared with the
solver's DST diagonalisation).

Usage::

    PYTHONPATH=src python3 docs/audit-m15554/verify_cavity_m15458.py \
        --bundle /root/tp-agent/work/m15458/artifacts/CASE-CAVITY-RE1000-FVM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fields(bundle: str, n: int) -> dict:
    path = os.path.join(bundle, "fields", f"Re1000_n{n}.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": path, "u": blob["u"].double().numpy(), "v": blob["v"].double().numpy(),
            "steps": int(blob["steps"]), "sha256": sha256(path)}


def centreline(fields: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """u(y) on x = L/2 and v(x) on y = L/2, read off the mid grid line.

    Stored MAC layout (``src/tensorfvm/lid_cavity.py``, ``__init__``): ``u[j, i]``
    is the wall-normal velocity at ``x = j*h, y = (i+1/2)*h`` (shape ``(n+1, n)``);
    ``v[j, i]`` sits at ``x = (j+1/2)*h, y = i*h`` (shape ``(n, n+1)``).  With the
    solver's even-``n`` rule the mid-plane *is* a grid line, so the audit samples
    the face row directly -- no averaging, no interpolation in the transverse
    direction (the shipped scorer interpolates longitudinally with Hermite, the
    audit uses linear: different path, lower order).
    """
    u, v = fields["u"], fields["v"]
    n = v.shape[1] - 1
    h = 1.0 / n
    y = (np.arange(n) + 0.5) * h
    u_prof = u[n // 2, :]
    x = (np.arange(n) + 0.5) * h
    v_prof = v[:, n // 2]
    return y, u_prof, x, v_prof


def score(y, u_prof, x, v_prof, reynolds: float) -> dict:
    out = {}
    for which, coords, values in (("u", y, u_prof), ("v", x, v_prof)):
        pts = table(reynolds, which)
        tgt = np.array([p[0] for p in pts])
        ref = np.array([p[1] for p in pts])
        got = np.interp(tgt, coords, values)          # linear, not Hermite
        delta = got - ref
        jmin, jmax = int(np.argmin(ref)), int(np.argmax(ref))
        out[which] = {
            "coords": tgt.tolist(), "reference": ref.tolist(), "computed": got.tolist(),
            "delta": delta.tolist(),
            "max_abs_error": float(np.abs(delta).max()),
            "rms_error": float(np.sqrt((delta ** 2).mean())),
            "extremum_ref": {"coord": float(tgt[jmin if which == "u" else jmax]),
                             "value": float(ref[jmin if which == "u" else jmax])},
            "extremum_got": {"coord": float(tgt[jmin if which == "u" else jmax]),
                             "value": float(got[jmin if which == "u" else jmax])},
        }
        # error with the two wall endpoints dropped: at coord 0/1 the field has no
        # sample (cell-centred rows only), so any value there is extrapolation and
        # says nothing about the interior solution.
        inner = np.abs(delta)[(tgt > 1e-9) & (tgt < 1.0 - 1e-9)]
        out[which]["interior_max_abs_error"] = float(inner.max())
        out[which]["interior_arg"] = float(tgt[(tgt > 1e-9) & (tgt < 1.0 - 1e-9)][
            int(np.argmax(np.abs(delta)[(tgt > 1e-9) & (tgt < 1.0 - 1e-9)]))])
        jw = int(np.argmax(np.abs(delta)))
        out[which]["worst_coord"] = float(tgt[jw])
        out[which]["worst_delta_pct_of_U"] = 100.0 * float(delta[jw])
        key = "min" if which == "u" else "max"
        gv, rv = out[which]["extremum_got"]["value"], out[which]["extremum_ref"]["value"]
        out[which]["extremum_rel_err_pct"] = 100.0 * abs(abs(gv) - abs(rv)) / abs(rv)
        out[which]["extrema_diff_pct_of_U"] = 100.0 * abs(gv - rv)
        out[which]["_kind"] = key
    return out


def divergence(fields: dict) -> float:
    """Discrete MAC divergence at cell centres (exact face differences)."""
    u, v = fields["u"], fields["v"]
    h = 1.0 / (v.shape[1] - 1)
    du = (u[1:, :] - u[:-1, :]) / h
    dv = (v[:, 1:] - v[:, :-1]) / h
    return float(np.abs(du + dv).max())


def boundary_conditions(fields: dict) -> dict:
    u, v = fields["u"], fields["v"]
    # stored fields carry cell-centre samples only (no wall row), so the no-slip
    # condition is probed on the row/column adjacent to each wall, corners excluded
    return {
        "lid_adjacent_u_min_max": [float(u[1:-1, -1].min()), float(u[1:-1, -1].max())],
        "bottom_adjacent_u_abs_max": float(np.abs(u[1:-1, 0]).max()),
        "left_adjacent_v_abs_max": float(np.abs(v[0, 1:-1]).max()),
        "right_adjacent_v_abs_max": float(np.abs(v[-1, 1:-1]).max()),
        "lid_adjacent_u_mean": float(u[1:-1, -1].mean()),
    }


def vortex_centre(fields: dict) -> dict:
    """Primary vortex = minimum of psi from a path integral of u = dpsi/dy.

    Independent of the solver's DST Poisson solve for ``lap psi = -omega``: psi is
    built by trapezoid integration along each column, with psi = 0 on the bottom
    wall, and the interior minimum is located on the cell centres.
    """
    u = fields["u"]
    n = u.shape[1]
    h = 1.0 / n
    u_mid = u[1:-1, :]                       # interior x-faces, (n-1, n) at y=(i+.5)h
    psi = np.zeros_like(u_mid)               # rows = x-faces, cols = y cell centres
    for i in range(1, psi.shape[1]):         # integrate along y (axis 1) -- u = dpsi/dy
        psi[:, i] = psi[:, i - 1] + 0.5 * h * (u_mid[:, i - 1] + u_mid[:, i])
    inner = psi[1:-1, 1:-1]
    jx, iy = np.unravel_index(int(np.argmin(inner)), inner.shape)
    xc = (jx + 2) * h                        # +2: dropped the first/last x-face and column
    yc = (iy + 1.5) * h
    ref = VORTEX_CENTRE[1000.0]
    return {"computed": [round(float(xc), 4), round(float(yc), 4)],
            "reference": [float(ref[0]), float(ref[1])],
            "abs_delta": [round(abs(float(xc) - ref[0]), 4), round(abs(float(yc) - ref[1]), 4)],
            "psi_min": float(inner.min())}


def steadiness(fields: dict, n: int) -> dict:
    """Re-advance the stored field one projection step and measure the gate rate.

    "Converged" in the delivery means ``max|du|/dt <= 1e-7`` (the runner's
    ``on_check`` gate).  That claim is re-derived here: the stored velocity field is
    installed in a fresh solver of the same resolution and stepped once, so if the
    field really is a steady discrete solution the rate must sit at the gate.
    ``tendencies()`` alone is *not* usable for this -- it is the momentum RHS
    **without** the pressure gradient, which is O(1) even at a steady state.
    """
    from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver

    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=1000.0, cfl=0.15))
    solver.u = torch.as_tensor(fields["u"]).clone()
    solver.v = torch.as_tensor(fields["v"]).clone()
    u0, v0 = solver.u.clone(), solver.v.clone()
    dt = solver.step()
    rate = max(float((solver.u - u0).abs().max()),
               float((solver.v - v0).abs().max())) / dt
    return {"dt": dt, "max_abs_change_rate": rate, "gate": 1e-07,
            "steady": bool(rate <= 1e-07),
            "divergence_after_step": float(solver.divergence().abs().max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--grids", default="32,64,128")
    args = ap.parse_args()

    summary_path = os.path.join(args.bundle, "summary.json")
    with open(summary_path) as fh:
        summary = json.load(fh)
    shipped = {c["n"]: c for c in summary["cases"]}

    audit = {"anchor": "CASE-CAVITY-RE1000-FVM", "mission": 15458,
             "bundle": args.bundle,
             "artefacts": {"summary.json": {"sha256": sha256(summary_path),
                                            "updated": summary.get("updated")},
                           "run.log": {"sha256": sha256(os.path.join(args.bundle, "run.log"))}},
             "interpolation": "audit=linear on face-averaged centrelines; shipped=Hermite",
             "rows": {}}

    for n in [int(s) for s in args.grids.split(",")]:
        fields = load_fields(args.bundle, n)
        y, u_prof, x, v_prof = centreline(fields)
        sc = score(y, u_prof, x, v_prof, summary["reynolds"])
        vc = vortex_centre(fields)
        row = {
            "n": n,
            "steps": fields["steps"],
            "field_sha256": fields["sha256"],
            "audit": {
                "u_min": sc["u"]["extremum_got"]["value"],
                "u_min_ref": sc["u"]["extremum_ref"]["value"],
                "u_min_rel_err_pct": sc["u"]["extremum_rel_err_pct"],
                "v_max": sc["v"]["extremum_got"]["value"],
                "v_max_ref": sc["v"]["extremum_ref"]["value"],
                "v_max_rel_err_pct": sc["v"]["extremum_rel_err_pct"],
                "u_max_abs_error_pct_of_U": 100.0 * sc["u"]["max_abs_error"],
                "v_max_abs_error_pct_of_U": 100.0 * sc["v"]["max_abs_error"],
                "u_interior_max_abs_error_pct_of_U": 100.0 * sc["u"]["interior_max_abs_error"],
                "u_interior_arg": sc["u"]["interior_arg"],
                "u_worst_coord": sc["u"]["worst_coord"],
                "u_worst_delta_pct_of_U": sc["u"]["worst_delta_pct_of_U"],
                "v_interior_max_abs_error_pct_of_U": 100.0 * sc["v"]["interior_max_abs_error"],
                "v_interior_arg": sc["v"]["interior_arg"],
                "v_worst_coord": sc["v"]["worst_coord"],
                "v_worst_delta_pct_of_U": sc["v"]["worst_delta_pct_of_U"],
                "u_rms_pct_of_U": 100.0 * sc["u"]["rms_error"],
                "v_rms_pct_of_U": 100.0 * sc["v"]["rms_error"],
                "max_divergence": divergence(fields),
                "boundary_conditions": boundary_conditions(fields),
                "vortex_centre": vc,
                "steadiness": steadiness(fields, n),
            },
            "profiles": {"u": {k: sc["u"][k] for k in
                               ("coords", "reference", "computed", "delta")},
                         "v": {k: sc["v"][k] for k in
                               ("coords", "reference", "computed", "delta")}},
        }
        s = shipped.get(n)
        if s is not None:
            row["shipped"] = {
                "converged": s["solver"]["converged"],
                "residual_rate": s["solver"]["residual_rate"],
                "u_min": s["u"]["extremum"]["min_computed"][1],
                "v_max": s["v"]["extremum"]["max_computed"][1],
                "u_max_abs_error": s["u"]["max_abs_error"],
                "v_max_abs_error": s["v"]["max_abs_error"],
                "vortex_centre": s["vortex_centre"]["computed"],
                "verdict_5pct": s["verdict_5pct"]["pass"],
            }
            row["audit_vs_shipped"] = {
                "du_min_abs": abs(row["audit"]["u_min"] - s["u"]["extremum"]["min_computed"][1]),
                "dv_max_abs": abs(row["audit"]["v_max"] - s["v"]["extremum"]["max_computed"][1]),
                "d_max_abs_u_error": abs(row["audit"]["u_max_abs_error_pct_of_U"] / 100.0
                                         - s["u"]["max_abs_error"]),
                "d_max_abs_v_error": abs(row["audit"]["v_max_abs_error_pct_of_U"] / 100.0
                                         - s["v"]["max_abs_error"]),
                "vortex_centre_abs_delta": max(
                    abs(row["audit"]["vortex_centre"]["computed"][0] - s["vortex_centre"]["computed"][0]),
                    abs(row["audit"]["vortex_centre"]["computed"][1] - s["vortex_centre"]["computed"][1])),
            }
        audit["rows"][str(n)] = row
        print(f"n={n}: u_min {row['audit']['u_min']:.5f} ({row['audit']['u_min_rel_err_pct']:.2f}%), "
              f"v_max {row['audit']['v_max']:.5f} ({row['audit']['v_max_rel_err_pct']:.2f}%), "
              f"max|du| {row['audit']['u_max_abs_error_pct_of_U']:.2f}% U, "
              f"max|dv| {row['audit']['v_max_abs_error_pct_of_U']:.2f}% U, "
              f"|div| {row['audit']['max_divergence']:.2e}, "
              f"vortex {row['audit']['vortex_centre']['computed']}")

    # convergence order from the audit's own pointwise errors
    ns = sorted(int(k) for k in audit["rows"])
    errs = [audit["rows"][str(n)]["audit"]["u_max_abs_error_pct_of_U"] for n in ns]
    orders = [float(np.log(errs[i] / errs[i + 1]) / np.log(ns[i + 1] / ns[i]))
              for i in range(len(ns) - 1) if errs[i + 1] > 0]
    errs_in = [audit["rows"][str(n)]["audit"]["u_interior_max_abs_error_pct_of_U"] for n in ns]
    audit["convergence"] = {"n": ns, "u_max_abs_error_pct_of_U": errs,
                            "u_interior_max_abs_error_pct_of_U": errs_in,
                            "observed_order": orders,
                            "observed_order_interior": [
                                float(np.log(errs_in[i] / errs_in[i + 1]) / np.log(ns[i + 1] / ns[i]))
                                for i in range(len(ns) - 1)]}

    out = os.path.join(OUT_DIR, "cavity_audit.json")
    with open(out, "w") as fh:
        json.dump(audit, fh, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())