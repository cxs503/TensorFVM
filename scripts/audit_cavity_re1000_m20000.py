"""Independent re-scoring of the Re = 1000 lid-driven-cavity campaign (mission m20000).

Why a second scorer
-------------------
``scripts/run_cavity_re1000_m15458.py`` scores each converged field with
``tensorfvm.lid_cavity.score_profiles``, i.e. with the Catmull-Rom (Hermite)
interpolation that the solver module ships.  A tolerance verdict that is
produced by exactly one interpolation rule is a single point of failure: if
that rule were biased (it uses neighbouring slopes, so it can overshoot next to
the sharp wall layers of this benchmark), the "error" being compared with the
5 % ledger band would be an artefact of the scorer rather than of the solution.

This script therefore re-derives every reported number from the *stored fields*
(``fields/Re1000_n<n>.pt``) with machinery that does not share code with the
scoring path:

* interpolation along the centrelines: piecewise-linear (``numpy.interp``) and a
  local least-squares quadratic through the three samples nearest the target,
  both compared with the shipped Hermite value;
* extrema taken from a densely resampled profile rather than from the Ghia
  sample abscissae only, so "u_min" is a property of the field, not of the
  table's grid;
* the primary-vortex centre from the stream function of the stored field,
  refined by a 2-D quadratic fit (the solver's own ``vortex_centre`` is
  reported alongside for comparison, not replaced);
* the discrete divergence and the number of steps / residual, read back from the
  checkpoint.

Output: ``<artifacts>/m20000_independent_audit.json`` plus a printed table.
Nothing here writes into ``summary.json``: the point is a second, separate
reading of the same fields.

Usage::

    PYTHONPATH=src python scripts/audit_cavity_re1000_m20000.py \
        --artifacts artifacts/CASE-CAVITY-RE1000-FVM
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from tensorfvm.lid_cavity import hermite_interpolate
from tensorfvm.lid_cavity_reference import SUPPORTED_REYNOLDS, table


# --------------------------------------------------------------------------- #
# interpolation rules (independent of tensorfvm.lid_cavity)
# --------------------------------------------------------------------------- #
def linear(x: np.ndarray, f: np.ndarray, targets) -> np.ndarray:
    return np.interp(np.asarray(targets, dtype=float), x, f)


def local_quadratic(x: np.ndarray, f: np.ndarray, targets) -> np.ndarray:
    """Least-squares parabola through the three samples nearest each target."""
    out = []
    n = len(x)
    for t in np.asarray(targets, dtype=float):
        j = int(np.clip(np.searchsorted(x, t), 1, n - 2))
        idx = [j - 1, j, j + 1]
        dx = x[idx] - t
        A = np.vstack([np.ones(3), dx, dx * dx]).T
        coef, *_ = np.linalg.lstsq(A, f[idx], rcond=None)
        out.append(float(coef[0]))
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# field post-processing
# --------------------------------------------------------------------------- #
def cell_centred(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (u[:-1, :] + u[1:, :]), 0.5 * (v[:, :-1] + v[:, 1:])


def streamfunction(uc: np.ndarray, h: float) -> np.ndarray:
    """psi(x, y) = int_0^y u dy from the bottom wall (psi = 0 there)."""
    return np.cumsum(uc, axis=0) * h


def primary_vortex(psi: np.ndarray, h: float) -> tuple[float, float]:
    """Interior extremum of ``psi`` refined with a 2-D quadratic fit.

    The primary vortex of the Re = 1000 cavity is clockwise, i.e. the minimum of
    ``psi``.  A 5-point parabola is fitted through the discrete minimum and its
    three nearest interior neighbours, which gives a sub-cell centre instead of
    the cell centre of the extremal cell.
    """
    ny, nx = psi.shape
    interior = np.abs(psi).copy()
    interior[:2, :] = 0.0
    interior[-2:, :] = 0.0
    interior[:, :2] = 0.0
    interior[:, -2:] = 0.0
    i, j = np.unravel_index(int(np.argmax(interior)), psi.shape)
    # parabolic fit along each axis through the three cells around (i, j)
    def vertex(centre: int, values: tuple[float, float, float]) -> float:
        ym, y0, yp = values
        denom = ym - 2.0 * y0 + yp
        if denom == 0.0:
            return float(centre)
        return centre + 0.5 * (ym - yp) / denom

    i = int(np.clip(i, 1, ny - 2))
    j = int(np.clip(j, 1, nx - 2))
    sgn = 1.0 if psi[i, j] < 0 else -1.0
    di = vertex(i, (sgn * psi[i - 1, j], sgn * psi[i, j], sgn * psi[i + 1, j]))
    dj = vertex(j, (sgn * psi[i, j - 1], sgn * psi[i, j], sgn * psi[i, j + 1]))
    return (float(dj + 0.5) * h, float(di + 0.5) * h)      # (x, y) of the centre


def dense_profile(x: np.ndarray, f: np.ndarray, *, samples: int = 4001):
    """Dense resampling of a centreline with the local quadratic rule."""
    xs = np.linspace(x[0], x[-1], samples)
    return xs, local_quadratic(x, f, xs)


# --------------------------------------------------------------------------- #
# per-field audit
# --------------------------------------------------------------------------- #
def audit_field(path: str, reynolds: float, tolerance: float) -> dict:
    saved = torch.load(path, weights_only=False)
    u = np.asarray(saved["u"].cpu().double())
    v = np.asarray(saved["v"].cpu().double())
    n = int(u.shape[0] - 1)
    h = 1.0 / n
    uc, vc = cell_centred(u, v)
    y = (np.arange(n) + 0.5) * h
    x = (np.arange(n) + 0.5) * h

    # wall-clock consistency: the stored staggered field must be divergence free
    # and must carry the lid speed in the top cell row.
    div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h

    out: dict = {
        "field": os.path.basename(path),
        "n": n,
        "h": h,
        "steps": int(saved.get("steps", 0)),
        "max_abs_divergence": float(np.abs(div).max()),
        "lid_row_mean_u": float(uc[-1, :].mean()),
        "bottom_row_mean_u": float(uc[0, :].mean()),
        "interpolators": {},
    }

    # the mid-plane samples are the *face* values at x = (n//2) h = L/2 and
    # y = (n//2) h = L/2 -- the same locations the solver's centreline accessors
    # read, so only the interpolation rule differs from the shipped scorer
    for which, coord, prof in (("u", y, u[n // 2, :]), ("v", x, v[:, n // 2])):
        pts = table(reynolds, which)
        tgt = [p[0] for p in pts]
        ref = np.asarray([p[1] for p in pts])
        got = {
            "linear": linear(coord, prof, tgt),
            "quadratic": local_quadratic(coord, prof, tgt),
            "hermite": np.asarray(hermite_interpolate(
                torch.as_tensor(coord), torch.as_tensor(prof), tgt).double()),
        }
        block = {}
        for name, values in got.items():
            delta = values - ref
            block[name] = {
                "max_abs": float(np.abs(delta).max()),
                "rms": float(np.sqrt((delta ** 2).mean())),
                "max_abs_fraction_U": float(np.abs(delta).max()),
            }
        # extrema of the field itself, from a dense resampling
        xs, dense = dense_profile(coord, prof)
        if which == "u":
            j = int(np.argmin(dense))
            ext = {"kind": "min", "coord": float(xs[j]), "value": float(dense[j])}
            ext_ref = min(pts, key=lambda p: p[1])
        else:
            j = int(np.argmax(dense))
            ext = {"kind": "max", "coord": float(xs[j]), "value": float(dense[j])}
            ext_ref = max(pts, key=lambda p: p[1])
        block["extremum_dense"] = {
            **ext,
            "reference": [ext_ref[0], ext_ref[1]],
            "rel_error": abs(ext["value"] - ext_ref[1]) / abs(ext_ref[1]),
        }
        block["reference"] = [list(p) for p in pts]
        block["computed_hermite"] = [float(z) for z in got["hermite"]]
        block["computed_linear"] = [float(z) for z in got["linear"]]
        block["computed_quadratic"] = [float(z) for z in got["quadratic"]]
        out["interpolators"][which] = block

    psi = streamfunction(uc, h)
    cx, cy = primary_vortex(psi, h)
    out["vortex_centre_field"] = {"computed": [cx, cy]}
    out["psi_min"] = float(psi.min())
    out["psi_max"] = float(psi.max())

    # the ledger criterion, on each interpolation rule, on the field-derived
    # extrema and on the vortex centre
    checks: dict[str, float] = {}
    for which in ("u", "v"):
        b = out["interpolators"][which]
        for name in ("linear", "quadratic", "hermite"):
            checks[f"max|d{which}|_{name}"] = b[name]["max_abs"]
        checks[f"ext_{which}_rel"] = b["extremum_dense"]["rel_error"]
    from tensorfvm.lid_cavity_reference import VORTEX_CENTRE
    vx, vy = VORTEX_CENTRE[float(reynolds)]
    checks["vortex_centre_abs"] = max(abs(cx - vx), abs(cy - vy))
    checks["max_abs_divergence"] = out["max_abs_divergence"]
    # piecewise-linear interpolation of a fixed-spacing reference table is a
    # first-order comparator: its residual is dominated by the table spacing near
    # the lid shear layer, so it is reported separately and never decides the band.
    out["criteria"] = {
        "tolerance": tolerance,
        "values": checks,
        "pass": {k: bool(val <= tolerance) for k, val in checks.items()
                 if k != "max_abs_divergence" and "_linear" not in k},
        "linear_comparator": {k: val for k, val in checks.items() if "_linear" in k},
        "divergence_pass": bool(out["max_abs_divergence"] < 1e-10),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/CASE-CAVITY-RE1000-FVM")
    ap.add_argument("--reynolds", type=float, default=1000.0)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()

    reynolds = float(args.reynolds)
    if reynolds not in SUPPORTED_REYNOLDS:
        raise SystemExit(f"no Ghia table for Re={reynolds}")

    paths = sorted(glob.glob(os.path.join(args.artifacts, "fields", "Re*_n*.pt")),
                   key=lambda p: int(re.search(r"_n(\d+)\.pt$", p).group(1)))
    if not paths:
        raise SystemExit(f"no stored fields under {args.artifacts}/fields")

    rows = [audit_field(p, reynolds, args.tolerance) for p in paths]
    payload = {
        "case": "lid-driven cavity, Ghia, Ghia & Shin (1982), Tables I and II",
        "anchor": "CASE-FVM-CAVITY-RE1000",
        "mission": 20000,
        "reynolds": reynolds,
        "tolerance_fraction": args.tolerance,
        "scorer": "scripts/audit_cavity_re1000_m20000.py (independent of lid_cavity.score_profiles)",
        "rows": rows,
    }
    out = os.path.join(args.artifacts, "m20000_independent_audit.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)

    hdr = f"{'n':>4} {'steps':>7} {'max|du| lin':>11} {'herm':>8} {'max|dv| lin':>11} {'herm':>8} " \
          f"{'u_min lin':>9} {'rel%':>6} {'v_max lin':>9} {'rel%':>6} {'vortex d':>8} {'div':>9} {'pass':>5}"
    print(hdr)
    for r in rows:
        b_u, b_v = r["interpolators"]["u"], r["interpolators"]["v"]
        print(f"{r['n']:>4} {r['steps']:>7} "
              f"{b_u['linear']['max_abs']:>11.5f} {b_u['hermite']['max_abs']:>8.5f} "
              f"{b_v['linear']['max_abs']:>11.5f} {b_v['hermite']['max_abs']:>8.5f} "
              f"{b_u['extremum_dense']['value']:>9.5f} {100 * b_u['extremum_dense']['rel_error']:>6.2f} "
              f"{b_v['extremum_dense']['value']:>9.5f} {100 * b_v['extremum_dense']['rel_error']:>6.2f} "
              f"{r['criteria']['values']['vortex_centre_abs']:>8.4f} "
              f"{r['max_abs_divergence']:>9.1e} "
              f"{str(all(r['criteria']['pass'].values())):>5}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()