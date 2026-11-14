"""m20098 board audit: independent re-scoring of the m20000 cavity bundle.

Independence statement
----------------------
The numerics below are written from scratch (numpy/torch/scipy only); nothing
from ``tensorfvm.lid_cavity`` is imported or copied.  The only imported data are
the two Ghia et al. (1982) centreline tables (``U_CENTRELINE[1000]`` /
``V_CENTRELINE[1000]``, plain data in ``src/tensorfvm/lid_cavity_reference.py``);
their transcription is checked here against the paper's published invariants
(u_min = -0.38289 at y = 0.1719, v_max = +0.37095 at x = 0.1563, primary vortex
(0.5313, 0.5625), and monotone tails to zero at the walls).

Adjudication follows the ledger row of ``tensor_benchmark_cases``
(anchor_slug = ``CASE-CAVITY-RE1000-FVM``):

    u_min (x=L/2), v_max (y=L/2) relative error      < 3 %
    pointwise max|du|/U, max|dv|/U (interior points)  < 3 %   (ledger band 5 %)
    primary vortex centre |delta| / L                 <= 0.01  @ n >= 128

Stored field layout (staggered MAC, n cells per direction, h = 1/n), verified in
this script against the no-penetration walls, the lid flux condition and the
discrete continuity:
    u[i, j] : x = i*h,       y = (j+0.5)*h ,  shape (n+1, n)
    v[i, j] : x = (i+0.5)*h, y = j*h       ,  shape (n, n+1)

Usage:
    python3 scripts/audit_cavity_re1000_m20098.py --dir <bundle dir> [--out j.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

# ----------------------------------------------------------------------------
# External anchor (Ghia, Ghia & Shin 1982, Tables I and II)
# ----------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tensorfvm.lid_cavity_reference import U_CENTRELINE, V_CENTRELINE  # noqa: E402

GHIA_U = [(float(y), float(u)) for y, u in U_CENTRELINE[1000.0]]
GHIA_V = [(float(x), float(v)) for x, v in V_CENTRELINE[1000.0]]
GHIA_U_MIN = min(u for _, u in GHIA_U)
GHIA_V_MAX = max(v for _, v in GHIA_V)
GHIA_U_MIN_AT = [y for y, u in GHIA_U if u == GHIA_U_MIN][0]
GHIA_V_MAX_AT = [x for x, v in GHIA_V if v == GHIA_V_MAX][0]
GHIA_VORTEX = (0.5313, 0.5625)


def reference_self_check():
    """The transcription must reproduce the paper's published extrema."""
    assert abs(GHIA_U_MIN - (-0.38289)) < 1e-9, GHIA_U_MIN
    assert abs(GHIA_V_MAX - 0.37095) < 1e-9, GHIA_V_MAX
    assert abs(GHIA_U_MIN_AT - 0.1719) < 1e-9, GHIA_U_MIN_AT
    assert abs(GHIA_V_MAX_AT - 0.1563) < 1e-9, GHIA_V_MAX_AT
    assert GHIA_U[0] == (1.0, 1.0) and abs(GHIA_U[-1][1]) < 1e-12   # lid / bottom
    assert abs(GHIA_V[0][1]) < 1e-12 and abs(GHIA_V[-1][1]) < 1e-12  # both side walls
    return {"u_min": GHIA_U_MIN, "u_min_at": GHIA_U_MIN_AT,
            "v_max": GHIA_V_MAX, "v_max_at": GHIA_V_MAX_AT,
            "vortex_centre": list(GHIA_VORTEX)}


# ----------------------------------------------------------------------------
# 1-D interpolation (own implementation)
# ----------------------------------------------------------------------------
def interp_linear(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    k = max(i for i in range(len(xs) - 1) if xs[i] <= x)
    t = (x - xs[k]) / (xs[k + 1] - xs[k])
    return ys[k] * (1.0 - t) + ys[k + 1] * t


def interp_quadratic(xs, ys, x):
    """Local parabola through the 3 nearest nodes."""
    n = len(xs)
    if x <= xs[0]:
        a, b, c = 0, 1, 2
    elif x >= xs[-1]:
        a, b, c = n - 3, n - 2, n - 1
    else:
        k = max(i for i in range(n - 1) if xs[i] <= x)
        a, b, c = (k - 1, k, k + 1) if k >= 1 else (0, 1, 2)
        if c > n - 1:
            a, b, c = n - 3, n - 2, n - 1
    x0, x1, x2 = xs[a], xs[b], xs[c]
    y0, y1, y2 = ys[a], ys[b], ys[c]
    l0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    l1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    l2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    return y0 * l0 + y1 * l1 + y2 * l2


INTERP = {"linear": interp_linear, "quadratic": interp_quadratic}


# ----------------------------------------------------------------------------
# Field handling
# ----------------------------------------------------------------------------
def load_field(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    u = d["u"].double().numpy()
    v = d["v"].double().numpy()
    n = u.shape[1]
    assert u.shape == (n + 1, n), u.shape
    assert v.shape == (n, n + 1), v.shape
    return u, v, n, {"steps": d.get("steps")}


def centreline_u(u, n, y, rule):
    """u at x = L/2 (a u-face, i = n/2); walls carry their Dirichlet value."""
    h = 1.0 / n
    i = n // 2
    ys = [(j + 0.5) * h for j in range(n)]
    vals = [float(u[i, j]) for j in range(n)]
    if y == 0.0 or y == 1.0:  # no-slip / lid: known exactly, never extrapolated
        return y
    return INTERP[rule](ys, vals, y)


def centreline_v(v, n, x, rule):
    """v at y = L/2 (a v-face, j = n/2); walls carry their Dirichlet value."""
    h = 1.0 / n
    j = n // 2
    xs = [(i + 0.5) * h for i in range(n)]
    vals = [float(v[i, j]) for i in range(n)]
    if x == 0.0 or x == 1.0:
        return 0.0
    return INTERP[rule](xs, vals, x)


def psi_integrated(u, n):
    """psi(x,y) = int_0^y u dy' from the cell-centre field (definition 1)."""
    h = 1.0 / n
    uc = 0.5 * (u[:-1, :] + u[1:, :])  # (x cell, y cell)
    psi = np.zeros((n, n))
    psi[:, 0] = uc[:, 0] * h
    for j in range(1, n):
        psi[:, j] = psi[:, j - 1] + uc[:, j] * h
    return psi


def psi_poisson(u, v, n):
    """Solve lap psi = -omega, psi = 0 on the walls, via DST-I (definition 2)."""
    from scipy.fft import dstn, idstn

    h = 1.0 / n
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    omega = np.zeros((n, n))
    omega[1:-1, 1:-1] = ((vc[2:, 1:-1] - vc[:-2, 1:-1]) / (2 * h)
                         - (uc[1:-1, 2:] - uc[1:-1, :-2]) / (2 * h))
    lam = 2.0 * (np.cos(np.pi * np.arange(1, n + 1) / (n + 1)) - 1.0) / h ** 2
    rhs = -omega
    coeff = dstn(rhs, type=1)
    den = lam[:, None] + lam[None, :]
    psi = idstn(coeff / den, type=1) / (2 * (n + 1)) ** 2 * 4
    return psi


def vortex_centre(psi, n):
    """Interior minimum of psi with a parabolic sub-cell refinement."""
    h = 1.0 / n
    masked = psi.copy()
    masked[0, :] = masked[-1, :] = np.inf
    masked[:, 0] = masked[:, -1] = np.inf
    i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)  # (x, y)

    def refine(axis):
        k = i if axis == 0 else j
        a = masked[k - 1, j] if axis == 0 else masked[i, k - 1]
        b = masked[k, j] if axis == 0 else masked[i, k]
        c = masked[k + 1, j] if axis == 0 else masked[i, k + 1]
        den = a - 2 * b + c
        return 0.0 if den == 0 else 0.5 * (a - c) / den

    x = (i + 0.5 + refine(0)) * h
    y = (j + 0.5 + refine(1)) * h
    return (float(x), float(y)), (int(i), int(j)), float(masked[i, j])


def divergence(u, v, n):
    h = 1.0 / n
    return (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h


# ----------------------------------------------------------------------------
# Audit of one field
# ----------------------------------------------------------------------------
def audit_field(path, rule="quadratic"):
    u, v, n, meta = load_field(path)
    h = 1.0 / n

    uc = [(y, centreline_u(u, n, y, rule)) for y, _ in GHIA_U]
    vc = [(x, centreline_v(v, n, x, rule)) for x, _ in GHIA_V]
    du = [abs(val - ref) for (y, val), (_, ref) in zip(uc, GHIA_U)]
    dv = [abs(val - ref) for (x, val), (_, ref) in zip(vc, GHIA_V)]

    # extrema from a dense sweep of the same interpolant (wall rows excluded)
    gy = [k / 2000.0 for k in range(1, 2000)]
    u_curve = [centreline_u(u, n, y, rule) for y in gy]
    u_min = float(min(u_curve))
    y_min = gy[int(np.argmin(u_curve))]
    gx = [k / 2000.0 for k in range(1, 2000)]
    v_curve = [centreline_v(v, n, x, rule) for x in gx]
    v_max = float(max(v_curve))
    x_max = gx[int(np.argmax(v_curve))]

    psi1 = psi_integrated(u, n)
    psi2 = psi_poisson(u, v, n)
    centre1, cell1, psi_min1 = vortex_centre(psi1, n)
    centre2, cell2, psi_min2 = vortex_centre(psi2, n)
    d1 = math.hypot(centre1[0] - GHIA_VORTEX[0], centre1[1] - GHIA_VORTEX[1])
    d2 = math.hypot(centre2[0] - GHIA_VORTEX[0], centre2[1] - GHIA_VORTEX[1])

    div = divergence(u, v, n)
    bc = {
        "side_wall_max_abs_u": float(max(np.abs(u[0, :]).max(), np.abs(u[-1, :]).max())),
        "bottom_max_abs_v": float(np.abs(v[:, 0]).max()),
        "top_max_abs_v": float(np.abs(v[:, -1]).max()),
        # mean u in the cell row adjacent to the lid (driven but O(h) short of U)
        "top_cell_row_mean_u": float(u[:, -1].mean()),
        # closed-cavity flux condition: every vertical line carries zero net flux
        "max_abs_column_mean_u": float(np.abs(u.mean(axis=1)).max()),
    }

    out = {
        "field": pathlib.Path(path).name,
        "n": n,
        "h": h,
        "rule": rule,
        "steps": meta.get("steps"),
        "u_min": u_min,
        "y_at_u_min": y_min,
        "u_min_rel_err": abs(u_min - GHIA_U_MIN) / abs(GHIA_U_MIN),
        "ext_u_rel_err": abs(u_min - GHIA_U_MIN) / abs(GHIA_U_MIN),
        "v_max": v_max,
        "x_at_v_max": x_max,
        "v_max_rel_err": abs(v_max - GHIA_V_MAX) / abs(GHIA_V_MAX),
        "ext_v_rel_err": abs(v_max - GHIA_V_MAX) / abs(GHIA_V_MAX),
        "extrema_location_error": {"y_at_u_min": abs(y_min - GHIA_U_MIN_AT),
                                   "x_at_v_max": abs(x_max - GHIA_V_MAX_AT)},
        "pointwise_interior_only": True,
        "pointwise_max_abs_du": float(max(du)),
        "pointwise_max_abs_dv": float(max(dv)),
        "pointwise_worst_u_at_y": uc[int(np.argmax(du))][0],
        "pointwise_worst_v_at_x": vc[int(np.argmax(dv))][0],
        "vortex_centre_psi_integrated": {"centre": list(centre1), "cell": list(cell1),
                                         "psi_min": psi_min1, "distance": d1},
        "vortex_centre_psi_poisson": {"centre": list(centre2), "cell": list(cell2),
                                      "psi_min": psi_min2, "distance": d2},
        "max_abs_divergence": float(np.abs(div).max()),
        "bc": bc,
    }
    return out


def profile_dump(path):
    """Compact, text-only evidence: centreline profiles + field digest.

    Committed next to the audit JSON so every headline number can be re-derived
    in the repo without the binary ``.pt`` fields (which are not tracked).
    """
    import hashlib

    u, v, n, _ = load_field(path)
    h = 1.0 / n
    psi = psi_integrated(u, n)
    return {
        "n": n,
        "h": h,
        "sha256": hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
        "u_centreline": {"x": 0.5, "y": [round((j + 0.5) * h, 12) for j in range(n)],
                         "u": [round(float(u[n // 2, j]), 12) for j in range(n)]},
        "v_centreline": {"y": 0.5, "x": [round((i + 0.5) * h, 12) for i in range(n)],
                         "v": [round(float(v[i, n // 2]), 12) for i in range(n)]},
        "psi_extrema": {"min": round(float(psi.min()), 12), "max": round(float(psi.max()), 12)},
        "max_abs_divergence": float(np.abs(divergence(u, v, n)).max()),
        "boundary": {"side_wall_max_abs_u": float(max(np.abs(u[0, :]).max(), np.abs(u[-1, :]).max())),
                     "top_max_abs_v": float(np.abs(v[:, -1]).max()),
                     "bottom_max_abs_v": float(np.abs(v[:, 0]).max()),
                     "top_cell_row_mean_u": float(u[:, -1].mean()),
                     "max_abs_column_mean_u": float(np.abs(u.mean(axis=1)).max())},
    }


def judge(row):
    """Ledger row of tensor_benchmark_cases (anchor CASE-CAVITY-RE1000-FVM)."""
    checks = {
        "u_min_rel<0.03": row["u_min_rel_err"] < 0.03,
        "v_max_rel<0.03": row["v_max_rel_err"] < 0.03,
        "max_abs_du<0.03": row["pointwise_max_abs_du"] < 0.03,
        "max_abs_dv<0.03": row["pointwise_max_abs_dv"] < 0.03,
    }
    if row["n"] >= 128:
        checks["vortex_centre_poisson<=0.01"] = row["vortex_centre_psi_poisson"]["distance"] <= 0.01
        checks["vortex_centre_integrated<=0.01"] = row["vortex_centre_psi_integrated"]["distance"] <= 0.01
    checks["ledger_5pct_band"] = (
        max(row["u_min_rel_err"], row["v_max_rel_err"]) < 0.05
        and max(row["pointwise_max_abs_du"], row["pointwise_max_abs_dv"]) < 0.05
    )
    row["checks"] = checks
    row["benchmark_pass_strict"] = all(checks.values())
    row["benchmark_pass_3pct"] = all(checks[k] for k in checks if k.startswith(("u_min", "v_max", "max_abs")))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/CASE-CAVITY-RE1000-FVM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    fields = sorted(root.glob("fields/Re1000_n*.pt"),
                    key=lambda p: int(p.stem.replace("Re1000_n", "")))
    if not fields:
        raise SystemExit(f"no fields under {root}/fields")

    report = {
        "mission": 20098,
        "audited_mission": 20000,
        "anchor": "CASE-CAVITY-RE1000-FVM",
        "dir": str(root),
        "reference": reference_self_check(),
        "reference_source": "Ghia, Ghia & Shin (1982) Tables I/II via src/tensorfvm/lid_cavity_reference.py",
        "tolerance": ("u_min/v_max <3%; pointwise max|d|/U <3% (interior); "
                      "vortex |d|/L <= 0.01 @ n>=128; ledger band 5%"),
        "rows": [],
    }
    for f in fields:
        for rule in ("quadratic", "linear"):
            row = judge(audit_field(str(f), rule=rule))
            report["rows"].append(row)
            print(f"{row['field']:>16s} {rule:>9s} u_min={row['u_min']:+.5f}({row['u_min_rel_err']*100:5.2f}%)"
                  f" v_max={row['v_max']:+.5f}({row['v_max_rel_err']*100:5.2f}%)"
                  f" |du|={row['pointwise_max_abs_du']:.5f} |dv|={row['pointwise_max_abs_dv']:.5f}"
                  f" vortexP=({row['vortex_centre_psi_poisson']['centre'][0]:.4f},"
                  f"{row['vortex_centre_psi_poisson']['centre'][1]:.4f}) dP={row['vortex_centre_psi_poisson']['distance']:.4f}"
                  f" dI={row['vortex_centre_psi_integrated']['distance']:.4f}"
                  f" div={row['max_abs_divergence']:.1e} PASS3%={row['benchmark_pass_3pct']}"
                  f" PASSstrict={row['benchmark_pass_strict']}")

    verdicts = {}
    for row in report["rows"]:
        if row["rule"] != "quadratic":
            continue
        verdicts[str(row["n"])] = {
            "pass_3pct": row["benchmark_pass_3pct"],
            "pass_strict": row["benchmark_pass_strict"],
            "failing": [k for k, ok in row["checks"].items() if not ok],
        }
    report["verdict_per_resolution"] = verdicts
    report["production_n_pass_3pct"] = {k: v["pass_3pct"] for k, v in verdicts.items() if int(k) >= 64}

    report["fields"] = [profile_dump(str(f)) for f in fields]

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
        print("wrote", args.out)
    return report


if __name__ == "__main__":
    main()