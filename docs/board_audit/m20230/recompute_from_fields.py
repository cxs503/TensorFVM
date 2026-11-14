#!/usr/bin/env python3
"""Independent recomputation of the CASE-FVM-CAVITY-RE1000 verdict from raw fields.

Audit mission m20230 (board audit of m20144).  Every number produced here is
derived from the shipped staggered velocity fields
(``refs/docs/artifacts/fields/Re1000_n*.pt``) by this script's own scoring code:
no audited scoring routine is imported and no value from ``summary.json`` is
reused.  ``summary.json`` is only *read afterwards* to diff against.

Checks
------
S1  field integrity: shapes, dtype, wall/lid boundary conditions, continuity
S2  independent Ghia Table I/II transcription typed from the paper (u at x=0.5,
    v at y=0.5, Re=1000) vs the module's committed transcription
S3  u_min, v_max, max|u-u_Ghia|, max|v-v_Ghia| recomputed from the fields
S4  primary-vortex centre recomputed from an independently integrated
    streamfunction
S5  cold vs continued n=64 field: is grid continuation steady-state neutral
S6  DB referee row (tensor_benchmark_cases) applied as the sole gate

Run:  python3 docs/board_audit/m20230/recompute_from_fields.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

HERE = pathlib.Path(__file__).resolve().parent
FIELDS = HERE / "refs" / "docs" / "artifacts" / "fields"
SUMMARY = HERE / "refs" / "docs" / "artifacts" / "summary.json"

# ---------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982) J. Comput. Phys. 48, 387-411, Re = 1000.
# This copy was typed by the auditor and then *corrected*: the first typing had
# four wrong u entries (y = 0.5, 0.4531, 0.2813) and five wrong v entries
# (x = 0.5, 0.0938, 0.0781, 0.0703, 0.0625) -- memory drift, caught because the
# converged field disagreed with it by up to 0.06 U in the smooth mid-cavity
# region while agreeing to < 0.005 everywhere else.  The final copy is
# corroborated entry-by-entry against an independent transcription that lives in
# another repository (TensorLBM src/tensorlbm/lid_driven_cavity.py GHIA_RE1000;
# check S2 below) and against the audited module's table.
GHIA_U_RE1000 = [  # (y, u)
    (1.0000, 1.00000), (0.9766, 0.65928), (0.9688, 0.57492), (0.9609, 0.51117),
    (0.9531, 0.46604), (0.8516, 0.33304), (0.7344, 0.18719), (0.6172, 0.05702),
    (0.5000, -0.06080), (0.4531, -0.10648), (0.2813, -0.27805), (0.1719, -0.38289),
    (0.1016, -0.29730), (0.0703, -0.22220), (0.0625, -0.20196), (0.0547, -0.18109),
    (0.0000, 0.00000),
]
GHIA_V_RE1000 = [  # (x, v)
    (1.0000, 0.00000), (0.9688, -0.21388), (0.9609, -0.27669), (0.9531, -0.33714),
    (0.9453, -0.39188), (0.9063, -0.51550), (0.8594, -0.42665), (0.8047, -0.31966),
    (0.5000, 0.02526), (0.2344, 0.32235), (0.2266, 0.33075), (0.1563, 0.37095),
    (0.0938, 0.32627), (0.0781, 0.30353), (0.0703, 0.29012), (0.0625, 0.27485),
    (0.0000, 0.00000),
]
GHIA_VORTEX_CENTRE = (0.5313, 0.5625)

# The registered referee row of tensor_benchmark_cases, anchor
# CASE-CAVITY-RE1000-FVM (read by sql in the audit driver, restated here so the
# gate this script applies is explicit):
DB_TOL = {
    "extremum_rel": 0.03,        # u_min and v_max relative error < 3 %
    "profile_max_abs": 0.03,     # pointwise max|du|/U, max|dv|/U < 3 % (interior)
    "vortex_centre_abs": 0.01,   # |d|/L <= 0.01 at n >= 128
    "ledger_band": 0.05,         # case-pool ledger band 5 %
}


def load(n: int, stem: str | None = None) -> dict:
    return torch.load(FIELDS / (stem or f"Re1000_n{n}.pt"), weights_only=False)


def cell_centred(f: dict) -> tuple[np.ndarray, np.ndarray, int]:
    u = f["u"].numpy().astype(float)          # (n+1, n) x-faces
    v = f["v"].numpy().astype(float)          # (n, n+1) y-faces
    uc = 0.5 * (u[:-1, :] + u[1:, :])         # (n, n) at cell centres
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc, u.shape[0] - 1


def divergence(f: dict, n: int) -> float:
    u = f["u"].numpy().astype(float)
    v = f["v"].numpy().astype(float)
    h = 1.0 / n
    div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
    return float(np.abs(div).max())


def profile_u(f: dict, n: int):
    """u along x = 0.5: u lives on x-faces, so face index n//2 is exactly x = 0.5."""
    u = f["u"].numpy().astype(float)
    h = 1.0 / n
    ys = (np.arange(n) + 0.5) * h
    us = u[n // 2, :].copy()
    return np.concatenate(([0.0], ys, [1.0])), np.concatenate(([0.0], us, [1.0]))


def profile_v(f: dict, n: int):
    """v along y = 0.5: v lives on y-faces, so face index n//2 is exactly y = 0.5."""
    v = f["v"].numpy().astype(float)
    h = 1.0 / n
    xs = (np.arange(n) + 0.5) * h
    vs = v[:, n // 2].copy()
    return np.concatenate(([0.0], xs, [1.0])), np.concatenate(([0.0], vs, [0.0]))


def score(n: int, f: dict, kind: str = "pchip") -> dict:
    ys, us = profile_u(f, n)
    xs, vs = profile_v(f, n)
    interp_u = (PchipInterpolator if kind == "pchip" else
                (lambda a, b: lambda q: np.interp(q, a, b)))(ys, us)
    interp_v = (PchipInterpolator if kind == "pchip" else
                (lambda a, b: lambda q: np.interp(q, a, b)))(xs, vs)

    gy = np.array([p[0] for p in GHIA_U_RE1000])
    gu = np.array([p[1] for p in GHIA_U_RE1000])
    gx = np.array([p[0] for p in GHIA_V_RE1000])
    gv = np.array([p[1] for p in GHIA_V_RE1000])
    uq = np.asarray(interp_u(gy), dtype=float)
    vq = np.asarray(interp_v(gx), dtype=float)
    du = uq - gu
    dv = vq - gv

    # extrema of the computed profiles over the published sample set + the
    # interior cell centres (a finer search cannot miss a deeper extremum)
    y_all = np.concatenate((ys, gy))
    u_all = np.concatenate((us, uq))
    x_all = np.concatenate((xs, gx))
    v_all = np.concatenate((vs, vq))
    u_min = float(u_all.min())
    y_umin = float(y_all[np.argmin(u_all)])
    v_max = float(v_all.max())
    x_vmax = float(x_all[np.argmax(v_all)])

    ref_u_min, ref_v_max = -0.38289, 0.37095
    res = {
        "n": n,
        "u_min_got": u_min, "u_min_at_y": y_umin, "u_min_ref": ref_u_min,
        "u_min_rel_err": abs(u_min - ref_u_min) / abs(ref_u_min),
        "v_max_got": v_max, "v_max_at_x": x_vmax, "v_max_ref": ref_v_max,
        "v_max_rel_err": abs(v_max - ref_v_max) / abs(ref_v_max),
        "u_max_abs_dev": float(np.abs(du).max()),
        "u_max_abs_dev_at_y": float(gy[int(np.argmax(np.abs(du)))]),
        "v_max_abs_dev": float(np.abs(dv).max()),
        "v_max_abs_dev_at_x": float(gx[int(np.argmax(np.abs(dv)))]),
        "u_rms_dev": float(np.sqrt(np.mean(du ** 2))),
        "v_rms_dev": float(np.sqrt(np.mean(dv ** 2))),
        "interior_only": {
            "u_max_abs_dev": float(np.abs(du[(gy > 0) & (gy < 1)]).max()),
            "v_max_abs_dev": float(np.abs(dv[(gx > 0) & (gx < 1)]).max()),
        },
        "interp": kind,
    }
    # S4: vortex centre from an independently integrated streamfunction,
    # psi(x, y) = int_0^y u dy with psi = 0 on the floor.  Two collocations are
    # reported because the stored location is quantised to the grid.
    h2 = 1.0 / n
    uf = f["u"].numpy().astype(float)                    # (n+1, n) x-faces
    psi_face = np.cumsum(uf, axis=1) * h2                # (n+1, n) at x = i*h
    inner = psi_face[1:n, :]
    kf = np.unravel_index(int(np.argmin(inner)), inner.shape)
    cf = ((kf[0] + 1) * h2, (kf[1] + 0.5) * h2)
    uc, _, _ = cell_centred(f)
    psi_cell = np.cumsum(uc, axis=1) * h2                # (n, n) at cell centres
    kc = np.unravel_index(int(np.argmin(psi_cell)), psi_cell.shape)
    cc = ((kc[0] + 0.5) * h2, (kc[1] + 0.5) * h2)
    res["vortex_centre_euclid"] = {
        "face_collocation": [float(cf[0]), float(cf[1])],
        "cell_collocation": [float(cc[0]), float(cc[1])],
        "psi_min": float(inner[kf]),
        "abs_error": float(max(abs(cf[0] - GHIA_VORTEX_CENTRE[0]),
                               abs(cf[1] - GHIA_VORTEX_CENTRE[1]))),
        "abs_error_cell_collocation": float(max(abs(cc[0] - GHIA_VORTEX_CENTRE[0]),
                                                abs(cc[1] - GHIA_VORTEX_CENTRE[1]))),
    }
    res["max_divergence"] = divergence(f, n)
    return res


def bc_report(f: dict, n: int) -> dict:
    u = f["u"].numpy().astype(float)
    v = f["v"].numpy().astype(float)
    return {
        "u_shape": list(u.shape), "v_shape": list(v.shape),
        "u_dtype": str(u.dtype), "steps_recorded": int(f.get("steps", -1)),
        "lid_u_minus_1": float(np.abs(u[-1, :] - 1.0).max()),
        "bottom_u_absmax": float(np.abs(u[0, :]).max()),
        "x_wall_u_absmax": float(np.abs(u[:, [0, -1]]).max()),
        "y_wall_v_absmax": float(np.abs(v[[0, -1], :]).max()),
        "side_wall_v_absmax": float(np.abs(v[:, [0, -1]]).max()),
        "u_max": float(u.max()), "v_max": float(v.max()),
        "u_interior_absmax": float(np.abs(u[1:-1, :]).max()),
    }


def db_gate(s: dict) -> dict:
    """Apply the registered referee row of tensor_benchmark_cases as the gate."""
    checks = {
        "u_min_rel<3%": s["u_min_rel_err"] <= DB_TOL["extremum_rel"],
        "v_max_rel<3%": s["v_max_rel_err"] <= DB_TOL["extremum_rel"],
        "max|du|/U<3%": s["interior_only"]["u_max_abs_dev"] <= DB_TOL["profile_max_abs"],
        "max|dv|/U<3%": s["interior_only"]["v_max_abs_dev"] <= DB_TOL["profile_max_abs"],
        "vortex|d|<=0.01@n>=128": (s["n"] < 128
                                   or s["vortex_centre_euclid"]["abs_error"]
                                   <= DB_TOL["vortex_centre_abs"]),
        "u_min_rel<5% ledger": s["u_min_rel_err"] <= DB_TOL["ledger_band"],
        "v_max_rel<5% ledger": s["v_max_rel_err"] <= DB_TOL["ledger_band"],
        "max|du|/U<5% ledger": s["u_max_abs_dev"] <= DB_TOL["ledger_band"],
        "max|dv|/U<5% ledger": s["v_max_abs_dev"] <= DB_TOL["ledger_band"],
        "divergence<1e-10": s["max_divergence"] < 1e-10,
    }
    return {"checks": checks, "pass": all(checks.values())}


def reference_crosscheck() -> dict:
    """S2: three independent copies of the Ghia Re = 1000 tables must agree."""
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    res: dict = {}
    try:
        mod = load(HERE / "refs" / "src" / "lid_cavity_reference.py", "aud_lid_ref")
        res["module_u"] = list(mod.table(1000.0, "u"))
        res["module_v"] = list(mod.table(1000.0, "v"))
        res["module_vortex"] = list(mod.VORTEX_CENTRE[1000.0])
    except Exception as exc:                       # pragma: no cover
        res["module_error"] = repr(exc)
        return res
    res["module_u_matches_auditor"] = (
        [(round(float(a), 4), round(float(b), 5)) for a, b in res["module_u"]]
        == [(round(a, 4), round(b, 5)) for a, b in GHIA_U_RE1000])
    res["module_v_matches_auditor"] = (
        [(round(float(a), 4), round(float(b), 5)) for a, b in res["module_v"]]
        == [(round(a, 4), round(b, 5)) for a, b in GHIA_V_RE1000])
    # second repository, independent lineage of the same published table
    lbm = pathlib.Path("/srv/tensor/workspace/TensorLBM/src/tensorlbm/lid_driven_cavity.py")
    if lbm.exists():
        try:
            src = lbm.read_text()
            ns: dict = {}
            start = src.index("GHIA_RE1000")
            block = src[start:src.index("}", src.index("\"v\": [", start)) + 1]
            exec(block, ns)                       # noqa: S102 - plain literal dict
            tab = ns["GHIA_RE1000"]
            lmb_u = [(round(float(a), 4), round(float(b), 5))
                     for a, b in zip(tab["y"], tab["u"])]
            lmb_v = [(round(float(a), 4), round(float(b), 5))
                     for a, b in zip(tab["x"], tab["v"])]
            res["tensorlbm_u_matches_auditor"] = lmb_u == [
                (round(a, 4), round(b, 5)) for a, b in GHIA_U_RE1000]
            res["tensorlbm_v_matches_auditor_any_order"] = (
                sorted(lmb_v) == sorted([(round(a, 4), round(b, 5))
                                         for a, b in GHIA_V_RE1000]))
            res["tensorlbm_source"] = str(lbm)
        except Exception as exc:                   # pragma: no cover
            res["tensorlbm_error"] = repr(exc)
    else:
        res["tensorlbm_source"] = "absent"
    return res


def main() -> int:
    out: dict = {"inputs": {}, "profiles": {}, "gates": {}, "diff_vs_summary": {}}
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    by_n = {int(c["n"]): c for c in summary.get("cases", [])}

    for n in (32, 64, 128):
        f = load(n)
        out["inputs"][f"n{n}"] = bc_report(f, n)
        rows = {}
        for kind in ("pchip", "linear"):
            rows[kind] = score(n, f, kind)
        out["profiles"][f"n{n}"] = rows
        out["gates"][f"n{n}"] = db_gate(rows["pchip"])
        c = by_n.get(n, {})
        if c:
            out["diff_vs_summary"][f"n{n}"] = {
                "u_min": {"shipped": c["u"]["extremum"]["min_computed"][1],
                          "recomputed": rows["pchip"]["u_min_got"]},
                "v_max": {"shipped": c["v"]["extremum"]["max_computed"][1],
                          "recomputed": rows["pchip"]["v_max_got"]},
                "u_max_abs": {"shipped": c["u"]["max_abs_error"],
                              "recomputed_pchip": rows["pchip"]["u_max_abs_dev"],
                              "recomputed_linear": rows["linear"]["u_max_abs_dev"]},
                "v_max_abs": {"shipped": c["v"]["max_abs_error"],
                              "recomputed_pchip": rows["pchip"]["v_max_abs_dev"],
                              "recomputed_linear": rows["linear"]["v_max_abs_dev"]},
                "vortex_centre": {"shipped": c["vortex_centre"]["computed"],
                                  "recomputed": rows["pchip"]["vortex_centre_euclid"]["face_collocation"]},
                "max_divergence": {"shipped": c["max_divergence"],
                                   "recomputed": rows["pchip"]["max_divergence"]},
                "steps": {"shipped": c["solver"]["steps"],
                          "fields_file": c["solver"]["fields_file"]},
            }

    # S2: the audited module's transcription vs this auditor copy vs TensorLBM's
    out["reference_crosscheck"] = reference_crosscheck()

    # S5: continuation neutrality, recomputed from the two n = 64 fields
    a, b = load(64), load(64, "Re1000_n64_from32.pt")
    out["continuation"] = {
        "max_abs_du": float((a["u"] - b["u"]).abs().max()),
        "max_abs_dv": float((a["v"] - b["v"]).abs().max()),
        "steps_cold": int(a["steps"]), "steps_continued": int(b["steps"]),
        "shipped": summary.get("continuation_check", {}),
    }
    out["continuation"]["agrees_with_shipped"] = (
        abs(out["continuation"]["max_abs_du"]
            - out["continuation"]["shipped"].get("max_abs_du", -1)) < 1e-12)
    out["gate_overall_pchip_n128"] = out["gates"]["n128"]["pass"]

    (HERE / "recompute.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gates", "continuation")}, indent=1))
    for n in (32, 64, 128):
        s = out["profiles"][f"n{n}"]["pchip"]
        print(f"n={n:3d} u_min={s['u_min_got']:+.6f} ({100*s['u_min_rel_err']:.3f}%) "
              f"v_max={s['v_max_got']:+.6f} ({100*s['v_max_rel_err']:.3f}%) "
              f"max|du|={s['u_max_abs_dev']:.5f} max|dv|={s['v_max_abs_dev']:.5f} "
              f"vortex={s['vortex_centre_euclid']['face_collocation']} "
              f"div={s['max_divergence']:.2e}")
    print("inputs:", json.dumps(out["inputs"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())