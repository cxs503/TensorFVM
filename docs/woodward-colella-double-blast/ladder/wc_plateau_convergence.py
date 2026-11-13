"""Diagnosis of the open CASE-WOODWARD-COLELLA metric ("low plateau") + a
well-posed replacement, from stored profiles only.

Background
----------
``docs/CASE-WOODWARD-COLELLA-reference-verification.md`` reports four metrics
inside 10 % of the digitised published reference and one outside it: the
minimum density of the shallow low-density region (``low_plateau``), 14.2 % high
at nx = 400 **and** at nx = 800.  A metric that does not move when the grid is
refined four times may be (a) a standing discretisation error of the scheme, or
(b) a badly posed statistic.  This script decides between the two *from data*:

1. **Where is the minimum?**  The comparison window is ``x in [0.42, 0.98]``
   and the digitised reference curve starts at ``x = 0.3999``.  If the minimum
   of the reference sits within a cell of the window border it is not a trough
   minimum at all, it is the value of a monotone ramp at a boundary — an
   extremum by truncation.  Then its value is a property of the window, not of
   the solution, and a relative error on it is meaningless (a 1 % change of
   window would move it by 1 %).
2. **Does it converge?**  The same statistic is computed on the whole ladder
   (nx = 200 ... 3200) and on a *second, algorithmically unrelated* solver
   (WENO5 / SSP-RK3 / Rusanov, ``wc_ladder_weno5.py``) as well as on limiter /
   CFL variants.  A standing 14 % that survives a change of scheme is not
   numerical diffusion.
3. **What is the well-posed statistic?**  Integrals over fixed interior windows
   (``plateau_mean``: the mean density of the low-density region;
   ``window_mass``: the mass inside the digitised window) are smooth
   functionals of the solution: they converge monotonically and can be
   Richardson-extrapolated.  The script reports them for every level, the
   observed order, the extrapolated continuum value, and the error against the
   reference.

Nothing here relaxes a tolerance: the ill-posed statistic is *diagnosed*, not
deleted, and the well-posed ones are held to the same 10 % acceptance bound.

    python docs/woodward-colella-double-blast/ladder/wc_plateau_convergence.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

#: the acceptance bound of the feature-ledger entry (CASE-WOODWARD-COLELLA)
TOLERANCE_PERCENT = 10.0
#: the comparison window of the original metric (unchanged, for the diagnosis)
ORIGINAL_WINDOW = (0.42, 0.98)
#: interior sub-windows for the well-posed ``plateau_mean``; all inside the
#: low-density region, away from the digitised-domain border (0.3999) and from
#: the first steep front (x ~ 0.57 where rho_ref rises from 0.17 to 0.25)
PLATEAU_WINDOWS = ((0.42, 0.56), (0.44, 0.54), (0.46, 0.52))
#: primary plateau window used in the verdict
PLATEAU_WINDOW = (0.44, 0.54)
#: the digitised reference window (mass functional)
MASS_WINDOW = (0.40, 1.00)
#: probe point (interior, low slope, so a pointwise comparison is meaningful)
PROBE_X = 0.50
#: a window extremum closer than this to a border is "edge-degenerate"
EDGE_FRACTION = 0.05

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"
DOCS = _HERE.parent
ARTIFACTS = DOCS / "artifacts"
ROOT = DOCS.parents[1]


# ----------------------------------------------------------------------
# metrics on stored profiles (no solver)
# ----------------------------------------------------------------------
def window_min(x, rho, window):
    lo, hi = window
    sel = (x >= lo) & (x <= hi)
    if sel.sum() < 2:
        raise ValueError(f"window {window} holds {sel.sum()} points")
    xs, rs = x[sel], rho[sel]
    i = int(np.argmin(rs))
    return float(rs[i]), float(xs[i]), (lo, hi)


def edge_degeneracy(x, rho, window, fraction=EDGE_FRACTION):
    """Is the window minimum sitting on a window border (i.e. a truncation)?"""
    _, xmin, (lo, hi) = window_min(x, rho, window)
    width = hi - lo
    d_lo, d_hi = (xmin - lo) / width, (hi - xmin) / width
    return {"x_min": xmin,
            "distance_to_low_edge": d_lo,
            "distance_to_high_edge": d_hi,
            "edge_degenerate": bool(min(d_lo, d_hi) <= fraction),
            "nearest_edge": "low" if d_lo <= d_hi else "high"}


def window_mean(x, rho, window):
    lo, hi = window
    sel = (x >= lo) & (x <= hi)
    return float(np.trapezoid(rho[sel], x[sel]) / (hi - lo))


def window_mass(x, rho, window=MASS_WINDOW):
    lo, hi = window
    sel = (x >= lo) & (x <= hi)
    return float(np.trapezoid(rho[sel], x[sel]))


def probe(x, rho, x0=PROBE_X):
    return float(np.interp(x0, x, rho))


def metrics(x, rho, reference, xref=None, rref=None):
    """All metrics of one profile; ``reference`` is the loaded reference dict."""
    xref = reference["x"] if xref is None else xref
    rref = reference["rho"] if rref is None else rref
    lo_min, x_min, _ = window_min(x, rho, ORIGINAL_WINDOW)
    lo_ref, x_ref_min, _ = window_min(xref, rref, ORIGINAL_WINDOW)
    plateau_window = PLATEAU_WINDOW
    out = {
        "window_min": {"ref": lo_ref, "got": lo_min,
                       "x_ref": x_ref_min, "x_got": x_min,
                       "rel_error_percent": abs(lo_min - lo_ref) / lo_ref * 100},
        "min_edge": edge_degeneracy(x, rho, ORIGINAL_WINDOW),
        "min_edge_ref": edge_degeneracy(xref, rref, ORIGINAL_WINDOW),
        "plateau_mean": {
            "ref": window_mean(xref, rref, plateau_window),
            "got": window_mean(x, rho, plateau_window),
            "window": list(plateau_window),
        },
        "plateau_mean_windows": {
            f"{lo:.2f}-{hi:.2f}": {
                "ref": window_mean(xref, rref, (lo, hi)),
                "got": window_mean(x, rho, (lo, hi)),
            } for lo, hi in PLATEAU_WINDOWS},
        "window_mass": {"ref": window_mass(xref, rref), "got": window_mass(x, rho),
                        "window": list(MASS_WINDOW)},
        "probe": {"x": PROBE_X, "ref": probe(xref, rref), "got": probe(x, rho)},
    }
    out["plateau_mean"]["rel_error_percent"] = (
        abs(out["plateau_mean"]["got"] - out["plateau_mean"]["ref"])
        / out["plateau_mean"]["ref"] * 100)
    out["window_mass"]["rel_error_percent"] = (
        abs(out["window_mass"]["got"] - out["window_mass"]["ref"])
        / out["window_mass"]["ref"] * 100)
    out["probe"]["rel_error_percent"] = (
        abs(out["probe"]["got"] - out["probe"]["ref"]) / out["probe"]["ref"] * 100)
    for key, entry in out["plateau_mean_windows"].items():
        entry["rel_error_percent"] = abs(entry["got"] - entry["ref"]) / entry["ref"] * 100
    # the published metrics, for continuity with the previous report
    base = wcr.metrics_from_profiles(xref, rref, x, rho)
    out["published"] = {
        "peak_density_rel_percent": base["peak_density"]["rel_error_percent"],
        "l1_percent": base["l1_relative"]["percent"],
        "low_plateau_rel_percent": base["low_plateau"]["rel_error_percent"],
        "verdict": base["verdict"],
    }
    return out


# ----------------------------------------------------------------------
# convergence helpers (pure functions, unit-tested on synthetic series)
# ----------------------------------------------------------------------
def observed_order(errors, sizes) -> float | None:
    """Mean observed order ``p`` from ``err ~ C n^-p`` over >=2 levels."""
    e = np.asarray([x for x in errors], dtype=float)
    n = np.asarray(sizes, dtype=float)
    ok = np.isfinite(e) & (e > 0)
    e, n = e[ok], n[ok]
    if e.size < 2:
        return None
    p = np.diff(np.log(e)) / np.diff(np.log(n))
    # errors decay like n^-p, so the slope is -p; report p > 0
    return float(-p.mean()) if np.isfinite(p).all() else None


def richardson(values, sizes, order) -> float | None:
    """Extrapolate the two finest levels to the continuum limit."""
    if order is None or not np.isfinite(order) or order <= 0:
        return None
    v = np.asarray(values, dtype=float)
    n = np.asarray(sizes, dtype=float)
    if v.size < 2:
        return None
    r = n[-1] / n[-2]
    denom = r ** order - 1.0
    if abs(denom) < 1e-12:
        return None
    return float(v[-1] + (v[-1] - v[-2]) / denom)


def monotone_decreasing(series) -> bool:
    s = np.asarray(series, dtype=float)
    return bool(np.all(np.diff(s) < 0))


# ----------------------------------------------------------------------
# profile discovery
# ----------------------------------------------------------------------
def load_profiles(data_dir: Path, artifacts: Path) -> dict:
    """All stored profiles keyed by tag (``nx800``, ``nx1600_hllc``, ...)."""
    out: dict[str, dict] = {}
    for nx in (400, 800):
        p = artifacts / f"wc_blast_profile_nx{nx}.npz"
        if p.exists():
            d = np.load(p)
            out[f"nx{nx}_hllc"] = {"x": d["x_got"], "rho": d["rho_got"],
                                   "path": str(p), "source": str(p.relative_to(ROOT))}
    for p in sorted(data_dir.glob("ladder_profile_*.npz")):
        tag = p.stem.replace("ladder_profile_", "")
        d = np.load(p)
        out[tag] = {"x": d["x"], "rho": d["rho"],
                    "path": str(p), "source": str(p.relative_to(ROOT))}
    return out


def family(tag: str) -> str:
    return "weno5" if "weno5" in tag else "fvm"


def size_of(tag: str) -> int:
    m = re.match(r"nx(\d+)", tag)
    if not m:
        raise ValueError(f"no nx in tag {tag!r}")
    return int(m.group(1))


def is_baseline(tag: str) -> bool:
    """The production scheme: HLLC + MC limiter + CFL 0.4 (or WENO5 defaults)."""
    return tag.endswith("_hllc") or tag.endswith("_weno5")


# ----------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------
def analyse(data_dir: Path = DATA, artifacts: Path = ARTIFACTS,
            outdir: Path | None = None) -> dict:
    outdir = outdir or data_dir
    reference = wcr.load_reference()
    profiles = load_profiles(data_dir, artifacts)
    if not profiles:
        raise SystemExit("no stored profiles found")

    rows = {}
    for tag, prof in profiles.items():
        x = np.asarray(prof["x"], dtype=float)
        rho = np.asarray(prof["rho"], dtype=float)
        order = np.argsort(x)
        m = metrics(x[order], rho[order], reference)
        m.update({"tag": tag, "nx": size_of(tag), "family": family(tag),
                  "baseline": is_baseline(tag), "source": prof["source"],
                  "path": prof["path"]})
        rows[tag] = m

    def ladder(fam, baseline=True):
        items = [r for r in rows.values()
                 if r["family"] == fam and (r["baseline"] or not baseline)]
        return sorted(items, key=lambda r: r["nx"])

    result: dict = {
        "case": "CASE-WOODWARD-COLELLA",
        "t_end": 0.038,
        "tolerance_percent": TOLERANCE_PERCENT,
        "reference": {"path": reference["path"],
                      "panel_spread": reference["panel_spread"],
                      "domain": [float(reference["x"].min()), float(reference["x"].max())]},
        "windows": {"original_min": list(ORIGINAL_WINDOW),
                    "plateau_mean": list(PLATEAU_WINDOW),
                    "plateau_mean_robustness": [list(w) for w in PLATEAU_WINDOWS],
                    "window_mass": list(MASS_WINDOW),
                    "probe_x": PROBE_X},
        "rows": rows,
    }

    # --- diagnosis 1: where does the reference minimum sit? -----------------
    ref_min = rows[next(iter(rows))]["window_min"]
    result["diagnosis"] = {
        "window_min_position": {
            "ref_x": ref_min["x_ref"], "ref_value": ref_min["ref"],
            "window": list(ORIGINAL_WINDOW),
            "digitised_domain": [float(reference["x"].min()), float(reference["x"].max())],
            "edge_degeneracy_ref": rows[next(iter(rows))]["min_edge_ref"],
            "edge_degenerate_solutions": {
                tag: r["min_edge"]["edge_degenerate"] for tag, r in rows.items()},
        },
        "window_min_is_edge_degenerate": {
            "ref": rows[next(iter(rows))]["min_edge_ref"]["edge_degenerate"],
            "all_solutions": all(r["min_edge"]["edge_degenerate"]
                                 for r in rows.values()),
        },
    }

    # --- diagnosis 2: convergence of every statistic -----------------------
    families = {}
    for fam in ("fvm", "weno5"):
        lad = ladder(fam)
        if len(lad) < 2:
            families[fam] = {"levels": [r["nx"] for r in lad], "note": "insufficient levels"}
            continue
        sizes = [r["nx"] for r in lad]
        series = {
            "window_min_rel_percent": [r["window_min"]["rel_error_percent"] for r in lad],
            "plateau_mean_rel_percent": [r["plateau_mean"]["rel_error_percent"] for r in lad],
            "window_mass_rel_percent": [r["window_mass"]["rel_error_percent"] for r in lad],
            "probe_rel_percent": [r["probe"]["rel_error_percent"] for r in lad],
            "peak_density_rel_percent": [r["published"]["peak_density_rel_percent"] for r in lad],
            "l1_percent": [r["published"]["l1_percent"] for r in lad],
        }
        entry = {"levels": sizes, "values": series,
                 "observed_order": {k: observed_order(v, sizes) for k, v in series.items()},
                 "monotone_decreasing": {k: monotone_decreasing(v) for k, v in series.items()}}
        # extrapolate the smooth statistics whose own values (not errors) we have
        extrap = {}
        for key, getter in (
            ("plateau_mean", lambda r: r["plateau_mean"]["got"]),
            ("window_mass", lambda r: r["window_mass"]["got"]),
            ("probe", lambda r: r["probe"]["got"]),
        ):
            vals = [getter(r) for r in lad]
            errs = [r["plateau_mean"]["rel_error_percent"] if key == "plateau_mean"
                    else r["window_mass"]["rel_error_percent"] if key == "window_mass"
                    else r["probe"]["rel_error_percent"] for r in lad]
            p = observed_order(errs, sizes)
            lim = richardson(vals, sizes, p)
            ref_val = (lad[-1]["plateau_mean"]["ref"] if key == "plateau_mean"
                       else lad[-1]["window_mass"]["ref"] if key == "window_mass"
                       else lad[-1]["probe"]["ref"])
            extrap[key] = {"order": p, "value_finest": vals[-1], "value_limit": lim,
                           "ref": ref_val,
                           "limit_rel_error_percent": (None if lim is None else
                                                       abs(lim - ref_val) / ref_val * 100)}
        entry["richardson"] = extrap
        families[fam] = entry
    result["families"] = families

    # --- diagnosis 3: scheme sensitivity at fixed resolution ---------------
    variants: dict = {}
    by_nx: dict[int, list] = {}
    for r in rows.values():
        if r["baseline"]:
            continue
        by_nx.setdefault(r["nx"], []).append(r)
    for nx, group in sorted(by_nx.items()):
        base = next((r for r in rows.values()
                     if r["nx"] == nx and r["baseline"] and r["family"] == "fvm"), None)
        if base is None:
            continue
        variants[nx] = {
            "baseline_tag": base["tag"],
            "baseline_plateau_mean": base["plateau_mean"]["got"],
            "variants": {r["tag"]: {
                "plateau_mean": r["plateau_mean"]["got"],
                "delta_percent": (r["plateau_mean"]["got"] - base["plateau_mean"]["got"])
                                 / base["plateau_mean"]["got"] * 100,
                "window_min": r["window_min"]["got"],
                "window_min_delta_percent": (r["window_min"]["got"] - base["window_min"]["got"])
                                            / base["window_min"]["got"] * 100,
            } for r in group},
        }
    # cross-family comparison at shared sizes
    cross = {}
    fvm_by = {r["nx"]: r for r in rows.values()
              if r["family"] == "fvm" and r["baseline"]}
    wen_by = {r["nx"]: r for r in rows.values() if r["family"] == "weno5"}
    for nx in sorted(set(fvm_by) & set(wen_by)):
        a, b = fvm_by[nx], wen_by[nx]
        cross[nx] = {
            "fvm": {"plateau_mean": a["plateau_mean"]["got"],
                    "plateau_mean_rel_percent": a["plateau_mean"]["rel_error_percent"],
                    "window_min": a["window_min"]["got"],
                    "window_min_rel_percent": a["window_min"]["rel_error_percent"]},
            "weno5": {"plateau_mean": b["plateau_mean"]["got"],
                      "plateau_mean_rel_percent": b["plateau_mean"]["rel_error_percent"],
                      "window_min": b["window_min"]["got"],
                      "window_min_rel_percent": b["window_min"]["rel_error_percent"]},
            "plateau_mean_diff_percent": abs(a["plateau_mean"]["got"] - b["plateau_mean"]["got"])
                                         / a["plateau_mean"]["got"] * 100,
        }
    result["scheme_sensitivity"] = {"fixed_resolution_variants": variants,
                                    "cross_family": cross}

    # --- verdicts ----------------------------------------------------------
    best = ladder("fvm")[-1] if len(ladder("fvm")) >= 1 else None
    verdict: dict = {"tolerance_percent": TOLERANCE_PERCENT}
    if best is not None:
        verdict["finest_fvm"] = {
            "tag": best["tag"], "nx": best["nx"],
            "peak_density_rel_percent": best["published"]["peak_density_rel_percent"],
            "l1_percent": best["published"]["l1_percent"],
            "plateau_mean_rel_percent": best["plateau_mean"]["rel_error_percent"],
            "window_mass_rel_percent": best["window_mass"]["rel_error_percent"],
            "probe_rel_percent": best["probe"]["rel_error_percent"],
            "window_min_rel_percent": best["window_min"]["rel_error_percent"],
        }
        verdict["pass_fail"] = {
            "peak_density": best["published"]["peak_density_rel_percent"] < TOLERANCE_PERCENT,
            "l1": best["published"]["l1_percent"] < TOLERANCE_PERCENT,
            "plateau_mean": best["plateau_mean"]["rel_error_percent"] < TOLERANCE_PERCENT,
            "window_mass": best["window_mass"]["rel_error_percent"] < TOLERANCE_PERCENT,
            "probe": best["probe"]["rel_error_percent"] < TOLERANCE_PERCENT,
            "window_min": best["window_min"]["rel_error_percent"] < TOLERANCE_PERCENT,
        }
    result["verdict"] = verdict

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plateau_convergence.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    (outdir / "plateau_tables.md").write_text(render_tables(result))
    return result


# ----------------------------------------------------------------------
# generated tables (programmatic injection: no hand-copied numbers)
# ----------------------------------------------------------------------
def render_tables(result: dict) -> str:
    L: list[str] = []
    tol = result["tolerance_percent"]
    L.append("<!-- generated by wc_plateau_convergence.py; do not edit by hand -->")
    L.append("")
    L.append(f"### T1 — 全部已存剖面（容差 {tol:g} %，基准=数字化文献 Reference）")
    L.append("")
    L.append("| profile | nx | family | peak ρ % | L1 % | plateau_mean % | window_mass % | probe(x=0.50) % | window_min % | x(ρ_min) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, r in sorted(result["rows"].items(), key=lambda kv: (kv[1]["family"], kv[1]["nx"], kv[0])):
        L.append("| {tag} | {nx} | {family} | {peak:.2f} | {l1:.2f} | {pm:.2f} | {wm:.3f} | {pr:.2f} | {mn:.2f} | {xmin:.4f} |".format(
            tag=tag, nx=r["nx"], family=r["family"],
            peak=r["published"]["peak_density_rel_percent"],
            l1=r["published"]["l1_percent"],
            pm=r["plateau_mean"]["rel_error_percent"],
            wm=r["window_mass"]["rel_error_percent"],
            pr=r["probe"]["rel_error_percent"],
            mn=r["window_min"]["rel_error_percent"],
            xmin=r["window_min"]["x_got"]))
    L.append("")
    L.append("### T2 — 收敛性：误差随网格（各自 family 的基线格式）")
    L.append("")
    L.append("| family | stat | values over nx | observed order | monotone ↓ |")
    L.append("|---|---|---|---|---|")
    for fam, entry in result["families"].items():
        if "values" not in entry:
            L.append(f"| {fam} | — | {entry.get('note','')} | — | — |")
            continue
        for key, vals in entry["values"].items():
            p = entry["observed_order"].get(key)
            mono = entry["monotone_decreasing"].get(key)
            L.append("| {} | {} | {} | {} | {} |".format(
                fam, key, ", ".join(f"{v:.2f}" for v in vals),
                "n/a" if p is None else f"{p:.2f}", "yes" if mono else "**no**"))
    L.append("")
    L.append("### T3 — Richardson 外推（continuum limit vs Reference）")
    L.append("")
    L.append("| family | 量 | 最细网格值 | 观测阶 | 外推极限 | Reference | 极限误差 % |")
    L.append("|---|---|---|---|---|---|---|")
    for fam, entry in result["families"].items():
        for key, e in (entry.get("richardson") or {}).items():
            lim = e["value_limit"]
            err = e["limit_rel_error_percent"]
            L.append("| {} | {} | {:.5f} | {} | {} | {:.5f} | {} |".format(
                fam, key, e["value_finest"],
                "n/a" if e["order"] is None else f"{e['order']:.2f}",
                "n/a" if lim is None else f"{lim:.5f}", e["ref"],
                "n/a" if err is None else f"{err:.2f}"))
    L.append("")
    L.append("### T4 — 同网格格式敏感度（limiter / CFL 变化）")
    L.append("")
    L.append("| nx | baseline | variant | Δplateau_mean % | Δwindow_min % |")
    L.append("|---|---|---|---|---|")
    for nx, e in result["scheme_sensitivity"]["fixed_resolution_variants"].items():
        for tag, v in e["variants"].items():
            L.append("| {} | {} | {} | {:+.3f} | {:+.3f} |".format(
                nx, e["baseline_tag"], tag, v["delta_percent"], v["window_min_delta_percent"]))
    L.append("")
    L.append("### T5 — 跨格式族（MUSCL/HLLC vs WENO5/Rusanov）")
    L.append("")
    L.append("| nx | FVM plateau_mean % | WENO5 plateau_mean % | 互差 % | FVM window_min % | WENO5 window_min % |")
    L.append("|---|---|---|---|---|---|")
    for nx, e in result["scheme_sensitivity"]["cross_family"].items():
        L.append("| {} | {:.2f} | {:.2f} | {:.3f} | {:.2f} | {:.2f} |".format(
            nx, e["fvm"]["plateau_mean_rel_percent"], e["weno5"]["plateau_mean_rel_percent"],
            e["plateau_mean_diff_percent"], e["fvm"]["window_min_rel_percent"],
            e["weno5"]["window_min_rel_percent"]))
    L.append("")
    v = result.get("verdict", {})
    if "pass_fail" in v:
        L.append("### T6 — 判定（最细 FVM 网格 {}）".format(v["finest_fvm"]["tag"]))
        L.append("")
        L.append("| 指标 | 误差 % | <{} % ? |".format(tol))
        L.append("|---|---|---|")
        f = v["finest_fvm"]
        for key, label, val in (
            ("peak_density", "峰值密度", f["peak_density_rel_percent"]),
            ("l1", "剖面 L1", f["l1_percent"]),
            ("plateau_mean", "低密度区均值（良态）", f["plateau_mean_rel_percent"]),
            ("window_mass", "窗口质量积分（良态）", f["window_mass_rel_percent"]),
            ("probe", "定点点探针 x=0.50", f["probe_rel_percent"]),
            ("window_min", "窗口最小值（**病态**，见诊断）", f["window_min_rel_percent"]),
        ):
            L.append("| {} | {:.3f} | {} |".format(label, val, "✅" if v["pass_fail"][key] else "❌"))
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    res = analyse(args.data, args.artifacts, args.outdir)
    print(render_tables(res))
    d = res["diagnosis"]["window_min_is_edge_degenerate"]
    print("\nwindow-min edge-degenerate: ref=%s all_solutions=%s"
          % (d["ref"], d["all_solutions"]))


if __name__ == "__main__":
    main()