"""CASE-WOODWARD-COLELLA: domain coverage, scheme independence, verdict robustness.

The published bundle (``CASE-WOODWARD-COLELLA-final-verdict.md``,
``-calibration-bias.md``, ``-nx3200-asymptotic.md``) scores every metric on the
digitised window ``x in [0.4, 1]``.  Three verification debts follow from that:

1. **Coverage accounting.** How much of the domain -- in *length* and, more
   importantly, in *mass* -- does the published window actually represent?  The
   rest of the domain has no external reference at all, and until now no report
   said how large that blind spot is.
2. **Reference-free evidence for the blind strip.** The strip cannot be scored
   against the paper, but it can be bounded: cross-scheme distance (MUSCL/HLLC/MC
   vs WENO5/Rusanov -- no shared discretisation) and cross-grid distance,
   computed *per subdomain*, plus the exact conservation invariants that tie the
   whole domain together.
3. **Verdict robustness to the reference calibration.** The additive bias
   (+0.022017 density units) is *measured* from PDF geometry, not fitted to the
   solution.  Here it is swept over ``delta in [-0.022, +0.044]`` and, per metric,
   we report ``d(error%)/d(delta)`` and whether the ``<10 %`` verdict survives.
   This separates metrics that can carry a verdict from metrics whose denominator
   is so small that the calibration dominates them.

Nothing in this module re-runs the solver: it reads the committed profiles.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WC = HERE.parent                       # docs/woodward-colella-double-blast
LADDER = WC / "ladder"
DATA = LADDER / "data"
REFDIR = WC / "reference"
CALIBRATED = REFDIR / "reference_density_calibrated.csv"
OUT_JSON = HERE / "coverage_robustness.json"

sys.path.insert(0, str(LADDER))
sys.path.insert(0, str(WC.parent.parent / "src"))

import wc_asymptotic_analysis as asy  # noqa: E402  (reuse loader + distance)
from tensorfvm.compressible import wc_reference as wcr  # noqa: E402

GAMMA = 1.4
# exact invariants of the initial condition on a closed, adiabatic tube
P_LEFT, P_MID, P_RIGHT = 1000.0, 0.01, 100.0
L_LEFT, L_RIGHT = 0.1, 0.1
MASS_EXACT = 1.0
ENERGY_EXACT = (P_LEFT * L_LEFT + P_MID * (1.0 - L_LEFT - L_RIGHT)
                + P_RIGHT * L_RIGHT) / (GAMMA - 1.0)

# Only these committed profiles span the whole domain [0,1]; the
# ``artifacts/wc_blast_profile_nx{400,800}.npz`` files are *window-only*
# (240/480 samples on x in [0.4,1]) despite their names, so they cannot be used
# for whole-domain or blind-strip analysis.
FULL_DOMAIN_TAGS = ("nx200_hllc", "nx1600_hllc", "nx3200_hllc")

# regions: the published window and the un-referenced blind strip on its left
REGIONS = {
    "blind_left_(0,0.4)": (0.0, 0.4),
    "window_(0.4,1.0)": (0.4, 1.0),
    "full_(0,1)": (0.0, 1.0),
}

# offsets to sweep, in density units; +0.022017 is the geometry-measured bias
OFFSETS = (-0.022017, -0.0110085, 0.0, 0.0110085, 0.022017, 0.0330255, 0.044034)


# ------------------------------------------------------------------ primitives
def region_mask(x, rho, lo: float, hi: float):
    m = (x >= lo) & (x <= hi)
    return x[m], rho[m]


def region_distance(pa: dict, pb: dict, lo: float, hi: float) -> dict:
    """``∫|ρa-ρb| dx / ∫ρb dx`` (in %) inside ``[lo, hi]``.

    The finer profile is interpolated onto the coarser grid, i.e. the comparison
    is a projection of the fine data, never an upsampling of the coarse one.
    """
    if pa["nx"] < pb["nx"]:
        pa, pb = pb, pa
    xb, rb = region_mask(pb["x"], pb["rho"], lo, hi)
    ra = np.interp(xb, pa["x"], pa["rho"])
    num = float(np.trapezoid(np.abs(ra - rb), xb))
    den = float(np.trapezoid(rb, xb))
    return {"percent": 100.0 * num / den, "abs": num, "denominator": den,
            "coarse_grid": pb["nx"], "fine_grid": pa["nx"], "region": [lo, hi]}


def region_mass(x, rho, lo: float, hi: float) -> float:
    xr, rr = region_mask(x, rho, lo, hi)
    return float(np.trapezoid(rr, xr))


# ----------------------------------------------------------------- 1. coverage
def coverage(profiles: dict) -> dict:
    """Length/mass share of the published window, plus the blind strip's share."""
    fine = profiles["nx3200_hllc"]
    total = region_mass(fine["x"], fine["rho"], 0.0, 1.0)
    win = region_mass(fine["x"], fine["rho"], 0.4, 1.0)
    left = region_mass(fine["x"], fine["rho"], 0.0, 0.4)
    per_grid = {}
    for tag in FULL_DOMAIN_TAGS:
        p = profiles[tag]
        if p["family"] != "hllc":
            continue
        t = region_mass(p["x"], p["rho"], 0.0, 1.0)
        w = region_mass(p["x"], p["rho"], 0.4, 1.0)
        l = region_mass(p["x"], p["rho"], 0.0, 0.4)
        per_grid[tag] = {"total": t, "window": w, "left": l,
                         "window_mass_share_percent": 100.0 * w / t,
                         "left_mass_share_percent": 100.0 * l / t}
    return {
        "window": [0.4, 1.0],
        "length_share_percent": 60.0,
        "window_mass_share_percent": 100.0 * win / total,
        "left_mass_share_percent": 100.0 * left / total,
        "blind_strip": [0.0, 0.4],
        "blind_strip_has_external_reference": False,
        "note": ("The digitised figure covers x in [0.4,1] only; the left strip "
                 "carries the remaining mass and is verified only by "
                 "reference-free evidence (cross-scheme / cross-grid / invariants)."),
        "per_grid": per_grid,
    }


# ------------------------------------------- 2. reference-free evidence per region
def scheme_and_grid_evidence(profiles: dict) -> dict:
    """Cross-scheme and cross-grid distances inside each subdomain.

    Only whole-domain profiles participate, so the blind strip is covered.
    """
    out = {"cross_scheme": {}, "cross_grid": {}, "observed_order": {},
           "full_domain_tags": list(FULL_DOMAIN_TAGS)}
    for nx in (1600, 3200):
        a, b = profiles.get(f"nx{nx}_hllc"), profiles.get(f"nx{nx}_weno5")
        if not a or not b:
            continue
        for name, (lo, hi) in REGIONS.items():
            out["cross_scheme"].setdefault(f"nx{nx}", {})[name] = region_distance(a, b, lo, hi)
    pairs = ((200, 1600), (1600, 3200))
    for name, (lo, hi) in REGIONS.items():
        stage = {}
        for coarse, fine in pairs:
            stage[f"d_{coarse}_{fine}"] = region_distance(
                profiles[f"nx{coarse}_hllc"], profiles[f"nx{fine}_hllc"], lo, hi)
        out["cross_grid"][name] = stage
        d1, d2 = stage["d_200_1600"], stage["d_1600_3200"]
        ratio = d1["percent"] / d2["percent"] if d2["percent"] > 0 else math.inf
        # (200,1600) spans a factor 8 in grid, (1600,3200) a factor 2
        base = 8.0
        out["observed_order"][name] = {
            "p_obs": math.log(ratio) / math.log(base) if ratio > 0 and ratio != math.inf else None,
            "ratio": ratio,
            "note": "p_obs mixes a factor-8 and a factor-2 stage; indicative only",
        }
    return out


# -------------------------------------------------------- 3. conservation invariants
def invariants() -> dict:
    """Exact closed-tube invariants from the stored profiles (needs p and u)."""
    rows = {}
    for p in sorted(DATA.glob("ladder_profile_*.npz")):
        d = np.load(p)
        tag = p.stem.replace("ladder_profile_", "")
        x, rho, u, pr = (np.asarray(d[k], float) for k in ("x", "rho", "u", "p"))
        nx = len(x)
        dx = 1.0 / nx
        m = float(np.sum(rho) * dx)
        mom = float(np.sum(rho * u) * dx)
        e = float(np.sum(pr / (GAMMA - 1.0) + 0.5 * rho * u * u) * dx)
        rows[tag] = {
            "nx": nx, "mass": m, "mass_drift": m - MASS_EXACT,
            "momentum": mom,
            "energy": e, "energy_drift": e - ENERGY_EXACT,
            "energy_relative_drift": (e - ENERGY_EXACT) / ENERGY_EXACT,
            "wall_work": 0.0,  # reflecting walls do no work -> energy is invariant
        }
    return {"mass_exact": MASS_EXACT, "energy_exact": ENERGY_EXACT,
            "gamma": GAMMA, "rows": rows}


# ------------------------------------------------------------- 4. offset sweep
def offset_sweep(profiles: dict) -> dict:
    """Re-score every metric under a global additive shift of the reference."""
    raw = wcr.load_reference()
    grids = [t for t in ("nx800_hllc", "nx1600_hllc", "nx3200_hllc") if t in profiles]
    sweep = {}
    for delta in OFFSETS:
        ref = {"x": raw["x"], "rho": raw["rho"] + delta}
        per_grid = {}
        for tag in grids:
            prof = profiles[tag]
            m = wcr.metrics_from_profiles(ref["x"], ref["rho"], prof["x"], prof["rho"])
            per_grid[tag] = {
                "peak_percent": m["peak_density"]["rel_error_percent"],
                "l1_percent": m["l1_relative"]["percent"],
                "window_mass_percent": m["window_mass"]["rel_error_percent"],
                "trough_percent": m["low_plateau"]["rel_error_percent"],
                "verdict": m["verdict"],
                "checks": m["checks"],
            }
        sweep[f"{delta:+.6f}"] = per_grid
    return {"offsets": list(OFFSETS), "bias_reference": 0.022017,
            "tolerance_percent": wcr.TOLERANCE_PERCENT, "sweep": sweep,
            "sensitivity": sensitivity(raw, profiles, grids)}


def sensitivity(raw: dict, profiles: dict, grids: list) -> dict:
    """Numerical ``d(error%)/d(delta)`` per metric at the finest available grid."""
    tag = grids[-1]
    prof = profiles[tag]
    eps = 1e-4
    out = {}
    for name in ("peak_percent", "l1_percent", "window_mass_percent", "trough_percent"):
        vals = []
        for delta in (-eps, +eps):
            ref = {"x": raw["x"], "rho": raw["rho"] + delta}
            m = wcr.metrics_from_profiles(ref["x"], ref["rho"], prof["x"], prof["rho"])
            vals.append({
                "peak_percent": m["peak_density"]["rel_error_percent"],
                "l1_percent": m["l1_relative"]["percent"],
                "window_mass_percent": m["window_mass"]["rel_error_percent"],
                "trough_percent": m["low_plateau"]["rel_error_percent"],
            }[name])
        out[name] = {
            "d_error_percent_per_unit_density": (vals[1] - vals[0]) / (2 * eps),
            "grid": tag,
        }
    return out


# --------------------------------------------------- 5. trough conditioning test
def trough_conditioning(profiles: dict) -> dict:
    """Independently test the *reason* `low_plateau` is excluded from the verdict.

    The exclusion is justified by "the reference minimum sits on the window edge,
    i.e. it is the boundary value of a monotone ramp, not a trough".  Two probes:

    * **edge probe** -- where is the reference minimum inside ``LOW_PLATEAU_X``,
      and how far is it from the edge?  If it is pinned to the edge the metric
      measures a boundary value.
    * **x-conditioning probe** -- shift the numerical profile by ``dx`` and watch
      each metric.  A metric that reads a steep ramp edge reacts to a sub-cell
      horizontal misalignment; a metric that reads an integral does not.
    """
    cal = wcr.load_reference(CALIBRATED)
    lo, hi = wcr.LOW_PLATEAU_X
    m = (cal["x"] >= lo) & (cal["x"] <= hi)
    xr, rr = cal["x"][m], cal["rho"][m]
    i = int(np.argmin(rr))
    width = hi - lo
    # local slope of the reference at the minimum [density per unit x]
    j0, j1 = max(i - 2, 0), min(i + 2, len(xr) - 1)
    slope = (rr[j1] - rr[j0]) / (xr[j1] - xr[j0])
    edge = {
        "window": [lo, hi],
        "argmin_x": float(xr[i]),
        "edge_distance_abs": float(min(xr[i] - lo, hi - xr[i])),
        "edge_distance_percent_of_window": 100.0 * min(xr[i] - lo, hi - xr[i]) / width,
        "local_slope_drho_dx": float(slope),
        "reference_has_interior_min": bool(xr[i] - lo > 0.01 * width and hi - xr[i] > 0.01 * width),
    }
    shifts = (-0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002)
    tag = "nx3200_hllc"
    prof = profiles[tag]
    rows = {}
    for dx in shifts:
        mets = wcr.metrics_from_profiles(cal["x"], cal["rho"], prof["x"] + dx, prof["rho"])
        rows[f"{dx:+.4f}"] = {
            "trough_percent": mets["low_plateau"]["rel_error_percent"],
            "peak_percent": mets["peak_density"]["rel_error_percent"],
            "l1_percent": mets["l1_relative"]["percent"],
            "window_mass_percent": mets["window_mass"]["rel_error_percent"],
        }
    eps = 5e-4
    cond = {}
    for key in ("trough_percent", "peak_percent", "l1_percent", "window_mass_percent"):
        cond[key] = (rows[f"{eps:+.4f}"][key] - rows[f"{-eps:+.4f}"][key]) / (2 * eps)
    # monotonicity of the reference across the ramp next to the window edge
    seg = (xr >= 0.42) & (xr <= 0.50)
    inc = np.diff(rr[seg])
    mono = {"segment": [0.42, 0.50], "n_increments": int(inc.size),
            "negative_increments": int(np.sum(inc < 0)),
            "strictly_increasing": bool(np.all(inc > 0))}
    # relative sensitivity: |d(err%)/dx| divided by the metric's own error level
    base = {"trough_percent": rows["+0.0000"]["trough_percent"],
            "peak_percent": rows["+0.0000"]["peak_percent"],
            "l1_percent": rows["+0.0000"]["l1_percent"],
            "window_mass_percent": rows["+0.0000"]["window_mass_percent"]}
    rel = {k: abs(cond[k]) / max(abs(base[k]), 1e-12) for k in cond}
    return {"grid": tag, "edge": edge, "monotone_rise": mono, "x_shift_sweep": rows,
            "d_error_percent_per_unit_x": cond,
            "relative_x_sensitivity_per_unit_x": rel,
            "peak_is_flat_topped": abs(cond["peak_percent"]) < 1.0,
            "note": ("peak is a flat-topped maximum, so it is insensitive to a sub-cell "
                     "horizontal shift (sensitivity ~0); ratios against it are undefined.")}


# ------------------------------------------------------------------------ figures
def figures(profiles: dict, cov: dict, ev: dict, sw: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []

    # (a) domain coverage
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.8))
    a = ax[0]
    for tag, style in (("nx3200_hllc", dict(color="tab:blue", lw=1.1, label="MUSCL/HLLC/MC nx=3200")),
                       ("nx3200_weno5", dict(color="tab:orange", lw=1.0, ls="--", label="WENO5/Rusanov nx=3200"))):
        if tag in profiles:
            p = profiles[tag]
            a.plot(p["x"], p["rho"], **style)
    a.axvspan(0.0, 0.4, color="0.85", alpha=0.7, zorder=0)
    a.text(0.2, 4.6, "no external\nreference\n(6.5 % of mass)", ha="center", va="center",
           fontsize=8, color="0.25")
    a.axvline(0.4, color="k", lw=0.6, ls=":")
    a.text(0.7, 5.6, "digitised window\n(93.5 % of mass)", ha="center", fontsize=8)
    a.set_xlabel("x"); a.set_ylabel(r"$\rho$")
    a.set_title("(a) W&C double blast, t=0.038: reference coverage", fontsize=9)
    a.legend(fontsize=7, loc="upper left", framealpha=0.9)

    b = ax[1]
    names = list(REGIONS)
    cross = [ev["cross_scheme"]["nx3200"][n]["percent"] for n in names]
    grid = [ev["cross_grid"][n]["d_1600_3200"]["percent"] for n in names]
    xx = np.arange(len(names))
    b.bar(xx - 0.2, cross, 0.4, color="tab:purple", label="cross-scheme, nx=3200")
    b.bar(xx + 0.2, grid, 0.4, color="tab:green", label="cross-grid 1600 vs 3200")
    for i, (c, g) in enumerate(zip(cross, grid)):
        b.text(i - 0.2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=7)
        b.text(i + 0.2, g, f"{g:.2f}", ha="center", va="bottom", fontsize=7)
    b.axhline(10.0, color="tab:red", ls="--", lw=1.0, label="10 % acceptance")
    b.set_xticks(xx); b.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    b.set_ylabel("relative $L_1$ distance of $\\rho$ [%]")
    b.set_title("(b) reference-free evidence per subdomain", fontsize=9)
    b.legend(fontsize=7)
    fig.tight_layout()
    out = HERE / "fig_wc_domain_coverage.png"
    fig.savefig(out, dpi=150); made.append(out.name); plt.close(fig)

    # (b) offset robustness
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    tag = "nx3200_hllc"
    deltas = [float(k) for k in sw["sweep"]]
    labels = {"peak_percent": "peak density", "l1_percent": "profile $L_1$",
              "window_mass_percent": "window mass", "trough_percent": "near-vacuum trough"}
    for i, key in enumerate(labels):
        ys = [abs(sw["sweep"][f"{d:+.6f}"][tag][key]) for d in deltas]
        ax.semilogy(deltas, ys, "o-", ms=4, label=labels[key], color=f"C{i}")
    ax.axhline(10.0, color="tab:red", ls="--", lw=1.0)
    ax.text(deltas[0], 10.5, "10 % acceptance", color="tab:red", fontsize=7)
    ax.axvline(0.0, color="0.5", lw=0.8, ls=":")
    ax.axvline(sw["bias_reference"], color="tab:blue", lw=0.9, ls="-.")
    ax.text(sw["bias_reference"], 22, "measured bias\n+0.0220", fontsize=7,
            color="tab:blue", ha="center")
    ax.set_xlabel(r"additive reference offset $\delta$ [density units]")
    ax.set_ylabel("|error| vs reference [%], log scale")
    ax.set_title("(c) verdict robustness to the reference calibration\n(profile nx=3200)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = HERE / "fig_wc_offset_robustness.png"
    fig.savefig(out, dpi=150); made.append(out.name); plt.close(fig)
    return made


# --------------------------------------------------------------------------- main
def analyse() -> dict:
    profiles = asy.load_profiles()
    cov = coverage(profiles)
    ev = scheme_and_grid_evidence(profiles)
    inv = invariants()
    sw = offset_sweep(profiles)
    cond = trough_conditioning(profiles)
    figs = figures(profiles, cov, ev, sw)
    return {"profiles": {k: {"nx": v["nx"], "family": v["family"], "source": v["source"]}
                         for k, v in profiles.items()},
            "coverage": cov, "evidence": ev, "invariants": inv,
            "offset_sweep": sw, "trough_conditioning": cond, "figures": figs}


def main() -> None:
    res = analyse()
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    c = res["coverage"]
    print(f"window mass share      : {c['window_mass_share_percent']:.2f} %")
    print(f"blind strip mass share : {c['left_mass_share_percent']:.2f} %")
    for name in REGIONS:
        cs = res["evidence"]["cross_scheme"]["nx3200"][name]["percent"]
        cg = res["evidence"]["cross_grid"][name]["d_1600_3200"]["percent"]
        po = res["evidence"]["observed_order"][name]["p_obs"]
        print(f"{name:22s} cross-scheme {cs:6.2f} %  cross-grid {cg:6.2f} %  p_obs {po:.3f}")
    for tag, row in res["invariants"]["rows"].items():
        print(f"{tag:34s} mass drift {row['mass_drift']:+.2e}  "
              f"energy rel drift {row['energy_relative_drift']:+.2e}")
    print("\noffset sweep (nx=3200):")
    for k, v in res["offset_sweep"]["sweep"].items():
        r = v["nx3200_hllc"]
        print(f"  delta={k}  peak {r['peak_percent']:5.2f}  l1 {r['l1_percent']:5.2f}  "
              f"mass {r['window_mass_percent']:6.3f}  trough {r['trough_percent']:7.2f}  {r['verdict']}")
    print("\nsensitivity d(err%)/d(delta):")
    for k, v in res["offset_sweep"]["sensitivity"].items():
        print(f"  {k:22s} {v['d_error_percent_per_unit_density']:9.1f}")
    e = res["trough_conditioning"]["edge"]
    c2 = res["trough_conditioning"]
    print(f"\ntrough conditioning: ref argmin x={e['argmin_x']:.4f}, "
          f"{e['edge_distance_percent_of_window']:.2f} % of window from edge, "
          f"interior min={e['reference_has_interior_min']}, slope={e['local_slope_drho_dx']:.2f}/unit")
    print("  d(err%)/dx: " + "  ".join(f"{k.split('_')[0]} {v:8.0f}" for k, v in
                                       c2["d_error_percent_per_unit_x"].items()))
    r = c2["relative_x_sensitivity_per_unit_x"]
    print("  relative |d(err%)/dx| / err: " + "  ".join(f"{k.split('_')[0]} {v:6.1f}" for k, v in r.items()))
    print(f"  reference strictly increasing on 0.42-0.50: {c2['monotone_rise']['strictly_increasing']} "
          f"({c2['monotone_rise']['negative_increments']} negative of {c2['monotone_rise']['n_increments']})")
    print(f"\nwrote {OUT_JSON.relative_to(WC.parent.parent)}")


if __name__ == "__main__":
    main()