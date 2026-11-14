"""Asymptotic-range convergence and reference-uncertainty budget (W&C double blast).

Why this script exists
----------------------
``wc_gci_analysis.py`` ran the ASME V&V-20 three-grid GCI study on nx = 200/400/800
and found the observed order far below the formal one (0.25 for the peak).  Those
three grids are *not* in the asymptotic range for a solution containing two
colliding shocks, so the extrapolated peak (9.02, i.e. 43 % above the published
reference) was an artefact and no honest claim could be made from it.  Since then
the ladder has been extended to nx = 1600 and 3200 (``wc_ladder.py``); this script
does the three things that the fine data finally allow:

1. **Fine-triplet GCI.**  Repeat the three-grid study on nx = 3200/1600/800 where
   the solution *is* closer to the asymptotic range, for three quantities that are
   physically meaningful and mesh-independent: the window mass ``∫ρ dx`` over
   ``[0.4, 1]``, the peak density in the window, and the L1 distance to the
   calibrated digitised reference.
2. **Field-level convergence.**  The quantities above can converge while the field
   does not, so the L1 distance between successive grids is measured directly and
   an observed order is extracted from it.  Richardson extrapolation then produces
   an extrapolated *field*, whose distance to (a) the finest solution and (b) the
   calibrated reference separates discretisation error from reference error.
3. **Uncertainty budget.**  For every judged quantity, the numerical uncertainty
   ``U_num`` (GCI band at the fine triplet) is set against the residual distance
   to the published reference.  If ``|extrapolated - reference|`` exceeds ``U_num``
   the residual is *not* our discretisation, and the <10 % acceptance can only be
   quoted to the accuracy of the reference itself - which is the honest statement
   the ledger entry needs.

Cross-code: when the independent WENO5/Rusanov profiles are present (``wc_ladder_weno5.py``)
the same distances are computed between the two families, and their extrapolated
fields are compared, so the *basis* (discretisation-family) error is bounded
independently of any published curve.

Nothing here calls a solver: it reads stored profiles only, so it re-runs in
seconds and every number in the report is reproducible from committed artefacts.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_asymptotic_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr
from tensorfvm.verification.richardson import reference_agreement, richardson_gci

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARTIFACTS = HERE.parent / "artifacts"
REFERENCE = HERE.parent / "reference"
CALIBRATED = REFERENCE / "reference_density_calibrated.csv"

THEORY_ORDER = 2.0          # MUSCL-Hancock / WENO5 formal orders differ; see per-family override
FINE_TRIPLET = (3200, 1600, 800)
COARSE_TRIPLET = (800, 400, 200)


# --------------------------------------------------------------------------- io
def load_profiles() -> dict[str, dict]:
    """Every stored density profile, keyed by tag, normalised to ``x, rho``."""
    out: dict[str, dict] = {}
    for nx in (400, 800):
        p = ARTIFACTS / f"wc_blast_profile_nx{nx}.npz"
        if p.exists():
            d = np.load(p)
            out[f"nx{nx}_hllc"] = {"x": d["x_got"], "rho": d["rho_got"], "nx": nx,
                                   "family": "hllc", "source": str(p.relative_to(HERE.parent))}
    for p in sorted(DATA.glob("ladder_profile_*.npz")):
        tag = p.stem.replace("ladder_profile_", "")
        d = np.load(p)
        nx = int(tag.split("_")[0].replace("nx", ""))
        family = "weno5" if "weno5" in tag else "hllc"
        out[tag] = {"x": d["x"], "rho": d["rho"], "nx": nx, "family": family,
                    "source": str(p.relative_to(HERE.parent))}
    for tag in out:
        order = np.argsort(out[tag]["x"])
        out[tag]["x"] = np.asarray(out[tag]["x"], float)[order]
        out[tag]["rho"] = np.asarray(out[tag]["rho"], float)[order]
    return out


def window_mask(x, rho, lo=None, hi=None):
    lo = wcr.REFERENCE_X[0] if lo is None else lo
    hi = wcr.REFERENCE_X[1] if hi is None else hi
    m = (x >= lo) & (x <= hi)
    return x[m], rho[m]


def window_mass(x, rho) -> float:
    """``∫ρ dx`` over the comparison window (trapezoid on the given grid)."""
    xw, rw = window_mask(x, rho)
    return float(np.trapezoid(rw, xw))


def peak_density(x, rho) -> float:
    _, rw = window_mask(x, rho)
    return float(rw.max())


def probe(x, rho, x0: float = 0.5) -> float:
    """Density at a fixed physical point (interpolated), a non-integral quantity."""
    xw, rw = window_mask(x, rho)
    return float(np.interp(x0, xw, rw))


def field_distance(pa: dict, pb: dict) -> dict:
    """Relative L1 distance ``∫|ρa-ρb| dx / ∫ρb dx`` on ``pb``'s grid inside the window.

    The finer profile is interpolated onto the coarser one so the comparison is
    always a projection of the fine data (no upsampling of the coarse data).
    """
    if pa["nx"] < pb["nx"]:
        pa, pb = pb, pa
    xb, rb = window_mask(pb["x"], pb["rho"])
    ra = np.interp(xb, pa["x"], pa["rho"])
    num = float(np.trapezoid(np.abs(ra - rb), xb))
    den = float(np.trapezoid(rb, xb))
    return {"percent": 100.0 * num / den, "abs": num, "denominator": den,
            "grid": pb["nx"]}


def extrapolate_field(fine: dict, medium: dict, order: float) -> dict:
    """Richardson-extrapolated field on the **fine** grid (ratio 2 assumed)."""
    r = fine["nx"] / medium["nx"]
    xf, rf = fine["x"], fine["rho"]
    rm = np.interp(xf, medium["x"], medium["rho"])
    factor = r ** order - 1.0
    return {"x": xf, "rho": rf + (rf - rm) / factor, "nx": fine["nx"],
            "order_used": order, "ratio": r, "within_window": True}


# ------------------------------------------------------------------ quantities
def digitisation_sampling_test(prof: dict, vertices=(174, 349, 698, 1398, 2796)) -> dict:
    """How much peak height a polyline reference loses at a given vertex count.

    The published curve was extracted from the PDF figure as a *polyline*: the
    vector path carries ~174 vertices across the plotted window, so between
    vertices the curve is straight even though the ``reference_density.csv`` grid
    is finer.  Sampling our own finest solution at that vertex density and
    interpolating back measures the peak height that representation alone would
    remove - a ceiling on how closely any solver can match a peak read off that
    figure, independent of the solver's own accuracy.
    """
    x, rho = window_mask(prof["x"], prof["rho"])
    native = float(rho.max())
    out = {"native_peak": native, "window": [float(x[0]), float(x[-1])],
           "dx_finest": float(np.median(np.diff(x))), "cases": []}
    for n in vertices:
        xs = np.linspace(x[0], x[-1], int(n))
        rs = np.interp(xs, x, rho)
        back = np.interp(x, xs, rs)
        peak = float(back.max())
        out["cases"].append({"vertices": int(n),
                             "vertex_spacing": float(xs[1] - xs[0]),
                             "peak_after_polyline": peak,
                             "peak_loss_percent": 100.0 * (native - peak) / native})
    return out


def quantities(prof: dict) -> dict:
    return {
        "window_mass": window_mass(prof["x"], prof["rho"]),
        "peak_density": peak_density(prof["x"], prof["rho"]),
        "probe_x0.5": probe(prof["x"], prof["rho"]),
    }


def study_of(values: dict[int, float], triplet, theory_order: float, tol: float = 0.2):
    fine, medium, coarse = (values[n] for n in triplet)
    spacings = tuple(1.0 / n for n in triplet)
    return richardson_gci(fine, medium, coarse, spacings,
                          theory_order=theory_order, order_tolerance=tol)


def family_analysis(profiles: dict, family: str, ref_cal: dict, ref_raw: dict,
                    theory_order: float) -> dict:
    """GCI + reference-agreement for one discretisation family."""
    tags = {p["nx"]: t for t, p in profiles.items() if p["family"] == family}
    missing = [n for n in FINE_TRIPLET if n not in tags]
    out: dict = {"family": family, "theory_order": theory_order,
                 "grids_available": sorted(tags), "missing_grids": missing}
    if missing:
        return out

    q = {n: quantities(profiles[tags[n]]) for n in tags}
    l1_cal = {}
    for n in tags:
        m = wcr.metrics_from_profiles(ref_cal["x"], ref_cal["rho"],
                                      profiles[tags[n]]["x"], profiles[tags[n]]["rho"])
        l1_cal[n] = m["l1_relative"]["percent"] / 100.0

    studies = {}
    for key, src in (("window_mass", {n: q[n]["window_mass"] for n in tags}),
                     ("peak_density", {n: q[n]["peak_density"] for n in tags}),
                     ("probe_x0.5", {n: q[n]["probe_x0.5"] for n in tags}),
                     ("l1_vs_calibrated_reference", l1_cal)):
        fine_tri = [n for n in FINE_TRIPLET if n in src]
        coarse_tri = [n for n in COARSE_TRIPLET if n in src]
        entry = {"values": {str(n): src[n] for n in sorted(src)}}
        for label, tri in (("fine", fine_tri), ("coarse", coarse_tri)):
            if len(tri) != 3:
                continue
            st = study_of(src, tuple(tri), theory_order)
            d = st.as_dict()
            entry[label] = {
                "triplet": list(tri),
                "observed_order": st.order,
                "order_used": st.order_used,
                "convergence": st.convergence,
                "fallback_reason": st.fallback_reason,
                "asymptotic": st.asymptotic,
                "extrapolated": st.extrapolated,
                "gci_percent": 100.0 * st.gci,
                "band": list(st.band),
            }
            ref_val = {"window_mass": window_mass(ref_cal["x"], ref_cal["rho"]),
                       "peak_density": peak_density(ref_cal["x"], ref_cal["rho"]),
                       "probe_x0.5": probe(ref_cal["x"], ref_cal["rho"]),
                       "l1_vs_calibrated_reference": 0.0}.get(key)
            if ref_val:
                entry[label]["reference_agreement"] = reference_agreement(st, ref_val)
                entry[label]["reference_agreement"]["reference_kind"] = (
                    "calibrated digitised reference")
        studies[key] = entry
    out["studies"] = studies

    # Field-level convergence: successive-grid distances and their order.
    dists = []
    for a, b in ((800, 1600), (1600, 3200)):
        if a in tags and b in tags:
            dists.append({"grids": [a, b],
                          **field_distance(profiles[tags[b]], profiles[tags[a]])})
    out["field_distance_successive"] = dists
    if len(dists) == 2:
        d_fine = dists[1]["percent"] / 100.0
        d_med = dists[0]["percent"] / 100.0
        p_obs = math.log(d_med / d_fine) / math.log(2.0) if d_fine > 0 else float("inf")
        out["field_order"] = {"observed_order": p_obs,
                              "comment": "log2(D(800,1600)/D(1600,3200)) on the window L1 norm"}

    # Extrapolated field with an order sensitivity sweep: the field order below
    # is shock-limited and therefore only a proxy, so the extrapolation is
    # repeated at p=1, p=p_field and p=2 and every distance is reported.  A
    # conclusion that survives the whole sweep is not an artefact of the order.
    order_field = out.get("field_order", {}).get("observed_order")
    orders = [1.0]
    if order_field and 0.2 < order_field < 6:
        orders.append(round(float(order_field), 4))
    orders.append(2.0)
    seen = []
    for o in orders:
        if all(abs(o - q) > 1e-6 for q in seen):
            seen.append(o)
    sweep = []
    for o in seen:
        ext = extrapolate_field(profiles[tags[3200]], profiles[tags[1600]], o)
        row = {"order_used": o,
               "distance_to_nx3200_percent": field_distance(ext, profiles[tags[3200]])["percent"],
               "peak_density": peak_density(ext["x"], ext["rho"]),
               "window_mass": window_mass(ext["x"], ext["rho"])}
        for label, ref in (("calibrated", ref_cal), ("committed_raw", ref_raw)):
            row[f"distance_to_{label}_reference_percent"] = field_distance(
                ext, {"x": ref["x"], "rho": ref["rho"], "nx": 3200})["percent"]
        sweep.append(row)
    out["extrapolated_field"] = {"source_grids": [1600, 3200], "order_sweep": sweep,
                                 "reference_peak_calibrated": peak_density(ref_cal["x"], ref_cal["rho"]),
                                 "reference_mass_calibrated": window_mass(ref_cal["x"], ref_cal["rho"]),
                                 "distance_nx3200_to_calibrated_percent": field_distance(
                                     profiles[tags[3200]],
                                     {"x": ref_cal["x"], "rho": ref_cal["rho"], "nx": 3200})["percent"],
                                 "distance_nx1600_to_calibrated_percent": field_distance(
                                     profiles[tags[1600]],
                                     {"x": ref_cal["x"], "rho": ref_cal["rho"], "nx": 1600})["percent"]}

    out["digitisation_sampling"] = digitisation_sampling_test(profiles[tags[3200]])

    # Budget per judged quantity: U_num (GCI at fine triplet) vs residual to reference.
    budget = {}
    for key in ("window_mass", "peak_density", "probe_x0.5"):
        st = studies[key]["fine"]
        ref_val = {"window_mass": window_mass(ref_cal["x"], ref_cal["rho"]),
                   "peak_density": peak_density(ref_cal["x"], ref_cal["rho"]),
                   "probe_x0.5": probe(ref_cal["x"], ref_cal["rho"])}[key]
        resid = abs(st["extrapolated"] - ref_val) / abs(ref_val) * 100.0
        budget[key] = {
            "reference": ref_val,
            "extrapolated": st["extrapolated"],
            "residual_to_reference_percent": resid,
            "numerical_uncertainty_percent": st["gci_percent"],
            "residual_over_unum": resid / st["gci_percent"] if st["gci_percent"] else float("inf"),
            "reference_inside_gci_band": st["reference_agreement"]["band_covered"],
        }
    out["uncertainty_budget"] = budget
    return out


def cross_code(profiles: dict) -> dict:
    """WENO5 vs MUSCL/HLLC: distances at each common grid, plus extrapolated-field agreement."""
    fam = {}
    for t, p in profiles.items():
        fam.setdefault(p["family"], {})[p["nx"]] = t
    if "weno5" not in fam or "hllc" not in fam:
        return {"available": False, "reason": "WENO5 ladder profiles not present"}
    out = {"available": True, "per_grid": []}
    for nx in sorted(set(fam["weno5"]) & set(fam["hllc"])):
        d = field_distance(profiles[fam["weno5"][nx]], profiles[fam["hllc"][nx]])
        out["per_grid"].append({"nx": nx, **d})
    # orders of the cross-code difference (does it vanish with refinement?)
    ds = out["per_grid"]
    if len(ds) >= 2:
        for a, b in zip(ds, ds[1:]):
            if b["percent"] > 0:
                a["order_to_next"] = math.log(a["percent"] / b["percent"]) / math.log(
                    b["nx"] / a["nx"])
    # Basis error: both families Richardson-extrapolated (theory order of each
    # scheme) and compared with each other.  This distance is independent of any
    # published curve: it can only shrink if the two discretisations converge to
    # the same limit, and it bounds how much of the reference gap is "scheme".
    if all(1600 in fam[f] and 3200 in fam[f] for f in ("hllc", "weno5")):
        ext = {f: extrapolate_field(profiles[fam[f][3200]], profiles[fam[f][1600]],
                                    {"hllc": THEORY_ORDER, "weno5": 5.0}[f])
               for f in ("hllc", "weno5")}
        out["extrapolated_field_agreement"] = {
            "hllc_order_used": THEORY_ORDER, "weno5_order_used": 5.0,
            "distance_percent": field_distance(ext["weno5"], ext["hllc"])["percent"],
            "peak_hllc": peak_density(ext["hllc"]["x"], ext["hllc"]["rho"]),
            "peak_weno5": peak_density(ext["weno5"]["x"], ext["weno5"]["rho"]),
            "mass_hllc": window_mass(ext["hllc"]["x"], ext["hllc"]["rho"]),
            "mass_weno5": window_mass(ext["weno5"]["x"], ext["weno5"]["rho"]),
        }
    elif 3200 in fam.get("hllc", {}) and 3200 in fam.get("weno5", {}):
        out["at_finest_grid"] = {"nx": 3200, "distance_percent":
                                 field_distance(profiles[fam["weno5"][3200]],
                                                profiles[fam["hllc"][3200]])["percent"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=DATA)
    ap.add_argument("--json", type=Path, default=DATA / "asymptotic_analysis.json")
    ap.add_argument("--markdown", type=Path, default=DATA / "asymptotic_tables.md")
    args = ap.parse_args()

    profiles = load_profiles()
    ref_cal = wcr.load_reference(CALIBRATED)
    ref_raw = wcr.load_reference()
    print(f"profiles: {', '.join(sorted(profiles))}")

    result = {
        "reference": {"calibrated": str(CALIBRATED.relative_to(HERE.parent)),
                      "committed_raw": str(wcr.DEFAULT_REFERENCE),
                      "window": list(wcr.REFERENCE_X),
                      "tolerance_percent": wcr.TOLERANCE_PERCENT},
        "profiles": {t: {"nx": p["nx"], "family": p["family"], "source": p["source"]}
                     for t, p in sorted(profiles.items())},
        "families": {
            "hllc": family_analysis(profiles, "hllc", ref_cal, ref_raw, THEORY_ORDER),
            "weno5": family_analysis(profiles, "weno5", ref_cal, ref_raw, 5.0),
        },
        "cross_code": cross_code(profiles),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    md = render_markdown(result)
    args.markdown.write_text(md)
    print(md)
    print(f"\njson -> {args.json}")


def render_markdown(res: dict) -> str:
    L = ["# CASE-WOODWARD-COLELLA — asymptotic-range GCI and uncertainty budget", "",
         f"Reference: calibrated digitised curve, comparison window x∈{res['reference']['window']}, "
         f"acceptance {res['reference']['tolerance_percent']:.0f} %.", ""]
    for fam, d in res["families"].items():
        L.append(f"## family: {fam}  (formal order {d['theory_order']}, grids {d['grids_available']})")
        if d.get("missing_grids"):
            L.append(f"_not enough grids (missing {d['missing_grids']})_")
            L.append("")
            continue
        L.append("")
        L.append("| quantity | triplet | p_obs | asymptotic? | extrapolated | GCI % | ref inside band |")
        L.append("|---|---|---|---|---|---|---|")
        for key, st in d["studies"].items():
            for label in ("coarse", "fine"):
                if label not in st:
                    continue
                e = st[label]
                cov = e.get("reference_agreement", {}).get("band_covered")
                L.append(f"| {key} | {e['triplet']} | {e['observed_order']:.3f} | "
                         f"{e['asymptotic']} | {e['extrapolated']:.6f} | "
                         f"{e['gci_percent']:.3f} | {cov} |")
        L.append("")
        fo = d.get("field_order")
        if fo:
            L.append(f"Field-level observed order (window L1): p = {fo['observed_order']:.3f}")
            for e in d["field_distance_successive"]:
                L.append(f"- D{e['grids']} = {e['percent']:.3f} %")
        ef = d.get("extrapolated_field", {})
        if ef:
            L.append(f"- nx3200 → calibrated reference: "
                     f"{ef['distance_nx3200_to_calibrated_percent']:.3f} %; "
                     f"nx1600 → calibrated reference: "
                     f"{ef['distance_nx1600_to_calibrated_percent']:.3f} %")
            L.append("")
            L.append("| extrapolation order p | → nx3200 % | → calibrated ref % | → raw ref % | peak | window mass |")
            L.append("|---|---|---|---|---|---|")
            for row in ef["order_sweep"]:
                L.append(f"| {row['order_used']:.3f} | {row['distance_to_nx3200_percent']:.3f} | "
                         f"{row['distance_to_calibrated_reference_percent']:.3f} | "
                         f"{row['distance_to_committed_raw_reference_percent']:.3f} | "
                         f"{row['peak_density']:.4f} | {row['window_mass']:.6f} |")
            L.append(f"| _calibrated reference_ | — | — | — | "
                     f"{ef['reference_peak_calibrated']:.4f} | {ef['reference_mass_calibrated']:.6f} |")
        L.append("")
        ds = d.get("digitisation_sampling")
        if ds:
            L.append("Polyline-representation ceiling on the peak (our own nx=3200 solution "
                     "sampled at the stated vertex count and interpolated back):")
            L.append("")
            L.append("| vertices | vertex spacing | peak after polyline | peak loss % |")
            L.append("|---|---|---|---|")
            for c in ds["cases"]:
                L.append(f"| {c['vertices']} | {c['vertex_spacing']:.5f} | "
                         f"{c['peak_after_polyline']:.4f} | {c['peak_loss_percent']:.3f} |")
            L.append("")
        L.append("| quantity | reference (calibrated) | extrapolated | residual % | U_num (GCI %) | residual/U_num |")
        L.append("|---|---|---|---|---|---|")
        for key, b in d["uncertainty_budget"].items():
            L.append(f"| {key} | {b['reference']:.6f} | {b['extrapolated']:.6f} | "
                     f"{b['residual_to_reference_percent']:.3f} | {b['numerical_uncertainty_percent']:.3f} | "
                     f"{b['residual_over_unum']:.2f} |")
        L.append("")
    cc = res["cross_code"]
    L.append("## cross-code (WENO5 vs MUSCL/HLLC)")
    if not cc.get("available"):
        L.append(f"not available: {cc.get('reason')}")
    else:
        for e in cc["per_grid"]:
            tail = f", order to next = {e['order_to_next']:.3f}" if "order_to_next" in e else ""
            L.append(f"- nx={e['nx']}: D = {e['percent']:.3f} %{tail}")
        if "extrapolated_field_agreement" in cc:
            a = cc["extrapolated_field_agreement"]
            L.append(f"- extrapolated fields (p={a['hllc_order_used']:.1f} HLLC / "
                     f"{a['weno5_order_used']:.1f} WENO5): D = {a['distance_percent']:.3f} %; "
                     f"peak {a['peak_hllc']:.4f} vs {a['peak_weno5']:.4f}; "
                     f"window mass {a['mass_hllc']:.6f} vs {a['mass_weno5']:.6f}")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()