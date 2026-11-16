"""Round-16: measure the pre-registered E2 -- h^1 shrinkage of max|D| at the
25600 rung (same-instrument CFL 0.8 reference ladder).

Why this exists
---------------
``W1-NORM-CERTIFICATION.md`` (round 5) closed with the attribution: the W1
residual of the reference family is a *net mass-flux defect* at the collision
(x~0.765), i.e. an O(h) quantity.  The falsifiable consequence, frozen in
``PREDECLARATION-r7.md`` / ``PREDECLARATION-r10.md`` (criterion **E2**) and
repeated in the debt ledger (D21), is:

    p_D = log2( max|D|(6400->12800) / max|D|(12800->25600) )  in [1.85, 2.15]

R7 and R10 both recorded ``E2 = not_measured`` because the 25600 rung was not
in the library yet.  It has since been backfilled -- but only as the CFL 0.8
variant (the R10 CFL 0.4 job ``cj4143`` was never fetched).  This script
measures E2 on the **same-instrument** CFL 0.8 ladder (6400/12800/25600),
whose 6400/12800 rungs were already admitted by the R15 E1r gates
(0.0231 % / 0.0255 % < 0.2 %).

Criteria E2/E2b-E2e are frozen in ``PREDECLARATION-r16.md`` **before** any
number below was computed.  E2 reuses the R7/R10 band verbatim; E2b-E2e are
new supporting checks declared in the same predeclaration.

No solver is run: everything is re-integrated from the stored npz profiles.
The metric functions are imported from ``wc_w1_norm.py`` (single
implementation, no copy) so the numbers are comparable with round 5/7.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/wc_w1_r16_flux_defect.py \
        --json docs/XVAL-FVM-M18737-ROUND2/r16_flux_defect.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # docs/
DATA = ROOT / "woodward-colella-double-blast" / "ladder" / "data"
XV = ROOT / "woodward-colella-double-blast" / "xval"

WINDOW = (0.4, 1.0)

# same-instrument ladder (frozen in PREDECLARATION-r16.md section 1)
CFL08_LADDER = (6400, 12800, 25600)
CFL08_FAMILY = "weno5_cfl0.8"
# context only (not a gate): the historical frozen-instrument family
CFL04_LADDER = (1600, 3200, 6400, 12800)
CFL04_FAMILY = "weno5"

# --------------------------------------------------------------------------- #
# FROZEN CRITERIA -- PREDECLARATION-r16.md (written before computing).
# --------------------------------------------------------------------------- #
FROZEN = {
    "E2_pD_band": (1.85, 2.15),        # R7/R10 band, reused verbatim
    "E2b_ratio_band": (0.45, 0.55),    # amplitude: ~0.5 x previous pair
    "E2c_xpeak_band": (0.70, 0.85),    # collision region
    "E2d_rel_net_mass_max": 1e-4,      # not a conservation defect
}


def _load_mod(name: str, base: Path):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


W1 = _load_mod("wc_w1_norm", HERE)
assert tuple(W1.WINDOW) == WINDOW


def load(nx: int, family: str) -> dict | None:
    path = DATA / f"ladder_profile_nx{nx}_{family}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    out = {k: np.asarray(d[k], float) for k in d.files}
    out["nx"] = nx
    out["family"] = family
    return out


def metrics_json(nx: int, family: str) -> dict | None:
    path = DATA / f"ladder_metrics_nx{nx}_{family}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def e0_artifact_gate() -> dict:
    """Pre-registered artifact gate for the 25600 rung (PREDECLARATION-r16 §4)."""
    prof = load(25600, CFL08_FAMILY)
    met = metrics_json(25600, CFL08_FAMILY)
    checks = {
        "profile_present": prof is not None,
        "metrics_present": met is not None,
        "nx": (met or {}).get("nx"),
        "cfl": (met or {}).get("cfl"),
        "status": (met or {}).get("status"),
        "profile_length": int(prof["x"].size) if prof is not None else None,
    }
    checks["pass"] = bool(
        prof is not None
        and met is not None
        and met.get("status") == "ok"
        and met.get("cfl") == 0.8
        and met.get("nx") == 25600
        and prof["x"].size == 25600
    )
    return checks


def pair_stats(fine: dict, coarse: dict) -> dict:
    """max|D| + localization + both norms for one (coarse, fine) pair."""
    loc = W1.localize(fine, coarse, "rho")
    return {
        "coarse_nx": coarse["nx"],
        "fine_nx": fine["nx"],
        "max_abs_D": loc["max_abs"],
        "x_peak": loc["x_peak"],
        "w_eff_coarse_cells": loc["w_eff_coarse_cells"],
        "rel_net_mass": loc["rel_net_mass"],
        "d_w1_percent": W1.d_w1_percent(fine, coarse, "rho"),
        "d_l1_percent": W1.d_l1_percent(fine, coarse, "rho"),
    }


def build(verbose: bool = True) -> dict:
    e0 = e0_artifact_gate()
    out: dict = {
        "anchor": "XVAL-FVM-M18737-ROUND2",
        "round": 16,
        "instrument": {
            "family": "WENO5 + SSP-RK3 + Rusanov",
            "cfl": 0.8,
            "rungs": list(CFL08_LADDER),
            "admission": "R15 E1r6400=0.0231% / E1r@12800=0.0255% (<0.2% gate)",
        },
        "e0_artifact_gate": e0,
        "frozen": FROZEN,
    }
    if not e0["pass"]:
        out["verdict"] = "not_measured"
        out["reason"] = "E0 artifact gate failed; 25600 rung not usable"
        if verbose:
            print(json.dumps(out, indent=2, default=float))
        return out

    profs = {nx: load(nx, CFL08_FAMILY) for nx in CFL08_LADDER}
    if any(p is None for p in profs.values()):
        out["verdict"] = "not_measured"
        out["reason"] = "missing CFL 0.8 rung"
        if verbose:
            print(json.dumps(out, indent=2, default=float))
        return out

    pairs = []
    for lo, hi in zip(CFL08_LADDER, CFL08_LADDER[1:]):
        pairs.append(pair_stats(profs[hi], profs[lo]))
    out["pairs_cfl08"] = pairs

    m_prev = pairs[0]["max_abs_D"]      # 6400 -> 12800
    m_new = pairs[1]["max_abs_D"]       # 12800 -> 25600
    p_D = math.log2(m_prev / m_new) if m_new > 0 else float("nan")
    ratio = m_new / m_prev if m_prev > 0 else float("nan")

    lo_b, hi_b = FROZEN["E2_pD_band"]
    e2 = {
        "quantity": "p_D = log2( max|D|(6400->12800) / max|D|(12800->25600) )",
        "max_abs_D_6400_12800": m_prev,
        "max_abs_D_12800_25600": m_new,
        "p_D": p_D,
        "band": list(FROZEN["E2_pD_band"]),
        "verdict": "PASS" if lo_b <= p_D <= hi_b else "FALSIFIED",
    }
    lo_r, hi_r = FROZEN["E2b_ratio_band"]
    e2b = {
        "quantity": "max|D|(12800->25600) / max|D|(6400->12800)",
        "ratio": ratio,
        "band": list(FROZEN["E2b_ratio_band"]),
        "verdict": "PASS" if lo_r <= ratio <= hi_r else "FAIL",
    }
    lo_x, hi_x = FROZEN["E2c_xpeak_band"]
    e2c = {
        "quantity": "x_peak(12800->25600)",
        "x_peak": pairs[1]["x_peak"],
        "band": list(FROZEN["E2c_xpeak_band"]),
        "verdict": "PASS" if lo_x <= pairs[1]["x_peak"] <= hi_x else "FAIL",
    }
    e2d = {
        "quantity": "rel_net_mass(12800->25600)",
        "rel_net_mass": pairs[1]["rel_net_mass"],
        "max": FROZEN["E2d_rel_net_mass_max"],
        "verdict": "PASS" if pairs[1]["rel_net_mass"] < FROZEN["E2d_rel_net_mass_max"] else "FAIL",
    }
    seq = [p["max_abs_D"] for p in pairs]
    e2e = {
        "quantity": "max|D| monotone decreasing on the CFL 0.8 ladder",
        "sequence": seq,
        "verdict": "PASS" if all(b < a for a, b in zip(seq, seq[1:])) else "FAIL",
    }
    out["criteria"] = {"E2": e2, "E2b": e2b, "E2c": e2c, "E2d": e2d, "E2e": e2e}

    # ---- cross-checks (declared, not gates) ------------------------------- #
    cfl04 = {nx: load(nx, CFL04_FAMILY) for nx in CFL04_LADDER}
    cfl04 = {nx: p for nx, p in cfl04.items() if p is not None}
    seq04 = []
    for lo, hi in zip(CFL04_LADDER, CFL04_LADDER[1:]):
        if lo in cfl04 and hi in cfl04:
            seq04.append(pair_stats(cfl04[hi], cfl04[lo]))
    out["crosscheck_cfl04_sequence"] = seq04

    # The R5 "next hand" prediction (0.004509->0.002352->0.001078) was computed
    # on the *graded* family (hllc vs weno5@12800), not the reference family.
    # Record it explicitly so the closure can separate the two quantities.
    ref12800 = load(12800, CFL04_FAMILY)
    graded_seq = []
    if ref12800 is not None:
        for nx in (1600, 3200, 6400):
            g = load(nx, "hllc")
            if g is None:
                continue
            loc = W1.localize(ref12800 := ref12800, g, "rho")
            graded_seq.append({
                "nx": nx,
                "max_abs_D": loc["max_abs"],
                "x_peak": loc["x_peak"],
                "rel_net_mass": loc["rel_net_mass"],
            })
    out["crosscheck_graded_sequence"] = graded_seq
    out["crosscheck_note"] = (
        "R5 'next hand' numbers (0.004509/0.002352/0.001078) are the GRADED family "
        "(hllc vs weno5@12800); the frozen E2 criterion is on the REFERENCE family "
        "(weno5 self-pairs). Both are reported; E2 adjudicates the reference family."
    )

    mixed = None
    if 12800 in cfl04 and profs[25600] is not None:
        mixed = pair_stats(profs[25600], cfl04[12800])
        mixed["note"] = "mixed-instrument probe (12800 CFL 0.4 -> 25600 CFL 0.8), not a gate"
    out["crosscheck_mixed_pair"] = mixed

    # d_W1 observed order on the CFL 0.8 ladder (context for the W1 instrument)
    if len(pairs) >= 2:
        p_w1, src = W1.WRU.observed_order(pairs[0]["d_w1_percent"], pairs[1]["d_w1_percent"])
        p_l1, _ = W1.WRU.observed_order(pairs[0]["d_l1_percent"], pairs[1]["d_l1_percent"])
        out["cfl08_orders"] = {"p_w1": p_w1, "p_w1_source": src, "p_l1": p_l1}

    out["verdict"] = e2["verdict"]
    out["all_supporting_pass"] = all(
        c["verdict"] == "PASS" for c in (e2b, e2c, e2d, e2e)
    )
    if verbose:
        print(json.dumps(out, indent=2, default=float))
    return out


def make_figure(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = out["pairs_cfl08"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) max|D| ladder, both instruments, h^1 reference slope
    axa = ax[0]
    xs04 = [p["fine_nx"] for p in out["crosscheck_cfl04_sequence"]]
    ys04 = [p["max_abs_D"] for p in out["crosscheck_cfl04_sequence"]]
    xs08 = [p["fine_nx"] for p in pairs]
    ys08 = [p["max_abs_D"] for p in pairs]
    axa.loglog(xs04, ys04, "o--", color="0.55", label="CFL 0.4 (frozen instrument)")
    axa.loglog(xs08, ys08, "o-", color="tab:blue", label="CFL 0.8 (same instrument)")
    xref = np.array([12800, 25600])
    axa.loglog(xref, [ys08[-2]] / 2.0 ** np.arange(1, 3), "k:",
               label=r"$h^1$ slope (ratio 2)")
    axa.set_xlabel("fine nx")
    axa.set_ylabel(r"$\max|D|$  [mass units]")
    axa.set_title("(a) accumulated-difference amplitude")
    axa.legend(fontsize=8)
    axa.grid(True, which="both", alpha=0.3)

    # (b) D(x) of the new pair
    axb = ax[1]
    fine = load(25600, CFL08_FAMILY)
    coarse = load(12800, CFL08_FAMILY)
    f = W1.cumulative_diff_field(fine, coarse, "rho")
    axb.plot(f["x"], f["D"], color="tab:blue", lw=1.2)
    axb.axhline(0.0, color="k", lw=0.6)
    axb.axvline(pairs[1]["x_peak"], color="tab:red", ls="--", lw=0.8,
                label=f"x_peak={pairs[1]['x_peak']:.4f}")
    axb.set_xlabel("x")
    axb.set_ylabel(r"$D(x)=\int_{0.4}^{x}(\rho_c-\rho_f)\,dx$")
    axb.set_title("(b) new pair 12800$\\to$25600")
    axb.legend(fontsize=8)
    axb.grid(True, alpha=0.3)

    # (c) both norms on the CFL 0.8 pairs
    axc = ax[2]
    labels = [f"{p['coarse_nx']}\u2192{p['fine_nx']}" for p in pairs]
    xx = np.arange(len(pairs))
    axc.semilogy(xx, [p["d_l1_percent"] for p in pairs], "s-", label="d_L1")
    axc.semilogy(xx, [p["d_w1_percent"] for p in pairs], "o-", label="d_W1")
    axc.set_xticks(xx, labels)
    axc.set_ylabel("percent")
    axc.set_title("(c) pair distances, CFL 0.8 ladder")
    axc.legend(fontsize=8)
    axc.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="write JSON result")
    ap.add_argument("--figure", default=None, help="write PNG figure")
    args = ap.parse_args(argv)
    out = build(verbose=args.json is None and args.figure is None)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
    if args.figure:
        make_figure(out, Path(args.figure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
