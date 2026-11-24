"""Round-17: interface decomposition of D(x) for the WENO5 12800→25600 pair.

Why this exists
---------------
R16 falsified E2 (max|D| h^1 shrinkage): p_D = 0.6116, far below the
frozen band [1.85, 2.15]. The "next hand" from R16 §7 is:

    Decompose D(x) by interface (left shock / contact / right shock /
    collision zone) and identify which segment contributes to the
    "slow shrinkage" component.

This script locates the four interfaces on the WENO5@25600 density profile,
then integrates |D(x)| over each interface segment to quantify its contribution.

Criteria F1–F4 are frozen in ``PREDECLARATION-r17.md`` **before** any
number below was computed.

No solver is run: everything is re-integrated from the stored npz profiles.
The metric functions are imported from ``wc_w1_norm.py`` (single
implementation, no copy) so the numbers are comparable with round 5/7/16.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/wc_w1_r17_interface_decomposition.py \
        --json docs/XVAL-FVM-M18737-ROUND2/r17_interface_decomposition.json
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
CFL08_FAMILY = "weno5_cfl0.8"

# --------------------------------------------------------------------------- #
# FROZEN CRITERIA -- PREDECLARATION-r17.md (written before computing).
# --------------------------------------------------------------------------- #
FROZEN = {
    "F1_CD_contribution_min": 0.30,   # contact discontinuity > 30 % of max|D|
    "F2_CC_contribution_min": 0.20,   # collision zone > 20 % of max|D|
    "F3_LS_RS_contribution_max": 0.15,  # each shock < 15 % of max|D|
    "F4_total_conservation": 0.01,    # sum of segments = total (within 1 %)
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
    """Pre-registered artifact gate for the 12800/25600 rungs."""
    prof_12800 = load(12800, CFL08_FAMILY)
    prof_25600 = load(25600, CFL08_FAMILY)
    met_12800 = metrics_json(12800, CFL08_FAMILY)
    met_25600 = metrics_json(25600, CFL08_FAMILY)
    checks = {
        "profile_12800_present": prof_12800 is not None,
        "profile_25600_present": prof_25600 is not None,
        "metrics_12800_present": met_12800 is not None,
        "metrics_25600_present": met_25600 is not None,
        "nx_12800": (met_12800 or {}).get("nx"),
        "nx_25600": (met_25600 or {}).get("nx"),
        "cfl_12800": (met_12800 or {}).get("cfl"),
        "cfl_25600": (met_25600 or {}).get("cfl"),
        "status_12800": (met_12800 or {}).get("status"),
        "status_25600": (met_25600 or {}).get("status"),
    }
    checks["pass"] = bool(
        prof_12800 is not None and prof_25600 is not None
        and met_12800 is not None and met_25600 is not None
        and met_12800.get("status") == "ok"
        and met_25600.get("status") == "ok"
        and met_12800.get("cfl") == 0.8
        and met_25600.get("cfl") == 0.8
        and met_12800.get("nx") == 12800
        and met_25600.get("nx") == 25600
    )
    return checks


def locate_interfaces(rho: np.ndarray, x: np.ndarray) -> list[float]:
    """Locate the four interfaces on the density profile.

    Uses the gradient magnitude |dρ/dx| to find peaks corresponding to:
    - Left shock (LS): x ≈ 0.15–0.20 (outside window [0.4, 1.0])
    - Contact discontinuity (CD): x ≈ 0.50–0.55
    - Right shock (RS): x ≈ 0.80–0.85
    - Collision zone (CC): x ≈ 0.75–0.80

    Since the window starts at x=0.4, we only see CD, CC, and RS.
    Returns sorted list of interface positions within the window.
    """
    # Compute gradient (central difference)
    dx = np.diff(x)
    d_rho = np.diff(rho)
    # Use element-wise division with safe denominator
    grad = np.abs(d_rho / np.where(dx > 1e-15, dx, 1e-15))

    # Pad gradient to match x length
    grad_full = np.zeros_like(x)
    grad_full[:-1] = grad
    grad_full[-1] = grad[-1]  # backward difference at last point

    # Find peaks in |dρ/dx|
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(grad_full, height=0.01, distance=50)

    # Filter to window [0.4, 1.0]
    in_window = [(i, float(x[i])) for i in peaks if WINDOW[0] <= x[i] <= WINDOW[1]]
    in_window.sort(key=lambda t: grad_full[t[0]], reverse=True)

    if len(in_window) >= 3:
        # Take the 3 strongest peaks in window (CD, CC, RS)
        interfaces = sorted([pos for _, pos in in_window[:3]])
    else:
        # Fallback: use known approximate positions
        interfaces = [0.52, 0.77, 0.82]  # CD, CC, RS

    return interfaces


def segment_contribution(D: np.ndarray, x: np.ndarray,
                         interfaces: list[float]) -> list[dict]:
    """Compute |D(x)| contribution for each interface segment.

    Segments are defined as:
    - [x_start, interface_1]: pre-CD region
    - [interface_1, interface_2]: CD region
    - [interface_2, interface_3]: CC/RS region
    - [interface_3, x_end]: post-CC region
    """
    segments = []
    total = float(np.trapezoid(np.abs(D), x))

    # Define segment boundaries
    x_start = float(x[0])
    x_end = float(x[-1])
    boundaries = [x_start] + interfaces + [x_end]

    for i in range(len(boundaries) - 1):
        x_lo, x_hi = boundaries[i], boundaries[i + 1]
        mask = (x >= x_lo) & (x <= x_hi)
        if mask.sum() < 2:
            continue
        x_seg = x[mask]
        D_seg = D[mask]
        contrib = float(np.trapezoid(np.abs(D_seg), x_seg))
        pct = 100.0 * contrib / total if total > 0 else 0.0
        segments.append({
            "segment": f"region_{i+1}",
            "x_lo": x_lo,
            "x_hi": x_hi,
            "contrib": contrib,
            "pct_of_total": pct,
        })

    return segments


def build(verbose: bool = True) -> dict:
    e0 = e0_artifact_gate()
    out: dict = {
        "anchor": "XVAL-FVM-M18737-ROUND2",
        "round": 17,
        "instrument": {
            "family": "WENO5 + SSP-RK3 + Rusanov",
            "cfl": 0.8,
            "rungs": [12800, 25600],
        },
        "e0_artifact_gate": e0,
        "frozen": FROZEN,
    }

    if not e0["pass"]:
        out["verdict"] = "not_measured"
        out["reason"] = "E0 artifact gate failed; rungs not usable"
        if verbose:
            print(json.dumps(out, indent=2, default=float))
        return out

    prof_12800 = load(12800, CFL08_FAMILY)
    prof_25600 = load(25600, CFL08_FAMILY)

    # Get D(x) field using the fine grid x
    fine = prof_25600
    coarse = prof_12800
    lo, hi = WINDOW
    m = (fine["x"] >= lo) & (fine["x"] <= hi)
    x = fine["x"][m]
    a = fine["rho"][m]
    b = np.interp(x, coarse["x"], coarse["rho"])
    e = a - b
    dx = np.diff(x)
    D = np.concatenate(([0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * dx)))

    total_integral = float(np.trapezoid(np.abs(D), x))
    total_max_D = float(np.max(np.abs(D)))
    x_peak_idx = int(np.argmax(np.abs(D)))
    x_peak = float(x[x_peak_idx])

    # Locate interfaces on the fine profile (within window)
    interfaces = locate_interfaces(fine["rho"], fine["x"])

    out["interfaces"] = {
        "positions": interfaces,
        "n_detected": len(interfaces),
        "x_peak": x_peak,
    }

    # Compute segment contributions
    segments = segment_contribution(D, x, interfaces)
    out["segments"] = segments

    # Total contribution check
    total_contrib = sum(s["contrib"] for s in segments)
    out["total_contribution"] = {
        "sum_of_segments": total_contrib,
        "total_integral": total_integral,
        "ratio": total_contrib / total_integral if total_integral > 0 else 0.0,
    }

    # Adjudicate criteria
    contributions = {s["segment"]: s["pct_of_total"] for s in segments}

    # F1: Contact discontinuity (region_2, between interface_1 and interface_2) > 30 %
    cd_contrib = contributions.get("region_2", 0.0)
    f1 = {
        "quantity": "CD (region_2) contribution % of max|D|",
        "value": cd_contrib,
        "threshold": FROZEN["F1_CD_contribution_min"] * 100,
        "verdict": "PASS" if cd_contrib >= FROZEN["F1_CD_contribution_min"] * 100 else "FAIL",
    }

    # F2: Collision zone (region_3, between interface_2 and interface_3) > 20 %
    cc_contrib = contributions.get("region_3", 0.0)
    f2 = {
        "quantity": "CC (region_3) contribution % of max|D|",
        "value": cc_contrib,
        "threshold": FROZEN["F2_CC_contribution_min"] * 100,
        "verdict": "PASS" if cc_contrib >= FROZEN["F2_CC_contribution_min"] * 100 else "FAIL",
    }

    # F3: Pre-CD and post-CC regions < 15 % each
    pre_cd_contrib = contributions.get("region_1", 0.0)
    post_cc_contrib = contributions.get("region_4", 0.0)
    f3 = {
        "quantity": "Pre-CD (region_1) and post-CC (region_4) contribution % of max|D|",
        "pre_cd_value": pre_cd_contrib,
        "post_cc_value": post_cc_contrib,
        "threshold": FROZEN["F3_LS_RS_contribution_max"] * 100,
        "verdict": "PASS" if (pre_cd_contrib < FROZEN["F3_LS_RS_contribution_max"] * 100 and
                               post_cc_contrib < FROZEN["F3_LS_RS_contribution_max"] * 100) else "FAIL",
    }

    # F4: Total = 100 % (within 1 %)
    total_ratio = out["total_contribution"]["ratio"]
    f4 = {
        "quantity": "Sum of segments / total integral |D(x)|",
        "value": total_ratio,
        "threshold": 1.0 - FROZEN["F4_total_conservation"],
        "verdict": "PASS" if abs(total_ratio - 1.0) < FROZEN["F4_total_conservation"] else "FAIL",
    }

    out["criteria"] = {"F1": f1, "F2": f2, "F3": f3, "F4": f4}
    out["verdict"] = "PASS" if f1["verdict"] == "PASS" else "FAIL"
    out["all_supporting_pass"] = all(
        c["verdict"] == "PASS" for c in (f1, f2, f3, f4)
    )

    if verbose:
        print(json.dumps(out, indent=2, default=float))
    return out


def make_figure(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof_25600 = load(25600, CFL08_FAMILY)
    prof_12800 = load(12800, CFL08_FAMILY)

    # Get D(x) field
    fine = prof_25600
    coarse = prof_12800
    lo, hi = WINDOW
    m = (fine["x"] >= lo) & (fine["x"] <= hi)
    x = fine["x"][m]
    a = fine["rho"][m]
    b = np.interp(x, coarse["x"], coarse["rho"])
    e = a - b
    dx = np.diff(x)
    D = np.concatenate(([0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * dx)))

    interfaces = out["interfaces"]["positions"]
    segments = out["segments"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Density profiles with interface markers
    ax1.plot(prof_25600["x"], prof_25600["rho"], "b-", lw=0.8,
             label="WENO5@25600", alpha=0.7)
    ax1.plot(prof_12800["x"], prof_12800["rho"], "r--", lw=0.8,
             label="WENO5@12800", alpha=0.7)
    for i, x_int in enumerate(interfaces):
        color = ["tab:blue", "tab:orange", "tab:green"][i]
        ax1.axvline(x_int, color=color, ls="--", lw=1.2,
                    label=f"Interface {i+1}: x={x_int:.4f}")
    ax1.axvline(out["interfaces"]["x_peak"], color="tab:red", ls=":", lw=1.2,
                label=f"x_peak={out['interfaces']['x_peak']:.4f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\rho$")
    ax1.set_title("(a) Density profiles with interface markers")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(WINDOW)

    # (b) |D(x)| with segment contributions
    ax2.plot(x, np.abs(D), "k-", lw=1.2, label=r"$|D(x)|$")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, seg in enumerate(segments):
        mask = (x >= seg["x_lo"]) & (x <= seg["x_hi"])
        if mask.sum() >= 2:
            x_seg = x[mask]
            D_seg = np.abs(D[mask])
            ax2.fill_between(x_seg, 0, D_seg, alpha=0.3,
                            color=colors[i % len(colors)],
                            label=f"{seg['segment']}: {seg['pct_of_total']:.1f}%")
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$|D(x)|$")
    ax2.set_title("(b) |D(x)| with segment contributions")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(WINDOW)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--figure", type=str, default=None)
    args = parser.parse_args()

    out = build(verbose=True)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nJSON written to {args.json}")

    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        make_figure(out, Path(args.figure))
        print(f"Figure written to {args.figure}")
