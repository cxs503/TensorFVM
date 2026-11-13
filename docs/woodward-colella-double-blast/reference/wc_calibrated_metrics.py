"""Re-score every stored Woodward-Colella profile against the calibrated reference.

The digitised published reference carries a constant vertical calibration bias
of +0.022017 in density (``axis_calibration.json``; the label bbox centres sit
0.337 pt above the drawn tick marks).  This script re-runs the existing metric
set unchanged on (i) the committed reference and (ii) the calibrated one, for
every stored profile, so the two verdicts can be compared side by side without
touching either the metric definitions or the raw digitisation.

Outputs ``metrics_calibrated_<tag>.json`` next to the profiles (default outdir:
this directory) and prints one row per profile.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/reference/wc_calibrated_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts"
LADDER = HERE.parent / "ladder" / "data"
CALIBRATED = HERE / "reference_density_calibrated.csv"


def profiles() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every stored density profile, keyed by grid tag."""
    out = {}
    for nx in (400, 800):
        p = ARTIFACTS / f"wc_blast_profile_nx{nx}.npz"
        if p.exists():
            d = np.load(p)
            out[f"nx{nx}_hllc"] = (d["x_got"], d["rho_got"])
    for p in sorted(LADDER.glob("ladder_profile_*.npz")):
        d = np.load(p)
        out[p.stem.replace("ladder_profile_", "")] = (d["x"], d["rho"])
    return out


def main() -> None:
    raw = wcr.load_reference()
    bias = json.loads((HERE / "axis_calibration.json").read_text())["bias_density"]
    cal = wcr.load_reference(CALIBRATED)
    if abs(float(np.max(np.abs(cal["rho"] - raw["rho"]))) - bias) > 1e-5:
        raise SystemExit("calibrated csv does not differ from raw by the bias")
    print(f"reference bias {bias:+.6f}  calibrated peak {cal['rho'].max():.4f} "
          f"(committed {raw['rho'].max():.4f})")
    hdr = ("tag", "peak%", "l1%", "plateau%", "probe%", "mass%", "peak_cal%",
           "l1_cal%", "plateau_cal%", "probe_cal%", "mass_cal%", "verdict_cal")
    print("{:>18} " + " ".join(f"{h:>9}" for h in hdr[1:]))
    rows = {}
    for tag, (x, rho) in sorted(profiles().items()):
        m_raw = wcr.metrics_from_profiles(raw["x"], raw["rho"], x, rho)
        m_cal = wcr.metrics_from_profiles(cal["x"], cal["rho"], x, rho)
        rows[tag] = {"committed_reference": m_raw, "calibrated_reference": m_cal,
                     "bias_density": bias}
        (HERE / f"metrics_calibrated_{tag}.json").write_text(
            json.dumps(rows[tag], indent=2) + "\n")
        vals = [m_raw["peak_density"]["rel_error_percent"], m_raw["l1_relative"]["percent"],
                m_raw["low_plateau"]["rel_error_percent"],
                m_raw["probe_at_0.5"]["rel_error_percent"] if "probe_at_0.5" in m_raw else float("nan"),
                m_raw["window_mass"]["rel_error_percent"] if "window_mass" in m_raw else float("nan"),
                m_cal["peak_density"]["rel_error_percent"], m_cal["l1_relative"]["percent"],
                m_cal["low_plateau"]["rel_error_percent"],
                m_cal["probe_at_0.5"]["rel_error_percent"] if "probe_at_0.5" in m_cal else float("nan"),
                m_cal["window_mass"]["rel_error_percent"] if "window_mass" in m_cal else float("nan")]
        verdict = "PASS" if all(float(v) < wcr.TOLERANCE_PERCENT
                                for v in vals[6:10] if v == v) else "FAIL"
        print(f"{tag:>18} " + " ".join(f"{v:9.3f}" for v in vals) + f" {verdict:>10}")


if __name__ == "__main__":
    main()
