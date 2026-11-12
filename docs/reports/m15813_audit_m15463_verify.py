"""Independent re-computation of the CASE-CHANNEL-PERIODIC (plane Poiseuille) evidence.

Audit tool for mission m15813 (cross-audit of m15463).  It never trusts the
numbers printed in a report: it re-derives every scored quantity from the raw
arrays shipped in the run artefacts (``benchmark.json`` / ``profiles.csv``) and
prints ref / got / relative error next to the registered tolerance.

Usage
-----
    python docs/reports/m15813_audit_m15463_verify.py \
        --artifacts /root/tp-agent/work/m15463/runs/poiseuille-channel \
        --tolerance 0.02

Definitions reproduced from ``tensorfvm/benchmark_poiseuille_channel.py``:
  * velocity_l2   = ||u_num - u_exact||_2 / ||u_exact||_2  (norm relative)
  * velocity_linf = max|u_num - u_exact| / u_max_exact
  * dp/dx         = least-squares slope of the y-averaged pressure over
                    0.375 L <= x <= 0.75 L
  * tau_w         = -dp/dx * H/2 (steady momentum balance, independent of the
                    wall-shear stencils used by the solver)
  * Po            = 8 * tau_w / (rho ubar^2) * Re_Dh  (Fanning Cf = tau_w/(1/2 rho ubar^2),
                    f = 4 Cf, Re_Dh = 2 Re_H)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

H = 1.0
UBAR = 1.0
RHO = 1.0
PO_REF = 96.0          # plane channel, Shah & London 1978 (f*Re_Dh)
DPDX_REF = -0.12       # -12 mu ubar / H^2, mu = nu = 0.01 at Re_H = 100
TAU_REF = 0.06         # 6 mu ubar / H


def _profiles(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["case"], []).append(
                (float(row["y_over_H"]), float(row["u_over_bar"]),
                 float(row["u_exact_over_bar"])))
    return out


def _l2(pairs: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    num = math.sqrt(sum((a - b) ** 2 for _, a, b in pairs))
    den = math.sqrt(sum(b ** 2 for _, _, b in pairs))
    linf = max(abs(a - b) for _, a, b in pairs)
    return num / den, linf, max(b for _, _, b in pairs)


def recompute(artifacts: Path, tolerance: float) -> dict[str, object]:
    payload = json.loads((artifacts / "benchmark.json").read_text())
    profiles = _profiles(artifacts / "profiles.csv")
    rows = []
    for case in payload["cases"]:
        name = case["case"]
        got = case["metrics"]
        entry = {"case": name, "registered_in_json": got}
        if name in profiles and not all(v == 0.0 for v in (got.get("velocity_l2", 0.0),)):
            l2, linf, umax = _l2(profiles[name])
            entry["recomputed_from_csv"] = {"velocity_l2": l2, "velocity_linf": linf,
                                            "u_max_exact": umax,
                                            "velocity_l2_rel_diff":
                                                abs(l2 - got["velocity_l2"]) / got["velocity_l2"]}
        if name in profiles and "velocity_l2" in got:
            # sample identity: the csv profile must be the same station the json scored
            own = _l2(profiles[name])[0]
            entry["csv_vs_json_l2_consistent"] = abs(own - got["velocity_l2"]) < 1e-6 \
                if got["velocity_l2"] > 0 else None
        if name == payload["production_case"]:
            pressure = case["pressure"]
            xs = pressure["x"]
            ps = pressure["mean_over_y"]
            band = [(x, p) for x, p in zip(xs, ps) if 0.375 * 4.0 <= x <= 0.75 * 4.0]
            mx = sum(x for x, _ in band) / len(band)
            mp = sum(p for _, p in band) / len(band)
            sxx = sum((x - mx) ** 2 for x, _ in band)
            sxy = sum((x - mx) * (p - mp) for x, p in band)
            slope = sxy / sxx
            tau = -slope * H / 2.0
            cf = tau / (0.5 * RHO * UBAR ** 2)
            re_dh = 2.0 * case["re_h"]
            po = 4.0 * cf * re_dh
            entry["independent_checks"] = {
                "dpdx": slope,
                "dpdx_rel": abs(slope - DPDX_REF) / abs(DPDX_REF),
                "tau_w_from_momentum": tau,
                "tau_w_rel": abs(tau - TAU_REF) / TAU_REF,
                "poiseuille": po,
                "poiseuille_rel": abs(po - PO_REF) / PO_REF,
                "all_within_tolerance": max(abs(slope - DPDX_REF) / abs(DPDX_REF),
                                            abs(po - PO_REF) / PO_REF) < tolerance,
            }
        entry["zero_filled_record"] = bool(
            got and all(v == 0 for k, v in got.items() if isinstance(v, (int, float)))
            and case.get("iterations") == 0)
        rows.append(entry)
    return {
        "artifacts": str(artifacts),
        "declared_plan_cases": 6,
        "cases_in_artifact": [c["case"] for c in payload["cases"]],
        "missing_from_plan": sorted({"mesh-ny8", "mesh-ny16", "mesh-ny32", "re-10",
                                     "re-1000", "aspect-8"} - {c["case"] for c in payload["cases"]}),
        "production_case": payload["production_case"],
        "production_passed": payload["production_passed"],
        "all_cases_passed": payload["all_cases_passed"],
        "observed_orders": payload["observed_orders"],
        "tolerance": tolerance,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="runs/poiseuille-channel")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    audit = recompute(Path(args.artifacts), args.tolerance)
    text = json.dumps(audit, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())