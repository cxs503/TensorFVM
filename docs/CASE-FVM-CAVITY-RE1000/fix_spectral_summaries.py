#!/usr/bin/env python3
"""Back-fill extrapolation and comparison data into spectral summary JSONs.

The ``cavity_spectral_summary_re1000.json`` file was generated before the
extrapolation pipeline was wired up; it carries ``extrapolation: null`` even
though ``spectral_report.json`` already contains the full Richardson analysis.
This script reads ``spectral_report.json`` and patches the Re=1000 summary
so that downstream consumers (reports, tests, dashboards) see a single source
of truth.

Re=100 summary is left untouched (no Ghia reference in the report, and the
extrapolation block is Re=1000-specific).

Usage::

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/fix_spectral_summaries.py
"""

from __future__ import annotations

import json
import pathlib

CASE_DIR = pathlib.Path(__file__).resolve().parent
RUNS = CASE_DIR / "runs"
SPECTRAL = RUNS / "spectral"


def _load(name: str) -> dict:
    return json.loads((SPECTRAL / name).read_text())


def _save(name: str, data: dict) -> None:
    (SPECTRAL / name).write_text(json.dumps(data, indent=2) + "\n")


def patch_re1000_summary(summary: dict, report: dict) -> dict:
    """Add extrapolation / vs_ghia / vs_fvm to the Re=1000 spectral summary."""
    extrapolation = report.get("extrapolation")
    verdict = report.get("verdict")

    if not extrapolation or not verdict:
        return summary

    # Build vs_ghia from the finest converged spectral grid
    ladder = report.get("ladder_re1000", [])
    finest = None
    for rec in ladder:
        if rec.get("ghia") and rec.get("status") == "ok":
            finest = rec

    vs_ghia = {}
    if finest:
        g = finest["ghia"]
        vs_ghia = {
            "u_min_rel_err": g["u_min_rel_err"],
            "v_max_rel_err": g["v_max_rel_err"],
            "u_profile_max_abs": g["u_profile_max_abs"],
            "v_profile_max_abs": g["v_profile_max_abs"],
            "within_5pct": g["within_5pct"],
            "finest_order": finest["order"],
        }

    vs_fvm = {
        "u_min": {
            "spectral_limit": verdict["u_min"]["spectral_limit"],
            "fvm_richardson_limit": verdict["u_min"]["fvm_richardson_limit"],
            "spectral_vs_fvm_rel": verdict["u_min"]["spectral_vs_fvm_rel"],
            "spectral_vs_ghia_rel": verdict["u_min"]["spectral_vs_ghia_rel"],
            "fvm_vs_ghia_rel": verdict["u_min"]["fvm_vs_ghia_rel"],
        },
        "v_max": {
            "spectral_limit": verdict["v_max"]["spectral_limit"],
            "fvm_richardson_limit": verdict["v_max"]["fvm_richardson_limit"],
            "spectral_vs_fvm_rel": verdict["v_max"]["spectral_vs_fvm_rel"],
            "spectral_vs_ghia_rel": verdict["v_max"]["spectral_vs_ghia_rel"],
            "fvm_vs_ghia_rel": verdict["v_max"]["fvm_vs_ghia_rel"],
        },
    }

    summary["extrapolation"] = extrapolation
    summary["vs_ghia"] = vs_ghia
    summary["vs_fvm"] = vs_fvm
    return summary


def main() -> None:
    report = _load("spectral_report.json")
    summary_name = "cavity_spectral_summary_re1000.json"
    summary = _load(summary_name)
    patched = patch_re1000_summary(summary, report)
    _save(summary_name, patched)
    ext = patched.get("extrapolation")
    if ext:
        print(f"{summary_name}: extrapolation u_min limit = {ext['u_min']['limit']:.6f}, "
              f"v_max limit = {ext['v_max']['limit']:.6f}")
        vf = patched.get("vs_fvm", {})
        if vf:
            print(f"  spectral vs FVM (u_min): {vf['u_min']['spectral_vs_fvm_rel']*100:.3f}%")
            print(f"  spectral vs FVM (v_max): {vf['v_max']['spectral_vs_fvm_rel']*100:.3f}%")
    else:
        print(f"{summary_name}: no extrapolation data available")


if __name__ == "__main__":
    main()
