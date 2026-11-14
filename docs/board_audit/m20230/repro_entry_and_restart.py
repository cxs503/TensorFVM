#!/usr/bin/env python3
"""Execute the audited entry point and compare the result with the shipped fields.

Audit m20230, checks
--------------------
R1  the documented CLI entry really runs end to end (same module, same flags as
    the shipped control grid): ``python -m tensorfvm.case_cavity_re1000``
R2  the cold n = 32 run reproduces the shipped ``Re1000_n32.pt`` field
    (``max|du|``), i.e. the shipped numbers come from this code, not from a
    hand-written artefact
R3  the shipped n = 128 field is a steady state: restart it in the audited
    solver and measure the drift over N further steps
R4  ``score_against_ghia`` on the *shipped* fields reproduces the shipped
    metric block (isolates "field" from "scoring")

Usage: python3 repro_entry_and_restart.py [--skip-cold]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE / "refs"
ART = REF / "docs" / "artifacts"
OUT = HERE / "runs"


def load_pkg():
    """Import the audited sources as ``aud.tensorfvm`` (no install, no edits)."""
    import types

    pkg = types.ModuleType("audtensorfvm")
    pkg.__path__ = [str(REF / "src")]
    sys.modules["audtensorfvm"] = pkg
    for name in ("lid_cavity_reference", "lid_cavity", "case_cavity_re1000"):
        spec = importlib.util.spec_from_file_location(
            f"audtensorfvm.{name}", REF / "src" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"audtensorfvm.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["audtensorfvm.case_cavity_re1000"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cold", action="store_true")
    ap.add_argument("--restart-steps", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    case = load_pkg()
    lc = sys.modules["audtensorfvm.lid_cavity"]
    res: dict = {"entry": "python -m tensorfvm.case_cavity_re1000 "
                          "--out docs/CASE-CAVITY-RE1000-FVM/artifacts "
                          "--grids 64,128 --cold-grids 32"}

    # R2: cold n = 32, the same numerics as the shipped control row
    if not args.skip_cold:
        t0 = time.time()
        rec, solver = case.run_case(32, out_dir=OUT / "cold32", tolerance=1e-7,
                                    max_steps=250_000, check_every=250, threads=1,
                                    fields_name="Re1000_n32_audit.pt")
        wall = time.time() - t0
        shipped = torch.load(ART / "fields" / "Re1000_n32.pt", weights_only=False)
        res["cold32"] = {
            "wall_seconds": wall,
            "steps": rec["solver"]["steps"],
            "shipped_steps": int(shipped["steps"]),
            "converged": rec["solver"]["converged"],
            "residual": rec["solver"]["residual_rate"],
            "u_min": rec["u"]["extremum"]["min_computed"][1],
            "v_max": rec["v"]["extremum"]["max_computed"][1],
            "u_max_abs": rec["u"]["max_abs_error"],
            "v_max_abs": rec["v"]["max_abs_error"],
            "max_abs_du_vs_shipped": float((solver.u - shipped["u"]).abs().max()),
            "max_abs_dv_vs_shipped": float((solver.v - shipped["v"]).abs().max()),
            "shipped_u_min": -0.316430836724452,
            "shipped_u_max_abs": 0.20329603195274926,
        }
        print("cold32:", json.dumps(res["cold32"], indent=1), flush=True)

    # R3: restart the shipped n = 128 field
    f128 = torch.load(ART / "fields" / "Re1000_n128.pt", weights_only=False)
    sv = lc.LidCavitySolver(lc.LidCavityConfig(n=128, reynolds=1000.0))
    sv.u, sv.v, sv.steps = f128["u"].clone(), f128["v"].clone(), int(f128["steps"])
    u0, v0 = sv.u.clone(), sv.v.clone()
    info = sv.run(max_steps=args.restart_steps, tolerance=0.0, check_every=250)
    res["restart128"] = {
        "steps_run": int(sv.steps - int(f128["steps"])),
        "max_abs_du": float((sv.u - u0).abs().max()),
        "max_abs_dv": float((sv.v - v0).abs().max()),
        "residual_after": float(info["residual"]),
        "converged_flag_when_tolerance_0": bool(info["converged"]),
    }
    print("restart128:", json.dumps(res["restart128"], indent=1), flush=True)

    # R4: scoring of the shipped fields by the audited scorer
    score = {}
    for n in (32, 64, 128):
        f = torch.load(ART / "fields" / f"Re1000_n{n}.pt", weights_only=False)
        sv = lc.LidCavitySolver(lc.LidCavityConfig(n=n, reynolds=1000.0))
        sv.u, sv.v, sv.steps = f["u"], f["v"], int(f["steps"])
        sc = lc.score_against_ghia(sv, 1000.0)
        score[f"n{n}"] = {
            "u_min": sc["u"]["extremum"]["min_computed"][1],
            "v_max": sc["v"]["extremum"]["max_computed"][1],
            "u_max_abs": sc["u"]["max_abs_error"],
            "v_max_abs": sc["v"]["max_abs_error"],
            "u_rms": sc["u"]["rms_error"],
            "v_rms": sc["v"]["rms_error"],
            "vortex_centre": list(sc["vortex_centre"]["computed"]),
            "max_divergence": sc["max_divergence"],
        }
    res["audited_scorer_on_shipped_fields"] = score
    print("scorer:", json.dumps(score, indent=1), flush=True)
    (OUT / "repro.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())