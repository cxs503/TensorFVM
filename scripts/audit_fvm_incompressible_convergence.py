#!/usr/bin/env python3
"""Continue each saved baseline field and measure how converged it actually is.

The run JSON records the continuity residual and the velocity change at the end of
the *iteration budget*, but the interesting question for a slowly marching
recirculation is different: does the reported quantity of interest (``L_r/H``,
``K``, ``Nu``) still drift?  This script reloads the saved field of every level,
continues it with the same settings for a fixed number of iterations, and writes
``runs/incompressible-baselines/convergence_audit.json`` with

* ``change``    -- ``max|du|`` over the last block of the continuation,
* ``continuity`` -- normalised continuity residual at the end,
* ``drift``     -- change of the quantity of interest over the continuation,
* ``qoi``       -- the quantity of interest before and after.

It is an *audit* only: the reported benchmark values stay those of the original
run, and a non-zero drift is reported instead of being hidden.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def _load_kernel():
    import importlib.util
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "src")
    pkg_name = "_tfvm_audit"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(src, "tensorfvm")]
    sys.modules[pkg_name] = pkg
    for mod in ("staggered_simple", "incompressible_baselines"):
        name = f"{pkg_name}.{mod}"
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(src, "tensorfvm", mod + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[pkg_name + ".incompressible_baselines"]


_k = _load_kernel()
METRICS = {
    "backward_facing_step": _k.backward_facing_step_metrics,
    "sudden_expansion": _k.sudden_expansion_metrics,
    "natural_convection_cavity": _k.natural_convection_metrics,
}
QOI = {
    "backward_facing_step": "reattachment_over_H",
    "sudden_expansion": "k_total_head",
    "natural_convection_cavity": "nusselt",
}


def build(rec):
    job = rec["job"]
    case = job["case"]
    if case == "backward_facing_step":
        solver = _k.build_backward_facing_step(
            job["level"], poisson_solver="sparse",
            outlet_length_H=job.get("outlet_length_H", 10.0))
    elif case == "sudden_expansion":
        solver = _k.build_sudden_expansion(
            job["level"], re_D=job.get("re_d", 100.0), poisson_solver="sparse",
            length_inlet_heights=job.get("length_inlet_heights", 12.0))
    else:
        solver = _k.build_natural_convection_cavity(job["level"],
                                                    rayleigh=job["rayleigh"],
                                                    poisson_solver="sparse")
    return solver


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/incompressible-baselines")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--poisson", default="sparse")
    ap.add_argument("--only", default=None,
                    help="comma separated tag prefixes to audit (default: all)")
    ap.add_argument("--out", default="convergence_audit.json")
    args = ap.parse_args(argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = os.path.join(root, args.runs)
    out = {}
    for path in sorted(glob.glob(os.path.join(runs, "*.json"))):
        tag = os.path.basename(path)[:-5]
        if tag.startswith("probe_") or tag.startswith("convergence_audit"):
            continue
        if args.only and not any(tag.startswith(p) for p in args.only.split(",")):
            continue
        fields = os.path.join(runs, tag + "_fields.npz")
        with open(path) as fh:
            rec = json.load(fh)
        case = rec["job"]["case"]
        qoi = QOI[case]
        before = rec["metrics"].get(qoi)
        if not os.path.exists(fields):
            out[tag] = dict(skipped="no saved field")
            continue
        solver = build(rec)
        solver.alpha_u = rec["settings"]["alpha_u"]
        solver.alpha_p = rec["settings"]["alpha_p"]
        solver.limited = rec["settings"]["limited"]
        with np.load(fields) as d:
            solver.u = d["u"].copy()
            solver.v = d["v"].copy()
            solver.p = d["p"].copy()
            solver.T = d["T"].copy()
        solver._outflow_initialised = True
        solver.apply_bc()
        if solver.limited and solver.energy:
            # Reproduce the delivered scheme: the limited (van Leer) correction is a
            # *frozen* defect there, so refreeze it on the loaded field instead of
            # re-evaluating it explicitly every iteration (which is unstable).
            solver.refresh_deferred(1.0)
        r = solver.run(max_iter=args.iterations, tol=0.0, check=10,
                       momentum_sweeps=4, p_cycles=2, max_stall=10 ** 6)
        after = METRICS[case](solver).get(qoi)
        drift = None if (before is None or after is None) else abs(after - before)
        rel = None if not before else (None if drift is None else drift / abs(before))
        out[tag] = dict(case=case, qoi=qoi, iterations=args.iterations,
                        qoi_before=before, qoi_after=after, drift=drift,
                        drift_rel=rel, change=float(r["change"]),
                        continuity=float(r["continuity"]))
        print(f"{tag}: {qoi} {before} -> {after} (drift {drift}, "
              f"{100 * (rel or 0):.3f} %), dU={r['change']:.2e}, "
              f"cont={r['continuity']:.2e}", flush=True)
    with open(os.path.join(runs, args.out), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())