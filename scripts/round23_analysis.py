#!/usr/bin/env python3
"""Round 23 analysis: what the refined ladder rungs actually contain.

Round 22 pre-registered a grid ladder for ``CASE-FVM-PERIODIC-HILLS``
(closure-free, ``Re = 2800``, constant CFL) and submitted four rungs.  The jobs
returned ``rc = 0``, so round 22 wrote ``verdict = inconclusive / fewer than two
levels`` -- its reader had silently dropped rungs B/C/D because the
pre-registered artefact stem (``round22_laminar_*``) never matched the producer's
name (``round20_laminar_*``).  Round 23 fixes that reader and then asks the next
question: **are the three refined artefacts usable at all?**

They are not: each carries ``config.diverged_at_step`` = 508 / 538 / 652 and
``samples = 0``, i.e. the solver's non-finite guard fired inside the startup
transient and the time mean is identically zero.  Averaging zeros is not a
measurement, so the ladder is *unusable*, which is a different (and stronger)
statement than "inconclusive".

This script produces the audit table and the figures:

* per rung: ``samples``, ``diverged_at_step``, ``t_diverged``, whether the stored
  mean wall shear is identically zero, and the resulting usability flag;
* the pre-registered attribution, re-read with the corrected reader;
* the dt-sweep probe (:mod:`scripts.round23_divergence_probe`) that shows the
  blow-up is a **time-step** limit, not a mesh-intrinsic failure;
* two figures: the ladder with usable/unusable markers, and the stability probe.

Usage
-----
    PYTHONPATH=src python scripts/round23_analysis.py \
        --ladder /tmp/r23out/round23_table.json \
        --probe /tmp/probe_a.json --probe /tmp/probe_b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
REFERENCE_X_R = 5.41
RUNGS = [
    ("A", "round20_laminar_Re2800_96x48"),
    ("Ap", "round20_laminar_Re2800_96x48dt01"),
    ("B", "round20_laminar_Re2800_144x72"),
    ("C", "round20_laminar_Re2800_192x96"),
    ("D", "round20_laminar_Re2800_240x120"),
]


def audit_rung(level, stem):
    js = CASEDIR / f"{stem}.json"
    npz = CASEDIR / f"{stem}.npz"
    payload = json.loads(js.read_text())
    cfg = payload.get("config") or {}
    with np.load(npz) as z:
        tau = np.asarray(z["mean_tau_b"], dtype=float)
    finite = np.isfinite(tau)
    nonzero = bool(np.any(np.abs(tau[finite]) > 0.0)) if finite.any() else False
    diverged = cfg.get("diverged_at_step")
    samples = payload.get("samples")
    dt = cfg.get("time_step_s")
    usable = bool(samples) and nonzero and diverged is None
    return {
        "level": level,
        "stem": stem,
        "nx": cfg.get("nx"),
        "ny": cfg.get("ny"),
        "dt": dt,
        "steps_requested": cfg.get("steps"),
        "samples": samples,
        "diverged_at_step": diverged,
        "t_diverged": None if not diverged or not dt else float(diverged * dt),
        "mean_tau_identically_zero": not nonzero,
        "usable": usable,
        "reason": ("ok" if usable else
                   ("solver diverged inside startup transient; time mean is zero"
                    if diverged is not None else "no samples / zero shear")),
        "x_r_over_H": (payload.get("time_averaged") or {}).get("x_r_over_H"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", default="/tmp/r23out/round23_table.json")
    ap.add_argument("--probe", action="append", default=None)
    ap.add_argument("--out-dir", default=str(CASEDIR))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ladder = json.loads(Path(args.ladder).read_text())
    audit = [audit_rung(lv, stem) for lv, stem in RUNGS]
    usable = [a for a in audit if a["usable"]]

    probes = []
    for path in (args.probe or []):
        probes.extend(json.loads(Path(path).read_text()).get("cases", []))
    probe_payload = {
        "case": "CASE-FVM-PERIODIC-HILLS round 23 dt-sweep probe",
        "closure": "laminar (nu_t == 0)",
        "reynolds": 2800.0,
        "note": ("Short single-core runs, no time averaging: reproduces the "
                 "cluster divergence step of rungs B/C/D within ~2 % and shows "
                 "that halving dt at fixed mesh removes it."),
        "cases": probes,
    }
    (out / "round23_probe.json").write_text(json.dumps(probe_payload, indent=1))

    distinct_meshes = sorted({(a["nx"], a["ny"]) for a in usable})
    verdict = {
        # a ladder needs >= 2 *distinct meshes*; two time steps on one mesh are a
        # temporal pair, not a spatial level (this is the defect round 23 fixed
        # in its own first draft, caught by tests/test_periodic_hill_round23.py)
        "ladder_usable": len(distinct_meshes) >= 2,
        "distinct_meshes": [list(m) for m in distinct_meshes],
        "usable_levels": [a["level"] for a in usable],
        "n_usable": len(usable),
        "attribution_status": ("measured" if len(distinct_meshes) >= 2
                               else "not_measurable_with_stored_artefacts"),
        "reason": ("3 of 5 rungs diverged inside the startup transient "
                   "(diverged_at_step = 508/538/652, samples = 0); the two "
                   "surviving rungs are 96x48 at dt = 0.02 and dt = 0.01, i.e. "
                   "a one-mesh ladder, which cannot support a spatial "
                   "order/extrapolation verdict.  The pre-registered rule "
                   "therefore returns non-finite/inconclusive and is NOT to be "
                   "reported as 'discretisation ruled out'.")
    }

    table = {
        "case": "CASE-FVM-PERIODIC-HILLS round 23 (ladder usability audit)",
        "reference_x_r_over_H": REFERENCE_X_R,
        "rungs": audit,
        "verdict": verdict,
        "pre_registered_attribution": {
            "largest_area_bubble": ladder.get("attribute_x_r_over_H"),
            "crest_bubble": ladder.get("attribute_x_r_over_H_crest_bubble"),
            "separation": ladder.get("attribute_x_s_over_H"),
        },
        "temporal_dt_half": ladder.get("temporal"),
        "dt_sweep": [
            {k: c[k] for k in ("label", "nx", "ny", "dt", "diverged_at_step",
                               "t_diverged", "max_abs_u_last", "advective_cfl",
                               "explicit_viscous_number")}
            for c in probes
        ],
    }
    (out / "round23_table.json").write_text(json.dumps(table, indent=1, default=float))

    print(json.dumps({"usable": verdict["usable_levels"],
                      "n_usable": verdict["n_usable"],
                      "diverged": {a["level"]: a["diverged_at_step"] for a in audit},
                      "temporal_dt_half": table["temporal_dt_half"]},
                     indent=1, default=float))

    if not args.no_figures:
        _figures(out, audit, probes, ladder)
    return 0


def _figures(out, audit, probes, ladder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for a in audit:
        h = 9.0 / a["nx"]
        if a["usable"] and a["x_r_over_H"] is not None:
            ax[0].plot(h, a["x_r_over_H"], "o", color="C0", ms=7,
                       label="usable rung" if a["level"] == "A" else None)
            ax[0].annotate(f"{a['level']} {a['nx']}x{a['ny']}",
                           (h, a["x_r_over_H"]), textcoords="offset points",
                           xytext=(6, 4), fontsize=7)
        else:
            ax[0].plot(h, 0.0, "x", color="C3", ms=9,
                       label="diverged (samples = 0)" if a["level"] == "B" else None)
            ax[0].annotate(f"{a['level']} {a['nx']}x{a['ny']} "
                           f"@step {a['diverged_at_step']}",
                           (h, 0.0), textcoords="offset points", xytext=(6, 4),
                           fontsize=7, color="C3")
    ax[0].axhline(REFERENCE_X_R, color="k", lw=1.0, label="Breuer 2009 DNS 5.41")
    ax[0].axhspan(REFERENCE_X_R * 0.9, REFERENCE_X_R * 1.1, color="0.87")
    ax[0].set_xlabel(r"$\Delta x/H = 9/n_x$")
    ax[0].set_ylabel(r"$x_r/H$")
    ax[0].set_title("round 22 ladder, re-read with the corrected artefact stems")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    for mesh, marker in (((144, 72), "o"), ((192, 96), "s"), ((96, 48), "^")):
        pts = [c for c in probes if (c["nx"], c["ny"]) == mesh]
        if not pts:
            continue
        pts.sort(key=lambda c: c["dt"])
        dts = [c["dt"] for c in pts]
        td = [c["t_diverged"] if c["diverged_at_step"] else
              c["max_steps"] * c["dt"] for c in pts]
        ax[1].semilogx(dts, td, marker + "-", label=f"{mesh[0]}x{mesh[1]}")
        for c, t in zip(pts, td):
            ok = c["diverged_at_step"] is None
            ax[1].annotate("stable" if ok else f"div@{c['diverged_at_step']}",
                           (c["dt"], t), textcoords="offset points",
                           xytext=(4, 5), fontsize=7,
                           color="C2" if ok else "C3")
    ax[1].set_xlabel(r"$\Delta t$")
    ax[1].set_ylabel(r"$t$ reached (or $t$ of blow-up)")
    ax[1].set_title("dt sweep: blow-up is step-size limited, not mesh-intrinsic")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("CASE-FVM-PERIODIC-HILLS round 23: ladder unusable, cause measured")
    fig.tight_layout()
    fig.savefig(out / "fig_round23_ladder_usability.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())