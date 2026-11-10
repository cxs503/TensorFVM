#!/usr/bin/env python
"""Time-domain seakeeping evidence driver for the sigma-grid wave tank.

One invocation = one head-wave case (one period, one grid, one damping scale).
Everything is written to JSON so the summary step can be machine-checked; the
driver never post-processes by hand.

    python scripts/wave_tank_td/run_seakeeping.py --period 0.9 --grid coarse \
        --outdir docs/wave-tank-td-seakeeping/artifacts

Reference coefficients (added mass, radiation damping, excitation) come from the
in-repo potential-flow solver :mod:`tensorfvm.wave_tank_td.potential2d` at the
same period; the hydrostatic stiffness comes from
:func:`tensorfvm.wave_tank_td.seakeeping.hydrostatic_stiffness`, which is
verified against the second derivative of the hydrostatic potential energy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from tensorfvm.wave_tank_td import seakeeping as sk  # noqa: E402
from tensorfvm.wave_tank_td.potential2d import section_hydrodynamics  # noqa: E402

GEOMETRY = {"length": 1.2, "draft": 0.2, "freeboard": 0.05}
GRIDS = {"coarse": (200, 20, 100), "fine": (280, 28, 140)}


def geometry_gate(outdir: Path | None) -> dict:
    """Cb / L / B / nabla pre-gate plus the 2-D section hydrostatics."""
    from tensorfvm.wave_tank_td.cases import run_geometry_gate

    hull = run_geometry_gate(None)
    section = sk.floating_strip(**GEOMETRY)
    k_analytic = sk.hydrostatic_stiffness(section)
    k_numeric = sk.potential_energy_stiffness(section.polygon, section.mass, section.ref)
    out = {
        "hull_3d": hull,
        "section_2d": section.as_dict(),
        "stiffness_analytic": k_analytic,
        "stiffness_numeric": k_numeric,
        "stiffness_relative_error": {
            "C33": abs(k_numeric["C33"] - k_analytic["C33"]) / k_analytic["C33"],
            "C55": abs(k_numeric["C55"] - k_analytic["C55"]) / k_analytic["C55"],
            "C35": abs(k_numeric["C35"]),
        },
        "check": {
            "C33_analytic_vs_energy_hessian": bool(
                abs(k_numeric["C33"] - k_analytic["C33"]) / k_analytic["C33"] <= 1e-5),
            "C55_analytic_vs_energy_hessian": bool(
                abs(k_numeric["C55"] - k_analytic["C55"]) / k_analytic["C55"] <= 1e-5),
            "GM_positive": bool(k_analytic["GM"] > 0.0),
        },
    }
    out["pass"] = all(out["check"].values())
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "geometry_gate.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    return out


def reference(period: float, section, nx: int = 200, nz: int = 52) -> dict:
    omega = 2.0 * math.pi / period
    centred = sk.floating_strip(section.length, section.draft, section.freeboard, x0=0.0)
    hydro = section_hydrodynamics(centred.polygon, omega, depth=1.0, nx=nx, nz=nz)
    k_stiff = sk.hydrostatic_stiffness(section)
    a_mat = np.asarray(hydro["A"], dtype=float)
    b_mat = np.asarray(hydro["B"], dtype=float)
    k_mat = np.array([[k_stiff["C33"], 0.0], [0.0, k_stiff["C55"]]])
    f_vec = np.array([complex(*hydro["F_exc"][0]), complex(*hydro["F_exc"][1])])
    d = float(section.ref[1] - section.cg[1])
    rao = sk.reference_rao(omega, section.mass, section.inertia, d, a_mat, b_mat,
                           k_mat, f_vec, wave_amp=hydro["amp"] if "amp" in hydro else 0.02,
                           k=hydro["k"])
    zeta = {
        "heave": float(b_mat[0, 0] / (2.0 * math.sqrt(k_mat[0, 0] * (section.mass + a_mat[0, 0])))),
        "pitch": float(b_mat[1, 1] / (2.0 * math.sqrt(k_mat[1, 1] * (section.inertia + a_mat[1, 1])))),
    }
    return {"period": period, "omega": omega, "k": hydro["k"],
            "wavelength": hydro["wavelength"], "A": a_mat.tolist(),
            "B": b_mat.tolist(), "C": k_mat.tolist(),
            "F_exc": [[f_vec[0].real, f_vec[0].imag], [f_vec[1].real, f_vec[1].imag]],
            "damping_ratio": zeta, "rao": rao,
            "note": ("potential-flow reference (potential2d) with the stiffness "
                     "from the verified hydrostatic energy Hessian")}


def run_one(period: float, grid: str, amp: float, periods: float,
            damp_scale: float, outdir: Path | None, label: str | None,
            length: float = 14.0, add_mass: bool = True,
            damp_waves: float = 1.5) -> dict:
    nx, ns, spp = GRIDS[grid]
    section = sk.floating_strip(**GEOMETRY, x0=0.5 * length - 1.0)
    ref = reference(period, section)
    added = np.asarray(ref["A"], dtype=float).diagonal() if add_mass else np.zeros(2)
    damping = damp_scale * np.asarray(ref["B"], dtype=float).diagonal()
    case = sk.run_seakeeping_case(amp=amp, period=period, nx=nx, ns=ns,
                                  length=length, spp=spp, periods=periods,
                                  section=section, added_mass=added,
                                  damping=damping, damp_waves=damp_waves,
                                  name=label or f"{grid}_T{period:.2f}_d{damp_scale:.2f}")
    rao_ref = ref["rao"]
    case["reference"] = ref
    case["comparison"] = {
        "rao_heave_td": case["rao"]["rao_heave"],
        "rao_heave_ref": rao_ref["rao_heave"],
        "rao_heave_rel_error": abs(case["rao"]["rao_heave"] - rao_ref["rao_heave"])
        / max(rao_ref["rao_heave"], 1e-12),
        "rao_pitch_td": case["rao"]["rao_pitch"],
        "rao_pitch_ref": rao_ref["rao_pitch"],
        "rao_pitch_rel_error": abs(case["rao"]["rao_pitch"] - rao_ref["rao_pitch"])
        / max(rao_ref["rao_pitch"], 1e-12),
        "damping_scale": damp_scale,
        "omega_over_omega_n": {
            "heave": ref["omega"] / rao_ref["natural_heave"],
            "pitch": ref["omega"] / rao_ref["natural_pitch"],
        },
    }
    keep = [k for k in case if k != "series"]
    slim = {k: case[k] for k in keep}
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{case['name']}.json").write_text(
            json.dumps(slim, indent=2, sort_keys=True, default=float))
    return slim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", type=float, default=0.9)
    ap.add_argument("--grid", choices=sorted(GRIDS), default="coarse")
    ap.add_argument("--amp", type=float, default=0.02)
    ap.add_argument("--periods", type=float, default=12.0)
    ap.add_argument("--damp-scale", type=float, default=1.0)
    ap.add_argument("--no-added-mass", action="store_true")
    ap.add_argument("--length", type=float, default=14.0)
    ap.add_argument("--no-beach", action="store_true",
                    help="negative control: remove the absorbing beach")
    ap.add_argument("--label", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--geometry-gate", action="store_true")
    ap.add_argument("--mode", choices=["seakeeping", "propagation"], default="seakeeping")
    a = ap.parse_args(argv)
    outdir = Path(a.outdir) if a.outdir else None
    if a.geometry_gate:
        gate = geometry_gate(outdir)
        print(json.dumps(gate["check"], indent=2, sort_keys=True))
        return 0 if gate["pass"] else 1
    if a.mode == "propagation":
        # body-free wave propagation: the tank-property gates (wave height,
        # dispersion k) must be measured without a scatterer in the field.
        from tensorfvm.wave_tank_td import cases as wt_cases

        nx, ns, spp = GRIDS[a.grid]
        out = wt_cases.run_propagation_case(
            amp=a.amp, period=a.period, nx=nx, ns=ns, length=a.length,
            periods=min(a.periods, 6.0), steps_per_period=spp,
            name=a.label or f"propagation_{a.grid}_T{a.period:.2f}",
            artifact_dir=outdir)
        print(json.dumps({k: out[k] for k in ("name", "amp_error_max", "k_error",
                                              "k_three_probe", "k_reference", "criteria")},
                         indent=2, sort_keys=True))
        return 0
    case = run_one(a.period, a.grid, a.amp, a.periods, a.damp_scale, outdir,
                   a.label, a.length, add_mass=not a.no_added_mass,
                   damp_waves=0.0 if a.no_beach else 1.5)
    print(json.dumps({"case": case["case"], "comparison": case["comparison"],
                      "reflection": case["reflection"]["C_R"],
                      "drift": case["drift"], "numerics": case["numerics"]},
                     indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())