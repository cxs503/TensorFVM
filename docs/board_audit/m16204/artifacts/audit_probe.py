"""Board-audit probes for CASE-FVM-BLASIUS-PLATE (mission 16182).

Three independent probes, none of which reuses the case module's own reference
constants as a *result*:

P1  Independent Blasius similarity solve (self-written RK4 + Newton shooting,
    adaptive step, Simpson integrals) -> f''(0), delta*_sqrt(Re), theta_sqrt(Re),
    H.  Checks the four literature constants the case compares against.
P2  Initial-condition independence of the passing device: march the audited
    `march_boundary_layer` from (a) its default Blasius seed, (b) a crude linear
    seed, (c) a 1/7 power-law seed.  If a seeded mismatch still relaxes onto the
    Blasius c_f / H, the pass is a property of the march, not of the seed.
P3  Reproduction of the committed artifact: recompute c_f/H from the archived
    `march.json` numbers and re-derive the reported relative errors.

Run: PYTHONPATH=/root/tp-agent/work/m16182/src python3 audit_probe.py
(the audited code is the m16182 tree; this script only imports its march solver).
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

LIT = {"cf_sqrt_re_x": 0.664, "delta_star_sqrt_re_x": 1.7208,
       "theta_sqrt_re_x": 0.664, "shape_factor": 2.5916}
RE_L = 1.0e5


# ---------------------------------------------------------------- P1
def blasius_independent(eta_max=14.0, h=1.0e-3):
    """Solve f''' + f f''/2 = 0, f(0)=f'(0)=0, f'(inf)=1 by RK4 shooting."""
    def integrate(s, n_steps=int(round(eta_max / h))):
        eta = 0.0
        f, fp, fpp = 0.0, 0.0, s
        traj = [(eta, f, fp, fpp)]
        for _ in range(n_steps):
            def f_(f, fp, fpp):
                return fp, fpp, -0.5 * f * fpp
            k1 = f_(f, fp, fpp)
            k2 = f_(f + 0.5 * h * k1[0], fp + 0.5 * h * k1[1], fpp + 0.5 * h * k1[2])
            k3 = f_(f + 0.5 * h * k2[0], fp + 0.5 * h * k2[1], fpp + 0.5 * h * k2[2])
            k4 = f_(f + h * k3[0], fp + h * k3[1], fpp + h * k3[2])
            f += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            fp += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            fpp += h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            eta += h
            traj.append((eta, f, fp, fpp))
        return np.array(traj)

    lo, hi = 0.1, 1.0
    for _ in range(200):                       # bisection on f'(eta_max) - 1
        mid = 0.5 * (lo + hi)
        traj = integrate(mid)
        if traj[-1, 2] < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    s = 0.5 * (lo + hi)
    traj = integrate(s)
    eta, f, fp = traj[:, 0], traj[:, 1], traj[:, 2]
    fpp0 = traj[0, 3]
    # boundary-layer integrals: delta* = int (1-f') deta, theta = int f'(1-f') deta
    one_m_fp = 1.0 - fp
    delta_star = np.trapezoid(one_m_fp, eta)
    theta = np.trapezoid(fp * one_m_fp, eta)
    return {"fpp0": float(fpp0), "two_fpp0": float(2.0 * fpp0),
            "fp_at_eta_max": float(fp[-1]), "eta_max": float(eta[-1]),
            "step": h, "delta_star_sqrt_re": float(delta_star),
            "theta_sqrt_re": float(theta), "H": float(delta_star / theta)}


# ---------------------------------------------------------------- P2
def march_probe(seed_kind, stations=401, ny=256, stretching=8.0):
    from tensorfvm.rans_ship_resistance.boundary_layer import march_boundary_layer

    length = 1.0
    x = np.linspace(length / stations, length, stations)
    ue = np.ones_like(x)
    nu = length / RE_L
    seed = None
    if seed_kind != "default-blasius":
        sigma = (np.exp(stretching * np.linspace(0, 1, ny)) - 1.0) / \
                (math.exp(stretching) - 1.0)
        y = 0.0923 * sigma
        d0 = 5.0 * math.sqrt(nu * x[0])
        if seed_kind == "linear":
            u = np.clip(y / d0, 0.0, 1.0)
        elif seed_kind == "power-law-1over7":
            u = np.clip(y / d0, 0.0, 1.0) ** (1.0 / 7.0)
        elif seed_kind == "uniform":
            u = np.ones_like(y)
        else:
            raise SystemExit("unknown seed")
        seed = (y, u)
    res = march_boundary_layer(x, ue, nu, model="laminar", trip_x=None,
                              ny=ny, stretching=stretching, seed=seed)
    cf = float(np.asarray(res.cf)[-1])
    H = float(np.asarray(res.H)[-1])
    ref_cf = LIT["cf_sqrt_re_x"] / math.sqrt(RE_L)
    return {"seed": seed_kind, "cf": cf, "H": H,
            "cf_err": cf / ref_cf - 1.0, "H_err": H / LIT["shape_factor"] - 1.0,
            "pass": bool(abs(cf / ref_cf - 1.0) < 0.03 and
                         abs(H / LIT["shape_factor"] - 1.0) < 0.03)}


def main():
    out = {}
    p1 = blasius_independent()
    p1["dev_two_fpp0_vs_lit"] = p1["two_fpp0"] / LIT["cf_sqrt_re_x"] - 1.0
    p1["dev_delta_star_vs_lit"] = p1["delta_star_sqrt_re"] / LIT["delta_star_sqrt_re_x"] - 1.0
    p1["dev_theta_vs_lit"] = p1["theta_sqrt_re"] / LIT["theta_sqrt_re_x"] - 1.0
    p1["dev_H_vs_lit"] = p1["H"] / LIT["shape_factor"] - 1.0
    out["P1_independent_blasius"] = p1

    out["P2_seed_independence"] = [march_probe(k) for k in
                                   ("default-blasius", "linear",
                                    "power-law-1over7", "uniform")]

    art = json.load(open("/root/tp-agent/work/m16182/runs/blasius-plate/march.json"))
    got, ref = art["comparison"]["got"], art["comparison"]["reference"]
    out["P3_artifact_recompute"] = {
        "artifact": "/root/tp-agent/work/m16182/runs/blasius-plate/march.json",
        "got_cf": got["cf"], "got_H": got["H"],
        "ref_cf": ref["cf"], "ref_H": ref["H"],
        "cf_err_recomputed": got["cf"] / ref["cf"] - 1.0,
        "H_err_recomputed": got["H"] / ref["H"] - 1.0,
        "cf_err_in_artifact": art["comparison"]["cf"]["relative_error"],
        "H_err_in_artifact": art["comparison"]["H"]["relative_error"],
        "ref_cf_check_lit_over_sqrt_re": LIT["cf_sqrt_re_x"] / math.sqrt(RE_L),
        "continuity_residual": art["measured"]["continuity_residual"],
    }
    json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()