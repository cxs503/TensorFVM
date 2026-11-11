#!/usr/bin/env python3
"""Run Sod shock tube with TensorFVM at multiple resolutions and extract metrics."""
import json
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tensorfvm.compressible.cases import sod, GAMMA_AIR
from tensorfvm.compressible.riemann import ExactRiemann

torch.set_num_threads(1)


def find_wave_positions(xc, rho, p, x0=0.5):
    """Locate shock, contact, and rarefaction head/tail from the numerical profile.
    
    For Sod shock tube (left: rho=1, p=1; right: rho=0.125, p=0.1):
    Structure: [left state] [rarefaction] [star-left] [contact] [star-right] [shock] [right state]
    """
    drho = torch.diff(rho)
    
    # Shock: rightmost large density jump (from ~0.266 to 0.125, so drho < 0)
    # Look for the most negative gradient in the right half (x > 0.7)
    right_mask = xc > 0.7
    right_indices = torch.where(right_mask)[0]
    if len(right_indices) > 2:
        # drho[i] = rho[i+1] - rho[i], so drho at index i corresponds to position xc[i]
        right_drho_indices = right_indices[right_indices < len(drho)]
        if len(right_drho_indices) > 0:
            shock_local_idx = torch.argmin(drho[right_drho_indices])
            shock_idx = right_drho_indices[shock_local_idx]
            shock_pos = float(xc[shock_idx])
        else:
            shock_pos = float('nan')
    else:
        shock_pos = float('nan')
    
    # Contact: density jump between star-left (~0.447) and star-right (~0.266)
    # This is in the middle region, between x=0.5 and the shock
    # Look for the largest negative gradient in the band [0.55, 0.80]
    contact_mask = (xc > 0.55) & (xc < 0.80)
    contact_indices = torch.where(contact_mask)[0]
    if len(contact_indices) > 2:
        contact_drho_indices = contact_indices[contact_indices < len(drho)]
        if len(contact_drho_indices) > 0:
            contact_local_idx = torch.argmin(drho[contact_drho_indices])
            contact_idx = contact_drho_indices[contact_local_idx]
            contact_pos = float(xc[contact_idx])
        else:
            contact_pos = float('nan')
    else:
        contact_pos = float('nan')
    
    # Rarefaction head: where rho first drops significantly from 1.0
    head_pos = float('nan')
    for i in range(len(rho)):
        if rho[i] < 0.99:
            head_pos = float(xc[i])
            break
    
    # Rarefaction tail: where rho reaches the star-left plateau (~0.42-0.45)
    # Look for where the density gradient becomes small after the rarefaction fan
    # The tail is where rho first reaches ~0.44 (star-left value)
    tail_pos = float('nan')
    for i in range(len(rho)):
        if rho[i] < 0.46 and rho[i] > 0.40:
            # Check if we're in a plateau (small gradient)
            if i + 5 < len(rho):
                local_var = rho[i:i+5].std()
                if local_var < 0.01:
                    tail_pos = float(xc[i])
                    break
    
    return {
        "shock": shock_pos,
        "contact": contact_pos,
        "rarefaction_head": head_pos,
        "rarefaction_tail": tail_pos,
    }


def get_star_region_values(xc, rho, u, p, contact_pos, shock_pos):
    """Extract star region values."""
    # Star-left: between rarefaction tail and contact
    # Star-right: between contact and shock
    
    # Use a band around the expected star region
    star_l_mask = (xc > 0.5) & (xc < contact_pos)
    star_r_mask = (xc > contact_pos) & (xc < shock_pos)
    
    if star_l_mask.sum() > 0:
        rho_star_l = float(rho[star_l_mask].mean())
        u_star_l = float(u[star_l_mask].mean())
        p_star_l = float(p[star_l_mask].mean())
    else:
        rho_star_l = u_star_l = p_star_l = float('nan')
    
    if star_r_mask.sum() > 0:
        rho_star_r = float(rho[star_r_mask].mean())
        u_star_r = float(u[star_r_mask].mean())
        p_star_r = float(p[star_r_mask].mean())
    else:
        rho_star_r = u_star_r = p_star_r = float('nan')
    
    return {
        "rho_star_l": rho_star_l,
        "u_star_l": u_star_l,
        "p_star_l": p_star_l,
        "rho_star_r": rho_star_r,
        "u_star_r": u_star_r,
        "p_star_r": p_star_r,
    }


def run_sod_resolution(nx, order=2, t_end=0.2, cfl=0.4):
    """Run Sod at a given resolution and return metrics."""
    solver, meta = sod(nx, order=order, t_end=t_end, cfl=cfl)
    initial = solver.integrals()
    solver.run(t_end)
    prim = solver.primitive()
    
    xc = meta["xc"]
    rho = prim["rho"][:, 0]
    u = prim["u"][:, 0]
    p = prim["p"][:, 0]
    
    # Exact solution
    xi = (xc - meta["x0"]) / t_end
    rho_e, u_e, p_e = meta["exact"].sample(xi)
    
    # L1 errors
    l1_rho = float((rho - rho_e).abs().mean())
    l1_u = float((u - u_e).abs().mean())
    l1_p = float((p - p_e).abs().mean())
    
    # L_inf errors
    linf_rho = float((rho - rho_e).abs().max())
    linf_u = float((u - u_e).abs().max())
    linf_p = float((p - p_e).abs().max())
    
    # Wave positions
    positions = find_wave_positions(xc, rho, p)
    
    # Star region values
    star = get_star_region_values(xc, rho, u, p, positions["contact"], positions["shock"])
    
    # Exact star region values
    exact_p_star = float(meta["exact"].pstar)
    exact_u_star = float(meta["exact"].ustar)
    exact_rho_star_r = float(meta["exact"].density_star_r())
    exact_rho_star_l = float(meta["exact"].density_star_l())
    
    # Exact wave positions
    exact_shock_speed = float(meta["exact"].shock_speed_r())
    exact_shock_pos = 0.5 + exact_shock_speed * t_end
    exact_contact_pos = 0.5 + exact_u_star * t_end
    c_left = math.sqrt(GAMMA_AIR * 1.0 / 1.0)
    exact_rarefaction_head = 0.5 - c_left * t_end
    c_star_l = math.sqrt(GAMMA_AIR * exact_p_star / exact_rho_star_l)
    exact_rarefaction_tail = 0.5 + (exact_u_star - c_star_l) * t_end
    
    # Conservation
    drift = solver.budget_drift(initial)
    
    return {
        "nx": nx,
        "order": order,
        "t_end": t_end,
        "nsteps": solver.nsteps,
        "l1_rho": l1_rho,
        "l1_u": l1_u,
        "l1_p": l1_p,
        "linf_rho": linf_rho,
        "linf_u": linf_u,
        "linf_p": linf_p,
        "positions": positions,
        "exact_positions": {
            "shock": exact_shock_pos,
            "contact": exact_contact_pos,
            "rarefaction_head": exact_rarefaction_head,
            "rarefaction_tail": exact_rarefaction_tail,
        },
        "star_region": star,
        "exact_star_region": {
            "p_star": exact_p_star,
            "u_star": exact_u_star,
            "rho_star_l": exact_rho_star_l,
            "rho_star_r": exact_rho_star_r,
        },
        "drift_mass": drift["mass"],
        "drift_energy": drift["energy"],
        "positivity_fixes": solver.positivity_fixes,
        "profile": {
            "x": xc.tolist(),
            "rho": rho.tolist(),
            "u": u.tolist(),
            "p": p.tolist(),
            "rho_exact": rho_e.tolist(),
            "u_exact": u_e.tolist(),
            "p_exact": p_e.tolist(),
        },
    }


def main():
    resolutions = [100, 200, 400, 800]
    results = []
    
    for nx in resolutions:
        print(f"Running TensorFVM Sod with nx={nx}...")
        result = run_sod_resolution(nx)
        results.append(result)
        print(f"  L1 rho={result['l1_rho']:.6e}, u={result['l1_u']:.6e}, p={result['l1_p']:.6e}")
        print(f"  Shock pos: num={result['positions']['shock']:.6f}, exact={result['exact_positions']['shock']:.6f}, err={abs(result['positions']['shock']-result['exact_positions']['shock'])/result['exact_positions']['shock']*100:.2f}%")
        print(f"  Contact pos: num={result['positions']['contact']:.6f}, exact={result['exact_positions']['contact']:.6f}, err={abs(result['positions']['contact']-result['exact_positions']['contact'])/result['exact_positions']['contact']*100:.2f}%")
        print(f"  Rarefaction head: num={result['positions']['rarefaction_head']:.6f}, exact={result['exact_positions']['rarefaction_head']:.6f}")
        print(f"  Rarefaction tail: num={result['positions']['rarefaction_tail']:.6f}, exact={result['exact_positions']['rarefaction_tail']:.6f}")
        print(f"  Star p_l={result['star_region']['p_star_l']:.6f} (exact={result['exact_star_region']['p_star']:.6f})")
        print(f"  Star p_r={result['star_region']['p_star_r']:.6f} (exact={result['exact_star_region']['p_star']:.6f})")
    
    # Compute convergence orders
    for order in (1, 2):
        sub = [r for r in results if r["order"] == order]
        if len(sub) >= 2:
            for i in range(len(sub) - 1):
                r1, r2 = sub[i], sub[i+1]
                order_rho = math.log(r1["l1_rho"] / r2["l1_rho"]) / math.log(r2["nx"] / r1["nx"])
                order_p = math.log(r1["l1_p"] / r2["l1_p"]) / math.log(r2["nx"] / r1["nx"])
                print(f"  Order {order}: nx {r1['nx']}->{r2['nx']}, rho order={order_rho:.3f}, p order={order_p:.3f}")
    
    output = {
        "tool": "TensorFVM",
        "version": "local",
        "case": "Sod shock tube (Sod 1978)",
        "gamma": GAMMA_AIR,
        "left_state": {"rho": 1.0, "u": 0.0, "p": 1.0},
        "right_state": {"rho": 0.125, "u": 0.0, "p": 0.1},
        "domain": [0.0, 1.0],
        "t_end": 0.2,
        "results": results,
    }
    
    outpath = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sod_tensorfvm.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
