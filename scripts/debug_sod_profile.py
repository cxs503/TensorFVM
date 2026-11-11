#!/usr/bin/env python3
"""Debug: examine the TensorFVM Sod profile to understand wave positions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import math
from tensorfvm.compressible.cases import sod, GAMMA_AIR

torch.set_num_threads(1)

solver, meta = sod(400, order=2, t_end=0.2, cfl=0.4)
solver.run(0.2)
prim = solver.primitive()

xc = meta["xc"]
rho = prim["rho"][:, 0]
u = prim["u"][:, 0]
p = prim["p"][:, 0]

# Exact solution
xi = (xc - meta["x0"]) / 0.2
rho_e, u_e, p_e = meta["exact"].sample(xi)

# Print key values
print("=== Density profile (selected points) ===")
for i in range(0, 400, 20):
    print(f"  x={xc[i]:.4f}  rho={rho[i]:.6f}  rho_exact={rho_e[i]:.6f}  p={p[i]:.6f}  u={u[i]:.6f}")

print("\n=== Around shock region (x > 0.7) ===")
mask = xc > 0.7
for i in torch.where(mask)[0]:
    print(f"  x={xc[i]:.4f}  rho={rho[i]:.6f}  rho_exact={rho_e[i]:.6f}  p={p[i]:.6f}")

print("\n=== Around contact region (0.6 < x < 0.75) ===")
mask = (xc > 0.6) & (xc < 0.75)
for i in torch.where(mask)[0]:
    print(f"  x={xc[i]:.4f}  rho={rho[i]:.6f}  rho_exact={rho_e[i]:.6f}  p={p[i]:.6f}")

# Exact positions
exact_p_star = float(meta["exact"].pstar)
exact_u_star = float(meta["exact"].ustar)
exact_rho_star_r = float(meta["exact"].density_star_r())
exact_rho_star_l = float(meta["exact"].density_star_l())
exact_shock_speed = float(meta["exact"].shock_speed_r())

print(f"\n=== Exact solution values ===")
print(f"  p_star = {exact_p_star:.6f}")
print(f"  u_star = {exact_u_star:.6f}")
print(f"  rho_star_l = {exact_rho_star_l:.6f}")
print(f"  rho_star_r = {exact_rho_star_r:.6f}")
print(f"  shock_speed = {exact_shock_speed:.6f}")
print(f"  shock_pos = {0.5 + exact_shock_speed * 0.2:.6f}")
print(f"  contact_pos = {0.5 + exact_u_star * 0.2:.6f}")

# Find shock: where density jumps from ~0.125 to ~0.265
# The shock is the rightmost discontinuity
drho = torch.diff(rho)
print(f"\n=== Density gradient (abs) ===")
# Find all significant jumps
threshold = 0.02
jumps = torch.where(drho.abs() > threshold)[0]
for j in jumps:
    print(f"  Jump at x={xc[j]:.4f}: drho={drho[j]:.6f}, rho[{j}]={rho[j]:.6f} -> rho[{j+1}]={rho[j+1]:.6f}")
