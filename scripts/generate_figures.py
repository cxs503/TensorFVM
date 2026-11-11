#!/usr/bin/env python3
"""Generate cross-validation report figures for Sod shock tube."""
import json
import os
import sys
import math
import base64
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load TensorFVM results
with open(os.path.join(os.path.dirname(__file__), "..", "artifacts", "sod_tensorfvm.json")) as f:
    tfvm_data = json.load(f)

# Exact solution (from TensorFVM's ExactRiemann)
from tensorfvm.compressible.cases import sod, GAMMA_AIR

torch = __import__('torch')
torch.set_num_threads(1)

solver, meta = sod(800, order=2, t_end=0.2, cfl=0.4)
solver.run(0.2)
prim = solver.primitive()

xc = meta["xc"].numpy()
rho_num = prim["rho"][:, 0].numpy()
u_num = prim["u"][:, 0].numpy()
p_num = prim["p"][:, 0].numpy()

xi = (meta["xc"].numpy() - 0.5) / 0.2
rho_exact, u_exact, p_exact = meta["exact"].sample(xi)
rho_exact = rho_exact.numpy()
u_exact = u_exact.numpy()
p_exact = p_exact.numpy()

outdir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(outdir, exist_ok=True)

# Figure 1: Density comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(xc, rho_num, 'b-', linewidth=1.5, label='TensorFVM (nx=800, order=2)')
axes[0].plot(xc, rho_exact, 'r--', linewidth=1.0, label='Exact Riemann')
axes[0].set_ylabel('Density')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Sod Shock Tube: Density Profile at t=0.2')

axes[1].plot(xc, u_num, 'b-', linewidth=1.5, label='TensorFVM')
axes[1].plot(xc, u_exact, 'r--', linewidth=1.0, label='Exact')
axes[1].set_ylabel('Velocity')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

axes[2].plot(xc, p_num, 'b-', linewidth=1.5, label='TensorFVM')
axes[2].plot(xc, p_exact, 'r--', linewidth=1.0, label='Exact')
axes[2].set_ylabel('Pressure')
axes[2].set_xlabel('x')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Mark wave positions
exact_p_star = float(meta["exact"].pstar)
exact_u_star = float(meta["exact"].ustar)
exact_rho_star_r = float(meta["exact"].density_star_r())
exact_rho_star_l = float(meta["exact"].density_star_l())
exact_shock_speed = float(meta["exact"].shock_speed_r())
c_left = math.sqrt(GAMMA_AIR)
c_star_l = math.sqrt(GAMMA_AIR * exact_p_star / exact_rho_star_l)

shock_pos = 0.5 + exact_shock_speed * 0.2
contact_pos = 0.5 + exact_u_star * 0.2
rare_head = 0.5 - c_left * 0.2
rare_tail = 0.5 + (exact_u_star - c_star_l) * 0.2

for ax in axes:
    ax.axvline(x=rare_head, color='g', linestyle=':', alpha=0.5, label='Rarefaction head')
    ax.axvline(x=rare_tail, color='g', linestyle='-.', alpha=0.5, label='Rarefaction tail')
    ax.axvline(x=contact_pos, color='m', linestyle=':', alpha=0.5, label='Contact')
    ax.axvline(x=shock_pos, color='r', linestyle=':', alpha=0.5, label='Shock')

plt.tight_layout()
png_path = os.path.join(outdir, "sod_density_velocity_pressure.png")
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Convergence plot
fig, ax = plt.subplots(figsize=(8, 6))

nx_list = [100, 200, 400, 800]
l1_rho_list = [r["l1_rho"] for r in tfvm_data["results"]]
l1_p_list = [r["l1_p"] for r in tfvm_data["results"]]

ax.loglog(nx_list, l1_rho_list, 'bo-', linewidth=2, markersize=8, label='L1(ρ)')
ax.loglog(nx_list, l1_p_list, 'rs-', linewidth=2, markersize=8, label='L1(p)')

# Reference slopes
nx_ref = np.array([100, 800])
ax.loglog(nx_ref, l1_rho_list[0] * (nx_ref / 100)**(-1), 'k--', alpha=0.5, label='O(Δx) slope')
ax.loglog(nx_ref, l1_rho_list[0] * (nx_ref / 100)**(-2), 'k:', alpha=0.5, label='O(Δx²) slope')

ax.set_xlabel('Grid resolution (nx)')
ax.set_ylabel('L1 error')
ax.set_title('TensorFVM Sod Shock Tube: Convergence Study')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

png_path2 = os.path.join(outdir, "sod_convergence.png")
fig.savefig(png_path2, dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Error distribution
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

err_rho = np.abs(rho_num - rho_exact)
err_u = np.abs(u_num - u_exact)
err_p = np.abs(p_num - p_exact)

axes[0].plot(xc, err_rho, 'b-', linewidth=1)
axes[0].set_ylabel('|ρ_num - ρ_exact|')
axes[0].set_title('Pointwise Error Distribution (nx=800)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(xc, err_u, 'b-', linewidth=1)
axes[1].set_ylabel('|u_num - u_exact|')
axes[1].grid(True, alpha=0.3)

axes[2].plot(xc, err_p, 'b-', linewidth=1)
axes[2].set_ylabel('|p_num - p_exact|')
axes[2].set_xlabel('x')
axes[2].grid(True, alpha=0.3)

# Mark discontinuities
for ax in axes:
    ax.axvline(x=contact_pos, color='m', linestyle=':', alpha=0.5)
    ax.axvline(x=shock_pos, color='r', linestyle=':', alpha=0.5)

plt.tight_layout()
png_path3 = os.path.join(outdir, "sod_error_distribution.png")
fig.savefig(png_path3, dpi=150, bbox_inches='tight')
plt.close()

# Encode figures as base64 for the report
def encode_png(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

figures = {
    "sod_profile": encode_png(png_path),
    "sod_convergence": encode_png(png_path2),
    "sod_error": encode_png(png_path3),
}

# Save figure metadata
fig_meta = {
    "sod_profile": {"path": png_path, "caption": "Density, velocity, and pressure profiles at t=0.2. TensorFVM (nx=800, order=2) vs exact Riemann solution. Vertical lines mark wave positions."},
    "sod_convergence": {"path": png_path2, "caption": "L1 error convergence with grid refinement. Observed order ~0.84-0.96 for density, ~0.93-1.01 for pressure."},
    "sod_error": {"path": png_path3, "caption": "Pointwise error distribution. Largest errors near discontinuities (contact and shock), as expected for shock-capturing schemes."},
}

with open(os.path.join(outdir, "figures_meta.json"), "w") as f:
    json.dump(fig_meta, f, indent=2)

print("Figures generated:")
print(f"  {png_path}")
print(f"  {png_path2}")
print(f"  {png_path3}")

# Print key metrics for the report
print("\n=== Key Metrics (nx=800, order=2) ===")
result_800 = [r for r in tfvm_data["results"] if r["nx"] == 800][0]
print(f"L1(ρ) = {result_800['l1_rho']:.6e}")
print(f"L1(u) = {result_800['l1_u']:.6e}")
print(f"L1(p) = {result_800['l1_p']:.6e}")
print(f"L∞(ρ) = {result_800['linf_rho']:.6e}")
print(f"L∞(u) = {result_800['linf_u']:.6e}")
print(f"L∞(p) = {result_800['linf_p']:.6e}")

# Wave position errors
positions = result_800["positions"]
exact_pos = result_800["exact_positions"]
print(f"\nWave position errors:")
print(f"  Shock:    num={positions['shock']:.6f}, exact={exact_pos['shock']:.6f}, err={abs(positions['shock']-exact_pos['shock'])/exact_pos['shock']*100:.3f}%")
print(f"  Contact:  num={positions['contact']:.6f}, exact={exact_pos['contact']:.6f}, err={abs(positions['contact']-exact_pos['contact'])/exact_pos['contact']*100:.3f}%")
print(f"  Rare head: num={positions['rarefaction_head']:.6f}, exact={exact_pos['rarefaction_head']:.6f}")
print(f"  Rare tail: num={positions['rarefaction_tail']:.6f}, exact={exact_pos['rarefaction_tail']:.6f}")

# Star region errors
star = result_800["star_region"]
exact_star = result_800["exact_star_region"]
print(f"\nStar region errors:")
print(f"  p_star_l: num={star['p_star_l']:.6f}, exact={exact_star['p_star']:.6f}, err={abs(star['p_star_l']-exact_star['p_star'])/exact_star['p_star']*100:.3f}%")
print(f"  p_star_r: num={star['p_star_r']:.6f}, exact={exact_star['p_star']:.6f}, err={abs(star['p_star_r']-exact_star['p_star'])/exact_star['p_star']*100:.3f}%")
print(f"  rho_star_l: num={star['rho_star_l']:.6f}, exact={exact_star['rho_star_l']:.6f}, err={abs(star['rho_star_l']-exact_star['rho_star_l'])/exact_star['rho_star_l']*100:.3f}%")
print(f"  rho_star_r: num={star['rho_star_r']:.6f}, exact={exact_star['rho_star_r']:.6f}, err={abs(star['rho_star_r']-exact_star['rho_star_r'])/exact_star['rho_star_r']*100:.3f}%")
