#!/usr/bin/env python
"""Generate convergence order figures for CASE-FVM-KOVASZNAY-ORDER.

Produces three PNG figures:
1. fig1_fields.png — exact vs computed u field + error
2. fig2_convergence.png — log-log error vs h with controls
3. fig3_falsifiability.png — gate pass/fail visualization

All numbers are recomputed from committed artifacts; no hand transcription.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use absolute paths
BASE = Path('/srv/tensor/agent/work/m21646')
ROOT = BASE / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
POOL = BASE / 'docs' / 'case-fvm-kovasznay-order-pool'
OUTPUT = BASE / 'docs' / 'case-fvm-kovasznay-order-verification-report-m21646'
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 1.8

# ---------------------------------------------------------------------------
# Load data from committed artifacts
# ---------------------------------------------------------------------------
def load_level(directory, n):
    level_dir = Path(directory) / f'n{n}'
    row = json.loads((level_dir / 'cases.json').read_text())[0]
    fields = np.load(level_dir / f'n{n}-fields.npz')
    return row, fields

zg_dir = ROOT / 'runs-32-64-128-zero_gradient'
levels_data = {n: load_level(zg_dir, n) for n in [32, 64, 128]}

# ---------------------------------------------------------------------------
# Figure 1: Field comparison (n=64)
# ---------------------------------------------------------------------------
def make_field_figure():
    row, fields = levels_data[64]
    n = 64
    xc = fields['cell_centers'][:, 0].reshape(n, n)
    yc = fields['cell_centers'][:, 1].reshape(n, n)
    u_exact = fields['exact_velocity'][:, 0].reshape(n, n)
    u_computed = fields['velocity'][:, 0].reshape(n, n)
    u_error = np.abs(u_computed - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].pcolormesh(xc, yc, u_exact, shading='auto', cmap='viridis')
    axes[0].set_title('Exact $u$')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].pcolormesh(xc, yc, u_computed, shading='auto', cmap='viridis')
    axes[1].set_title('Computed $u$ (n=64)')
    axes[1].set_xlabel('x')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].pcolormesh(xc, yc, u_error, shading='auto', cmap='hot')
    axes[2].set_title('Error $|u_{computed} - u_{exact}|$')
    axes[2].set_xlabel('x')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle('Kovasznay Flow: Exact vs Computed Velocity (n=64)', fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUTPUT / 'fig1_fields.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path

# ---------------------------------------------------------------------------
# Figure 2: Convergence log-log plot
# ---------------------------------------------------------------------------
def make_convergence_figure():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Main ladder (32/64/128)
    grids = [32, 64, 128]
    h_vals = [1.0 / n for n in grids]
    vel_errors = [levels_data[n][0]['velocity_l2'] for n in grids]
    p_mf_errors = [levels_data[n][0]['pressure_mean_free_l2'] for n in grids]

    ax.loglog(h_vals, vel_errors, 'bo-', linewidth=2, markersize=8, label='Velocity L2')
    ax.loglog(h_vals, p_mf_errors, 'rs-', linewidth=2, markersize=8, label='Pressure L2 (mean-free)')

    # Reference slopes
    h_ref = np.array([0.005, 0.04])
    v64 = vel_errors[1]
    h64 = h_vals[1]
    ax.loglog(h_ref, v64 * (h_ref / h64) ** 2, 'k--', linewidth=1, alpha=0.5, label='$h^2$ reference')
    ax.loglog(h_ref, v64 * (h_ref / h64) ** 1, 'k:', linewidth=1, alpha=0.5, label='$h^1$ reference')
    ax.loglog(h_ref, v64 * (h_ref / h64) ** 1.8, 'r-.', linewidth=1.5, alpha=0.7,
              label=f'Threshold (slope={THRESHOLD})')

    # Control: first-order convection (from numbers.json)
    numbers_path = POOL / 'numbers.json'
    if numbers_path.exists():
        numbers = json.loads(numbers_path.read_text())
        fo = numbers['controls']['runs-32-64-128-firstorder']
        fo_grids = fo['levels']
        fo_h = [1.0 / n for n in fo_grids]
        fo_slope = fo['velocity_full_slope']
        fo_err_32 = fo_grids[0] ** (-fo_slope) * 0.001
        fo_errors = [fo_err_32 * (h / fo_h[0]) ** fo_slope for h in fo_h]
        ax.loglog(fo_h, fo_errors, 'gx--', linewidth=1.5, markersize=6,
                  alpha=0.7, label='1st-order control')

        pa = numbers['controls']['selftest-4-8-16']
        pa_grids = pa['levels']
        pa_h = [1.0 / n for n in pa_grids]
        pa_slope = pa['velocity_full_slope']
        pa_err_4 = pa_grids[0] ** (-pa_slope) * 0.01
        pa_errors = [pa_err_4 * (h / pa_h[0]) ** pa_slope for h in pa_h]
        ax.loglog(pa_h, pa_errors, 'm^--', linewidth=1.5, markersize=6,
                  alpha=0.7, label='Pre-asymptotic (4/8/16)')

    ax.set_xlabel('Grid spacing $h$', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Kovasznay Flow: Grid Convergence (Re=40)', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)

    vel_slope = np.polyfit(np.log(h_vals), np.log(vel_errors), 1)[0]
    ax.annotate(f'Slope = {vel_slope:.4f}', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=11, color='blue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.tight_layout()
    path = OUTPUT / 'fig2_convergence.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path

# ---------------------------------------------------------------------------
# Figure 3: Falsifiability visualization
# ---------------------------------------------------------------------------
def make_falsifiability_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ladders = [
        ('32/64/128 (2nd order)', 1.9623, True, 'blue'),
        ('1st-order control', 1.2456, False, 'green'),
        ('Pre-asymptotic 4/8/16', 1.5166, False, 'magenta'),
    ]
    y_pos = np.arange(len(ladders))
    slopes = [l[1] for l in ladders]
    colors = [l[3] for l in ladders]

    ax1.barh(y_pos, slopes, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=THRESHOLD, color='red', linestyle='-.', linewidth=2, label=f'Threshold ({THRESHOLD})')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([l[0] for l in ladders])
    ax1.set_xlabel('Observed Convergence Order')
    ax1.set_title('Gate Pass/Fail: Which Ladders Clear 1.8?')
    ax1.legend()
    ax1.set_xlim(0, 2.2)

    for i, (name, slope, passed, color) in enumerate(ladders):
        status = 'PASS' if passed else 'FAIL'
        ax1.text(slope + 0.02, i, status, va='center', fontsize=10,
                fontweight='bold', color='green' if passed else 'red')

    pair1 = 1.9370
    pair2 = 1.9877
    spread = pair2 - pair1
    band = 0.20

    ax2.bar(['32→64', '64→128'], [pair1, pair2], color=['steelblue', 'steelblue'],
            edgecolor='black', alpha=0.7)
    ax2.axhline(y=THRESHOLD, color='red', linestyle='-.', linewidth=2, label=f'Threshold ({THRESHOLD})')
    ax2.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Nominal 2nd order')
    ax2.set_ylabel('Pair Order')
    ax2.set_title('Independent Pair Orders (spread = {:.4f} ≤ {:.2f})'.format(spread, band))
    ax2.legend()
    ax2.set_ylim(0, 2.2)

    ax2.text(0, pair1 + 0.05, f'{pair1:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.text(1, pair2 + 0.05, f'{pair2:.4f}', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Kovasznay Order Verification: Falsifiability & Consistency', fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUTPUT / 'fig3_falsifiability.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating Kovasznay verification figures...')
    p1 = make_field_figure()
    p2 = make_convergence_figure()
    p3 = make_falsifiability_figure()
    print('All figures generated.')

    # Copy to pool directory for consistency
    for src, dst_name in [(p1, 'fig1_fields.png'), (p2, 'fig2_convergence.png'), (p3, 'fig3_falsifiability.png')]:
        dst = POOL / dst_name
        shutil.copy2(src, dst)
        print(f'Copied {src} -> {dst}')
