"""Generate figures for the wave tank metrology report."""
import sys
sys.path.insert(0, '/root/tp-agent/work/m9065/src')

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorfvm.wave_tank_td import SigmaWaveTank, TankConfig
from tensorfvm.wave_tank.theory import dispersion_k, stokes2_elevation
from tensorfvm.wave_tank_td.wave_target import fit_harmonics

# ============================================================================
# Figure 1: Wave propagation and probe signals
# ============================================================================
def fig_wave_propagation():
    cfg = TankConfig(length=10.0, depth=1.0, nx=120, ns=20,
                     relax_waves=1.5, damp_waves=2.0, damp_strength=5.0)
    tank = SigmaWaveTank(cfg)
    tank.set_wave(0.02, 1.2)  # H=0.04m, T=1.2s
    
    dt = 0.005
    T = 1.2
    n_steps = int(4 * T / dt)
    
    probe_x = [2.5, 4.0, 5.5, 7.0]
    etas = {x: [] for x in probe_x}
    times = []
    snapshots = []
    
    for i in range(n_steps):
        tank.step(dt)
        if i % 4 == 0:
            times.append(tank.t)
            for x in probe_x:
                etas[x].append(float(tank.eta_at(x)))
        if i % 40 == 0:
            snapshots.append((tank.t, tank.eta.copy()))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: surface snapshots
    ax = axes[0]
    x = tank.x
    for t_snap, eta_snap in snapshots[::2]:
        ax.plot(x, eta_snap, label=f't={t_snap:.1f}s', alpha=0.7)
    ax.axvspan(0, tank.x_relax if hasattr(tank, 'x_relax') else 2.0, 
               alpha=0.1, color='red', label='Relaxation zone')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('η (m)')
    ax.set_title('Free Surface Evolution (H=0.04m, T=1.2s, h=1.0m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Bottom: probe time series
    ax = axes[1]
    times_arr = np.array(times)
    for x in probe_x:
        ax.plot(times_arr, np.array(etas[x]), label=f'x={x}m')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('η (m)')
    ax.set_title('Wave Probe Time Series')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m9065/docs/fig1_wave_propagation.png', dpi=150)
    plt.close()
    print("Figure 1 saved: wave propagation")

# ============================================================================
# Figure 2: Harmonic analysis and Stokes II verification
# ============================================================================
def fig_harmonic_analysis():
    cfg = TankConfig(length=10.0, depth=1.0, nx=120, ns=20,
                     relax_waves=1.5, damp_waves=2.0, damp_strength=5.0)
    tank = SigmaWaveTank(cfg)
    
    # Stokes II wave with ka ≈ 0.05
    omega = 2.0 * math.pi / 1.2
    k = dispersion_k(omega, 1.0)
    a = 0.05 / k  # ka = 0.05
    tank.set_wave(a, 1.2)
    
    dt = 0.005
    T = 1.2
    n_steps = int(6 * T / dt)
    
    probe_x = 4.0
    etas = []
    times = []
    
    for i in range(n_steps):
        tank.step(dt)
        if i % 2 == 0:
            times.append(tank.t)
            etas.append(float(tank.eta_at(probe_x)))
    
    times_arr = np.array(times)
    eta_arr = np.array(etas)
    
    # Harmonic fit
    mask = times_arr > 2 * T
    result = fit_harmonics(times_arr[mask], eta_arr[mask], omega, n_harm=2)
    
    # Theoretical Stokes II elevation
    x_probe = np.array([probe_x])
    t_fine = np.linspace(2*T, times_arr[-1], 500)
    eta_theory = stokes2_elevation(x_probe, t_fine, a, omega, k, 1.0, order2=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: time series with fit
    ax = axes[0]
    ax.plot(times_arr, eta_arr, 'b.', markersize=2, label='CFD probe')
    ax.plot(t_fine, eta_theory.flatten(), 'r-', linewidth=1.5, label='Stokes II theory')
    
    # Reconstruct fit
    t_fit = times_arr[mask]
    eta_fit = (result["mean"] + 
               result["amp"][0] * np.cos(omega * t_fit - result["phase"][0]) +
               result["amp"][1] * np.cos(2 * omega * t_fit - result["phase"][1]))
    ax.plot(t_fit, eta_fit, 'g--', linewidth=1, label='Harmonic fit (2 harmonics)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('η (m)')
    ax.set_title(f'Harmonic Analysis (ka={k*a:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom: spectrum
    ax = axes[1]
    eta_centered = eta_arr[mask] - eta_arr[mask].mean()
    N = len(eta_centered)
    dt_sample = times_arr[1] - times_arr[0]
    eta_fft = np.fft.rfft(eta_centered)
    psd = np.abs(eta_fft) ** 2 * dt_sample / N
    freqs = np.fft.rfftfreq(N, d=dt_sample)
    
    ax.semilogy(freqs, psd, 'b-')
    ax.axvline(1/T, color='r', linestyle='--', label=f'f₁=1/T={1/T:.3f} Hz')
    ax.axvline(2/T, color='g', linestyle='--', label=f'f₂=2/T={2/T:.3f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3/T)
    
    # Add text box with results
    a2_over_a1 = result["amp"][1] / result["amp"][0] if result["amp"][0] > 1e-10 else 0.0
    theory_a2_a1 = (k * a / 4.0) * (3 - math.tanh(k*1.0)**2) / math.tanh(k*1.0)**3
    textstr = (f'a₁ = {result["amp"][0]:.5f} m\n'
               f'a₂ = {result["amp"][1]:.6f} m\n'
               f'a₂/a₁ = {a2_over_a1:.4f}\n'
               f'Theory a₂/a₁ = {theory_a2_a1:.4f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.7, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m9065/docs/fig2_harmonic_analysis.png', dpi=150)
    plt.close()
    print("Figure 2 saved: harmonic analysis")

# ============================================================================
# Figure 3: Volume conservation
# ============================================================================
def fig_volume_conservation():
    cfg = TankConfig(length=8.0, depth=1.0, nx=100, ns=16,
                     relax_waves=1.0, damp_waves=1.5, damp_strength=5.0)
    tank = SigmaWaveTank(cfg)
    tank.set_wave(0.02, 1.0)
    
    dt = 0.005
    T = 1.0
    n_steps = int(8 * T / dt)
    
    vol_history = []
    times = []
    
    for i in range(n_steps):
        tank.step(dt)
        if i % 10 == 0:
            times.append(tank.t)
            vol_history.append(tank.volume())
    
    vol0 = tank.volume0
    times_arr = np.array(times)
    vol_arr = np.array(vol_history)
    drift_percent = (vol_arr - vol0) / vol0 * 100
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_arr, drift_percent, 'b-')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Volume Drift (%)')
    ax.set_title(f'Volume Conservation (drift = {drift_percent[-1]:.4f}% after {times_arr[-1]:.1f}s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m9065/docs/fig3_volume_conservation.png', dpi=150)
    plt.close()
    print("Figure 3 saved: volume conservation")

# ============================================================================
# Figure 4: Grid convergence
# ============================================================================
def fig_grid_convergence():
    H_target = 0.04
    amp = H_target / 2.0
    T = 1.2
    
    resolutions = [(60, 12), (80, 16), (100, 20), (120, 24)]
    H_values = []
    dx_values = []
    
    for nx, ns in resolutions:
        cfg = TankConfig(length=8.0, depth=1.0, nx=nx, ns=ns,
                         relax_waves=1.0, damp_waves=1.5)
        tank = SigmaWaveTank(cfg)
        tank.set_wave(amp, T)
        
        dt = 0.005
        n_steps = int(3 * T / dt)
        probe_x = 4.0
        etas = []
        times = []
        for i in range(n_steps):
            tank.step(dt)
            if i % 4 == 0:
                times.append(tank.t)
                etas.append(float(tank.eta_at(probe_x)))
        
        times_arr = np.array(times)
        eta_arr = np.array(etas)
        mask = times_arr > 1.5 * T
        if mask.sum() > 10:
            result = fit_harmonics(times_arr[mask], eta_arr[mask], 
                                  tank.wave.omega, n_harm=1)
            H_meas = 2.0 * result["amp"][0]
        else:
            H_meas = 2.0 * float(np.abs(eta_arr).max())
        H_values.append(H_meas)
        dx_values.append(8.0 / nx)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dx_values, [abs(H - H_target) for H in H_values], 'bo-', 
              markersize=8, linewidth=2, label='Wave height error')
    
    # Reference slopes
    dx_ref = np.array(dx_values)
    ax.loglog(dx_ref, 0.01 * dx_ref / dx_ref[0], 'r--', alpha=0.5, label='O(h)')
    ax.loglog(dx_ref, 0.01 * (dx_ref / dx_ref[0])**2, 'g--', alpha=0.5, label='O(h²)')
    
    ax.set_xlabel('Grid spacing dx (m)')
    ax.set_ylabel('|H_measured - H_target| (m)')
    ax.set_title('Grid Convergence: Wave Height Error')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m9065/docs/fig4_grid_convergence.png', dpi=150)
    plt.close()
    print("Figure 4 saved: grid convergence")

# ============================================================================
# Run all
# ============================================================================
if __name__ == '__main__':
    print("Generating figures...")
    fig_wave_propagation()
    fig_harmonic_analysis()
    fig_volume_conservation()
    fig_grid_convergence()
    print("All figures generated in artifacts/")
