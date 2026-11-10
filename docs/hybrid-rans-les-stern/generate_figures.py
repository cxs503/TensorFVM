"""Generate verification figures for hybrid RANS-LES stern flow report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorfvm.des_iddes_model import (
    HybridSSTDES, des97_length_scale, iddes_length_scale,
    iddes_shielding_function, C_DES_SST_1, BETA_STAR
)
from tensorfvm.stern_hybrid_verification import (
    SternFlowCase, MESH_COARSE, MESH_FINE,
    suboff_stern_separation_region, backward_facing_step_reference,
    synthesize_stern_flow_field, run_verification_suite,
    compare_rans_vs_des, grid_convergence_check, LENGTH_M,
)
from tensorfvm.suboff_geometry import LENGTH_M as SUBOFF_L

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ============================================================================
# Figure 1: SUBOFF stern geometry + separation region
# ============================================================================
def fig1_stern_geometry():
    stern = suboff_stern_separation_region(0.5, 1.0, 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(stern['x_m'], stern['r_m'], 'k-', linewidth=2, label='SUBOFF hull (upper)')
    ax.plot(stern['x_m'], -stern['r_m'], 'k-', linewidth=2, label='SUBOFF hull (lower)')
    
    # Mark separation region
    sep_mask = stern['xi'] >= 0.74
    ax.fill_between(stern['x_m'][sep_mask], -stern['r_m'][sep_mask],
                    stern['r_m'][sep_mask], alpha=0.15, color='red',
                    label='Separation-prone region')
    
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('r [m]', fontsize=12)
    ax.set_title('SUBOFF Bare Hull Stern Geometry', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(stern['x_m'][0], stern['x_m'][-1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig1_stern_geometry.png'), dpi=150)
    plt.close(fig)
    print("Fig 1 saved")


# ============================================================================
# Figure 2: DES/IDDES length scale ratio vs wall distance
# ============================================================================
def fig2_length_scales():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameters representative of stern flow
    k = 0.01  # m^2/s^2
    omega = 100.0  # 1/s
    nu = 1e-6
    nu_t = 0.01
    G = np.zeros((1, 3, 3))
    G[0, 0, 1] = 50.0
    
    d = np.logspace(-5, -1, 200)
    cell_size = np.full_like(d, 0.005)
    
    # DES97
    des97 = des97_length_scale(k * np.ones_like(d), omega * np.ones_like(d),
                                d, cell_size, nu=nu)
    
    # IDDES
    iddes = iddes_length_scale(k * np.ones_like(d), omega * np.ones_like(d),
                                d, cell_size, nu_t * np.ones_like(d), nu,
                                np.broadcast_to(G, (len(d), 3, 3)))
    
    # Left: Length scales
    ax = axes[0]
    ax.loglog(d * 1000, des97['L_RANS'] * 1000, 'b-', linewidth=2, label='L_RANS (SST)')
    ax.loglog(d * 1000, des97['L_LES'] * 1000, 'r--', linewidth=2, label='L_LES (C_DES·Δ)')
    ax.loglog(d * 1000, des97['L_DES'] * 1000, 'g-', linewidth=2, label='L_DES97')
    ax.loglog(d * 1000, iddes['L_IDDES'] * 1000, 'm-', linewidth=2, label='L_IDDES')
    ax.set_xlabel('Wall distance d [mm]', fontsize=11)
    ax.set_ylabel('Length scale [mm]', fontsize=11)
    ax.set_title('DES/IDDES Length Scales', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Right: Shielding function
    ax = axes[1]
    ax.semilogx(d * 1000, iddes['f_d'], 'k-', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(d * 1000, 0, iddes['f_d'], where=iddes['f_d'] < 0.5,
                    alpha=0.2, color='blue', label='RANS mode (f_d < 0.5)')
    ax.fill_between(d * 1000, 0, iddes['f_d'], where=iddes['f_d'] >= 0.5,
                    alpha=0.2, color='red', label='LES mode (f_d ≥ 0.5)')
    ax.set_xlabel('Wall distance d [mm]', fontsize=11)
    ax.set_ylabel('Shielding function f_d', fontsize=11)
    ax.set_title('IDDES Shielding Function', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig2_length_scales.png'), dpi=150)
    plt.close(fig)
    print("Fig 2 saved")


# ============================================================================
# Figure 3: Velocity profiles - RANS vs DES vs IDDES
# ============================================================================
def fig3_velocity_profiles():
    case = SternFlowCase(name='SUBOFF_Re1.9M', U_m_s=1.2, L_m=SUBOFF_L,
                        nu_m2_s=1e-6, rho_kg_m3=1025.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model in enumerate(['rans_sst', 'des97', 'iddes']):
        flow = synthesize_stern_flow_field(case, MESH_FINE, model, seed=42)
        ax = axes[idx]
        
        # Plot velocity profiles at selected stations
        stations = [5, 15, 25, 35, 45]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(stations)))
        
        for s, c in zip(stations, colors):
            if s < flow['u_profiles'].shape[0]:
                ax.plot(flow['u_profiles'][s, :], flow['y_wall'] * 1000,
                       color=c, linewidth=1.5,
                       label=f'x/L={flow["x_m"][s]/SUBOFF_L:.2f}')
        
        ax.set_xlabel('u [m/s]', fontsize=11)
        ax.set_ylabel('y [mm]', fontsize=11)
        ax.set_title(f'{model.upper().replace("_", " ")}', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_xlim(-0.3, 1.5)
    
    fig.suptitle('Stern Velocity Profiles: RANS vs DES vs IDDES', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig3_velocity_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Fig 3 saved")


# ============================================================================
# Figure 4: Reynolds stress comparison
# ============================================================================
def fig4_reynolds_stress():
    case = SternFlowCase(name='SUBOFF_Re1.9M', U_m_s=1.2, L_m=SUBOFF_L,
                        nu_m2_s=1e-6, rho_kg_m3=1025.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model in enumerate(['rans_sst', 'des97', 'iddes']):
        flow = synthesize_stern_flow_field(case, MESH_FINE, model, seed=42)
        ax = axes[idx]
        
        # Plot u_rms profiles at mid-stern
        mid = flow['u_profiles'].shape[0] // 2
        ax.plot(flow['u_rms'][mid, :], flow['y_wall'] * 1000,
               'b-', linewidth=2, label="u'")
        ax.plot(flow['v_rms'][mid, :], flow['y_wall'] * 1000,
               'r--', linewidth=2, label="v'")
        
        ax.set_xlabel('Turbulence intensity [m/s]', fontsize=11)
        ax.set_ylabel('y [mm]', fontsize=11)
        ax.set_title(f'{model.upper().replace("_", " ")}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Reynolds Stress Profiles at Mid-Stern (x/L ≈ 0.86)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig4_reynolds_stress.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Fig 4 saved")


# ============================================================================
# Figure 5: Reynolds stress spectra
# ============================================================================
def fig5_spectra():
    case = SternFlowCase(name='SUBOFF_Re1.9M', U_m_s=1.2, L_m=SUBOFF_L,
                        nu_m2_s=1e-6, rho_kg_m3=1025.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model, color, ls in [('rans_sst', 'blue', '-'), ('des97', 'red', '--'), ('iddes', 'green', '-.')]:
        flow = synthesize_stern_flow_field(case, MESH_FINE, model, seed=42)
        
        # Compute spectrum from time series
        u_prime = flow['u_prime']
        v_prime = flow['v_prime']
        dt = flow['t'][1] - flow['t'][0]
        n_fft = 1024
        window = np.hanning(n_fft)
        
        spectrum = np.zeros(n_fft // 2 + 1)
        n_seg = len(u_prime) // n_fft
        for seg in range(n_seg):
            s = seg * n_fft
            U_f = np.fft.rfft(u_prime[s:s+n_fft] * window)
            V_f = np.fft.rfft(v_prime[s:s+n_fft] * window)
            spectrum += np.real(U_f * np.conj(V_f))
        spectrum /= max(n_seg, 1)
        
        freqs = np.fft.rfftfreq(n_fft, d=dt)
        
        ax.loglog(freqs[1:], np.abs(spectrum[1:]), color=color, linestyle=ls,
                 linewidth=1.5, label=model.upper())
    
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_ylabel('|E_uv(f)|', fontsize=12)
    ax.set_title('Reynolds Stress Cross-Spectrum at Mid-Stern Probe', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig5_spectra.png'), dpi=150)
    plt.close(fig)
    print("Fig 5 saved")


# ============================================================================
# Figure 6: Grid convergence + metrics summary
# ============================================================================
def fig6_grid_convergence():
    results = run_verification_suite()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: L_sep comparison across models and meshes
    ax = axes[0]
    models = ['rans_sst', 'des97', 'iddes']
    case_name = 'SUBOFF_Re1.9M'
    x_pos = np.arange(len(models))
    width = 0.35
    
    coarse_vals = [results[f'{case_name}_coarse_{m}'].separation.L_sep_m * 1000 for m in models]
    fine_vals = [results[f'{case_name}_fine_{m}'].separation.L_sep_m * 1000 for m in models]
    
    bars1 = ax.bar(x_pos - width/2, coarse_vals, width, label='Coarse mesh', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, fine_vals, width, label='Fine mesh', color='coral')
    
    ax.set_xlabel('Turbulence Model', fontsize=11)
    ax.set_ylabel('L_sep [mm]', fontsize=11)
    ax.set_title('Separation Bubble Length', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['RANS SST', 'DES97', 'IDDES'], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Reynolds stress deviation (DES vs RANS)
    ax = axes[1]
    deviations = []
    for m in ['des97', 'iddes']:
        rans = results[f'{case_name}_fine_rans_sst']
        des = results[f'{case_name}_fine_{m}']
        comp = compare_rans_vs_des(rans, des)
        deviations.append(comp['reynolds_stress_deviation'] * 100)
    
    bars = ax.bar(['DES97', 'IDDES'], deviations, color=['red', 'green'], alpha=0.7)
    ax.axhline(30, color='gray', linestyle='--', linewidth=1, label='30% threshold')
    ax.set_ylabel('Reynolds Stress Deviation from RANS [%]', fontsize=11)
    ax.set_title('DES/IDDES Improvement over RANS', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig6_grid_convergence.png'), dpi=150)
    plt.close(fig)
    print("Fig 6 saved")


# ============================================================================
# Figure 7: Verification metrics summary table
# ============================================================================
def fig7_summary_table():
    results = run_verification_suite()
    case_name = 'SUBOFF_Re1.9M'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Build table data
    col_labels = ['Model', 'Mesh', 'L_sep [mm]', 'f_peak [Hz]', 'St', 
                  "u'v' [m²/s²]", 'CPU/RANS']
    table_data = []
    
    for model in ['rans_sst', 'des97', 'iddes']:
        for mesh_name in ['coarse', 'fine']:
            r = results[f'{case_name}_{mesh_name}_{model}']
            table_data.append([
                model.upper().replace('_', ' '),
                mesh_name.capitalize(),
                f'{r.separation.L_sep_m*1000:.2f}',
                f'{r.spectral.f_peak_Hz:.1f}',
                f'{r.spectral.St_peak:.4f}',
                f'{r.spectral.reynolds_stress_rms:.6f}',
                f'{r.cpu_ratio_vs_rans:.1f}×',
            ])
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Color model groups
    colors = ['#E2EFDA', '#FCE4D6', '#D6DCE4']
    for i in range(len(table_data)):
        row_color = colors[i // 2]
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(row_color)
    
    ax.set_title('Verification Metrics Summary (SUBOFF Re≈1.83M)', fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig7_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Fig 7 saved")


if __name__ == '__main__':
    fig1_stern_geometry()
    fig2_length_scales()
    fig3_velocity_profiles()
    fig4_reynolds_stress()
    fig5_spectra()
    fig6_grid_convergence()
    fig7_summary_table()
    print("\nAll figures generated successfully.")
