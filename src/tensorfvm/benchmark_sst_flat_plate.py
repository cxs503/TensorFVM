"""SST-2003 ZPG flat plate benchmark bound to anchor FEAT-TURB-SST-GATE.

Anchor: FEAT-TURB-SST-GATE

**STATUS: HONEST FAIL** - The flat plate benchmark requires a proper 2-D
boundary-layer solver with streamwise development. The current implementation
uses the SST channel flow solver as a proxy, which is not physically equivalent
to a flat plate boundary layer.

The channel flow at Re_tau=180 has Cf ≈ 0.0084, while the Schultz-Grunow
correlation for a flat plate at Re_L=1e7 gives Cf ≈ 0.0024. The 3.5x difference
is due to fundamentally different flow physics, not a bug in the SST closure.

This file documents the limitation and provides the infrastructure for when
a proper flat plate solver is implemented.

Reference
---------
NASA TMR SST-2003: https://tmbwg.github.io/turbmodels/sst.html#sst-2003
Schultz-Grunow correlation: Cf = 0.370 / (log10(Re_x))^2.584
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ANCHOR_SLUG = "FEAT-TURB-SST-GATE"
RE_L = 1e7  # Plate Reynolds number
RESOLUTIONS = [64, 128, 256]

# Pre-declared acceptance thresholds
TOLERANCE_CF_RMS = 0.10    # 10 % RMS error in Cf distribution
TOLERANCE_LOG_SLOPE = 0.05  # 5 % error in log-law slope
MAX_Y_PLUS = 1.0           # First cell y+ must be < 1


def schultz_grunow_cf(re_x):
    """Schultz-Grunow empirical correlation for local skin friction.

    Cf = 0.370 / (log10(Re_x))^2.584

    Valid for turbulent boundary layer from leading edge, Re_x > 1e6.
    """
    return 0.370 / (np.log10(re_x) ** 2.584)


def von_karman_cf(re_x):
    """Von Karman-Schlichting correlation for average skin friction.

    Cf_avg = 0.074 / Re_L^0.2
    """
    return 0.074 / (re_x ** 0.2)


def solve_flat_plate_sst(n, re_l=RE_L, stretch=4.0):
    """Attempt to solve the 1-D turbulent boundary layer on a ZPG flat plate.

    **LIMITATION**: This uses the SST channel flow solver as a proxy.
    A proper flat plate solver would integrate the boundary-layer equations
    with streamwise development, not the channel flow equations.

    Returns a result dict with converged=False to indicate the proxy limitation.
    """
    from tensorfvm.rans_validation.channel import solve_channel

    # The channel flow at Re_tau=180 is used as a proxy
    re_tau_equiv = 180.0

    result = solve_channel('sst', n, re_tau=re_tau_equiv, max_iterations=40000)

    if not result['converged']:
        return dict(
            n=n, re_l=re_l, converged=False,
            verdict='FAIL', reason='solver did not converge',
        )

    # Compute reference Cf
    cf_ref = schultz_grunow_cf(re_l)
    cf_channel = result['Cf']

    # The channel Cf is not comparable to the flat plate Cf
    # Document the discrepancy
    cf_error = abs(cf_channel - cf_ref) / cf_ref

    # Compute u+ profile
    y_plus = result['y_plus']
    u_plus = result['u_plus']

    # Check log-law slope
    log_mask = (y_plus >= 30) & (y_plus <= 0.2 * re_tau_equiv)
    if np.count_nonzero(log_mask) >= 3:
        y_log = y_plus[log_mask]
        u_log = u_plus[log_mask]
        coeffs = np.polyfit(np.log(y_log), u_log, 1)
        kappa_fitted = 1.0 / coeffs[0]
        b_fitted = coeffs[1]
        log_slope_error = abs(kappa_fitted - 0.41) / 0.41
    else:
        kappa_fitted = np.nan
        b_fitted = np.nan
        log_slope_error = np.nan

    return dict(
        n=n,
        re_l=re_l,
        re_tau_equiv=re_tau_equiv,
        converged=True,
        iterations=result['iterations'],
        first_cell_y_plus=result['first_cell_y_plus'],
        cf_channel=cf_channel,
        cf_reference=cf_ref,
        cf_relative_error=cf_error,
        bulk_u_plus=result['bulk_u_plus'],
        kappa_fitted=kappa_fitted,
        b_fitted=b_fitted,
        log_slope_error=log_slope_error,
        y_plus=y_plus.tolist(),
        u_plus=u_plus.tolist(),
        # Honest verdict: FAIL due to proxy limitation
        overall_verdict='FAIL',
        fail_reason='Channel flow proxy not physically equivalent to flat plate boundary layer; '
                    f'Cf_channel={cf_channel:.6f} vs Cf_flat_plate={cf_ref:.6f} (ratio={cf_error:.1%})',
    )


def make_figures(cases, out_dir):
    """Generate comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = []

    # Figure 1: Cf comparison (documenting the proxy limitation)
    fig, ax = plt.subplots(figsize=(8, 5))

    re_x_range = np.logspace(6, 7, 100)
    ax.loglog(re_x_range, schultz_grunow_cf(re_x_range), 'k-', linewidth=2, label='Schultz-Grunow (flat plate)')

    colors = {64: 'blue', 128: 'green', 256: 'red'}
    markers = {64: 'o', 128: 's', 256: '^'}

    for case in cases:
        if not case.get('converged'):
            continue
        c = colors.get(case['n'], 'gray')
        m = markers.get(case['n'], 'x')
        ax.plot(case['re_l'], case['cf_channel'], c=c, marker=m, markersize=10,
                linestyle='none', label=f"SST channel n={case['n']} (proxy)")

    ax.axhline(y=schultz_grunow_cf(RE_L), color='red', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Flat plate Cf at Re_L={RE_L:.0e}')

    ax.set_xlabel('Re_x')
    ax.set_ylabel('Cf')
    ax.set_title('Cf Comparison: SST Channel (proxy) vs Flat Plate Correlation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    path = out_dir / 'fig_cf_comparison.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    # Figure 2: Log-law comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    y_log = np.logspace(1, 3, 100)
    ax.semilogx(y_log, np.log(y_log) / 0.41 + 5.0, 'k-', linewidth=2, label='log law (kappa=0.41, B=5.0)')

    for case in cases:
        if not case.get('converged'):
            continue
        c = colors.get(case['n'], 'gray')
        m = markers.get(case['n'], 'x')
        ax.semilogx(case['y_plus'], case['u_plus'], c=c, marker=m, markersize=3,
                    linestyle='-', linewidth=1, label=f"SST channel n={case['n']}")

    ax.set_xlabel('y+')
    ax.set_ylabel('U+')
    ax.set_title('Log-Law: SST Channel Flow (proxy for flat plate)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    path = out_dir / 'fig_log_law.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    return figures


def render_report(summary, figures):
    """Generate a markdown report."""
    lines = [
        "# CASE-FVM-SST-FLAT-PLATE: SST-2003 ZPG Flat Plate (HONEST FAIL)",
        "",
        f"**Anchor**: `{ANCHOR_SLUG}`",
        "",
        "**Verdict: FAIL** - The flat plate benchmark requires a proper 2-D boundary-layer",
        "solver with streamwise development. The current implementation uses the SST",
        "channel flow solver as a proxy, which is not physically equivalent.",
        "",
        "## Reference",
        "",
        "NASA TMR SST-2003: https://tmbwg.github.io/turbmodels/sst.html#sst-2003",
        "",
        "Schultz-Grunow correlation: Cf = 0.370 / (log10(Re_x))^2.584",
        "",
        "## Root Cause Diagnosis",
        "",
        "The channel flow at Re_tau=180 has Cf ≈ 0.0084, while the Schultz-Grunow",
        f"correlation for a flat plate at Re_L={RE_L:.0e} gives Cf ≈ {schultz_grunow_cf(RE_L):.6f}.",
        f"The ratio is {0.0084/schultz_grunow_cf(RE_L):.1f}x, which is due to fundamentally",
        "different flow physics:",
        "",
        "- **Channel flow**: Pressure-driven flow between two walls, constant shear stress",
        "  distribution, Re_tau based on friction velocity and half-height.",
        "- **Flat plate**: Zero-pressure-gradient boundary layer, spatially developing,",
        "  Re_L based on free-stream velocity and plate length.",
        "",
        "The SST closure itself is correctly implemented (validated against DNS in the",
        "channel flow benchmark). The failure is in the benchmark design, not the closure.",
        "",
        "## Acceptance Criteria (pre-declared)",
        "",
        f"1. Local Cf RMS deviation from empirical correlation < {TOLERANCE_CF_RMS:.0%}",
        f"2. Log-law slope within {TOLERANCE_LOG_SLOPE:.0%} of kappa=0.41",
        f"3. First-cell y+ < {MAX_Y_PLUS}",
        "",
        "## Results",
        "",
        "| Resolution | Converged | y+ (first) | Cf (channel) | Cf (flat plate) | Ratio | Verdict |",
        "|-----------|-----------|------------|--------------|-----------------|-------|---------|",
    ]

    for case in summary['cases']:
        if case.get('converged'):
            ratio = case['cf_channel'] / case['cf_reference']
            lines.append(
                f"| n={case['n']} | Yes | {case['first_cell_y_plus']:.3f} | "
                f"{case['cf_channel']:.6f} | {case['cf_reference']:.6f} | "
                f"{ratio:.1f}x | FAIL (proxy) |"
            )
        else:
            lines.append(f"| n={case['n']} | No | - | - | - | - | FAIL |")

    lines.extend([
        "",
        "## Figures",
        "",
    ])

    for fig_path in figures:
        fig_name = Path(fig_path).name
        lines.append(f"![{fig_name}]({fig_name})")
        lines.append("")

    lines.extend([
        "## Path Forward",
        "",
        "To properly validate SST-2003 against the NASA TMR flat plate reference:",
        "",
        "1. Implement a 2-D boundary-layer solver with streamwise development",
        "2. Use the SST closure for the turbulence model",
        "3. Compare Cf(x) distribution along the plate",
        "4. Compare u+ profiles at multiple streamwise stations",
        "5. Perform grid convergence study with three resolutions",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m tensorfvm.benchmark_sst_flat_plate --output results/sst-flat-plate",
        "```",
        "",
    ])

    return "\n".join(lines)


def run_benchmark(directory, resolutions=None):
    """Run the SST flat plate benchmark."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if resolutions is None:
        resolutions = RESOLUTIONS

    cases = []

    for n in resolutions:
        print(f"Running SST flat plate n={n}...")
        case = solve_flat_plate_sst(n)
        cases.append(case)
        print(f"  -> {case.get('overall_verdict', 'UNKNOWN')}: {case.get('fail_reason', '')}")

    # Write benchmark.json
    summary = {
        "anchor_slug": ANCHOR_SLUG,
        "benchmark": "sst-flat-plate-ReL1e7",
        "reference": "Schultz-Grunow empirical correlation",
        "re_l": RE_L,
        "tolerance_cf_rms": TOLERANCE_CF_RMS,
        "tolerance_log_slope": TOLERANCE_LOG_SLOPE,
        "max_y_plus": MAX_Y_PLUS,
        "cases": cases,
        "all_passed": False,  # Honest: not all passed
        "verdict": "FAIL",
        "fail_reason": "Channel flow proxy not physically equivalent to flat plate boundary layer",
    }

    (directory / "benchmark.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Write profiles.csv
    with (directory / "profiles.csv").open("w") as f:
        f.write("case,x_over_l,Cf_channel,Cf_flat_plate,y_plus,u_plus\n")
        for case in cases:
            if case.get('converged'):
                for yp, up in zip(case['y_plus'], case['u_plus']):
                    f.write(f"n={case['n']},1.0,{case['cf_channel']},{case['cf_reference']},{yp},{up}\n")

    # Write metrics.csv
    with (directory / "metrics.csv").open("w") as f:
        f.write("n,converged,first_cell_y_plus,cf_channel,cf_reference,cf_ratio,kappa_fitted,log_slope_error,verdict\n")
        for case in cases:
            ratio = case.get('cf_channel', 0) / case.get('cf_reference', 1) if case.get('cf_reference', 0) > 0 else 0
            f.write(f"{case['n']},{case.get('converged',False)},{case.get('first_cell_y_plus','')},{case.get('cf_channel','')},{case.get('cf_reference','')},{ratio:.2f},{case.get('kappa_fitted','')},{case.get('log_slope_error','')},{case.get('overall_verdict','')}\n")

    # Generate figures
    figures = make_figures(cases, directory)

    # Write report
    report = render_report(summary, figures)
    (directory / "REPORT.md").write_text(report, encoding="utf-8")

    # Write manifest
    manifest = {
        "anchor_slug": ANCHOR_SLUG,
        "benchmark": "sst-flat-plate-ReL1e7",
        "artifacts": ["benchmark.json", "profiles.csv", "metrics.csv", "REPORT.md"] + [Path(f).name for f in figures],
        "sha256": {},
    }
    for name in manifest["artifacts"]:
        path = directory / name
        if path.exists():
            manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="SST-2003 flat plate benchmark (HONEST FAIL)")
    parser.add_argument("--output", type=Path, default=Path("results/sst-flat-plate"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)

    summary = run_benchmark(args.output, args.resolutions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2  # Always return 2 (FAIL) for this benchmark


if __name__ == "__main__":
    raise SystemExit(main())
