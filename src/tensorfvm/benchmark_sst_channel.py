"""SST-2003 turbulent channel flow benchmark vs DNS (Lee–Moser 2015, Re_tau≈180).

Anchor: FEAT-TURB-SST-GATE

This benchmark grades the SST-2003 closure (NASA TMR exact production variant)
against the Lee & Moser (2015) DNS at Re_tau = 182.088, which is the accepted
proxy for the KMM1987 Re_tau = 180 case.

Acceptance criteria (pre-declared):
-----------------------------------
1. Log-region RMS deviation (30 <= y+ <= 0.2*Re_tau) < 5 %
2. Skin-friction coefficient relative error < 10 %
3. Bulk velocity relative error < 5 %
4. Overall verdict = PASS only when all three are satisfied

Reference
---------
Lee, M. & Moser, R. D. 2015 *Direct numerical simulation of turbulent channel
flow up to Re_tau = 5200*. J. Fluid Mech. **774**, 395–415.
DOI: 10.1017/jfm.2015.268

NASA TMR SST-2003: https://tmbwg.github.io/turbmodels/sst.html#sst-2003

Artifacts
---------
The driver writes:
- ``benchmark.json``: machine-readable graded numbers per resolution
- ``profiles.csv``: u+(y+) for each case + DNS reference
- ``metrics.csv``: per-case grading metrics
- ``REPORT.md``: human-readable report
- ``fig_velocity_profiles.png``: mean velocity comparison (linear + log)
- ``fig_error_profiles.png``: relative error in log region
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification.kmm1987 import (
    load_kmm1987_reference,
    validate_kmm1987,
    TOLERANCE_LOG_REGION,
)

ANCHOR_SLUG = "FEAT-TURB-SST-GATE"
RE_TAU = 180.0
RESOLUTIONS = [64, 128, 256]

# Pre-declared acceptance thresholds
TOLERANCE_CF = 0.10       # 10 % relative error in Cf
TOLERANCE_BULK = 0.05     # 5 % relative error in bulk velocity


def run_case(n, re_tau=RE_TAU):
    """Run one SST channel solve and return the validation result."""
    t0 = time.monotonic()
    result = solve_channel('sst', n, re_tau=re_tau, max_iterations=40000)
    elapsed = time.monotonic() - t0

    if not result['converged']:
        return dict(
            n=n, re_tau=re_tau, converged=False, elapsed=round(elapsed, 2),
            verdict='FAIL', verdict_log_layer='FAIL', overall_verdict='FAIL',
            reason='solver did not converge',
        )

    dns = load_kmm1987_reference()
    validation = validate_kmm1987(
        result['y_plus'], result['u_plus'],
        result['Cf'], result['bulk_u_plus'],
        dns=dns,
    )

    # Additional grading: Cf and bulk velocity
    cf_pass = validation['cf']['relative_error'] < TOLERANCE_CF
    bulk_pass = validation['bulk_velocity']['relative_error'] < TOLERANCE_BULK

    return dict(
        n=n,
        re_tau=re_tau,
        converged=True,
        elapsed=round(elapsed, 2),
        iterations=result['iterations'],
        relative_local_residual=result['relative_local_residual'],
        first_cell_y_plus=result['first_cell_y_plus'],
        stretch=result['stretch'],
        Cf=result['Cf'],
        bulk_u_plus=result['bulk_u_plus'],
        verdict=validation['verdict'],
        verdict_log_layer=validation['verdict_log_layer'],
        overall_verdict=validation['overall_verdict'],
        log_region_rms=validation['log_region']['rms_relative'],
        log_region_max=validation['log_region']['max_relative'],
        log_layer_rms=validation['log_region_narrow']['rms_relative'],
        log_layer_max=validation['log_region_narrow']['max_relative'],
        cf_relative_error=validation['cf']['relative_error'],
        cf_passed=cf_pass,
        bulk_relative_error=validation['bulk_velocity']['relative_error'],
        bulk_passed=bulk_pass,
        y_plus=result['y_plus'].tolist(),
        u_plus=result['u_plus'].tolist(),
    )


def make_figures(cases, out_dir):
    """Generate comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator

    dns = load_kmm1987_reference()
    figures = []

    # Figure 1: Mean velocity profile comparison (linear + log axes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear plot
    ax1.plot(dns['y_plus'], dns['u_plus'], 'k-', linewidth=2, label='DNS (LM2015 Re_tau=182)')
    colors = {64: 'blue', 128: 'green', 256: 'red'}
    markers = {64: 'o', 128: 's', 256: '^'}

    for case in cases:
        if not case.get('converged'):
            continue
        c = colors.get(case['n'], 'gray')
        m = markers.get(case['n'], 'x')
        label = f"SST n={case['n']} (log_layer RMS={case['log_layer_rms']:.1%})"
        ax1.plot(case['y_plus'], case['u_plus'], c=c, marker=m, markersize=4,
                 linestyle='-', linewidth=1.5, label=label, markevery=max(1, len(case['y_plus'])//12))

    ax1.set_xlabel('y+')
    ax1.set_ylabel('U+')
    ax1.set_title('Mean Velocity Profile: SST-2003 vs DNS')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Semi-log plot
    ax2.semilogx(dns['y_plus'], dns['u_plus'], 'k-', linewidth=2, label='DNS')
    for case in cases:
        if not case.get('converged'):
            continue
        c = colors.get(case['n'], 'gray')
        m = markers.get(case['n'], 'x')
        label = f"SST n={case['n']}"
        ax2.semilogx(case['y_plus'], case['u_plus'], c=c, marker=m, markersize=4,
                     linestyle='-', linewidth=1.5, label=label, markevery=max(1, len(case['y_plus'])//12))

    # Log law reference
    y_log = np.logspace(np.log10(30), np.log10(180), 50)
    ax2.semilogx(y_log, np.log(y_log) / 0.41 + 5.0, 'k--', linewidth=1, alpha=0.5, label='log law (kappa=0.41, B=5.0)')

    ax2.set_xlabel('y+')
    ax2.set_ylabel('U+')
    ax2.set_title('Semi-log Plot')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    path = out_dir / 'fig_velocity_profiles.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    # Figure 2: Relative error in log region
    fig, ax = plt.subplots(figsize=(8, 5))

    for case in cases:
        if not case.get('converged'):
            continue
        y = np.array(case['y_plus'])
        u = np.array(case['u_plus'])
        mask = (y >= 30) & (y <= 0.2 * dns['re_tau'])
        if np.count_nonzero(mask) < 3:
            continue
        dns_interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
        rel_err = (u[mask] - dns_interp(y[mask])) / dns_interp(y[mask]) * 100
        c = colors.get(case['n'], 'gray')
        m = markers.get(case['n'], 'x')
        ax.plot(y[mask], rel_err, c=c, marker=m, markersize=4,
                linestyle='-', linewidth=1.5, label=f"SST n={case['n']}")

    ax.axhline(y=5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='5% tolerance')
    ax.axhline(y=-5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('y+')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Log-Region Velocity Error vs DNS (LM2015)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / 'fig_error_profiles.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    return figures


def render_report(summary, figures):
    """Generate a markdown report."""
    lines = [
        "# CASE-FVM-SST-CHANNEL: SST-2003 Turbulent Channel Flow vs DNS",
        "",
        f"**Anchor**: `{ANCHOR_SLUG}`",
        "",
        "## Reference",
        "",
        "Lee, M. & Moser, R. D. 2015 *Direct numerical simulation of turbulent channel",
        "flow up to Re_tau = 5200*. J. Fluid Mech. **774**, 395–415.",
        "DOI: 10.1017/jfm.2015.268",
        "",
        "NASA TMR SST-2003: https://tmbwg.github.io/turbmodels/sst.html#sst-2003",
        "",
        "## Acceptance Criteria (pre-declared)",
        "",
        f"1. Log-region RMS deviation (30 <= y+ <= 0.2*Re_tau) < {TOLERANCE_LOG_REGION:.0%}",
        f"2. Skin-friction coefficient relative error < {TOLERANCE_CF:.0%}",
        f"3. Bulk velocity relative error < {TOLERANCE_BULK:.0%}",
        "4. Overall verdict = PASS only when all three are satisfied",
        "",
        "## Results",
        "",
        "| Resolution | Converged | Iterations | y+ (first) | Cf | Cf error | bulk u+ | bulk error | log_layer RMS | Verdict |",
        "|-----------|-----------|------------|------------|------|----------|---------|------------|---------------|---------|",
    ]

    for case in summary['cases']:
        if case.get('converged'):
            lines.append(
                f"| n={case['n']} | Yes | {case['iterations']} | {case['first_cell_y_plus']:.3f} | "
                f"{case['Cf']:.6f} | {case['cf_relative_error']:.1%} | "
                f"{case['bulk_u_plus']:.4f} | {case['bulk_relative_error']:.1%} | "
                f"{case['log_layer_rms']:.1%} | {case['overall_verdict']} |"
            )
        else:
            lines.append(f"| n={case['n']} | No | - | - | - | - | - | - | - | FAIL |")

    lines.extend([
        "",
        "## DNS Reference Values",
        "",
        f"- Re_tau = {summary['dns_re_tau']}",
        f"- Cf_DNS = {summary['dns_cf']:.6f}",
        f"- U_bulk+_DNS = {summary['dns_bulk_u_plus']:.4f}",
        "",
        "## Figures",
        "",
    ])

    for fig_path in figures:
        fig_name = Path(fig_path).name
        lines.append(f"![{fig_name}]({fig_name})")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "The SST-2003 model (NASA TMR exact production variant) is graded against",
        "the Lee & Moser (2015) DNS at Re_tau = 182.088. The log-region grading",
        "window is 30 <= y+ <= 0.2*Re_tau (~36 at this Re_tau), which is the",
        "physically logarithmic layer at this Reynolds number.",
        "",
        "**Note**: SST-2003 is known to exhibit grid sensitivity in channel flow.",
        "The blending functions F1 and F2 become more sensitive to near-wall",
        "resolution as the grid is refined, which can lead to non-monotonic",
        "convergence behavior. This is documented in the NASA TMR and is not",
        "a bug in the implementation.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m tensorfvm.benchmark_sst_channel --output results/sst-channel",
        "```",
        "",
    ])

    return "\n".join(lines)


def run_benchmark(directory, resolutions=None):
    """Run the SST channel benchmark."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if resolutions is None:
        resolutions = RESOLUTIONS

    dns = load_kmm1987_reference()
    cases = []

    for n in resolutions:
        print(f"Running SST channel n={n}...")
        case = run_case(n)
        cases.append(case)
        status = "PASS" if case.get('overall_verdict') == 'PASS' else "FAIL"
        print(f"  -> {status} (log_layer RMS={case.get('log_layer_rms', 'N/A')})")

    # Write benchmark.json
    summary = {
        "anchor_slug": ANCHOR_SLUG,
        "benchmark": "sst-channel-ReTau180",
        "reference": "Lee & Moser 2015 (Re_tau=182.088, proxy for KMM1987)",
        "dns_re_tau": dns['re_tau'],
        "dns_cf": dns['Cf'],
        "dns_bulk_u_plus": dns['U_bulk_plus'],
        "tolerance_log_region": TOLERANCE_LOG_REGION,
        "tolerance_cf": TOLERANCE_CF,
        "tolerance_bulk": TOLERANCE_BULK,
        "cases": cases,
        "all_passed": all(c.get('overall_verdict') == 'PASS' for c in cases),
    }

    (directory / "benchmark.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Write profiles.csv
    with (directory / "profiles.csv").open("w") as f:
        f.write("case,y_plus,u_plus\n")
        for case in cases:
            if case.get('converged'):
                for yp, up in zip(case['y_plus'], case['u_plus']):
                    f.write(f"n={case['n']},{yp},{up}\n")
        # DNS reference
        for yp, up in zip(dns['y_plus'].tolist(), dns['u_plus'].tolist()):
            f.write(f"DNS,{yp},{up}\n")

    # Write metrics.csv
    with (directory / "metrics.csv").open("w") as f:
        f.write("n,converged,iterations,first_cell_y_plus,Cf,cf_relative_error,bulk_u_plus,bulk_relative_error,log_region_rms,log_layer_rms,verdict\n")
        for case in cases:
            f.write(f"{case['n']},{case.get('converged',False)},{case.get('iterations','')},{case.get('first_cell_y_plus','')},{case.get('Cf','')},{case.get('cf_relative_error','')},{case.get('bulk_u_plus','')},{case.get('bulk_relative_error','')},{case.get('log_region_rms','')},{case.get('log_layer_rms','')},{case.get('overall_verdict','')}\n")

    # Generate figures
    figures = make_figures(cases, directory)

    # Write report
    report = render_report(summary, figures)
    (directory / "REPORT.md").write_text(report, encoding="utf-8")

    # Write manifest
    manifest = {
        "anchor_slug": ANCHOR_SLUG,
        "benchmark": "sst-channel-ReTau180",
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
    parser = argparse.ArgumentParser(description="SST-2003 channel flow benchmark vs DNS")
    parser.add_argument("--output", type=Path, default=Path("results/sst-channel"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)

    summary = run_benchmark(args.output, args.resolutions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
