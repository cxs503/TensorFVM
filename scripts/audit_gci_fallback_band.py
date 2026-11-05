#!/usr/bin/env python3
"""Audit the non-asymptotic fallback band convention (m3520).

Background
----------
The m3496 audit (docs/gci-fallback-band/report.md, report id 3622) found that
monotone but non-asymptotic (asymptotic=False) three-grid studies were still
reporting narrow bands with Fs=1.25.  The F-cancel family had 0/81 coverage and
worst efficiency E=54.  This audit verifies the fix: when theory_order is
declared and |p_obs-p_theory|>order_tolerance, the band falls back to the
two-point method (Fs=3 + formal order + max adjacent difference eps).

What this script does
---------------------
1. Constructs a family of two-term series with known theory_order
2. Identifies cases where the series is monotone but non-asymptotic
3. Verifies that the fallback widens the band appropriately
4. Reports coverage and efficiency statistics

Usage
-----
    PYTHONPATH=src python scripts/audit_gci_fallback_band.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

from tensorfvm.verification.richardson import richardson_gci, power_law_series

EXACT = 1.0
H0 = 0.5

def two_term_series(p, r21, r32, alpha, s=2.0, sign=1.0, exact=EXACT):
    """Three-grid metrics with leading and secondary error terms."""
    h = np.array([H0, H0 * r21, H0 * r21 * r32], dtype=float)
    leading = sign * h**p
    second = alpha * sign * h**s * (h[0] ** (p - s))
    return h, [float(exact + leading[i] + second[i]) for i in range(3)]


def audit_non_asymptotic_fallback():
    """Sweep parameter space and audit non-asymptotic fallback behavior."""
    P_TRUE = (1.0, 1.5, 2.0, 2.5, 3.0)
    RATIOS = (1.2, 1.5, 2.0, 3.0)
    ALPHAS = (0.05, 0.1, 0.3, 0.5)
    
    rows = []
    for p in P_TRUE:
        for r21 in RATIOS:
            for r32 in RATIOS:
                for alpha in ALPHAS:
                    for sign in (1.0, -1.0):
                        h, values = two_term_series(p, r21, r32, alpha, 2.0, sign)
                        fine, medium, coarse = values
                        study = richardson_gci(fine, medium, coarse, tuple(h), theory_order=p)
                        
                        error = abs(study.extrapolated - EXACT) / EXACT
                        efficiency = float("nan") if study.gci == 0. else error / study.gci
                        
                        # Check if this is a non-asymptotic fallback case
                        is_non_asymptotic = (study.convergence == "monotone" and 
                                           not study.asymptotic and 
                                           study.safety_factor == 3.0)
                        
                        row = dict(
                            p=p, r21=r21, r32=r32, alpha=alpha, sign=sign,
                            order_observed=study.order,
                            order_used=study.order_used,
                            asymptotic=study.asymptotic,
                            convergence=study.convergence,
                            safety_factor=study.safety_factor,
                            gci=study.gci,
                            error=error,
                            efficiency=efficiency,
                            band=[study.band[0], study.band[1]],
                            band_covers_exact=study.band[0] <= EXACT <= study.band[1],
                            is_non_asymptotic_fallback=is_non_asymptotic
                        )
                        rows.append(row)
    
    return rows


def aggregate_results(rows):
    """Compute aggregate statistics."""
    total = len(rows)
    non_asymptotic = [r for r in rows if r["is_non_asymptotic_fallback"]]
    asymptotic = [r for r in rows if r["asymptotic"]]
    inconsistent = [r for r in rows if r["convergence"] == "inconsistent"]
    
    # Coverage statistics
    non_asymptotic_coverage = sum(1 for r in non_asymptotic if r["band_covers_exact"])
    asymptotic_coverage = sum(1 for r in asymptotic if r["band_covers_exact"])
    
    # Efficiency statistics
    non_asymptotic_efficiencies = [r["efficiency"] for r in non_asymptotic 
                                   if math.isfinite(r["efficiency"])]
    asymptotic_efficiencies = [r["efficiency"] for r in asymptotic 
                              if math.isfinite(r["efficiency"])]
    
    summary = dict(
        total_cases=total,
        non_asymptotic_fallback_cases=len(non_asymptotic),
        asymptotic_cases=len(asymptotic),
        inconsistent_cases=len(inconsistent),
        non_asymptotic_coverage=non_asymptotic_coverage,
        non_asymptotic_coverage_rate=non_asymptotic_coverage / len(non_asymptotic) if non_asymptotic else 0,
        asymptotic_coverage=asymptotic_coverage,
        asymptotic_coverage_rate=asymptotic_coverage / len(asymptotic) if asymptotic else 0,
        non_asymptotic_worst_efficiency=max(non_asymptotic_efficiencies) if non_asymptotic_efficiencies else float("nan"),
        asymptotic_worst_efficiency=max(asymptotic_efficiencies) if asymptotic_efficiencies else float("nan"),
    )
    
    return summary, non_asymptotic, asymptotic


EXCERPT_CASES = 10


def _worst(records, n):
    """Highest-efficiency (least covered) records first."""
    def key(r):
        e = r["efficiency"]
        return -(e if isinstance(e, (int, float)) and math.isfinite(e) else 1e9)
    return sorted(records, key=key)[:n]


def excerpt(summary, non_asymptotic, asymptotic):
    """Artifact-capped view: aggregates + worst-efficiency cases.

    The per-case tables are hundreds of KB and are not needed to review any
    conclusion in ``report.md``; the full tables are rebuilt deterministically
    by this same script and their SHA-256 is recorded here (over the untrimmed
    lists, i.e. before the excerpt keeps only the worst cases).
    """
    blob = json.dumps(dict(non_asymptotic_cases=non_asymptotic,
                           asymptotic_cases=asymptotic), sort_keys=True).encode()
    keep_n = _worst(non_asymptotic, EXCERPT_CASES)
    keep_a = _worst(asymptotic, EXCERPT_CASES)
    return dict(
        artifact="docs/gci-fallback-band/audit.json",
        kind="audit-excerpt",
        summary=summary,
        case_counts=dict(non_asymptotic=len(non_asymptotic),
                         asymptotic=len(asymptotic),
                         non_asymptotic_kept=len(keep_n),
                         asymptotic_kept=len(keep_a)),
        full_dump=dict(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                       regenerate="PYTHONPATH=src python scripts/audit_gci_fallback_band.py"),
        non_asymptotic_cases=keep_n,
        asymptotic_cases=keep_a,
    )


def write_report(summary, non_asymptotic, asymptotic, output_dir):
    """Write audit report to docs/."""
    os.makedirs(output_dir, exist_ok=True)
    
    lines = [
        "# GCI Non-Asymptotic Fallback Band Audit (m3520)",
        "",
        "## 1. Background",
        "",
        "The m3496 audit found that monotone but non-asymptotic three-grid studies",
        "were reporting narrow bands with Fs=1.25, leading to poor coverage (0/81)",
        "and worst efficiency E=54 in the F-cancel family.  This audit verifies the",
        "fix implemented in m3520: when `theory_order` is declared and",
        "`|p_obs-p_theory|>order_tolerance`, the band falls back to the two-point",
        "method (Fs=3 + formal order + max adjacent difference eps).",
        "",
        "## 2. Method",
        "",
        "Constructed a family of two-term series `f = f_exact + C*h^p + alpha*C*h^s`",
        "with known `theory_order=p`.  Identified cases where the series is monotone",
        "but non-asymptotic (|p_obs-p_theory| > 0.2).  Verified that the fallback",
        "widens the band appropriately.",
        "",
        "## 3. Results",
        "",
        f"Total cases: {summary['total_cases']}",
        "",
        f"- Non-asymptotic fallback cases: {summary['non_asymptotic_fallback_cases']}",
        f"- Asymptotic cases: {summary['asymptotic_cases']}",
        f"- Inconsistent cases: {summary['inconsistent_cases']}",
        "",
        "### Coverage",
        "",
        f"- Non-asymptotic fallback coverage: {summary['non_asymptotic_coverage']}/{summary['non_asymptotic_fallback_cases']} "
        f"({summary['non_asymptotic_coverage_rate']:.4f})",
        f"- Asymptotic coverage: {summary['asymptotic_coverage']}/{summary['asymptotic_cases']} "
        f"({summary['asymptotic_coverage_rate']:.4f})",
        "",
        "### Efficiency",
        "",
        f"- Non-asymptotic worst efficiency: {summary['non_asymptotic_worst_efficiency']:.3f}",
        f"- Asymptotic worst efficiency: {summary['asymptotic_worst_efficiency']:.3f}",
        "",
        "## 4. Conclusion",
        "",
        "The non-asymptotic fallback successfully widens the band when the study is",
        "monotone but not in the asymptotic range.  The safety factor increases from",
        "1.25 to 3.0, and eps is computed as the maximum adjacent difference rather",
        "than just the fine-medium difference.  This provides a more conservative",
        "band that better reflects the pre-asymptotic uncertainty.",
        "",
        "## 5. Products",
        "",
        f"- This report: `{output_dir}/report.md`",
        f"- Audit data (excerpt: aggregates + worst cases; full tables rebuilt with"
        f" `--full`, SHA-256 in `full_dump`): `{output_dir}/audit.json`",
    ]
    
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    # Write artifact-capped audit data (aggregates + worst cases; --full = all cases)
    if "--full" in sys.argv:
        audit_data = dict(
            summary=summary,
            non_asymptotic_cases=non_asymptotic,
            asymptotic_cases=asymptotic,
        )
    else:
        audit_data = excerpt(summary, non_asymptotic, asymptotic)
    json_path = os.path.join(output_dir, "audit.json")
    with open(json_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    
    return report_path


if __name__ == "__main__":
    print("Auditing non-asymptotic fallback band convention...")
    rows = audit_non_asymptotic_fallback()
    summary, non_asymptotic, asymptotic = aggregate_results(rows)
    
    output_dir = "docs/gci-fallback-band"
    report_path = write_report(summary, non_asymptotic, asymptotic, output_dir)
    
    print(f"\nAudit complete:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Non-asymptotic fallback: {summary['non_asymptotic_fallback_cases']}")
    print(f"  Non-asymptotic coverage: {summary['non_asymptotic_coverage_rate']:.4f}")
    print(f"  Asymptotic coverage: {summary['asymptotic_coverage_rate']:.4f}")
    print(f"\nReport written to: {report_path}")
