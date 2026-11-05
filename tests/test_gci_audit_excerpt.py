"""Git-artifact contract for the GCI audit dumps (m3807 slimming).

The per-case audit tables were 1.4 MB + 340 KB of JSON, which trips the 200 KB
per-artifact merge gate for zero review value: every conclusion in the reports
is carried by the aggregates plus the worst-efficiency cases.  The dumps are
therefore stored as excerpts that point at a deterministic regeneration path.
These tests pin that contract:

* the committed artifacts stay under the cap;
* an excerpt keeps the aggregates untouched and the *worst* cases;
* the recorded SHA-256 is the digest of the full (untrimmed) payload, so a
  regenerated full dump can be checked against the audited one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_CAP = 200 * 1024


def _load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_committed_audit_artifacts_under_cap():
    for rel in ("outputs/gci-band-coverage/audit.json",
                "docs/gci-fallback-band/audit.json"):
        size = (REPO / rel).stat().st_size
        assert size <= ARTIFACT_CAP, (rel, size)


def test_committed_audit_artifacts_are_excerpts_with_pointer():
    for rel in ("outputs/gci-band-coverage/audit.json",
                "docs/gci-fallback-band/audit.json"):
        data = json.loads((REPO / rel).read_text())
        assert data["kind"] == "audit-excerpt"
        assert data["full_dump"]["sha256"]
        assert data["full_dump"]["bytes"] > 0
        assert "scripts/audit_gci_" in data["full_dump"]["regenerate"]


def test_band_coverage_excerpt_keeps_aggregates_and_worst_cases():
    mod = _load("scripts/audit_gci_band_coverage.py", "audit_gci_band_coverage")
    rows = [dict(p=2.0, r21=2.0, r32=2.0, alpha=0.05, efficiency=0.01 * i,
                 implementation=True, published=True)
            for i in range(mod.EXCERPT_ROWS + 5)]
    noise_rows = [dict(amplitude=0.05, seed=i, efficiency=0.02 * i)
                  for i in range(mod.EXCERPT_NOISE_ROWS + 3)]
    payload = dict(summary={"cases": len(rows)}, by_alpha=[{"alpha": 0.0}],
                   by_order=[{"p": 2.0}], noise=[{"amplitude": 0.0}],
                   rows=rows, noise_rows=noise_rows)

    ex = mod.excerpt(payload)

    # aggregates are untouched -> the report's numbers cannot drift
    for key in ("summary", "by_alpha", "by_order", "noise"):
        assert ex[key] == payload[key]
    # truncation is lossy only for the tail of the efficiency ranking
    assert ex["case_counts"]["rows"] == len(rows)
    assert ex["case_counts"]["rows_kept"] == mod.EXCERPT_ROWS
    kept = [r["efficiency"] for r in ex["worst_efficiency_rows"]]
    assert kept == sorted((r["efficiency"] for r in rows), reverse=True)[:mod.EXCERPT_ROWS]
    assert len(ex["worst_efficiency_noise_rows"]) == mod.EXCERPT_NOISE_ROWS
    # the pointer identifies the full payload it was cut from
    assert ex["full_dump"]["sha256"] == _payload_digest(payload)
    assert ex["full_dump"]["bytes"] == len(json.dumps(payload, sort_keys=True).encode())


def test_band_coverage_excerpt_does_not_choke_on_non_finite_efficiency():
    mod = _load("scripts/audit_gci_band_coverage.py", "audit_gci_band_coverage")
    payload = dict(summary={"cases": 2}, by_alpha=[], by_order=[], noise=[],
                   rows=[dict(efficiency=float("nan")), dict(efficiency=0.5)],
                   noise_rows=[dict(efficiency=float("inf"))])
    ex = mod.excerpt(payload)
    assert ex["full_dump"]["sha256"] == _payload_digest(payload)
    assert math.isnan(ex["worst_efficiency_rows"][0]["efficiency"])
    assert math.isfinite(ex["worst_efficiency_noise_rows"][0]["efficiency"]) is False


def test_fallback_excerpt_keeps_worst_cases_and_digest():
    mod = _load("scripts/audit_gci_fallback_band.py", "audit_gci_fallback_band")
    non_asym = [dict(efficiency=0.1 * i) for i in range(mod.EXCERPT_CASES + 4)]
    asym = [dict(efficiency=0.2 * i) for i in range(mod.EXCERPT_CASES + 2)]
    summary = {"total_cases": len(non_asym) + len(asym)}

    ex = mod.excerpt(summary, non_asym, asym)

    assert ex["summary"] == summary
    assert ex["case_counts"] == dict(non_asymptotic=len(non_asym),
                                     asymptotic=len(asym),
                                     non_asymptotic_kept=mod.EXCERPT_CASES,
                                     asymptotic_kept=mod.EXCERPT_CASES)
    assert [r["efficiency"] for r in ex["non_asymptotic_cases"]] == \
        sorted((r["efficiency"] for r in non_asym), reverse=True)[:mod.EXCERPT_CASES]
    full = json.dumps(dict(non_asymptotic_cases=non_asym,
                           asymptotic_cases=asym), sort_keys=True).encode()
    assert ex["full_dump"]["sha256"] == hashlib.sha256(full).hexdigest()
    assert ex["full_dump"]["bytes"] == len(full)
