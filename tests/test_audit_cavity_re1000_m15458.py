"""Audit assertions for the m15458 (CASE-CAVITY-RE1000-FVM) evidence bundle.

These tests are the machine-checkable part of the delivery audit: they re-derive
every headline number from the frozen field snapshot in
``artifacts/audit-m15458`` instead of trusting the delivered scorer, and they pin
the two defects the audit found (BC-node metric artefact; bundle absent from the
merged upstream tree) so a future change cannot silently restore them.

Run: PYTHONPATH=src python3 -m pytest tests/test_audit_cavity_re1000_m15458.py -q
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from tensorfvm.lid_cavity_reference import table


def _audit_module():
    """Load scripts/audit_cavity_re1000_m15458.py without a package import."""
    import importlib.util

    path = os.path.join(ROOT, "scripts", "audit_cavity_re1000_m15458.py")
    spec = importlib.util.spec_from_file_location("audit_cavity_re1000_m15458", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "artifacts", "audit-m15458")
SUMMARY = os.path.join(BUNDLE, "summary.json")
TOL = 0.05  # case-ledger band for CASE-CAVITY-RE1000-FVM


@pytest.fixture(scope="module")
def summary() -> dict:
    with open(SUMMARY) as fh:
        return json.load(fh)


def _case(summary: dict, n: int) -> dict:
    return next(c for c in summary["cases"] if c["n"] == n)


def _field(n: int):
    import torch

    blob = torch.load(os.path.join(BUNDLE, "fields", f"Re1000_n{n}.pt"),
                      map_location="cpu", weights_only=False)
    return blob["u"].numpy(), blob["v"].numpy(), 1.0 / n, blob.get("steps")


def test_reference_arrays_are_the_repo_ghia_tables(summary):
    """The delivered bundle must quote the committed Ghia tables verbatim."""
    for n in (32, 64):
        case = _case(summary, n)
        assert np.array_equal(np.array(case["u"]["profiles"]["reference"]),
                              np.array(table(1000.0, "u"))[:, 1])
        assert np.array_equal(np.array(case["v"]["profiles"]["reference"]),
                              np.array(table(1000.0, "v"))[:, 1])


def test_reported_numbers_are_traceable_to_the_field(summary):
    """Shipped profiles must be reproducible from the .pt field, not typed in.

    Two-part check.  (a) At Ghia sample coordinates that coincide with a grid
    node the shipped value must equal the field value to round-off - that pins
    the numbers to the artefact.  (b) Between nodes the shipped Hermite and an
    independent linear interpolation may differ, so only a loose bound is
    asserted; that deviation is what makes the headline error
    interpolation-sensitive (documented in the audit report).
    """
    for n in (32, 64):
        case = _case(summary, n)
        u, v, h, _ = _field(n)
        cells = (np.arange(n) + 0.5) * h
        for which, series in (("u", u[n // 2, :]), ("v", v[:, n // 2])):
            coord = np.array(case[which]["profiles"]["coord"])
            shipped = np.array(case[which]["profiles"]["computed"])
            on_node = 0
            for c, got in zip(coord, shipped):
                j = int(np.argmin(np.abs(cells - c)))
                if abs(cells[j] - c) < 5e-4:   # Ghia coords are 4-decimal rounded
                    assert abs(got - series[j]) < 1e-3, (n, which, c, got, series[j])
                    on_node += 1
            assert on_node >= 3, (n, which, on_node)  # n=32: fewer coords land on nodes
            assert np.abs(shipped[1:-1] - np.interp(coord[1:-1], cells, series)).max() < 2e-2


def test_independent_metrics_inside_ledger_band(summary):
    """Audit-side metrics (linear interp, exact BC) must clear the 5 % band."""
    audit = _audit_module()

    row = audit.metrics_for(64, summary)
    m = row["independent_metric"]
    assert m["u_min_relative_error_at_ghia_y"] < TOL
    assert m["v_max_relative_error_at_ghia_x"] < TOL
    assert m["max_abs_du_all_points"] < TOL
    assert m["max_abs_dv_all_points"] < TOL
    assert row["vortex"]["independent_euclid_dist"] < TOL
    assert row["incompressibility"]["max_abs_cell_divergence"] < 1e-10


def test_bc_node_artefact_is_documented(summary):
    """Defect pin: the shipped max|du| is dominated by a lid-node extrapolation.

    Shipped u(y=1) = 0.9707 although the lid is Dirichlet u = 1 exactly, so the
    published 2.93 % U "pointwise error" is an interpolation artefact at a node
    where the physical error is zero.  The real interior maximum is ~1.9 % U.
    """
    audit = _audit_module()

    row = audit.metrics_for(64, summary)
    bc = row["boundary_node_artefact"]
    assert bc["shipped_max_abs_du_concentrated_at_bc_node"] is True
    assert abs(bc["shipped_u_at_lid_(y=1)"] - 1.0) > 0.02
    assert abs(bc["shipped_u_at_bottom_(y=0)"]) > 1e-3
    assert row["independent_metric"]["max_abs_du_interior_points"] < 0.02


def test_field_hashes_match_the_frozen_manifest():
    """Snapshot integrity: the audited bytes are pinned by SHA256SUMS."""
    manifest = {}
    with open(os.path.join(BUNDLE, "SHA256SUMS")) as fh:
        for line in fh:
            digest, path = line.split()
            manifest[os.path.basename(path)] = digest
    targets = {"summary.json": os.path.join(BUNDLE, "summary.json")}
    for n in (32, 64):
        targets[f"Re1000_n{n}.pt"] = os.path.join(BUNDLE, "fields", f"Re1000_n{n}.pt")
    for name, path in targets.items():
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert digest == manifest[name], name
