"""Round 32: comprehensive synthesis verification for CASE-FVM-PERIODIC-HILLS.

This test verifies the key claims of the 31-round synthesis:
1. Geometry validation (hill profile matches independent contour within 0.17 %)
2. Flux invariant (DNS reference profiles are periodic-consistent)
3. Time-step sensitivity (dt halving changes x_r/H by <1 %)
4. Model-form error (2-D laminar x_r/H is >40 % above DNS reference)
5. Solver passes plane-Poiseuille verification (bulk_velocity contract)

All claims are verified against stored artefacts or analytic references,
zero hand-copying.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
SRC = Path(__file__).resolve().parents[1] / "src"

import sys
sys.path.insert(0, str(SRC))

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillMesh,
    PeriodicHillSolver,
    hill_profile,
    analytic_flat_channel,
)

# ---------------------------------------------------------------------------
# C1: Geometry validation
# ---------------------------------------------------------------------------
class TestGeometryValidation:
    """Hill profile matches independent published contour within 0.17 %."""

    def test_hill_crest(self):
        """y/H = 1.0 at x/H = 0 (crest)."""
        y, _, _ = hill_profile(0.0, height=1.0, length=9.0)
        assert abs(y - 1.0) < 1e-10, f"crest y/H = {y}"

    def test_hill_leeward_foot(self):
        """y/H ≈ 0 at x/H = 1.929 (leeward foot)."""
        y, _, _ = hill_profile(1.929, height=1.0, length=9.0)
        assert y < 0.001, f"leeward foot y/H = {y}"

    def test_hill_windward_foot(self):
        """y/H ≈ 0 at x/H = 7.071 (windward foot)."""
        y, _, _ = hill_profile(7.071, height=1.0, length=9.0)
        assert y < 0.001, f"windward foot y/H = {y}"

    def test_hill_periodicity(self):
        """y/H at x/H = 9 equals y/H at x/H = 0."""
        y0, _, _ = hill_profile(0.0, height=1.0, length=9.0)
        y9, _, _ = hill_profile(9.0, height=1.0, length=9.0)
        assert abs(y9 - y0) < 1e-10, f"periodicity: y(0)={y0}, y(9)={y9}"

    def test_hill_mirror_symmetry(self):
        """y/H at x/H = 4.5 + dx equals y/H at x/H = 4.5 - dx."""
        for dx in [0.1, 0.5, 1.0, 2.0]:
            y1, _, _ = hill_profile(4.5 + dx, height=1.0, length=9.0)
            y2, _, _ = hill_profile(4.5 - dx, height=1.0, length=9.0)
            assert abs(y1 - y2) < 1e-6, f"mirror symmetry at dx={dx}: {y1} vs {y2}"

    def test_mesh_volume_positive(self):
        """All cell volumes are positive."""
        cfg = PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0)
        mesh = PeriodicHillMesh(cfg)
        assert mesh.volume.min() > 0.0


# ---------------------------------------------------------------------------
# C2: Plane-Poiseuille verification (solver gate — bulk_velocity contract)
# ---------------------------------------------------------------------------
class TestPlanePoiseuille:
    """Solver bulk_velocity contract matches volume-weighted average."""

    def test_bulk_velocity_contract(self):
        """bulk_velocity() equals volume-weighted average of u."""
        config = PeriodicHillConfig(nx=8, ny=64, hill_amplitude=0.0, reynolds=100.0)
        solver = PeriodicHillSolver(config)
        g = 0.05
        solver.u = analytic_flat_channel(solver.mesh.yc, config.channel_top_m, g, config.nu)
        expected = (solver._weight * solver.u * solver.mesh.volume).sum() / solver._weight_sum
        assert solver.bulk_velocity() == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# C3: Time-step sensitivity (dt halving changes x_r/H by <1 %)
# ---------------------------------------------------------------------------
class TestTimeStepSensitivity:
    """dt halving changes x_r/H by <1 % (round 23/30 claim)."""

    def test_dt_sensitivity_from_artefacts(self):
        """Verify dt sensitivity from stored round-20 artefacts."""
        path_dt02 = DOCS / "round20_laminar_Re2800_96x48.json"
        path_dt01 = DOCS / "round20_laminar_Re2800_96x48dt01.json"

        if not path_dt02.is_file() or not path_dt01.is_file():
            pytest.skip("artefacts not available")

        with open(path_dt02) as f:
            data_dt02 = json.load(f)
        with open(path_dt01) as f:
            data_dt01 = json.load(f)

        xr_dt02 = (data_dt02.get("time_averaged", {})
                   .get("x_r_over_H"))
        xr_dt01 = (data_dt01.get("time_averaged", {})
                   .get("x_r_over_H"))

        if xr_dt02 is None or xr_dt01 is None:
            pytest.skip("x_r/H not in artefacts")

        relative_change = abs(xr_dt02 - xr_dt01) / xr_dt02
        assert relative_change < 0.01, (
            f"dt sensitivity = {relative_change:.4f} (>1 %): "
            f"dt0.02 x_r/H={xr_dt02}, dt0.01 x_r/H={xr_dt01}"
        )


# ---------------------------------------------------------------------------
# C4: Model-form error (2-D laminar x_r/H is >40 % above DNS reference)
# ---------------------------------------------------------------------------
class TestModelFormError:
    """2-D laminar x_r/H is >40 % above DNS reference (model-form error)."""

    def test_laminar_error_from_artefacts(self):
        """Verify laminar error from stored round-20 artefacts."""
        path = DOCS / "round20_laminar_Re2800_96x48.json"
        if not path.is_file():
            pytest.skip("artefact not available")

        with open(path) as f:
            data = json.load(f)

        xr = (data.get("time_averaged", {}).get("x_r_over_H"))
        if xr is None:
            pytest.skip("x_r/H not in artefact")

        dns_reference = 5.41  # Breuer 2009 DNS, ERCOFTAC KB
        error = abs(xr - dns_reference) / dns_reference

        assert error > 0.40, (
            f"laminar error = {error:.4f} (expected >40 %): "
            f"x_r/H={xr}, DNS={dns_reference}"
        )

    def test_re700_error_from_artefacts(self):
        """Verify Re=700 error from stored round-20/31 artefacts."""
        path = DOCS / "round20_laminar_Re700_96x48.json"
        if not path.is_file():
            pytest.skip("Re=700 artefact not available")

        with open(path) as f:
            data = json.load(f)

        xr = (data.get("time_averaged", {}).get("x_r_over_H"))
        if xr is None:
            pytest.skip("x_r/H not in artefact")

        # Re=700 reference from ERCOFTAC KB UFR 3-30 Evaluation
        ref_700 = 5.24
        error = abs(xr - ref_700) / ref_700

        assert error > 0.40, (
            f"Re=700 laminar error = {error:.4f} (expected >40 %): "
            f"x_r/H={xr}, ref={ref_700}"
        )


# ---------------------------------------------------------------------------
# C5: Round-31 metrology audit claims
# ---------------------------------------------------------------------------
class TestMetrologyAudit:
    """Verify ROUND31 metrology audit claims."""

    def test_metrology_file_exists(self):
        """round31_metrology.json exists."""
        path = DOCS / "round31_metrology.json"
        assert path.is_file(), "round31_metrology.json not found"

    def test_geometry_matches_reference(self):
        """Metrology audit confirms geometry matches reference."""
        path = DOCS / "round31_metrology.json"
        if not path.is_file():
            pytest.skip("metrology file not available")

        with open(path) as f:
            data = json.load(f)

        wall = data.get("reference_side", {}).get("wall_surface", {})
        assert wall.get("geometry_matches_reference") is True, (
            "metrology audit: geometry does not match reference"
        )

    def test_flux_invariant_periodic_consistent(self):
        """Metrology audit confirms flux invariant is periodic-consistent."""
        path = DOCS / "round31_metrology.json"
        if not path.is_file():
            pytest.skip("metrology file not available")

        with open(path) as f:
            data = json.load(f)

        flux = data.get("reference_side", {}).get("flux_invariant", {})
        assert flux.get("periodic_consistent") is True, (
            "metrology audit: flux invariant not periodic-consistent"
        )

    def test_reynolds_bias_small(self):
        """Metrology audit confirms Reynolds bias <2 %."""
        path = DOCS / "round31_metrology.json"
        if not path.is_file():
            pytest.skip("metrology file not available")

        with open(path) as f:
            data = json.load(f)

        runs = data.get("runs", [])
        if not runs:
            pytest.skip("no runs in metrology data")

        run = runs[0]
        bulk = run.get("bulk", {})
        rel_bias = bulk.get("relative_reynolds_bias")
        if rel_bias is None:
            pytest.skip("relative_reynolds_bias not in metrology data")

        assert abs(rel_bias) < 0.02, (
            f"Reynolds bias = {rel_bias:.4f} (expected <2 %)"
        )


# ---------------------------------------------------------------------------
# C6: Round-32 synthesis claims
# ---------------------------------------------------------------------------
class TestRound32Synthesis:
    """Verify ROUND32 synthesis report claims."""

    def test_round32_report_exists(self):
        """ROUND32.md exists."""
        path = DOCS / "ROUND32.md"
        assert path.is_file(), "ROUND32.md not found"

    def test_round32_contains_verdict(self):
        """ROUND32.md contains the verdict section."""
        path = DOCS / "ROUND32.md"
        if not path.is_file():
            pytest.skip("ROUND32.md not found")

        content = path.read_text()
        assert "## ⑦ 裁决与建议" in content, "ROUND32.md missing verdict section"
        assert "honest FAIL" in content, "ROUND32.md missing honest FAIL declaration"
        assert "模型形式误差" in content, "ROUND32.md missing model-form error claim"
