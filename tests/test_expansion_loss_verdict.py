"""Gates for the sudden-expansion loss-coefficient verdict (m7573).

Two kinds of checks:

* **Pure post-processing invariants** of :mod:`tensorfvm.expansion_loss` -- the
  flux weighting reduces to the area average on a uniform profile, the
  kinetic-energy bias of a parabolic profile is the analytic ``(alpha-1)/AR**2``,
  the step-plane extrapolation is the identity for a zero slope, and the 3 %
  verdict is exact at its boundary.  These need no stored fields.
* **Evidence gates** -- the committed ``results.json`` must be reproducible
  byte-for-byte from the committed run fields plus the archived artifacts, and
  the re-derived station ``K`` must reproduce the archived ``k_total_head``.
  The measured development ratio is pinned below unity, which is the machine
  statement of the honest-WITHHELD verdict (the 12 D channel is not a developed
  friction balance, so no step-plane extrapolation is justified).
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfvm.expansion_loss import (  # noqa: E402
    borda_carnot,
    dynamic_head,
    kinetic_energy_factor,
    linear_slope,
    relative_error,
    section_average,
    step_plane_value,
    verdict,
)

OUT = REPO / "docs" / "sudden-expansion-loss-verdict"
#: snapshot taken at import time, i.e. before any fixture rewrites the file
COMMITTED_RESULTS = (OUT / "results.json").read_text() if (OUT / "results.json").exists() else None


def _audit_module():
    path = REPO / "scripts" / "audit_expansion_loss_verdict.py"
    spec = importlib.util.spec_from_file_location("audit_expansion_loss_verdict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# pure post-processing invariants
# --------------------------------------------------------------------------- #
def test_borda_carnot_reference():
    assert borda_carnot(4.0) == pytest.approx(0.5625, abs=1e-15)
    assert borda_carnot(2.0) == pytest.approx(0.25, abs=1e-15)


def test_dynamic_head_is_vectorised():
    u = np.array([0.0, 1.0, 2.0])
    assert np.allclose(dynamic_head(u), [0.0, 0.5, 2.0])


def test_flux_weighting_reduces_to_area_average_on_uniform_profile():
    values = np.array([1.0, 3.0, 5.0])
    weights = np.full(3, 0.25)
    assert section_average(values, weights) == pytest.approx(values.mean(), abs=1e-15)


def test_parabolic_kinetic_energy_bias():
    """A parabolic 2-D profile must show alpha = 1.8 and the exact flux bias."""
    y = np.linspace(0.0, 1.0, 2001)
    u = 4.0 * y * (1.0 - y)                     # mean = 2/3, peak = 1
    alpha = kinetic_energy_factor(u)
    assert alpha == pytest.approx(54.0 / 35.0, rel=3e-3)  # 2-D parabolic: alpha = 54/35
    # p_t flux-weighted minus area-averaged, in units of 0.5 * <u>**2
    pt_flux = section_average(0.5 * u ** 2, u)
    pt_bulk = 0.5 * section_average(u) ** 2
    bias = (pt_flux - pt_bulk) / pt_bulk
    assert bias == pytest.approx(alpha - 1.0, rel=2e-3)


def test_step_plane_extrapolation_identities():
    assert step_plane_value(0.7, 0.0, 10.0) == pytest.approx(0.7)
    # a positive dK/dx lowers the extrapolated value
    assert step_plane_value(0.7, 0.02, 10.0) == pytest.approx(0.5)
    assert step_plane_value(0.7, -0.02, 5.0) == pytest.approx(0.8)


def test_linear_slope_and_degenerate_guard():
    assert linear_slope(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        linear_slope(np.array([1.0, 1.0]), np.array([0.0, 1.0]))


def test_verdict_boundary_is_the_3pct_bar():
    assert verdict(0.5625, 0.5625) == "PASS"
    assert verdict(0.5625 * 1.0299, 0.5625) == "PASS"
    assert verdict(0.5625 * 1.0301, 0.5625) == "FAIL"
    assert math.isnan(relative_error(float("nan"), 0.5625))
    assert verdict(float("nan"), 0.5625) == "WITHHELD"


# --------------------------------------------------------------------------- #
# evidence gates
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    payload = _audit_module().build_report()
    return payload


def test_committed_results_are_reproducible(report):
    assert COMMITTED_RESULTS is not None
    committed = json.loads(COMMITTED_RESULTS)
    assert json.loads(json.dumps(report, default=float)) == committed, (
        "results.json drifted from what the artifacts regenerate; rerun "
        "`python scripts/audit_expansion_loss_verdict.py --mode report`")


def test_station_route_reproduces_the_archived_metric(report):
    archived = report["archived"]["k_total_head"]["expansion_n010"]
    assert report["routes"]["station_bulk_D10"]["value"] == pytest.approx(archived, rel=1e-9)
    # the static-recovery identity must reproduce the same station value exactly
    assert report["routes"]["static_station_D10"]["value"] == pytest.approx(archived, abs=1e-8)


def test_channel_is_not_developed_so_the_verdict_is_withheld(report):
    ratio = report["slopes"]["development_ratio"]
    # measured dp/dx over the last two diameters still has the wrong magnitude
    # (and even the wrong sign) versus the fully developed Poiseuille value
    assert abs(ratio) < 1.0, (
        "the 12 D channel now behaves like a developed friction balance; the "
        "WITHHELD verdict and its extrapolation caveat must be re-derived")
    assert report["slopes"]["r2_pressure"] > 0.9  # the fit itself is fine


def test_all_six_routes_are_reported_and_none_passes(report):
    routes = report["routes"]
    assert set(routes) == {
        "station_bulk_D10", "station_flux_D10", "static_station_D10",
        "static_step_D10", "bulk_step_D10", "flux_step_D10",
    }
    assert all(row["verdict"] == "FAIL" for row in routes.values())
    spread = max(r["value"] for r in routes.values()) - min(r["value"] for r in routes.values())
    assert spread > 0.5, (
        "the route spread collapsed; the 3 % bar may now be reachable and the "
        "archived FAIL/PASS disagreement has to be re-adjudicated")
    figure = OUT / "K_routes_D10.png"
    assert figure.exists() and figure.stat().st_size > 10_000