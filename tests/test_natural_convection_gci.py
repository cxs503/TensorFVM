"""Tests for the buoyant-cavity grid-convergence tooling.

Three independent claims are checked here without running the flow solver:

1. the two wall-flux estimators are exact for the profiles they are designed for
   (half-cell for a linear profile, quadratic for a quadratic profile);
2. the half-cell estimator is first order and the quadratic estimator second
   order on a smooth boundary layer, so the estimator truncation can be separated
   from the solver error;
3. ``summarise`` turns a synthetic ``C h^p`` series into the expected observed
   order and extrapolated limit (no solver involved, so a bug in the bookkeeping
   cannot hide behind a solver run).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.verification.natural_convection_gci import (
    BENCHMARK_NU,
    mean_nusselt_estimators,
    nusselt_half_cell,
    nusselt_quadratic,
    summarise,
)


def _cell_averages(profile, n: int) -> np.ndarray:
    """Exact cell averages of ``profile(x)`` on ``n`` cells of ``[0, 1]`` (x = axis 0)."""
    from numpy.polynomial.legendre import leggauss

    xs, ws = leggauss(6)                       # 6-point Gauss: exact for smooth data
    h = 1.0 / n
    centres = (np.arange(n) + 0.5) * h
    averages = np.empty(n)
    for i, xc in enumerate(centres):
        xq = xc + 0.5 * h * xs
        averages[i] = 0.5 * np.sum(ws * profile(xq))   # (h/2)/h * Gauss sum
    return averages


def _field_from_x_values(values: np.ndarray) -> np.ndarray:
    """Broadcast an x-profile (one value per x-cell) onto an ``(nx, ny)`` field."""
    return np.repeat(values[:, None], values.size, axis=1)


def test_linear_profile_both_estimators_exact() -> None:
    n = 16
    h = 1.0 / n
    # theta = 0.5 - x  =>  Nu = 1 on both walls, estimators must agree.
    x = (np.arange(n) + 0.5) * h
    theta = _field_from_x_values(0.5 - x)
    assert np.allclose(nusselt_half_cell(theta, h, "hot"), 1.0, atol=1e-12)
    assert np.allclose(nusselt_quadratic(theta, h, "hot"), 1.0, atol=1e-12)
    metrics = mean_nusselt_estimators(theta, h)
    assert metrics["nusselt_cold_half_cell"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["nusselt_cold_quadratic"] == pytest.approx(1.0, abs=1e-12)


def test_quadratic_profile_quadratic_estimator_exact() -> None:
    n = 16
    h = 1.0 / n
    # theta(x) = 0.5 - x - 0.5 x^2 -> dtheta/dx(0) = -1 -> Nu(0) = 1 exactly.
    # The estimator consumes *cell averages*, not centroid samples.
    averages = _cell_averages(lambda x: 0.5 - x - 0.5 * x ** 2, n)
    theta = _field_from_x_values(averages)
    assert np.allclose(nusselt_quadratic(theta, h, "hot"), 1.0, atol=1e-12)
    # ... while the half-cell estimator carries its O(h) curvature bias.
    assert abs(float(np.mean(nusselt_half_cell(theta, h, "hot"))) - 1.0) > 0.15 * h


def test_boundary_layer_estimator_orders() -> None:
    """Half-cell flux is O(h); the quadratic reconstruction is O(h^2)."""
    delta = 0.30
    profile = lambda x: 0.5 * np.exp(-x / delta)        # noqa: E731

    def errors(n: int) -> tuple[float, float]:
        h = 1.0 / n
        averages = _cell_averages(profile, n)
        # Wall value: theta_w = 0.5 * exp(0) = 0.5 is the *Dirichlet* value, but
        # the estimators normalise against a wall value of +-0.5, so the field
        # is rescaled to theta_w = +-0.5 by construction (profile(0) = 0.5).
        theta = _field_from_x_values(averages)
        exact = -(-0.5 / delta) / 1.0                    # -dtheta/dx(0) = 0.5/delta
        half = abs(float(np.mean(nusselt_half_cell(theta, h, "hot"))) - exact) / exact
        quad = abs(float(np.mean(nusselt_quadratic(theta, h, "hot"))) - exact) / exact
        return half, quad

    half_errors, quad_errors = zip(*(errors(n) for n in (16, 32, 64)))
    order_half = math.log(half_errors[0] / half_errors[1], 2.0)
    order_quad = math.log(quad_errors[1] / quad_errors[2], 2.0)
    assert 0.7 <= order_half <= 1.3, (order_half, half_errors)
    assert order_quad >= 1.7, (order_quad, quad_errors)
    # The second-order flux must be strictly closer on the finest grid.
    assert quad_errors[-1] < half_errors[-1]


def test_summarise_recovers_synthetic_order_and_limit() -> None:
    """A manufactured ``Nu(N) = 4.5 + 0.9 h^2`` series must come back as p = 2."""
    reference = BENCHMARK_NU[1.0e5]
    records = []
    for n in (16, 32, 64, 128):
        h = 1.0 / n
        nu = reference + 0.9 * h ** 2
        records.append({"metrics": {
            "rayleigh": 1.0e5, "cells": n, "h": h,
            "nusselt_hot_half_cell": nu, "nusselt_hot_quadratic": nu,
            "psi_extremum_abs": 9.111 + 1.0 * h ** 2,
        }})
    summary = summarise(records)
    block = summary["families"]["Ra1e+05"]
    refinement = block["nusselt_hot_half_cell"]["refinement"]
    assert refinement["enough_grids"] is True
    assert refinement["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert refinement["richardson"]["extrapolated"] == pytest.approx(reference, abs=1e-6)
    assert block["verdict"]["extrapolated_error"] < 1e-6
    # The gate is judged on the finest grid, which here is already inside 3 %.
    assert block["verdict"]["finest_grid"] == 128
    assert block["verdict"]["finest_error"] <= 0.03
    assert BENCHMARK_NU[1.0e5] == 4.519


def test_summarise_reports_inconsistent_series_as_such() -> None:
    """A non-monotone metric must not be dressed up as a converged study."""
    values = {16: 10.4, 32: 9.0, 64: 9.2}
    records = [{"metrics": {"rayleigh": 1.0e6, "cells": n, "h": 1.0 / n,
                            "nusselt_hot_half_cell": v, "nusselt_hot_quadratic": v,
                            "psi_extremum_abs": v}} for n, v in values.items()]
    block = summarise(records)["families"]["Ra1e+06"]["nusselt_hot_half_cell"]
    assert block["refinement"]["richardson"]["convergence"] in (
        "inconsistent", "order-implausible")