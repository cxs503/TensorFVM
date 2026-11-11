"""Tests of the Armaly (1983) reproduction surface (:mod:`tensorfvm.inc_bench.armaly`).

The checks are structural plus one short solver smoke run: the experimental
anchors must be the certified ones (no interpolation, no silent extrapolation),
the development metric must read ~zero for a genuinely parabolic field, and a
short run must produce a complete, JSON-serialisable reproduction record.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tensorfvm.inc_bench.armaly import (
    ARMALY_1983_CITATION, ARMALY_1983_EXPERIMENT, TOLERANCE_PROFILE,
    TOLERANCE_REATTACHMENT, ReproductionRun, column_at, experiment_point,
    profile_development)
from tensorfvm.inc_bench.cases import backward_step
from tensorfvm.inc_bench.solver import SolverOptions


def test_experimental_anchors_are_certified_values():
    assert sorted(ARMALY_1983_EXPERIMENT) == [100.0, 389.0]
    values = [p.reattachment_over_step for p in ARMALY_1983_EXPERIMENT.values()]
    assert values == sorted(values), "L_r/S must grow with Re in the 2-D regime"
    for point in ARMALY_1983_EXPERIMENT.values():
        assert point.band_over_step > 0.0
        assert point.reattachment_over_step > point.band_over_step
    assert "J. Fluid Mech. 127" in ARMALY_1983_CITATION
    assert "1983" in ARMALY_1983_CITATION
    assert TOLERANCE_REATTACHMENT == pytest.approx(0.15)
    assert TOLERANCE_PROFILE == pytest.approx(0.05)


def test_no_interpolation_between_anchors():
    assert experiment_point(100.0).reattachment_over_step == pytest.approx(3.0)
    assert experiment_point(389.0).reattachment_over_step == pytest.approx(7.9)
    with pytest.raises(ValueError):
        experiment_point(200.0)


def test_development_metric_is_zero_for_a_parabolic_profile(monkeypatch):
    """The 5 % gate must read ~0 for a genuinely parabolic field and O(1) for a plug."""
    from types import SimpleNamespace

    from tensorfvm.inc_bench import armaly as armaly_module

    setup = backward_step("coarse", reynolds_dh=100.0)
    ny = int(setup.fluid.shape[0])
    column = int(setup.fluid.shape[1]) // 2
    heights = float(setup.fluid[:, column].sum()) * setup.dy
    eta = (torch.arange(ny, dtype=torch.float64) + 0.5) * setup.dy / heights
    bulk = 0.6
    profile = 6.0 * bulk * eta * (1.0 - eta)

    def fake_cell_velocity(u, v):
        return u.clone(), torch.zeros_like(u)

    monkeypatch.setattr(armaly_module, "cell_velocity", fake_cell_velocity)

    def metric(field):
        solver = SimpleNamespace(u=field.clone().expand(ny, len(setup.fluid[0])),
                                 v=None, ny=ny)
        return profile_development(solver, setup, column)

    parabolic = metric(profile[:, None].expand(ny, int(setup.fluid.shape[1])))
    # The bulk velocity is the cell-average of the field (midpoint rule), so the
    # residual is the O(dy^2) quadrature error of the parabola, not a shape error.
    assert parabolic["max_rel"] < 5e-3
    assert parabolic["l2_rel"] < 5e-3
    assert parabolic["bulk_velocity"] > 0.0
    plug = metric(torch.full((ny, int(setup.fluid.shape[1])), bulk))
    assert plug["max_rel"] > 0.3


def test_column_at_locates_the_step_plane_and_outlet():
    setup = backward_step("coarse", reynolds_dh=100.0)
    nx = int(setup.fluid.shape[1])
    x_up = setup.geometry["inlet_heights_upstream"] * setup.params["h_in"]
    outlet = (nx - 0.5) * setup.dx - x_up
    assert column_at(setup, outlet) == nx - 1
    entry = column_at(setup, 0.0)
    assert 0 < entry < nx - 1


def test_short_run_produces_a_complete_record():
    options = SolverOptions(pressure_iterations=300, tolerance=1e-6)
    record = ReproductionRun(100.0, "coarse", 12.0, options=options,
                             max_iterations=25).run()
    for key in ("reattachment_over_step", "experiment_over_step",
                "reattachment_rel_error", "profile_outlet_column",
                "profile_stations", "field", "friction_profile", "converged"):
        assert key in record, key
    assert math.isfinite(record["reattachment_over_step"])
    assert record["experiment_over_step"] == pytest.approx(3.0)
    assert record["expansion_ratio_rel_error"] < 0.005
    assert len(record["profile_stations"]) == 3
    assert record["field"]["shape"] == [int(record["nx"]), int(record["ny"])]
    assert len(record["field"]["u"]) == record["fluid_cells"]
    json.dumps(record)  # must be artifact-serialisable
    # A Reynolds number without an experimental anchor must not invent one.
    other = ReproductionRun(200.0, "coarse", 12.0, options=options,
                            max_iterations=5).run()
    assert other["experiment_over_step"] is None
    assert other["reattachment_rel_error"] is None