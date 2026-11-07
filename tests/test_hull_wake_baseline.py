"""Pytest tests for hull wake baseline."""

import json
import pytest
import torch

from tensorfvm.hull_wake_baseline import HullWakeConfig, run_all


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    yield


class TestHullWakeBaseline:
    """Test the simplified hull wake baseline."""

    def test_convergence(self, tmp_path):
        summary = run_all(output_dir=tmp_path)
        assert summary["passed"]
        assert summary["metrics"]["converged"]

    def test_wake_deficit_positive(self, tmp_path):
        summary = run_all(output_dir=tmp_path)
        assert summary["metrics"]["max_wake_deficit_m_s"] > 0

    def test_wake_profiles_exist(self, tmp_path):
        summary = run_all(output_dir=tmp_path)
        assert len(summary["metrics"]["wake_stations"]) == 3
        assert (tmp_path / "wake_profiles.json").exists()

    def test_reynolds_number(self, tmp_path):
        summary = run_all(output_dir=tmp_path)
        assert summary["metrics"]["reynolds_number"] == 2.0
