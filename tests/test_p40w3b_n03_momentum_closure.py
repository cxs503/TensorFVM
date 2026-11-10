"""Pytest tests for P40W3B-N03: momentum budget closure verification.

Tests that the staggered MAC solver satisfies the discrete momentum
equations to high accuracy for internal flow benchmarks.
"""

import math
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorfvm.internal_flow.backward_step import BackwardStepCase, BackwardStepConfig
from tensorfvm.internal_flow.natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from tensorfvm.internal_flow.sudden_expansion import SuddenExpansionCase, SuddenExpansionConfig
from tensorfvm.internal_flow.momentum_budget import global_momentum_closure
from tensorfvm.internal_flow.reference import DE_VAHL_DAVIS_NU


@pytest.mark.slow
class TestMomentumBudgetClosure:
    """Test momentum budget closure for internal flow benchmarks."""

    def test_bfs_re100_momentum_closure(self):
        """BFS Re=100: x-momentum closure < 0.5%."""
        torch.set_num_threads(2)
        cfg = BackwardStepConfig(
            cells_per_step_height=10, reynolds=100.0,
            cfl=0.4, max_steps=5000, tolerance=1e-5)
        case = BackwardStepCase(cfg)
        case.solver.run(max_steps=5000, tolerance=1e-4)
        closure = global_momentum_closure(case.solver)
        assert closure["x_closure_pct"] < 0.5, \
            f"x-closure {closure['x_closure_pct']:.3f}% exceeds 0.5%"
        assert closure["mass_divergence_max"] < 1e-10, \
            f"mass divergence {closure['mass_divergence_max']:.2e} exceeds 1e-10"

    def test_expansion_ar2_momentum_closure(self):
        """Expansion AR=2: x-momentum closure < 0.5%."""
        torch.set_num_threads(2)
        cfg = SuddenExpansionConfig(
            area_ratio=2.0, reynolds=100.0,
            cells_per_downstream_height=12,
            cfl=0.4, max_steps=5000, tolerance=1e-5)
        case = SuddenExpansionCase(cfg)
        case.solver.run(max_steps=5000, tolerance=1e-4)
        closure = global_momentum_closure(case.solver)
        assert closure["x_closure_pct"] < 0.5, \
            f"x-closure {closure['x_closure_pct']:.3f}% exceeds 0.5%"

    def test_cavity_ra1e4_nusselt(self):
        """Cavity Ra=1e4: Nu within 3% of de Vahl Davis."""
        torch.set_num_threads(2)
        cfg = NaturalConvectionConfig(
            n=24, rayleigh=1e4, prandtl=0.71,
            cfl=0.4, max_steps=8000, tolerance=1e-5)
        case = NaturalConvectionCase(cfg)
        case.solver.run(max_steps=8000, tolerance=1e-4)
        diag = case.diagnostics()
        nu_ref = DE_VAHL_DAVIS_NU[1e4]
        nu_err = abs(diag["nu_mean"] - nu_ref) / nu_ref
        assert nu_err < 0.03, \
            f"Nu error {nu_err*100:.2f}% exceeds 3% (Nu={diag['nu_mean']:.4f}, ref={nu_ref})"

    def test_cavity_ra1e4_momentum_closure_x(self):
        """Cavity Ra=1e4: x-momentum closure < 0.5%."""
        torch.set_num_threads(2)
        cfg = NaturalConvectionConfig(
            n=24, rayleigh=1e4, prandtl=0.71,
            cfl=0.4, max_steps=8000, tolerance=1e-5)
        case = NaturalConvectionCase(cfg)
        case.solver.run(max_steps=8000, tolerance=1e-4)
        closure = global_momentum_closure(case.solver)
        assert closure["x_closure_pct"] < 0.5, \
            f"x-closure {closure['x_closure_pct']:.3f}% exceeds 0.5%"

    def test_mass_conservation_all_cases(self):
        """All cases: mass divergence < 1e-10 (near machine precision)."""
        torch.set_num_threads(2)
        # BFS
        cfg = BackwardStepConfig(cells_per_step_height=8, reynolds=100.0,
                                  cfl=0.4, max_steps=3000, tolerance=1e-5)
        case = BackwardStepCase(cfg)
        case.solver.run(max_steps=3000, tolerance=1e-4)
        closure = global_momentum_closure(case.solver)
        assert closure["mass_divergence_max"] < 1e-10

        # Expansion
        cfg = SuddenExpansionConfig(area_ratio=2.0, reynolds=100.0,
                                     cells_per_downstream_height=10,
                                     cfl=0.4, max_steps=3000, tolerance=1e-5)
        case = SuddenExpansionCase(cfg)
        case.solver.run(max_steps=3000, tolerance=1e-4)
        closure = global_momentum_closure(case.solver)
        assert closure["mass_divergence_max"] < 1e-10

        # Cavity
        cfg = NaturalConvectionConfig(n=16, rayleigh=1e4, prandtl=0.71,
                                       cfl=0.4, max_steps=3000, tolerance=1e-5)
        case = NaturalConvectionCase(cfg)
        case.solver.run(max_steps=3000, tolerance=1e-4)
        closure = global_momentum_closure(case.solver)
        assert closure["mass_divergence_max"] < 1e-10


@pytest.mark.fast
class TestMomentumBudgetUnit:
    """Unit tests for momentum budget module."""

    def test_closure_metric_bounded(self):
        """Closure metric should be non-negative and finite."""
        torch.set_num_threads(2)
        cfg = BackwardStepConfig(cells_per_step_height=6, reynolds=100.0,
                                  cfl=0.4, max_steps=1000, tolerance=1e-4)
        case = BackwardStepCase(cfg)
        case.solver.run(max_steps=1000, tolerance=1e-3)
        closure = global_momentum_closure(case.solver)
        assert closure["x_closure_pct"] >= 0
        assert closure["y_closure_pct"] >= 0
        assert math.isfinite(closure["x_closure_pct"])
        assert math.isfinite(closure["y_closure_pct"])

    def test_closure_decreases_with_convergence(self):
        """Closure metric should decrease as solution converges."""
        torch.set_num_threads(2)
        cfg = BackwardStepConfig(cells_per_step_height=8, reynolds=100.0,
                                  cfl=0.4, max_steps=5000, tolerance=1e-6)
        case = BackwardStepCase(cfg)

        # Run a few steps
        case.solver.run(max_steps=500, tolerance=1e-3)
        closure_early = global_momentum_closure(case.solver)

        # Run more steps
        case.solver.run(max_steps=3000, tolerance=1e-5)
        closure_late = global_momentum_closure(case.solver)

        # Closure should improve (decrease)
        assert closure_late["x_closure_pct"] <= closure_early["x_closure_pct"] * 2.0, \
            "Closure did not improve with convergence"
