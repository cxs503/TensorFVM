"""Grid-convergence verification of the curved-channel manufactured solution."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "curved_simple_order", ROOT / "scripts/verification_curved_simple_order.py")
order = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(order)


def _solve(nx, ny):
    c = order.config(nx, ny, 1., 4000)
    s, steps = order.solve(c, c.max_iterations)
    return c, s, steps


def test_manufactured_reference_is_divergence_free_and_force_consistent():
    """The manufactured solution must hold its own identities before it is used.

    Tolerances are the finite-difference floor of the independent audit, not the
    solver's own residual."""
    torch.set_num_threads(1)
    c = order.config(16, 8, 1., 1)
    from tensorfvm.sparse_simple import SparseBodyFittedSolver
    s = SparseBodyFittedSolver(c)
    audit = order.audit_reference(c, s.mesh)
    assert audit["divergence_max"] < 1e-5
    assert audit["momentum_residual_relative"] < 5e-1  # documented consistency defect
    assert audit["wall_no_slip_max"] < 5e-2  # polygonal wall chords, O(h^2) sagitta
    assert audit["cross_section_flux_relative_error"] < 1e-3
    # No-slip walls and one constant cross-section flux: the physical closure used
    # by the solver is itself well posed for this manufactured problem.
    assert np.isclose(audit["cross_section_flux_exact"], c.inlet_velocity * c.height)


def test_curved_simple_observed_order_on_coarse_pair():
    """Two grids already bracket the second-order range of the production solver."""
    torch.set_num_threads(1)
    rows = []
    for nx, ny in ((16, 8), (32, 16)):
        c, s, steps = _solve(nx, ny)
        assert s.converged, f"{nx}x{ny} did not converge in {steps} steps"
        errors = order.field_errors(s, c)
        rows.append(dict(resolution=f"{nx}x{ny}", spacing_m=c.height / ny, **errors))
    for key in ("velocity_volume_l2", "pressure_volume_l2"):
        assert rows[1][key] < rows[0][key], key
    assert rows[-1]["velocity_volume_l2"] < order.GATE
    assert rows[-1]["pressure_volume_l2"] < order.GATE
    velocity_order = order.orders(rows, "velocity_volume_l2")[0]
    pressure_order = order.orders(rows, "pressure_volume_l2")[0]
    assert 1.4 < velocity_order < 2.6
    assert 1.4 < pressure_order < 2.6