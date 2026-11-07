"""Geometry, reference and acceptance-logic tests for the benchmark cases.

The platform discipline is that a benchmark is only meaningful once its geometry
has been verified on the mesh (a stair-step step is either right or wrong by a
whole cell) and once the acceptance criteria are shown to reject wrong answers.
Both are tested here with cheap, deterministic checks.
"""


import pytest
import torch

from tensorfvm.inc_bench import cases, reference


# --------------------------------------------------------------- case 1: BFS
@pytest.mark.parametrize("level", sorted(cases.BFS_LEVELS))
def test_backward_step_geometry(level):
    setup = cases.backward_step(level)
    geometry = setup.geometry
    # The step must land exactly on a grid line, otherwise the modelled
    # expansion ratio is not the intended one.
    assert abs(geometry["step_height_over_dy"] - round(geometry["step_height_over_dy"])) < 1e-12
    assert geometry["expansion_ratio_rel_error"] <= 6e-4
    # Inlet mass flux must equal the nominal mean velocity times the inlet height
    # up to midpoint quadrature of the parabolic profile (measured 0.17 % here).
    nominal = setup.params["h_in"]
    assert abs(geometry["inlet_flux"] - nominal) / nominal < 5e-3
    # Downstream of the step the fluid column must be the full outlet height.
    fluid = setup.fluid
    assert int(fluid[:, -1].sum()) == fluid.shape[0]
    assert int(fluid[:, 0].sum()) == round(setup.params["h_in"] / setup.dy)


def test_backward_step_step_count_is_exact():
    for level, (n_s, n_in) in cases.BFS_LEVELS.items():
        setup = cases.backward_step(level)
        assert abs(setup.dy * n_s - 1.0) < 1e-12
        assert abs(setup.dy * n_in - setup.params["h_in"]) < 1e-12


# --------------------------------------------------------- case 2: expansion
@pytest.mark.parametrize("level", sorted(cases.EXPANSION_LEVELS))
def test_expansion_geometry(level):
    setup = cases.sudden_expansion(level)
    geometry = setup.geometry
    assert geometry["area_ratio_rel_error"] <= 2e-3
    fluid = setup.fluid
    # Symmetric expansion: equal solid bands top and bottom at the inlet plane.
    column = fluid[:, 0]
    assert int(column.sum()) == geometry["inlet_cells"]
    first = int(torch.nonzero(column).flatten()[0])
    last = int(torch.nonzero(column).flatten()[-1])
    assert abs(first - (fluid.shape[0] - 1 - last)) <= 1
    nominal = setup.params["h_in"]
    # Only 8/12/16 cells span the inlet here, so midpoint quadrature of the
    # parabolic profile leaves a sub-percent flux deficit (0.78 % on the coarsest).
    assert abs(geometry["inlet_flux"] - nominal) / nominal < 1.5e-2


def test_expansion_borda_carnot_value():
    table = reference.sudden_expansion()
    assert abs(table["borda_carnot"].value - 0.5625) < 1e-12
    assert table["area_ratio"].value == 4.0


# ------------------------------------------------------------- case 3: cavity
@pytest.mark.parametrize("level", sorted(cases.CAVITY_LEVELS))
def test_cavity_geometry(level):
    setup = cases.natural_cavity(1.0e5, level=level)
    n = cases.CAVITY_LEVELS[level]
    assert setup.fluid.all()
    assert abs(setup.dx * n - 1.0) < 1e-12
    # Rayleigh/Prandtl bookkeeping: Ra = g beta dT L^3 / (nu alpha) with
    # g beta dT = L = 1 and alpha = nu / Pr.
    nu, alpha = setup.viscosity, setup.conductivity
    assert abs(nu * alpha * 1.0e5 - 1.0) < 1e-9
    assert abs(nu / alpha - 0.71) < 1e-9


def test_cavity_reference_table_is_monotone():
    values = [reference.cavity(ra) for ra in (1e4, 1e5, 1e6)]
    nusselt = [item.nusselt for item in values]
    psi = [item.psi_max for item in values]
    assert nusselt == sorted(nusselt) and psi == sorted(psi)
    assert abs(reference.cavity(1e6).nusselt - 8.800) < 1e-12
    with pytest.raises(ValueError):
        reference.cavity(3e5)


# ------------------------------------------------------- acceptance criteria
def _fake_bfs(reattach, friction_error=0.01):
    return {
        "reattachment_over_step": reattach,
        "friction_far": 1.0, "friction_far_analytic": 1.0,
        "friction_far_rel_error": friction_error,
        "friction_min": -0.005,
    }


def test_backward_step_verdict_positive_and_negative_control():
    good = {"coarse": _fake_bfs(5.9), "medium": _fake_bfs(6.05), "fine": _fake_bfs(6.02)}
    assert cases.backward_step_verdict(good)["passed"] is True
    # Mutation: a 20 % error in the reattachment length must be rejected.
    bad = {"coarse": _fake_bfs(7.2), "medium": _fake_bfs(7.3), "fine": _fake_bfs(7.4)}
    verdict = cases.backward_step_verdict(bad)
    assert verdict["passed"] is False
    assert any(item["id"].startswith("bfs.reattachment") and not item["passed"]
               for item in verdict["criteria"])
    # Mutation: a friction coefficient off by 30 % must be rejected as well.
    friction = {"coarse": _fake_bfs(6.0), "medium": _fake_bfs(6.0),
                "fine": _fake_bfs(6.0, friction_error=0.30)}
    assert cases.backward_step_verdict(friction)["passed"] is False


def _fake_cavity(nu, psi, energy=0.001):
    ref = reference.cavity(1e4)
    return {"nusselt_hot": nu, "nusselt_reference": ref.nusselt,
            "nusselt_rel_error": abs(nu - ref.nusselt) / ref.nusselt,
            "psi_max": psi, "psi_max_reference": ref.psi_max,
            "psi_max_rel_error": abs(psi - ref.psi_max) / ref.psi_max,
            "energy_balance_rel": energy}


def test_cavity_verdict_rejects_large_error():
    good = {"1e+04": _fake_cavity(2.25, 5.05)}
    verdict = cases.cavity_verdict(good)
    assert verdict["passed"] is True and verdict["meets_target"] is True
    bad = {"1e+04": _fake_cavity(3.10, 5.05)}
    assert cases.cavity_verdict(bad)["passed"] is False
    # Energy imbalance alone must fail an otherwise accurate case.
    imbalanced = {"1e+04": _fake_cavity(2.25, 5.05, energy=0.05)}
    assert cases.cavity_verdict(imbalanced)["passed"] is False


def test_expansion_verdict_requires_viscous_excess():
    setups = {level: cases.sudden_expansion(level) for level in cases.EXPANSION_LEVELS}
    good = {level: {"reattachment_over_inlet_height": 2.0, "loss_coefficient": 0.9,
                    "loss_coefficient_borda_carnot": 0.5625,
                    "loss_excess_over_borda_carnot": 0.3375,
                    "mass_imbalance": 1e-9} for level in setups}
    assert cases.expansion_verdict(good, setups)["passed"] is True
    bad = {level: dict(value, loss_coefficient=0.40,
                       loss_excess_over_borda_carnot=-0.1625)
           for level, value in good.items()}
    assert cases.expansion_verdict(bad, setups)["passed"] is False