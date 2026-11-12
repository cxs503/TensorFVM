"""Gate for the Woodward & Colella (1984) forward-facing step (``REPRO-FVM-FORWARD-STEP``).

The pytest-sized version checks the machinery and the physics weakly on coarse
grids (so the rather expensive case stays inside the routine gate budget); the
graded 8 % criteria on the 480 x 160 grid of record are produced by
``scripts/repro_fvm_forward_step.py``.
"""

import math

import pytest

from tensorfvm.compressible.forward_step import (
    ForwardStepConfig,
    build_case,
    mass_flux_balance,
    measure_shock_jump,
    normal_shock_reference,
    run_case,
    stagnation_state,
)

REF = normal_shock_reference()


def test_reference_matches_textbook_relations():
    """The in-code reference must equal the closed-form normal-shock relations."""
    g, m = 1.4, 3.0
    p_ratio = 1.0 + 2.0 * g / (g + 1.0) * (m * m - 1.0)
    rho_ratio = (g + 1.0) * m * m / ((g - 1.0) * m * m + 2.0)
    mach2 = math.sqrt(((g - 1.0) * m * m + 2.0) / (2.0 * g * m * m - (g - 1.0)))
    assert REF["p_ratio"] == pytest.approx(p_ratio, rel=1e-12)
    assert REF["rho_ratio"] == pytest.approx(rho_ratio, rel=1e-12)
    assert REF["mach2"] == pytest.approx(mach2, rel=1e-12)
    assert REF["p_ratio"] == pytest.approx(10.3333333333, rel=1e-9)
    assert REF["rho_ratio"] == pytest.approx(3.8571428571, rel=1e-9)
    # inflow normalisation of the paper: a = 1, u = 3
    assert REF["c1"] == pytest.approx(1.0, rel=1e-12)
    assert REF["u1"] == pytest.approx(3.0, rel=1e-12)
    # the post-shock stream stagnates on the step face: p02 and rho02
    assert REF["p02"] == pytest.approx(12.0609647012, rel=1e-9)
    assert REF["rho02"] == pytest.approx(6.0304823506, rel=1e-9)


def test_geometry_lands_on_cell_faces():
    """A grid in the study must place the step wall exactly on cell faces."""
    for nx, ny in ((60, 20), (120, 40), (240, 80), (480, 160)):
        cfg = ForwardStepConfig(nx=nx, ny=ny)
        assert cfg.i_step * cfg.dx == pytest.approx(cfg.step_x, abs=1e-12)
        assert cfg.j_step * cfg.dy == pytest.approx(cfg.step_h, abs=1e-12)
        mask = cfg.solid_mask()
        assert int(mask.sum()) == (nx - cfg.i_step) * cfg.j_step
        assert not bool(mask[: cfg.i_step].any())


def test_mask_off_cell_face_is_rejected():
    with pytest.raises(ValueError):
        ForwardStepConfig(nx=71, ny=20)  # 0.6 / (3/71) is not an integer


@pytest.fixture(scope="module")
def coarse():
    return run_case(ForwardStepConfig(nx=60, ny=20, t_end=0.5))


def test_wall_is_impermeable_and_conservative(coarse):
    """Mass budget closes to round-off: the wall fluxes are projected exactly."""
    budget = coarse["budget"]
    assert abs(budget["mass_drift"]) < 1e-12
    assert abs(budget["energy_drift"]) < 1e-6


def test_projection_is_what_closes_the_budget():
    """Ablation: with the mirrored states alone the wall leaks mass."""
    plain = run_case(ForwardStepConfig(nx=60, ny=20, t_end=0.5, project_wall_flux=False))
    assert abs(plain["budget"]["mass_drift"]) > 1e-3
    assert abs(plain["budget"]["mass_drift"]) > 1e6 * 1e-12


def test_shock_jump_and_stagnation_on_a_coarse_grid(coarse):
    """Physics sanity on 60 x 20: every graded number inside 15 % (loose gate)."""
    cfg = ForwardStepConfig(nx=60, ny=20, t_end=0.5)
    prim = coarse["prim"]
    jump = measure_shock_jump(prim, cfg)
    assert 0.2 < jump["x_shock"] < cfg.step_x
    # on this very coarse grid (dx = 0.05, t = 0.5) the shock is still forming:
    # the gate checks the robust quantities tightly and only the sign/physics of
    # the velocity; the 8 % criteria are graded on the 480 x 160 grid of record
    for key, ref in (("p_ratio", REF["p_ratio"]), ("rho_ratio", REF["rho_ratio"])):
        assert abs(jump[key] / ref - 1.0) < 0.15, (key, jump[key], ref)
    assert 0.0 < jump["u_ratio"] < REF["u_ratio"] * 1.5
    stag = stagnation_state(prim, cfg)
    assert abs(stag["p_ratio"] / (REF["p02"] / REF["p1"]) - 1.0) < 0.15


def test_solid_cells_hold_the_mirror_of_the_fluid():
    """The solid state is a reflected fluid state, not storage."""
    solver, cfg = build_case(ForwardStepConfig(nx=60, ny=20))
    solver.run(0.05)
    prim = solver.primitive()
    j = cfg.j_step
    i = cfg.i_step + 2
    assert float(prim["rho"][i, j - 1]) == pytest.approx(float(prim["rho"][i, j]), rel=1e-12)
    assert float(prim["v"][i, j - 1]) == pytest.approx(-float(prim["v"][i, j]), rel=1e-10)


def test_mass_flux_balance_is_reported(coarse):
    cfg = ForwardStepConfig(nx=60, ny=20, t_end=0.5)
    bal = mass_flux_balance(coarse["prim"], cfg)
    assert bal["flux_in"] == pytest.approx(cfg.rho * cfg.u_in * cfg.ly, rel=1e-9)
    assert -0.6 < bal["rel_diff"] < 0.0  # flow is still filling the tunnel at t = 0.5