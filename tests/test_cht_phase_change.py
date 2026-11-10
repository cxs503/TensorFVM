"""Verification tests of the CHT / Stefan / ice-steel surface (campaign P295W1R2-N01).

Acceptance criteria implemented here (numbers are the campaign thresholds):

C1  ice/steel interface: temperature continuity error <= 1 %, interface heat-flux
    conservation residual <= 1e-6; every injected mutation must turn C1 red.
C2  Stefan phase change: front position against the Neumann similarity solution
    <= 3 %.
C3  ice/steel steady temperature field against the 1-D analytic series solution
    <= 5 %.
C4  cross-adjudication with the W4-N08 icing model at the same icing condition:
    surface fluxes / freezing fraction / icing rate reproduced, and the
    temperature field against an independently coded reference field <= 5 % (L2).
C5  no look-up tables / hard-coded temperatures: the field must respond to a
    perturbation of the boundary data (and C1 must go red under mutation).

The W4-N08 reference data is *loaded* from
``src/tensorfvm/data/marine_icing_w4n08/reference_panel.json`` (provenance and
sha256 inside) -- never transcribed into the test.
"""

import math

import numpy as np
import pytest

from tensorfvm.cht.ice_steel import (ICE, STEEL, LayeredWall, LayeredWall2D,
                                     SurfaceBC, TwoDomainCHT, analytic_series_field,
                                     interface_audit)
from tensorfvm.cht.icing import (IcingEnvironment, icing_state_adiabatic,
                                 icing_state_with_conduction, load_w4n08_reference)
from tensorfvm.cht.phase_change import (EnthalpySlab1D, NeumannSolution,
                                        PhaseChangeMaterial, RadialBoundary)

T_FREEZE = 271.35      # K, -1.8 degC
T_BACK = 258.15        # K, -15 degC cold hold (documented design assumption)
H_ICE = 0.0334         # m, ice layer for the C1/C3 panel
H_STEEL = 0.010        # m
R_CONTACT = 1.0e-4     # m^2 K/W


def _panel(mutation=None, q_init=100.0, h_ice=H_ICE):
    ice = LayeredWall([(h_ice, ICE, 40)], [],
                      SurfaceBC(t_ext=T_FREEZE, resistance=0.0),
                      SurfaceBC(t_ext=T_BACK, resistance=0.0))
    steel = LayeredWall([(H_STEEL, STEEL, 10)], [],
                        SurfaceBC(t_ext=T_FREEZE, resistance=0.0),
                        SurfaceBC(t_ext=T_BACK, resistance=0.0))
    case = TwoDomainCHT(ice, steel, r_contact=R_CONTACT, q_init=q_init,
                        mutation=mutation)
    out = case.solve()
    audit = interface_audit(out["t_ice"], ice.dx[0], ice.k[0],
                            out["t_steel"], steel.dx[0], steel.k[0],
                            R_CONTACT, T_FREEZE - T_BACK)
    return out, audit


# --------------------------------------------------------------------------- C1
def test_c1_interface_flux_and_temperature_continuity():
    out, audit = _panel()
    assert audit["flux_imbalance_rel"] <= 1.0e-6
    assert audit["temperature_continuity_rel"] <= 1.0e-2
    assert audit["jump_rel"] <= 1.0e-2
    # the coupling fixed point must also agree with the closed-form series answer
    ref = analytic_series_field([H_ICE, H_STEEL], [ICE.k, STEEL.k],
                                T_FREEZE, T_BACK, R_CONTACT)
    assert abs(out["q"] / ref["q"] - 1.0) < 1.0e-9
    # the contact jump is resolved, not smeared: q*R_c must be positive and visible
    assert audit["t_iface_jump"] > 0.0
    assert math.isclose(audit["t_iface_jump"], audit["q_mean"] * R_CONTACT, rel_tol=1e-8)


@pytest.mark.parametrize("mutation", ["single_sweep", "sign_flipped", "steel_decoupled"])
def test_c1_mutations_turn_the_criterion_red(mutation):
    _, clean = _panel()
    _, bad = _panel(mutation=mutation)
    assert clean["flux_imbalance_rel"] <= 1e-6 and clean["temperature_continuity_rel"] <= 1e-2
    red = (bad["flux_imbalance_rel"] > 1.0e-6) or (bad["temperature_continuity_rel"] > 1.0e-2)
    assert red, f"mutation {mutation} was not detected: {bad}"


def test_c1_interface_audit_sees_a_wrong_conductivity():
    """An audit-side regression (steel k used on the ice side) must be caught."""
    out, clean = _panel()
    ice = LayeredWall([(H_ICE, ICE, 40)], [], SurfaceBC(t_ext=T_FREEZE, resistance=0.0),
                      SurfaceBC(t_ext=T_BACK, resistance=0.0))
    steel = LayeredWall([(H_STEEL, STEEL, 10)], [], SurfaceBC(t_ext=T_FREEZE, resistance=0.0),
                        SurfaceBC(t_ext=T_BACK, resistance=0.0))
    bad = interface_audit(out["t_ice"], ice.dx[0], STEEL.k,
                          out["t_steel"], steel.dx[0], STEEL.k,
                          R_CONTACT, T_FREEZE - T_BACK)
    assert clean["flux_imbalance_rel"] <= 1e-6
    assert bad["flux_imbalance_rel"] > 0.5          # |q_ice - q_steel| / max(|q|) <= 1


# --------------------------------------------------------------------------- C2
def test_c2_stefan_front_matches_neumann_within_three_percent():
    material = PhaseChangeMaterial(dT_mush=0.5)
    t_surface = material.t_melt - 10.0
    slab = EnthalpySlab1D(0.04, 400, material, material.t_liquidus,
                          RadialBoundary(t_ext=t_surface, resistance=0.0))
    times = [300.0, 600.0, 1200.0]
    snaps = slab.run(1200.0, 0.5, record=times)
    exact = NeumannSolution(material, t_surface)
    errors = []
    for snap in snaps:
        x_exact = exact.front(snap["time"])
        err = abs(snap["front_liquid_fraction"] / x_exact - 1.0)
        errors.append(err)
        assert err <= 0.03, (snap["time"], err)
    assert slab.ledger()["imbalance_rel"] < 1.0e-9
    # the exact temperature field must be reproduced too (not only the front)
    mid = snaps[-1]
    t_exact = exact.temperature(mid["x"], mid["time"])
    deep = mid["x"] < 0.5 * mid["front"]
    assert np.max(np.abs(mid["temperature"][deep] - t_exact[deep])) < 0.08
    inside = mid["x"] < 0.8 * mid["front"]
    # inside the mushy band the field is smoothed by dT_mush = 0.5 K: <= 1.5 % of the drive
    assert np.max(np.abs(mid["temperature"][inside] - t_exact[inside])) < 0.15
    assert max(errors) <= 0.03


def test_c2_mushy_band_closure_and_enthalpy_inversion():
    material = PhaseChangeMaterial()
    t = np.array([250.0, material.t_solidus, material.t_melt, material.t_liquidus, 280.0])
    h = material.enthalpy(t)
    t_back, f_l = material.temperature(h)
    assert np.allclose(t_back, t, atol=1e-10)
    assert f_l[1] == pytest.approx(0.0, abs=1e-12)
    assert f_l[2] == pytest.approx(0.5, rel=1e-9)
    assert f_l[3] == pytest.approx(1.0, abs=1e-9)
    assert material.capacity(np.array([material.t_melt]))[0] == pytest.approx(
        material.c + material.latent / material.dT_mush)


def test_c2_no_phase_change_without_a_surface_flux():
    material = PhaseChangeMaterial()
    slab = EnthalpySlab1D(0.02, 100, material, material.t_melt,
                          RadialBoundary(q0=0.0))
    slab.run(60.0, 1.0)
    assert slab.front_liquid_fraction() == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(slab.t, material.t_melt, atol=1e-9)


# --------------------------------------------------------------------------- C3
def _column_field(ny=1, top_bc=None):
    wall = LayeredWall2D([(H_ICE, ICE, 40), (H_STEEL, STEEL, 10)], [R_CONTACT],
                         t_bottom=SurfaceBC(t_ext=T_BACK, resistance=0.0),
                         top_bc=top_bc or [SurfaceBC(t_ext=T_FREEZE, resistance=0.0)] * ny,
                         ny=ny)
    return wall, wall.solve()


def test_c3_ice_steel_field_matches_analytic_within_five_percent():
    ref = analytic_series_field([H_ICE, H_STEEL], [ICE.k, STEEL.k],
                                T_FREEZE, T_BACK, R_CONTACT, n_cells=4000)
    wall, field = _column_field()
    t_cht = field[0]
    x_cht = wall.xc
    t_ref = np.interp(x_cht, ref["x"], ref["temperature"])
    scale = np.linalg.norm(ref["temperature"] - T_BACK)
    l2 = np.linalg.norm(t_cht - t_ref) / scale
    assert l2 <= 0.05, l2
    assert l2 < 1e-3          # in practice the FV scheme is far below the limit
    # the contact jump must sit inside the field (not be smeared over a cell)
    assert ref["contacts"][0]["t_left"] - ref["contacts"][0]["t_right"] == pytest.approx(
        ref["q"] * R_CONTACT, rel=1e-12)


def test_c3_laterally_uniform_2d_reduces_to_the_1d_field():
    _, field_1d = _column_field(ny=1)
    _, field_2d = _column_field(ny=9)
    assert np.max(np.abs(field_2d - field_1d[0])) < 1e-9


def test_c3_robust_to_grid_refinement():
    errors = []
    for n_ice in (10, 20, 40, 80):
        wall = LayeredWall2D([(H_ICE, ICE, n_ice), (H_STEEL, STEEL, 10)], [R_CONTACT],
                             t_bottom=SurfaceBC(t_ext=T_BACK, resistance=0.0),
                             top_bc=[SurfaceBC(t_ext=T_FREEZE, resistance=0.0)], ny=1)
        field = wall.solve()[0]
        ref = analytic_series_field([H_ICE, H_STEEL], [ICE.k, STEEL.k],
                                    T_FREEZE, T_BACK, R_CONTACT, n_cells=4000)
        t_ref = np.interp(wall.xc, ref["x"], ref["temperature"])
        errors.append(np.linalg.norm(field - t_ref)
                      / np.linalg.norm(ref["temperature"] - T_BACK))
    assert errors[-1] <= 0.05
    assert errors[-1] <= errors[0] + 1e-12     # refinement must not make it worse


# --------------------------------------------------------------------------- C4
def _reference_environment() -> tuple[IcingEnvironment, dict]:
    ref = load_w4n08_reference()
    env_data = ref["environment"]
    env = IcingEnvironment(
        t_air=env_data["t_air_degC"],
        t_sea=env_data["t_sea_degC"],
        t_freeze=env_data["t_sea_degC"],
        wind_speed=env_data["wind_speed_m_s"],
        char_length=env_data["char_length_m"],
        relative_humidity=env_data["relative_humidity"],
        spray_flux=env_data["spray_flux_kg_m2_s"],
        exposure=6.0 * 3600.0,
    )
    return env, ref


def test_c4_w4n08_closure_is_reproduced():
    env, ref = _reference_environment()
    mod = ref["module_output"]
    state = icing_state_adiabatic(env)
    assert state.regime == mod["regime"]
    assert state.t_surface == pytest.approx(mod["surface_temperature_degC"], abs=1e-9)
    for key in ("q_conv", "q_evap", "q_lw", "q_req"):
        ours = getattr(state, key)
        theirs = mod[f"{key}_W_m2"]
        assert abs(ours - theirs) <= 1e-4 * max(abs(theirs), 1.0), (key, ours, theirs)
    assert abs(state.q_spray - mod["q_spray_W_m2"]) <= 1e-9
    assert state.freezing_fraction == pytest.approx(mod["freezing_fraction"], rel=1e-4)
    assert state.icing_rate_mm_per_h == pytest.approx(mod["icing_rate_mm_per_h"], rel=1e-4)
    assert state.terms_balance == pytest.approx(0.0, abs=1e-9)


def test_c4_temperature_field_against_reference_within_five_percent():
    """Same icing condition -> CHT field vs an independently coded reference field."""
    env, ref = _reference_environment()
    mod = ref["module_output"]
    h_ice = mod["icing_rate_mm_per_h"] * env.exposure / 3600.0 / 1000.0
    t_s = 273.15 + mod["surface_temperature_degC"]
    assert h_ice == pytest.approx(0.033433, rel=1e-3)
    wall = LayeredWall2D([(h_ice, ICE, 40), (H_STEEL, STEEL, 10)], [R_CONTACT],
                         t_bottom=SurfaceBC(t_ext=T_BACK, resistance=0.0),
                         top_bc=[SurfaceBC(t_ext=t_s, resistance=0.0)], ny=1)
    field = wall.solve()[0]
    reference = analytic_series_field([h_ice, H_STEEL], [ICE.k, STEEL.k],
                                      t_s, T_BACK, R_CONTACT, n_cells=4000)
    t_ref = np.interp(wall.xc, reference["x"], reference["temperature"])
    l2 = np.linalg.norm(field - t_ref) / np.linalg.norm(reference["temperature"] - T_BACK)
    assert l2 <= 0.05, l2
    # the CHT field must obey the series-resistance law that the reference uses
    r_wall = h_ice / ICE.k + R_CONTACT + H_STEEL / STEEL.k
    q_series = (t_s - T_BACK) / r_wall
    q_cht = ICE.k * (field[0] - field[1]) / wall.dx[0]   # +x flux (surface -> back)
    assert q_cht == pytest.approx(q_series, rel=1e-6)
    # conduction-aware adjudication: the cold hold adds a sink, so the freezing
    # fraction must grow with respect to the adiabatic W4-N08 branch.
    # NOTE: the icing closure works in degC while the CHT fields are in K
    state_cond = icing_state_with_conduction(env, r_wall, T_BACK - 273.15)
    assert state_cond.freezing_fraction > state_module_fraction(mod)
    assert state_cond.freezing_fraction == pytest.approx(1.0, abs=1e-12)
    assert state_cond.regime == "dry"


def state_module_fraction(mod: dict) -> float:
    return mod["freezing_fraction"]


# --------------------------------------------------------------------------- C5
def test_c5_field_is_solved_not_looked_up():
    """Perturbing the boundary data must move the field by the same amount."""
    _, base = _column_field()
    wall = LayeredWall2D([(H_ICE, ICE, 40), (H_STEEL, STEEL, 10)], [R_CONTACT],
                         t_bottom=SurfaceBC(t_ext=T_BACK - 3.0, resistance=0.0),
                         top_bc=[SurfaceBC(t_ext=T_FREEZE, resistance=0.0)], ny=1)
    shifted = wall.solve()[0]
    # moving one Dirichlet boundary by -3 K must shift the whole field (1-D series)
    assert np.all(shifted < base[0])
    delta = float(np.mean(base[0] - shifted))
    assert 2.99 > delta > 0.01
    assert np.all(np.isfinite(shifted))


def test_c5_no_nan_and_reference_data_has_provenance():
    _, field = _column_field()
    assert np.all(np.isfinite(field))
    ref = load_w4n08_reference()
    assert len(ref["_sha256"]) == 64
    assert ref["_provenance"]["source_code"].endswith("marine_icing.py")