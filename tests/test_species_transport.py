"""Tests for the species-transport / mixture-fraction module and its verification driver."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.special import erf, erfc, erfinv

from tensorfvm import species as S
from tensorfvm.verification import species_transport as V


ST = S.Stoichiometry()


# ---------------------------------------------------------------- chemistry algebra
def test_molar_masses_and_element_fractions():
    assert S.molar_mass("CH4") == pytest.approx(16.043, abs=1e-3)
    assert S.molar_mass("H2O") == pytest.approx(18.015, abs=1e-3)
    Y = {"CH4": 1.0}
    z = S.element_mass_fractions(Y)
    assert sum(z.values()) == pytest.approx(1.0, abs=1e-14)
    assert z["C"] == pytest.approx(12.011 / 16.043, rel=1e-14)


def test_formation_enthalpy_units():
    """CH4 must be -4.667 MJ/kg: the J/mol -> J/kg factor is a real trap."""
    assert S.CNUB["CH4"] == pytest.approx(-4.667e6, rel=2e-3)
    assert S.CNUB["H2O"] == pytest.approx(-13.42e6, rel=2e-3)
    assert S.CNUB["CO2"] == pytest.approx(-8.94e6, rel=2e-3)


def test_bilger_beta_vanishes_at_stoichiometry():
    assert abs(S.bilger_beta(ST.stoichiometric_products())) < 1e-15


def test_stoichiometric_mixture_fraction_matches_literature():
    assert ST.Z_st == pytest.approx(0.055, rel=1e-2)
    # two independent routes (element scalar vs coupling function)
    assert ST.Z_st == pytest.approx(ST.Z_st_coupling, rel=1e-12)
    assert ST.mass_stoichiometric_ratio == pytest.approx(3.989, rel=1e-3)


def test_burke_schumann_endpoints_and_sum():
    bs = S.burke_schumann(np.array([0.0, 1.0]), ST)
    assert bs["Y_O"][0] == pytest.approx(0.232, rel=1e-14)     # pure air
    assert bs["Y_F"][0] == pytest.approx(0.0, abs=1e-15)
    assert bs["Y_F"][1] == pytest.approx(1.0, rel=1e-14)       # pure fuel
    assert bs["Y_O"][1] == pytest.approx(0.0, abs=1e-15)
    Z = np.linspace(0.0, 1.0, 51)
    b = S.burke_schumann(Z, ST)
    total = b["Y_F"] + b["Y_O"] + b["Y_CO2"] + b["Y_H2O"] + b["Y_N2"]
    assert np.max(np.abs(total - 1.0)) < 1e-14


def test_burke_schumann_species_name_view():
    """The flat Y_F/Y_O keys are shorthands; the Y view is species-indexed."""
    b = S.burke_schumann(np.array([0.02]), ST)
    assert set(b["Y"]) == {"CH4", "O2", "CO2", "H2O", "N2"}
    assert b["Y"]["O2"][0] == pytest.approx(float(b["Y_O"][0]))


def test_burke_schumann_rejects_out_of_range_Z():
    with pytest.raises(ValueError):
        S.burke_schumann(np.array([-0.1]), ST)
    with pytest.raises(ValueError):
        S.burke_schumann(np.array([1.1]), ST)


def test_mixture_fraction_is_a_conserved_scalar():
    """Z must be invariant under the single-step reaction to round-off."""
    sc = S.BilgerScalar({"CH4": 1.0}, {"O2": 0.232, "N2": 0.768})
    for z in (0.02, 0.055, 0.2):
        fresh = {"CH4": z, "O2": 0.232 * (1.0 - z), "N2": 0.768 * (1.0 - z)}
        burnt = {k: float(v[0]) for k, v in
                 S.burke_schumann(np.array([z]), ST)["Y"].items()}
        assert sc(fresh) == pytest.approx(z, abs=1e-14)
        assert sc(burnt) == pytest.approx(z, abs=1e-14)


# ---------------------------------------------------------------- fluxes
def test_diffusive_flux_zero_sum_needs_correction():
    Y = {"A": np.array([0.5]), "B": np.array([0.3]), "C": np.array([0.2])}
    grad = {"A": np.array([1.0]), "B": np.array([-2.0]), "C": np.array([0.5])}
    D = {"A": 1e-5, "B": 3e-5, "C": 7e-5}
    jc = S.diffusive_flux(Y, grad, D, 1.0, correct=True)
    ju = S.diffusive_flux(Y, grad, D, 1.0, correct=False)
    scale = sum(abs(v)[0] for v in ju.values())
    assert abs(sum(jc.values())[0]) / scale < 1e-14
    assert abs(sum(ju.values())[0]) / scale > 1e-2


def test_correction_guard_rejects_incomplete_mixture():
    """A single scalar must not be "corrected": that multiplies flux by (1-Y)."""
    with pytest.raises(ValueError, match="complete species set"):
        S.diffusive_flux({"A": np.array([0.4])}, {"A": np.array([1.0])},
                         {"A": 1e-5}, 1.0, correct=True)
    # the same call is fine (and correct) without the correction
    j = S.diffusive_flux({"A": np.array([0.4])}, {"A": np.array([1.0])},
                         {"A": 1e-5}, 1.0, correct=False)
    assert j["A"][0] == pytest.approx(-1e-5, rel=1e-14)


# ---------------------------------------------------------------- steady solver
def test_pure_diffusion_is_exact():
    n, L, D = 40, 1.0, 1e-3
    phi = S.solve_steady_species_1d(n=n, L=L, D=D,
                                    bc_left=("dirichlet", 0.0),
                                    bc_right=("dirichlet", 1.0))
    assert np.max(np.abs(phi - (np.arange(n) + 0.5) * L / n)) < 1e-12


def test_zero_gradient_bc_gives_uniform_field():
    phi = S.solve_steady_species_1d(n=20, L=1.0, D=1e-3,
                                    bc_left=("zerogradient", None),
                                    bc_right=("dirichlet", 1.0))
    assert np.max(np.abs(phi - 1.0)) < 1e-14


def test_steady_advection_diffusion_second_order():
    u, D, L = 0.01, 1e-4, 0.1
    errs = []
    for n in (100, 200):
        Pe = u * L / D
        x = (np.arange(n) + 0.5) * L / n
        exact = (np.exp(Pe * x / L) - 1.0) / (math.exp(Pe) - 1.0)
        phi = S.solve_steady_species_1d(n=n, L=L, D=D, u=u,
                                        bc_left=("dirichlet", 0.0),
                                        bc_right=("dirichlet", 1.0))
        errs.append(np.linalg.norm(phi - exact) / np.linalg.norm(exact))
    order = math.log(errs[0] / errs[1]) / math.log(2.0)
    assert order > 1.8
    assert errs[-1] < 5e-3


def test_upwind_is_first_order():
    u, D, L = 0.01, 1e-4, 0.1
    errs = []
    for n in (100, 200):
        Pe = u * L / D
        x = (np.arange(n) + 0.5) * L / n
        exact = (np.exp(Pe * x / L) - 1.0) / (math.exp(Pe) - 1.0)
        phi = S.solve_steady_species_1d(n=n, L=L, D=D, u=u, scheme="upwind",
                                        bc_left=("dirichlet", 0.0),
                                        bc_right=("dirichlet", 1.0))
        errs.append(np.linalg.norm(phi - exact) / np.linalg.norm(exact))
    order = math.log(errs[0] / errs[1]) / math.log(2.0)
    assert 0.8 < order < 1.2


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        S.solve_steady_species_1d(n=10, L=1.0, D=-1.0,
                                  bc_left=("dirichlet", 0.0),
                                  bc_right=("dirichlet", 1.0))
    with pytest.raises(ValueError):
        S.solve_steady_species_1d(n=10, L=1.0, D=1e-3, scheme="quick",
                                  bc_left=("dirichlet", 0.0),
                                  bc_right=("dirichlet", 1.0))


# ---------------------------------------------------------------- transient solver
def test_transient_diffusion_matches_erfc():
    D, t, n = 1e-5, 1.0, 200
    L = 8.0 * math.sqrt(D * t)
    dx = L / n
    dt = 0.2 * dx * dx / D
    Y, _ = S.solve_transient_mixture_1d(
        {"A": np.zeros(n)}, L=L, D={"A": D}, dt=dt,
        n_steps=int(round(t / dt)), correct_fluxes=False,
        bc_left={"A": ("dirichlet", 1.0)},
        bc_right={"A": ("zerogradient", None)})
    x = (np.arange(n) + 0.5) * dx
    assert np.max(np.abs(Y["A"] - erfc(x / (2 * math.sqrt(D * t))))) < 5e-3


def test_transient_keeps_sum_Y_and_species_mass():
    D = {"A": 1e-5, "B": 3e-5, "C": 7e-5}
    n, L = 80, 0.05
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    Y0 = {"A": 0.5 + 0.3 * np.tanh((x - 0.02) / 0.002),
          "B": 0.3 + 0.1 * np.tanh((0.02 - x) / 0.002)}
    Y0["C"] = 1.0 - Y0["A"] - Y0["B"]
    dt = 0.1 * dx * dx / max(D.values())
    Y, _ = S.solve_transient_mixture_1d(Y0, L=L, D=D, dt=dt, n_steps=200,
                                        correct_fluxes=True)
    assert np.max(np.abs(Y["A"] + Y["B"] + Y["C"] - 1.0)) < 1e-12
    for k in Y0:                      # closed domain: zero-gradient faces
        assert abs(np.sum(Y[k]) - np.sum(Y0[k])) / np.sum(np.abs(Y0[k])) < 1e-12


def test_transient_history_recording():
    n, L = 20, 0.01
    dt = 1e-3
    Y, hist = S.solve_transient_mixture_1d(
        {"A": np.zeros(n)}, L=L, D={"A": 1e-5}, dt=dt, n_steps=10,
        correct_fluxes=False, store_every=5,
        bc_left={"A": ("dirichlet", 1.0)})
    assert len(hist) == 2
    assert hist[-1][0] == 10


# ---------------------------------------------------------------- chemistry driver
def test_strained_layer_matches_similarity_solution():
    phi, xc, delta = V._strained_layer(400)
    exact = 0.5 * (1.0 + erf(xc / delta))
    assert np.linalg.norm(phi - exact) / np.linalg.norm(exact) < 5e-3


def test_strained_layer_without_transverse_term_is_broken():
    """The quantified trap: the 1-D conservative form alone is not this problem."""
    phi, xc, delta = V._strained_layer(200, transverse=False)
    exact = 0.5 * (1.0 + erf(xc / delta))
    assert np.linalg.norm(phi - exact) / np.linalg.norm(exact) > 0.2


def test_flame_standoff_matches_closed_form():
    c3 = V._c3_layer()
    c4 = V._c4_standoff(c3)
    assert c4["rel_err"] < 1e-2
    assert c4["x_f_closed"] == pytest.approx(
        c3["delta"] * float(erfinv(2.0 * ST.Z_st - 1.0)), rel=1e-14)


def test_driver_all_criteria_pass_and_json_serialisable():
    summary = V.run()
    failed = {k: v for k, v in summary["criteria"].items() if not v["pass"]}
    assert not failed, f"verification criteria failed: {failed}"
    assert summary["all_pass"] is True
    payload = V._public(summary)
    json.dumps(payload)          # must be serialisable for the report pipeline
    text = V.render_report(summary)
    assert "误差判定" in text and "WITHHELD" in text


def test_report_states_withheld_items():
    summary = V.run()
    withheld = summary["withheld"]
    assert "cantera" in withheld["laminar_flame_speed_vs_cantera"].lower()
    assert "2226" in withheld["adiabatic_flame_temperature_vs_literature"]