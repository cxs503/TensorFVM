"""Seakeeping evidence layer: hydrostatics, wave decomposition, reference RAO.

The tests are anchored on quantities that are known independently of the code
under test:

* the hydrostatic stiffness is the second derivative of the hydrostatic
  potential energy, which is evaluated from the rotated wetted polygon alone;
* the 2-D far-field drift force must reduce to ``rho g a^2 / 4`` for a plane
  progressive wave (deep water radiation stress) and to the momentum balance
  ``(rho g/2)(a_i^2 + a_r^2 - a_t^2)(2n - 1/2)`` in finite depth;
* the Goda two-point split must recover a synthetic reflected amplitude exactly;
* the frequency-domain RAO must reproduce the single-degree-of-freedom closed
  form when the coupling is switched off.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.wave_tank.theory import GRAVITY
from tensorfvm.wave_tank_td import seakeeping as sk
from tensorfvm.wave_tank_td.potential2d import hydrostatic_restoring

SECTION = sk.floating_strip(1.2, 0.2, freeboard=0.05)


# ------------------------------------------------------------- hydrostatics
def test_stiffness_formula_matches_potential_energy_hessian():
    analytic = sk.hydrostatic_stiffness(SECTION)
    numeric = sk.potential_energy_stiffness(SECTION.polygon, SECTION.mass, SECTION.ref)
    assert abs(numeric["C33"] - analytic["C33"]) / analytic["C33"] < 1e-5
    assert abs(numeric["C55"] - analytic["C55"]) / analytic["C55"] < 1e-5
    assert abs(numeric["C35"]) < 1e-3 * analytic["C55"]


def test_metacentric_height_of_box_section():
    """GM = I_wl/nabla - BG for a box; K55 = rho g nabla GM."""
    sec = SECTION
    bm = sec.length ** 2 / (12.0 * sec.draft)
    bg = sec.cg[1] - (-0.5 * sec.draft)
    expected = sk.SECTION_DENSITY * GRAVITY * sec.wetted_area * (bm - bg)
    out = sk.hydrostatic_stiffness(sec)
    assert abs(out["C55"] - expected) / expected < 1e-12
    assert out["C33"] == pytest.approx(sk.SECTION_DENSITY * GRAVITY * sec.length)


def test_potential_flow_reference_now_uses_the_energy_hessian():
    """Regression: ``potential2d`` used to drop the buoyancy-centre term.

    ``C55 = rho g (I_wl - nabla z_G)`` doubles the true value for this section
    (294.2 vs 147.1 N m/rad on the 0.6 x 0.2 m box); the corrected expression
    ``rho g (I_wl + nabla (z_B - z_G))`` must agree with the independent
    potential-energy Hessian.
    """
    sec = sk.floating_strip(0.6, 0.2, freeboard=0.05)
    ref = hydrostatic_restoring(sec.polygon)
    num = sk.potential_energy_stiffness(sec.polygon, sec.mass, sec.ref)
    old_wrong = sk.SECTION_DENSITY * GRAVITY * (sec.length ** 3 / 12.0
                                                - sec.wetted_area * sec.cg[1])
    assert abs(ref["C55"] - num["C55"]) / num["C55"] < 1e-6
    # the trap really is far off: the old expression missed the z_B term
    assert abs(ref["C55"] - old_wrong) / ref["C55"] > 0.5


def test_zero_freeboard_section_has_no_heave_restoring_once_submerged():
    """The degenerate section that made the first campaign node diverge.

    With the deck exactly on the waterline the displaced area cannot grow when
    the body sinks, so ``d^2U/dh^2`` vanishes for ``h < 0`` while the linear
    waterplane theory still reports ``rho g B``.  A floating section therefore
    needs a freeboard (this is the geometry gate, not a numerical detail).
    """
    degenerate = sk.floating_strip(0.6, 0.2, freeboard=0.0)
    c33 = sk.hydrostatic_stiffness(degenerate)["C33"]
    h = 1e-4

    def curvature(h0, step):
        return (sk.potential_energy(degenerate.polygon, degenerate.mass, h0 + step, 0.0, degenerate.ref)
                - 2 * sk.potential_energy(degenerate.polygon, degenerate.mass, h0, 0.0, degenerate.ref)
                + sk.potential_energy(degenerate.polygon, degenerate.mass, h0 - step, 0.0, degenerate.ref)) / step ** 2

    # strictly below the waterline the displaced area cannot grow -> no restoring
    assert abs(curvature(-3 * h, h)) < 0.02 * c33
    # above it the waterplane does the work
    assert curvature(3 * h, h) > 0.98 * c33


# ---------------------------------------------------------------- wave field
def test_deep_water_drift_reduces_to_radiation_stress():
    a = 0.02
    f = sk.far_field_drift_2d(a, 0.0, 0.0, k=1e3, depth=1.0)
    assert f == pytest.approx(0.25 * sk.SECTION_DENSITY * GRAVITY * a * a, rel=1e-9)


def test_finite_depth_factor_and_energy_identity():
    k, depth = 2.0, 1.0
    kh = k * depth
    n = 0.5 * (1.0 + 2 * kh / math.sinh(2 * kh))
    assert sk.wave_energy_factor(k, depth) == pytest.approx(2 * n - 0.5, rel=1e-12)
    jm = sk.gerritsma_meijers_drift(0.03, 0.01, 0.028, k, depth)
    assert jm["momentum_balance"] == pytest.approx(
        jm["energy_content"] * (2 * n - 0.5), rel=1e-12)
    # the factor-of-two trap: energy density is not a force
    assert jm["ratio_energy_to_force"] == pytest.approx(1.0 / (2 * n - 0.5), rel=1e-12)


def test_goda_split_recovers_a_synthetic_reflection():
    from tensorfvm.wave_tank.theory import goda_two_point

    k, x1, x2 = 3.1, 0.7, 0.95
    a_i, a_r = 1.0, 0.05
    h1 = complex(a_i * np.exp(1j * k * x1) + a_r * np.exp(-1j * k * x1))
    h2 = complex(a_i * np.exp(1j * k * x2) + a_r * np.exp(-1j * k * x2))
    aI, aR = goda_two_point(x1, x2, h1, h2, k)
    assert abs(aI - a_i) < 1e-10
    assert abs(aR - a_r) < 1e-10


# --------------------------------------------------------------- reference
def test_reference_rao_matches_sdof_closed_form():
    omega, mass, added, b, k_stiff, exc, amp = 4.0, 240.0, 100.0, 50.0, 11768.0, 100.0, 0.02
    out = sk.reference_rao(omega, mass, 1.0, 0.0,
                           added_mass=[[added, 0.0], [0.0, 1e9]],
                           damping=[[b, 0.0], [0.0, 1e9]],
                           stiffness=[[k_stiff, 0.0], [0.0, 1e9]],
                           excitation=[exc + 0.0j, 0.0j], wave_amp=amp, k=1.0)
    expect = exc / math.hypot(k_stiff - omega ** 2 * (mass + added), omega * b)
    assert out["rao_heave"] == pytest.approx(expect / amp, rel=1e-10)


def test_natural_frequencies_of_the_reference_are_the_ratio_used_by_the_gate():
    omega, mass, inertia, added, k33, k55 = 4.0, 240.0, 31.4, [[500.0, 0.0], [0.0, 30.0]], 11768.0, 1353.0
    out = sk.reference_rao(omega, mass, inertia, 0.0, added_mass=added,
                           damping=[[10.0, 0.0], [0.0, 10.0]],
                           stiffness=[[k33, 0.0], [0.0, k55]],
                           excitation=[1.0 + 0j, 0.0j], wave_amp=0.02, k=1.0)
    assert out["natural_heave"] == pytest.approx(math.sqrt(k33 / (mass + 500.0)))
    assert out["natural_pitch"] == pytest.approx(math.sqrt(k55 / (inertia + 30.0)))


def test_gci_two_grid_is_zero_for_identical_grids():
    assert sk.gci_two_grid(0.5, 0.5)["gci_fine"] == 0.0
    g = sk.gci_two_grid(0.5, 0.55, ratio=1.4, order=2.0)
    assert g["gci_fine"] == pytest.approx(1.25 * 0.1 / (1.4 ** 2 - 1.0))


# ------------------------------------------------------------ force integrator
def test_contour_pressure_integrator_agrees_with_cut_cell_in_still_water():
    """Both force integrators must agree in the static limit (no flow)."""
    from tensorfvm.wave_tank_td import SigmaWaveTank, TankConfig

    sec = sk.floating_strip(0.6, 0.2, freeboard=0.05, x0=3.0)
    tank = SigmaWaveTank(TankConfig(length=6.0, depth=1.0, nx=60, ns=14))
    tank.set_body(sec.polygon, mass=sec.mass, inertia=sec.inertia, ref=sec.ref)
    weight = sec.mass * GRAVITY
    for heave in (-0.01, 0.0, 0.01):
        tank.body["xi"][:] = [heave, 0.0]
        tank._build_grid()
        tank._update_mask()
        fx_cut, fy_cut, m_cut = tank.body_force()
        fx_c, fy_c, m_c = tank.body_force_interpolated()
        assert abs(fx_c) < 1e-9 and abs(fx_cut) < 1e-9
        assert fy_c == pytest.approx(fy_cut, rel=2e-3, abs=1e-6)
        assert m_c == pytest.approx(m_cut, rel=2e-3, abs=1e-6)
        # exact hydrostatic slope: dF/dh = -rho g B
        assert (fy_c - weight) == pytest.approx(
            -sk.SECTION_DENSITY * GRAVITY * sec.length * heave, rel=5e-3, abs=1.0)


def test_seakeeping_case_smoke():
    """One very small run: the machinery returns finite, physical numbers."""
    out = sk.run_seakeeping_case(amp=0.01, period=1.0, nx=60, ns=12, length=6.0,
                                 spp=40, periods=3.0,
                                 section=sk.floating_strip(0.6, 0.2, 0.05, x0=2.5),
                                 motion="fixed", name="smoke")
    for key in ("rao_heave", "rao_pitch"):
        assert math.isfinite(out["rao"][key])
    assert out["reflection"]["C_R"] >= 0.0
    assert out["numerics"]["div_residual_max"] < 1e-6
    assert math.isfinite(out["drift"]["far_field"])