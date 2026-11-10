"""M3 Rayleigh-Plesset and cavitation void-transport verification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.two_phase.cavitation_transport import (
    BubbleConfig,
    CavitationTransportConfig,
    free_oscillation,
    integrate_bubble,
    linear_damping,
    linear_frequency,
    linear_room_frequency,
    measured_frequency,
    plesset_solution,
    rayleigh_growth,
    solve_void_transport,
)


def test_frequencies_reduce_to_the_classical_forms():
    cfg = BubbleConfig(r0=1.0e-3)          # 1 mm bubble: surface tension is small
    omega = linear_frequency(cfg)
    room = math.sqrt(3.0 * cfg.gamma * cfg.p_gas0 / (cfg.rho * cfg.r0 ** 2))
    assert omega == pytest.approx(room, rel=0.05)
    # the surface-tension correction must lower the frequency
    assert omega < room or abs(omega - room) / room < 1e-3
    tiny = BubbleConfig(r0=1.0e-6)         # surface tension dominates
    assert linear_frequency(tiny) < linear_room_frequency(tiny)
    assert linear_damping(cfg) == pytest.approx(
        2.0 * cfg.mu / (cfg.rho * cfg.r0 ** 2))


def test_small_amplitude_oscillation_matches_plesset():
    cfg = BubbleConfig()
    t, radius = free_oscillation(cfg, eps=1.0e-3, n_periods=6.0)
    reference = plesset_solution(cfg, t, 1.0e-3)
    deviation = float(np.max(np.abs(radius - reference)) / cfg.r0)
    assert deviation <= 0.05              # J3 gate
    assert deviation < 1e-3               # measured: ~3e-6
    frequency = measured_frequency(t, radius)
    assert abs(frequency - linear_frequency(cfg)) / linear_frequency(cfg) <= 0.05


def test_oscillation_damping_matches_the_viscous_rate():
    cfg = BubbleConfig()
    t, radius = free_oscillation(cfg, eps=2.0e-3, n_periods=8.0)
    peaks = [i for i in range(1, radius.size - 1)
             if radius[i] > radius[i - 1] and radius[i] >= radius[i + 1]]
    peak_t = t[peaks]
    amplitude = (radius[peaks] - cfg.r0) / cfg.r0
    slope = np.polyfit(peak_t, np.log(amplitude), 1)[0]
    assert slope == pytest.approx(-linear_damping(cfg), rel=0.05)


def test_zero_pressure_difference_control_is_static():
    cfg = BubbleConfig()
    t, radius = integrate_bubble(cfg, cfg.r0,
                                 4.0 * 2.0 * math.pi / linear_frequency(cfg))
    assert float(np.max(np.abs(radius - cfg.r0)) / cfg.r0) <= 1e-12


def test_rayleigh_growth_is_the_late_time_asymptote():
    vapour = BubbleConfig(r0=1.0e-3, p_gas_override=1.0)
    t, radius = integrate_bubble(vapour, vapour.r0, 2.0e-2, p_inf=1.0e3,
                                 n_samples=401)
    reference = rayleigh_growth(vapour, t, 1.0e3)
    mask = reference > 5.0 * vapour.r0
    assert np.any(mask)
    assert float(np.max(np.abs(radius[mask] - reference[mask])
                        / reference[mask])) < 0.1


def test_void_transport_conserves_bubbles_and_closes_the_mass_budget():
    cfg = CavitationTransportConfig(n_cells=80)
    out = solve_void_transport(cfg)
    assert abs(out["nb_total"] - out["nb_total_initial"]) / out["nb_total_initial"] \
        <= 1e-10
    change = out["gas_mass"] - out["gas_mass_initial"]
    assert change > 0.0                    # bubbles do grow
    residual = abs(change - out["source_total"]) / abs(change)
    assert residual <= 1e-10
    assert np.all(out["void_fraction"] >= 0.0)
    assert float(np.max(out["void_fraction"])) < 0.5
    # monotone growth towards the local equilibrium radius
    assert np.all(out["radius"] >= 1.0e-4 * 0.999)


def test_transport_without_source_is_pure_advection():
    cfg = CavitationTransportConfig(n_cells=80, source="none")
    out = solve_void_transport(cfg)
    assert out["source_total"] == 0.0
    assert abs(out["gas_mass"] - out["gas_mass_initial"]) <= 1e-12
    # the pulse has moved downstream but not left the duct
    centre = float(np.sum(out["z"] * out["n_bubbles"])
                   / max(float(np.sum(out["n_bubbles"])), 1e-30))
    assert centre > cfg.pulse_centre


def test_invalid_bubble_configuration_is_rejected():
    with pytest.raises(ValueError):
        BubbleConfig(p_inf0=1.0e3, p_vapour=2.3e3).validate()
    with pytest.raises(ValueError):
        CavitationTransportConfig(source="magic").validate()