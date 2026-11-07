"""Acceptance tests for the KVLCC2 PMM virtual-test chain (``pmm_cfd``).

The tests are grouped by the machine-checkable criteria of the campaign node:

* geometry gate (Cb / L / B / T / LCB / Cwp) - must pass before any CFD,
* analytic verification of the section hydrodynamics,
* exact-recovery tests of the derivative extractor on synthetic signals
  (the only way to test an extractor: the answer must be reproduced to machine
  precision when the force signal is synthesised from known derivatives),
* amplitude-sensitivity and quadrature-convergence bookkeeping,
* honest recording of the two criteria that are *not* met (criterion 1 against
  the in-repo SIMMAN table, criterion 2 for the phase), so that a silent
  regression to "passed" is impossible.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tensorfvm.pmm_cfd import (
    KVLCC2Hull,
    apply_strip_model,
    extract_derivatives,
    pmm_kinematics,
    reference_consistency_audit,
    run_pmm_campaign,
    section_added_mass_ellipse,
    spheroid_added_mass,
)


@pytest.fixture(scope="module")
def hull() -> KVLCC2Hull:
    h = KVLCC2Hull()
    h.calibrate()
    return h


@pytest.fixture(scope="module")
def campaign():
    return run_pmm_campaign(amplitudes_over_L=(0.005, 0.01, 0.02),
                            station_counts=(21, 41, 81), n_samples=240)


# ---------------------------------------------------------------------------
# Geometry gate
# ---------------------------------------------------------------------------

class TestGeometryGate:
    def test_principal_dimensions(self, hull):
        assert hull.L == pytest.approx(7.0)
        assert hull.B == pytest.approx(0.936)
        assert hull.T == pytest.approx(0.342)
        assert hull.Cb == pytest.approx(0.708)

    def test_block_coefficient_verified(self, hull):
        ver = hull.integrals(nx=241, nz=121)
        assert ver.Cb == pytest.approx(0.708, rel=5e-3)
        assert ver.volume == pytest.approx(0.708 * 7.0 * 0.936 * 0.342, rel=5e-3)

    def test_lcb_and_waterplane(self, hull):
        ver = hull.integrals(nx=241, nz=121)
        assert ver.lcb_fraction == pytest.approx(-0.01688, abs=5e-4)
        assert ver.cwp == pytest.approx(0.9005, rel=5e-3)

    def test_inrepo_offsets_would_fail_the_gate(self):
        """The legacy offset table gives Cb = 0.636: it cannot pass the gate."""
        from tensorfvm.kvlcc2_geometry import KVLCC2HullOffsets, KVLCC2PrincipalDimensions

        offs = KVLCC2HullOffsets()
        dims = KVLCC2PrincipalDimensions()
        f = offs.get_full_offsets(41, 21)
        x = np.linspace(0.0, dims.L_pp, 41)
        z = np.linspace(-dims.T, 0.0, 21)
        vol = float(np.trapezoid(np.trapezoid(2.0 * f * dims.B / 2, z, axis=0), x))
        cb = vol / (dims.L_pp * dims.B * dims.T)
        assert abs(cb - dims.Cb) / dims.Cb > 0.05

    def test_hull_fairness(self, hull):
        """Half-breadth must be monotone-decreasing towards both perpendiculars."""
        xi = np.linspace(0.0, 1.0, 101)
        wl = np.array([hull.wl_half_breadth_fraction(v) for v in xi])
        assert wl[0] == 0.0 and wl[-1] == 0.0
        assert np.all(np.diff(wl[15:35]) >= -1e-12)
        assert np.all(np.diff(wl[65:85]) <= 1e-12)
        assert wl.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Analytic verification of the section hydrodynamics
# ---------------------------------------------------------------------------

class TestSectionHydrodynamics:
    def test_section_added_mass_is_rho_pi_T2(self):
        assert section_added_mass_ellipse(beam=0.936, draught=0.342, rho=998.0) == \
            pytest.approx(998.0 * np.pi * 0.342 ** 2, rel=1e-12)

    def test_sphere_limit(self):
        """A sphere (a -> b) must reproduce the classical 0.5 rho V added mass."""
        vals = spheroid_added_mass(1.0, 1.0 - 1e-6)
        assert vals["m22_over_displaced"] == pytest.approx(0.5, rel=2e-3)
        near_sphere = spheroid_added_mass(1.0, 0.90)
        assert near_sphere["m22_over_displaced"] == pytest.approx(0.52, abs=0.03)

    def test_slender_limit(self):
        """A very slender spheroid approaches the 2D slender-body limit m/(rho V) -> 1."""
        vals = spheroid_added_mass(10.0, 0.5)
        assert vals["m22_over_displaced"] > 0.9


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

class TestKinematics:
    def test_pure_sway_uniform_velocity(self):
        t = np.linspace(0.0, 10.0, 101)
        kin = pmm_kinematics(t, amplitude=0.05, omega=2 * np.pi / 10.0,
                             length=7.0, mode="sway")
        assert np.allclose(kin["r"], 0.0)
        assert kin["v_mid"].max() == pytest.approx(0.05 * 2 * np.pi / 10.0, rel=1e-9)

    def test_pure_yaw_zero_mid_velocity(self):
        t = np.linspace(0.0, 10.0, 101)
        kin = pmm_kinematics(t, amplitude=np.radians(3.0), omega=2 * np.pi / 10.0,
                             length=7.0, mode="yaw")
        assert np.allclose(kin["v_mid"], 0.0)
        assert kin["r"].max() == pytest.approx(np.radians(3.0) * 2 * np.pi / 10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Derivative extraction: exact recovery from synthetic signals
# ---------------------------------------------------------------------------

class TestDerivativeExtraction:
    def _synthetic(self, *, mode, Yv, Nv, Yvdot, Nvdot, Yr, Nr, Yrdot, Nrdot,
                   amp, omega, U, rho, L, T, n_cycles=4.0, n=800):
        t = np.linspace(0.0, n_cycles * 2 * np.pi / omega, n + 1)
        if mode == "sway":
            v = amp * omega * np.cos(omega * t)
            a = -amp * omega ** 2 * np.sin(omega * t)
            q_y = 0.5 * rho * U ** 2 * L * T
            q_n = 0.5 * rho * U ** 2 * L ** 2 * T
            Y = (Yv * v + Yvdot * a) * q_y
            N = (Nv * v + Nvdot * a) * q_n
        else:
            r = amp * omega * np.cos(omega * t)
            rd = -amp * omega ** 2 * np.sin(omega * t)
            q_y = 0.5 * rho * U ** 2 * L * T
            q_n = 0.5 * rho * U ** 2 * L ** 2 * T
            Y = (Yr * r + Yrdot * rd) * q_y
            N = (Nr * r + Nrdot * rd) * q_n
        return t, Y, N

    def test_sway_recovery(self):
        kw = dict(amp=0.05, omega=2 * np.pi / 10.0, U=0.771, rho=998.0, L=7.0, T=0.342)
        truth = dict(Yv=-0.31, Nv=-0.12, Yvdot=-1.7, Nvdot=-0.22)
        t, Y, N = self._synthetic(mode="sway", Yr=0.0, Nr=0.0, Yrdot=0.0, Nrdot=0.0,
                                  **truth, **kw)
        der = extract_derivatives(t, Y, N, mode="sway", amplitude=kw["amp"],
                                  omega=kw["omega"], U=kw["U"], rho=kw["rho"],
                                  L=kw["L"], T=kw["T"], n_cycles=2.0)
        for key, val in truth.items():
            assert der[key] == pytest.approx(val, rel=1e-6), key

    def test_yaw_recovery(self):
        kw = dict(amp=np.radians(3.0), omega=2 * np.pi / 10.0, U=0.771, rho=998.0,
                  L=7.0, T=0.342)
        truth = dict(Yr=0.03, Nr=-0.06, Yrdot=0.02, Nrdot=-0.4)
        t, Y, N = self._synthetic(mode="yaw", Yv=0.0, Nv=0.0, Yvdot=0.0, Nvdot=0.0,
                                  **truth, **kw)
        der = extract_derivatives(t, Y, N, mode="yaw", amplitude=kw["amp"],
                                  omega=kw["omega"], U=kw["U"], rho=kw["rho"],
                                  L=kw["L"], T=kw["T"], n_cycles=2.0)
        for key, val in truth.items():
            assert der[key] == pytest.approx(val, rel=1e-6), key

    def test_noise_rejection(self):
        """A harmonic at 3x the PMM frequency must not leak into the derivatives."""
        kw = dict(amp=0.05, omega=2 * np.pi / 10.0, U=0.771, rho=998.0, L=7.0, T=0.342)
        t, Y, N = self._synthetic(mode="sway", Yv=-0.31, Nv=-0.12, Yvdot=-1.7,
                                  Nvdot=-0.22, Yr=0.0, Nr=0.0, Yrdot=0.0, Nrdot=0.0, **kw)
        Y = Y + 0.30 * np.hypot(Y.max(), 1.0) * np.sin(3 * kw["omega"] * t)
        der = extract_derivatives(t, Y, N, mode="sway", amplitude=kw["amp"],
                                  omega=kw["omega"], U=kw["U"], rho=kw["rho"],
                                  L=kw["L"], T=kw["T"], n_cycles=2.0)
        assert der["Yv"] == pytest.approx(-0.31, rel=0.02)


# ---------------------------------------------------------------------------
# Campaign bookkeeping / honest acceptance flags
# ---------------------------------------------------------------------------

class TestCampaign:
    def test_runs_complete(self, campaign):
        n = (len(campaign["conditions"]["amplitudes_over_L"])
             * 2 * len(campaign["conditions"]["station_counts"]))
        assert len(campaign["runs"]) == n

    def test_geometry_gate_passed(self, campaign):
        assert campaign["criteria"]["geometry_gate"] is True

    def test_added_mass_matches_strip_integral(self, campaign, hull):
        """Yvdot from the campaign must equal -int m22 dx / (0.5 rho U^2 L T)."""
        U = campaign["conditions"]["U"]
        rho = campaign["conditions"]["rho"]
        run = next(r for r in campaign["runs"] if r["mode"] == "sway"
                   and r["n_stations"] == 81 and r["A_over_L"] == 0.01)
        sig = apply_strip_model(hull, np.array([0.0]), mode="sway", amplitude=0.07,
                                omega=2 * np.pi / 10.0, U=U, rho=rho, n_stations=81)
        analytic = -float(np.trapezoid(sig["m22"], sig["x_stations"])) / \
            (0.5 * rho * U ** 2 * hull.L * hull.T)
        assert run["derivatives"]["Yvdot"] == pytest.approx(analytic, rel=2e-2)

    def test_amplitude_sensitivity_recorded(self, campaign):
        sens = campaign["amplitude_sensitivity"]
        assert set(sens) == {"Yv", "Nv", "Yvdot", "Nvdot", "Yr", "Nr", "Yrdot", "Nrdot"}
        # the inertial derivatives must be amplitude independent (linear model)
        assert sens["Yvdot"] < 1e-6
        assert sens["Nrdot"] < 1e-6
        # the drag derivatives cannot be, because the drag is quadratic
        assert sens["Yv"] > 0.05

    def test_quadrature_convergence_recorded(self, campaign):
        assert campaign["quadrature_convergence"]["Yvdot"] < 0.03

    def test_json_serialisable(self, campaign, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(campaign, default=float))
        data = json.loads(path.read_text())
        assert data["hull"]["Cb_computed"] == pytest.approx(0.708, rel=5e-3)

    def test_honest_flags(self, campaign):
        """Criteria that are *not* met must stay recorded as not met."""
        crit = campaign["criteria"]
        assert crit["C1_derivatives_within_15pct_of_simman"] is False
        assert crit["C2_phase_within_5deg"] is False
        assert crit["reference_table_self_consistent"] is False
        assert crit["reference_added_mass_plausible"] is False

    def test_evidence_grades_present(self, campaign):
        ev = campaign["evidence"]
        assert "REAL" in ev["section_added_mass"]
        assert "EMPIRICAL" in ev["viscous_damping"]
        assert "NOT_SUPPORTED" in ev["3d_hull_effects"]
        assert "WITHHELD" in ev["phase_vs_experiment"]


# ---------------------------------------------------------------------------
# Reference-data audit
# ---------------------------------------------------------------------------

class TestReferenceAudit:
    def test_derivative_table_inconsistent_with_drift_table(self):
        audit = reference_consistency_audit()
        assert audit["consistent"] is False
        assert abs(audit["drift_slope_Yv"] / audit["table_Yv"]) > 2.0
        assert abs(audit["drift_slope_Nv"] / audit["table_Nv"]) > 2.0

    def test_reference_added_mass_implausible(self, campaign):
        audit = campaign["reference_audit"]
        # potential theory: m22 / (rho * nabla) is 0.5 - 1.0 for a tanker
        assert audit["m22_reference_over_rho_nabla"] < 0.2
        assert audit["m22_slender_over_rho_nabla"] > 0.5