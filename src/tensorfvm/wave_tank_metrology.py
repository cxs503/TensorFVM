"""VOF Numerical Wave Tank Metrology Surface.

This module provides the measurement and analysis framework for the
numerical wave tank campaign node P189-N04. It ties together:

- Wave generation (relaxation zone with Stokes I/II targets)
- Wave absorption (damping zone / sponge)
- Wave probe analysis (FFT, harmonic fit, reflection coefficient)
- Grid convergence (GCI on three-level ladder)
- Volume conservation audit
- Cross-validation with potential flow (added resistance)

All criteria are machine-verifiable through pytest.

References
----------
- Dean & Dalrymple, "Water Wave Mechanics for Engineers and Scientists"
- Goda & Suzuki (1976), "Estimation of incident and reflected waves in random wave trains"
- Fenton (1985), "A fifth-order Stokes theory for steady waves"
- Roache (1997), "Perspective: A Method for Uniform Reporting of Grid Refinement Studies"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .wave_tank_td import SigmaWaveTank, TankConfig, WaveTarget
from .wave_tank.theory import (
    dispersion_k, harmonic_fit, three_probe_wavenumber,
    goda_two_point, added_resistance_energy, GRAVITY,
)
from .wave_tank_td.wave_target import fit_harmonics

__all__ = [
    "WaveTankMetrology",
    "MetrologyResult",
    "run_linear_wave_test",
    "run_stokes2_harmonic_test",
    "run_reflection_coefficient_test",
    "run_gci_study",
    "run_volume_conservation_test",
    "compute_stokes2_theoretical_harmonic_ratio",
]


def compute_stokes2_theoretical_harmonic_ratio(
    k: float, a: float, h: float
) -> float:
    """Theoretical second-order harmonic amplitude ratio a2/a1 for Stokes II.

    From the Stokes II elevation:
        eta = a cos(theta) + (k a^2 / 4) * (3 - s^2) / s^3 * cos(2*theta)
    where s = tanh(kh).

    The second harmonic amplitude is B2 = (k a^2 / 4) * (3 - s^2) / s^3
    and the first harmonic amplitude is a, so:
        a2/a1 = (k a / 4) * (3 - s^2) / s^3
    """
    s = math.tanh(k * h)
    if s < 1e-10:
        return 0.0
    return (k * a / 4.0) * (3.0 - s * s) / (s ** 3)


@dataclass
class MetrologyResult:
    """Container for metrology results with pass/fail criteria."""
    name: str
    value: float
    target: float
    tolerance: float
    unit: str = ""
    passed: bool = False
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        error = abs(self.value - self.target)
        if self.target != 0:
            rel_error = error / abs(self.target)
        else:
            rel_error = error
        self.passed = rel_error <= self.tolerance
        self.details["absolute_error"] = error
        self.details["relative_error"] = rel_error

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"{self.name}: {self.value:.6g} {self.unit} "
                f"(target={self.target:.6g}, error={self.details['relative_error']*100:.2f}%) [{status}]")


class WaveTankMetrology:
    """Comprehensive metrology for the numerical wave tank.

    Provides methods to verify all campaign criteria:
    (a) Linear wave amplitude and dispersion
    (b) Second-order Stokes harmonic ratio
    (c) Reflection coefficient
    (d) Grid convergence (GCI)
    (e) Added resistance cross-validation
    (f) Volume conservation
    """

    def __init__(self, length: float = 10.0, depth: float = 1.0,
                 gravity: float = GRAVITY):
        self.length = length
        self.depth = depth
        self.gravity = gravity

    def run_simulation(
        self,
        amp: float,
        period: float,
        nx: int = 200,
        ns: int = 30,
        n_periods: float = 5.0,
        dt: float = 0.005,
        probe_positions: Optional[list[float]] = None,
        relax_waves: float = 1.5,
        damp_waves: float = 2.0,
        damp_strength: float = 5.0,
        order2: bool = True,
    ) -> dict:
        """Run a wave tank simulation and collect probe data.

        Returns a dict with:
        - tank: the SigmaWaveTank object
        - wave: WaveTarget object
        - times: list of sample times
        - etas: dict mapping probe position to elevation time series
        - probe_positions: list of probe x-positions
        """
        cfg = TankConfig(
            length=self.length, depth=self.depth, nx=nx, ns=ns,
            relax_waves=relax_waves, damp_waves=damp_waves,
            damp_strength=damp_strength,
        )
        tank = SigmaWaveTank(cfg)
        tank.set_wave(amp, period)
        wt = tank.wave

        if probe_positions is None:
            # Place probes in the working zone (between relaxation and damping)
            x_relax_end = relax_waves * wt.wavelength
            x_damp_start = self.length - damp_waves * wt.wavelength
            working_length = x_damp_start - x_relax_end
            if working_length > 0:
                probe_positions = [
                    x_relax_end + 0.25 * working_length,
                    x_relax_end + 0.50 * working_length,
                    x_relax_end + 0.75 * working_length,
                ]
            else:
                probe_positions = [
                    self.length * 0.3,
                    self.length * 0.5,
                    self.length * 0.7,
                ]

        T = period
        n_steps = int(n_periods * T / dt)
        sample_interval = max(1, int(0.02 / dt))  # sample every ~0.02s

        times = []
        etas = {x: [] for x in probe_positions}

        for i in range(n_steps):
            tank.step(dt)
            if i % sample_interval == 0:
                times.append(tank.t)
                for x in probe_positions:
                    etas[x].append(float(tank.eta_at(x)))

        return {
            "tank": tank,
            "wave": wt,
            "times": np.array(times),
            "etas": {x: np.array(e) for x, e in etas.items()},
            "probe_positions": probe_positions,
            "dt": dt,
            "n_periods": n_periods,
        }

    def analyze_harmonics(
        self, times: np.ndarray, eta: np.ndarray, omega: float,
        skip_periods: float = 2.0, period: float = 1.0, n_harm: int = 2,
    ) -> dict:
        """Analyze wave probe signal with multi-harmonic fit.

        Returns dict with:
        - amp: list of harmonic amplitudes [a1, a2, ...]
        - phase: list of harmonic phases
        - mean: mean water level
        - residual_rel: relative residual
        - a2_over_a1: second-to-first harmonic ratio
        """
        T = period
        mask = times > skip_periods * T
        if mask.sum() < 10:
            mask = np.ones(len(times), dtype=bool)

        result = fit_harmonics(times[mask], eta[mask], omega, n_harm=n_harm)
        if len(result["amp"]) >= 2:
            result["a2_over_a1"] = (
                result["amp"][1] / result["amp"][0] if result["amp"][0] > 1e-10 else 0.0
            )
        else:
            result["a2_over_a1"] = 0.0
        return result

    def check_dispersion(
        self, times: np.ndarray, etas: dict, probe_positions: list[float],
        omega: float, period: float,
    ) -> dict:
        """Check dispersion relation using three-probe phase method.

        Returns dict with:
        - k_measured: wavenumber from probe phases
        - k_theory: wavenumber from dispersion relation
        - error_rel: relative error
        """
        T = period
        mask = times > 2.0 * T

        phases = []
        for x in probe_positions[:3]:
            result = harmonic_fit(times[mask], etas[x][mask], omega)
            phases.append(result[1])  # phase

        k_measured, _ = three_probe_wavenumber(
            np.array(probe_positions[:3]),
            np.ones(3),  # amplitudes not needed for phase fit
            np.array(phases),
        )
        k_theory = dispersion_k(omega, self.depth, self.gravity)

        return {
            "k_measured": k_measured,
            "k_theory": k_theory,
            "error_rel": abs(k_measured - k_theory) / k_theory,
        }

    def check_reflection_coefficient(
        self, times: np.ndarray, etas: dict, probe_positions: list[float],
        omega: float, k: float,
    ) -> float:
        """Compute reflection coefficient using Goda two-point method.

        Uses the first two probes. Returns C_R (0 = perfect, 1 = total reflection).
        """
        if len(probe_positions) < 2:
            return 1.0

        x1, x2 = probe_positions[0], probe_positions[1]
        eta1 = etas[x1]
        eta2 = etas[x2]

        # Compute complex amplitudes via harmonic fit
        amp1, phase1, _, _ = harmonic_fit(times, eta1, omega)
        amp2, phase2, _, _ = harmonic_fit(times, eta2, omega)

        h1 = amp1 * np.exp(1j * phase1)
        h2 = amp2 * np.exp(1j * phase2)

        try:
            aI, aR = goda_two_point(x1, x2, h1, h2, k)
            C_R = abs(aR) / abs(aI) if abs(aI) > 1e-10 else 1.0
        except ValueError:
            C_R = 1.0

        return min(float(C_R), 1.0)

    def check_volume_conservation(self, tank: SigmaWaveTank) -> dict:
        """Check volume conservation of the tank.

        Returns dict with:
        - volume_initial: initial volume
        - volume_current: current volume
        - drift_rel: relative drift
        - drift_per_period: drift per wave period (if wave is set)
        """
        vol0 = tank.volume0
        vol = tank.volume()
        drift = abs(vol - vol0) / vol0

        result = {
            "volume_initial": vol0,
            "volume_current": vol,
            "drift_rel": drift,
        }

        if tank.wave is not None:
            T = tank.wave.period
            n_periods = tank.t / T
            if n_periods > 0:
                result["drift_per_period"] = drift / n_periods
            else:
                result["drift_per_period"] = drift

        return result


def run_linear_wave_test(
    H: float = 0.04, T: float = 1.2, h: float = 1.0,
    nx: int = 200, ns: int = 30, n_periods: float = 5.0,
) -> dict:
    """Run linear wave test (criterion a).

    Checks:
    - Wave height relative error < 3%
    - Dispersion relation error < 1%

    Returns dict with all results and pass/fail status.
    """
    amp = H / 2.0
    metro = WaveTankMetrology(length=10.0, depth=h)
    sim = metro.run_simulation(
        amp=amp, period=T, nx=nx, ns=ns,
        n_periods=n_periods, order2=False,
    )

    wt = sim["wave"]
    probe_x = sim["probe_positions"][1]  # middle probe
    eta = sim["etas"][probe_x]
    times = sim["times"]

    # Harmonic analysis
    harm = metro.analyze_harmonics(times, eta, wt.omega, skip_periods=2.0, period=T, n_harm=1)
    H_measured = 2.0 * harm["amp"][0]

    # Dispersion check
    disp = metro.check_dispersion(times, sim["etas"], sim["probe_positions"], wt.omega, T)

    # Volume conservation
    vol = metro.check_volume_conservation(sim["tank"])

    # Criteria
    H_error = abs(H_measured - H) / H
    disp_error = disp["error_rel"]

    results = {
        "H_target": H,
        "H_measured": H_measured,
        "H_error_rel": H_error,
        "H_pass": H_error < 0.03,
        "dispersion_k_theory": disp["k_theory"],
        "dispersion_k_measured": disp["k_measured"],
        "dispersion_error_rel": disp_error,
        "dispersion_pass": disp_error < 0.01,
        "volume_drift_per_period": vol.get("drift_per_period", vol["drift_rel"]),
        "volume_pass": vol.get("drift_per_period", vol["drift_rel"]) < 0.001,
        "all_pass": H_error < 0.03 and disp_error < 0.01,
    }
    return results


def run_stokes2_harmonic_test(
    ka: float = 0.05, h: float = 1.0, T: float = 1.2,
    nx: int = 200, ns: int = 30, n_periods: float = 8.0,
) -> dict:
    """Run second-order Stokes harmonic test (criterion b).

    Checks:
    - a2/a1 vs theoretical Stokes II value, error < 10%

    Returns dict with all results and pass/fail status.
    """
    # Compute wave parameters for target ka
    omega = 2.0 * math.pi / T
    k = dispersion_k(omega, h)
    a = ka / k
    H = 2.0 * a

    metro = WaveTankMetrology(length=12.0, depth=h)
    sim = metro.run_simulation(
        amp=a, period=T, nx=nx, ns=ns,
        n_periods=n_periods, order2=True,
    )

    wt = sim["wave"]
    probe_x = sim["probe_positions"][1]
    eta = sim["etas"][probe_x]
    times = sim["times"]

    # Multi-harmonic analysis
    harm = metro.analyze_harmonics(times, eta, wt.omega, skip_periods=3.0, period=T, n_harm=2)

    a1_measured = harm["amp"][0]
    a2_measured = harm["amp"][1]
    a2_over_a1_measured = harm["a2_over_a1"]

    # Theoretical value
    a2_over_a1_theory = compute_stokes2_theoretical_harmonic_ratio(k, a, h)

    # Error
    if a2_over_a1_theory > 1e-10:
        harmonic_error = abs(a2_over_a1_measured - a2_over_a1_theory) / a2_over_a1_theory
    else:
        harmonic_error = abs(a2_over_a1_measured)

    results = {
        "ka": ka,
        "a": a,
        "H": H,
        "k": k,
        "a1_measured": a1_measured,
        "a2_measured": a2_measured,
        "a2_over_a1_measured": a2_over_a1_measured,
        "a2_over_a1_theory": a2_over_a1_theory,
        "harmonic_error_rel": harmonic_error,
        "harmonic_pass": harmonic_error < 0.10,
    }
    return results


def run_reflection_coefficient_test(
    H: float = 0.04, T: float = 1.2, h: float = 1.0,
    nx: int = 200, ns: int = 30, n_periods: float = 25.0,
) -> dict:
    """Run reflection coefficient test (criterion c).

    Checks:
    - Reflection coefficient R < 5% after 20 periods

    Returns dict with all results and pass/fail status.
    """
    amp = H / 2.0
    metro = WaveTankMetrology(length=14.0, depth=h)
    sim = metro.run_simulation(
        amp=amp, period=T, nx=nx, ns=ns,
        n_periods=n_periods, order2=True,
        relax_waves=1.5, damp_waves=2.5, damp_strength=6.0,
    )

    wt = sim["wave"]
    times = sim["times"]

    # Reflection coefficient from probes
    C_R = metro.check_reflection_coefficient(
        times, sim["etas"], sim["probe_positions"], wt.omega, wt.k
    )

    results = {
        "C_R": C_R,
        "C_R_target": 0.0,
        "C_R_tolerance": 0.05,
        "reflection_pass": C_R < 0.05,
    }
    return results


def run_gci_study(
    H: float = 0.04, T: float = 1.2, h: float = 1.0,
    refinement_ratios: list[float] = None,
    n_periods: float = 5.0,
) -> dict:
    """Run grid convergence study (criterion d).

    Uses three grid levels and computes GCI.

    Returns dict with:
    - H_values: wave height at each grid level
    - refinement_ratios: actual refinement ratios
    - order_of_convergence: estimated order
    - gci: Grid Convergence Index
    - gci_pass: GCI < 3%
    """
    if refinement_ratios is None:
        refinement_ratios = [1.0, 2.0, 4.0]  # coarse, medium, fine

    amp = H / 2.0
    base_nx = 100
    base_ns = 15

    H_values = []
    dx_values = []

    for r in refinement_ratios:
        nx = int(base_nx * r)
        ns = int(base_ns * r)
        metro = WaveTankMetrology(length=10.0, depth=h)
        sim = metro.run_simulation(
            amp=amp, period=T, nx=nx, ns=ns,
            n_periods=n_periods, order2=False,
        )
        wt = sim["wave"]
        probe_x = sim["probe_positions"][1]
        eta = sim["etas"][probe_x]
        times = sim["times"]

        harm = metro.analyze_harmonics(times, eta, wt.omega, skip_periods=2.0, period=T, n_harm=1)
        H_meas = 2.0 * harm["amp"][0]
        H_values.append(H_meas)
        dx_values.append(10.0 / nx)

    # Richardson extrapolation and GCI
    # Using three-grid method (Roache 1997)
    r1 = dx_values[0] / dx_values[1]  # coarse/medium
    r2 = dx_values[1] / dx_values[2]  # medium/fine

    eps1 = H_values[1] - H_values[0]  # medium - coarse
    eps2 = H_values[2] - H_values[1]  # fine - medium

    if abs(eps2) > 1e-15 and abs(eps1) > 1e-15:
        # Estimate order of convergence
        p = math.log(abs(eps1 / eps2)) / math.log(r1)
        p = max(0.5, min(p, 4.0))  # clamp to reasonable range

        # Richardson extrapolated value
        if abs(r1**p - 1.0) > 1e-10:
            H_extrap = H_values[2] + eps2 / (r2**p - 1.0)
        else:
            H_extrap = H_values[2]

        # GCI (fine grid)
        Fs = 1.25  # safety factor for 3 grids
        GCI_fine = Fs * abs(eps2 / H_values[2]) / (r2**p - 1.0) * 100.0
    else:
        p = 2.0
        H_extrap = H_values[2]
        GCI_fine = 0.0

    results = {
        "H_values": H_values,
        "dx_values": dx_values,
        "refinement_ratios": refinement_ratios,
        "order_of_convergence": p,
        "H_extrapolated": H_extrap,
        "gci_fine_percent": GCI_fine,
        "gci_pass": GCI_fine < 3.0,
    }
    return results


def run_volume_conservation_test(
    H: float = 0.04, T: float = 1.2, h: float = 1.0,
    nx: int = 200, ns: int = 30, n_periods: float = 10.0,
) -> dict:
    """Run volume conservation test (criterion f).

    Checks:
    - Volume drift < 0.1% per period

    Returns dict with all results and pass/fail status.
    """
    amp = H / 2.0
    metro = WaveTankMetrology(length=10.0, depth=h)
    sim = metro.run_simulation(
        amp=amp, period=T, nx=nx, ns=ns,
        n_periods=n_periods, order2=True,
    )

    vol = metro.check_volume_conservation(sim["tank"])

    results = {
        "volume_initial": vol["volume_initial"],
        "volume_current": vol["volume_current"],
        "drift_rel": vol["drift_rel"],
        "drift_per_period": vol.get("drift_per_period", vol["drift_rel"]),
        "volume_pass": vol.get("drift_per_period", vol["drift_rel"]) < 0.001,
    }
    return results
