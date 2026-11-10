"""Wave probe instrumentation for numerical wave tank analysis.

Provides point probes that record free-surface elevation over time,
supporting spectral analysis, reflection coefficient calculation (Goda
two-point method), and wave number inversion from dispersion relation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "WaveProbe",
    "WaveProbeArray",
    "goda_reflection_coefficient",
    "invert_wavenumber",
    "spectral_analysis",
]


@dataclass
class WaveProbe:
    """Single-point wave elevation recorder."""
    x: float
    h: float
    dy: float
    ny: int
    device: str = "cpu"

    def __post_init__(self):
        self.history: list[float] = []
        self.times: list[float] = []

    def record(self, eta: float, t: float):
        self.history.append(float(eta))
        self.times.append(float(t))


@dataclass
class WaveProbeArray:
    """Array of wave probes at known x-positions on the grid."""
    x_positions: list[float]
    h: float
    dx: float
    dy: float
    ny: int
    device: str = "cpu"

    def __post_init__(self):
        self.n_probes = len(self.x_positions)
        self.probes: list[WaveProbe] = [
            WaveProbe(x=x, h=self.h, dy=self.dy, ny=self.ny, device=self.device)
            for x in self.x_positions
        ]
        self.col_indices: list[int] = []
        for x in self.x_positions:
            col = int(round(x / self.dx - 0.5))
            self.col_indices.append(col)

    def measure_all(self, alpha: torch.Tensor, t: float) -> list[float]:
        """Measure free-surface elevation at all probes."""
        ny, nx = alpha.shape
        etas = []
        for idx, col in enumerate(self.col_indices):
            col = max(0, min(col, nx - 1))
            col_alpha = alpha[:, col]
            above = (col_alpha >= 0.5).nonzero(as_tuple=True)[0]
            if len(above) == 0:
                eta = -self.h
            else:
                j_max = int(above[-1].item())
                if j_max < ny - 1:
                    a0 = float(col_alpha[j_max].item())
                    a1 = float(col_alpha[j_max + 1].item())
                    if abs(a1 - a0) > 1e-12:
                        frac_interface = (0.5 - a0) / (a1 - a0)
                    else:
                        frac_interface = 0.5
                else:
                    frac_interface = 0.5
                z_jmax = (j_max + 0.5) * self.dy - self.h
                eta = z_jmax + frac_interface * self.dy
            etas.append(eta)
            self.probes[idx].record(eta, t)
        return etas

    def get_history(self, probe_idx: int = 0) -> tuple[list[float], list[float]]:
        p = self.probes[probe_idx]
        return list(p.times), list(p.history)


def goda_reflection_coefficient(eta1: torch.Tensor, eta2: torch.Tensor,
                                 dx_probe: float, omega: float,
                                 h: float, g: float = 9.81,
                                 dt: float = 1.0) -> float:
    """Goda two-point method for reflection coefficient.
    
    Parameters
    ----------
    eta1, eta2 : torch.Tensor
        Time series of surface elevation at two probes.
    dx_probe : float
        Distance between probes (m).
    omega : float
        Angular frequency (rad/s).
    h : float
        Water depth (m).
    g : float
        Gravitational acceleration (m/s²).
    dt : float
        Time step of the data (s). Default 1.0.
    """
    from .wave_theory import dispersion_relation

    k = dispersion_relation(omega, h, g)

    N = len(eta1)
    if N < 4:
        return 0.0

    # Time array
    t = torch.arange(N, dtype=torch.float64, device=eta1.device) * dt
    cos_wt = torch.cos(omega * t)
    sin_wt = torch.sin(omega * t)

    A1_real = torch.sum(eta1 * cos_wt) / N
    A1_imag = torch.sum(eta1 * sin_wt) / N
    A2_real = torch.sum(eta2 * cos_wt) / N
    A2_imag = torch.sum(eta2 * sin_wt) / N

    cos_kdx = math.cos(k * dx_probe)
    sin_kdx = math.sin(k * dx_probe)

    A1_mag = math.sqrt(A1_real ** 2 + A1_imag ** 2)
    if A1_mag < 1e-12:
        return 0.0

    A1_exp_real = A1_real * cos_kdx + A1_imag * sin_kdx
    A1_exp_imag = A1_imag * cos_kdx - A1_real * sin_kdx

    diff_real = A2_real - A1_exp_real
    diff_imag = A2_imag - A1_exp_imag

    cos_2kdx = math.cos(2 * k * dx_probe)
    sin_2kdx = math.sin(2 * k * dx_probe)
    denom_real = 1.0 - cos_2kdx
    denom_imag = sin_2kdx

    denom_mag = math.sqrt(denom_real ** 2 + denom_imag ** 2)
    if denom_mag < 1e-12:
        return 0.0

    diff_mag = math.sqrt(diff_real ** 2 + diff_imag ** 2)
    C_R = diff_mag / denom_mag

    return min(float(C_R), 1.0)


def invert_wavenumber(eta1: torch.Tensor, eta2: torch.Tensor, eta3: torch.Tensor,
                       dx12: float, dx23: float, omega: float,
                       h: float, g: float = 9.81,
                       dt: float = 1.0) -> tuple[float, float]:
    """Three-probe method to invert wave number from measured elevations.
    
    Parameters
    ----------
    eta1, eta2, eta3 : torch.Tensor
        Time series at three probes.
    dx12, dx23 : float
        Distances between probes (m).
    omega : float
        Angular frequency (rad/s).
    h : float
        Water depth (m).
    g : float
        Gravitational acceleration (m/s²).
    dt : float
        Time step of the data (s). Default 1.0.
    """
    from .wave_theory import dispersion_relation

    N = len(eta1)
    if N < 4:
        return 0.0, 0.0

    # Time array
    t = torch.arange(N, dtype=torch.float64, device=eta1.device) * dt
    cos_wt = torch.cos(omega * t)
    sin_wt = torch.sin(omega * t)

    A1_real = torch.sum(eta1 * cos_wt) / N
    A1_imag = torch.sum(eta1 * sin_wt) / N
    A2_real = torch.sum(eta2 * cos_wt) / N
    A2_imag = torch.sum(eta2 * sin_wt) / N
    A3_real = torch.sum(eta3 * cos_wt) / N
    A3_imag = torch.sum(eta3 * sin_wt) / N

    phi1 = math.atan2(float(A1_imag.item()), float(A1_real.item()))
    phi2 = math.atan2(float(A2_imag.item()), float(A2_real.item()))
    phi3 = math.atan2(float(A3_imag.item()), float(A3_real.item()))

    delta_phi12 = phi2 - phi1
    delta_phi23 = phi3 - phi2

    # Unwrap phase differences
    while delta_phi12 > math.pi:
        delta_phi12 -= 2 * math.pi
    while delta_phi12 < -math.pi:
        delta_phi12 += 2 * math.pi
    while delta_phi23 > math.pi:
        delta_phi23 -= 2 * math.pi
    while delta_phi23 < -math.pi:
        delta_phi23 += 2 * math.pi

    k12 = abs(delta_phi12 / dx12)
    k23 = abs(delta_phi23 / dx23)

    k_measured = (k12 + k23) / 2.0
    k_theory = dispersion_relation(omega, h, g)
    return k_measured, k_theory


def spectral_analysis(eta: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute power spectral density of wave elevation time series."""
    N = len(eta)
    eta_centered = eta - eta.mean()
    eta_fft = torch.fft.rfft(eta_centered)
    psd = (eta_fft.abs() ** 2) * dt / N
    freqs = torch.fft.rfftfreq(N, d=dt)
    return freqs, psd
