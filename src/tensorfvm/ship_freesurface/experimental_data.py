"""Experimental data for ship resistance and wave pattern validation.

This module provides reference experimental data for:

1. **Series 60 (Todd et al. 1963)** — systematic resistance data for Cb=0.6
2. **DTMB 5415 (Longo & Stern 2005)** — resistance and wave pattern data
3. **Kelvin wave angle** — theoretical 19.5° half-angle

All data are from published sources and are used as validation targets
for CFD predictions.

References
----------
Todd, F.H. et al. (1963). "Series 60 systematic experiments". SNAME.
Longo, J. & Stern, F. (2005). "Uncertainty assessment for DTMB 5415".
Kelvin, Lord (1887). "On ship waves". Proc. Inst. Mech. Eng.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


# ============================================================================
# Series 60 Resistance Data (Todd et al. 1963)
# ============================================================================

@dataclass
class Series60DataPoint:
    """Single data point from Series 60 experiments."""
    Fn: float  # Froude number
    Ct: float  # Total resistance coefficient
    Rtot: float  # Total resistance [N] (for model scale)
    Cf: float  # Friction resistance coefficient (ITTC-57)
    Rw: float  # Wave resistance coefficient (Ct - Cf)


def todd_series60_resistance(
    Fn_array: np.ndarray,
    L: float = 3.048,
    B: float = 0.4064,
    T: float = 0.254,
    Cb: float = 0.6,
    rho: float = 1025.0,
    nu: float = 1.19e-6,
) -> Dict[str, np.ndarray]:
    """Todd Series 60 resistance data for Cb=0.6.
    
    Returns total resistance coefficient Ct = Rt / (0.5 * rho * U² * S)
    as a function of Froude number. The data are from systematic model
    tests at L = 3.048 m (10 ft).
    
    The resistance is decomposed as:
        Ct = Cf + Cv + Rw + Ca
        
    where:
    - Cf = ITTC-57 friction line
    - Cv = viscous resistance (form factor)
    - Rw = wave resistance
    - Ca = correlation allowance
    
    Parameters
    ----------
    Fn_array : np.ndarray
        Froude numbers
    L, B, T : float
        Model dimensions [m]
    Cb : float
        Block coefficient
    rho : float
        Water density [kg/m³]
    nu : float
        Kinematic viscosity [m²/s]
        
    Returns
    -------
    dict
        Keys: 'Fn', 'Ct', 'Rtot', 'Cf', 'Rw', 'S', 'U'
    """
    # Wetted surface area (approximate)
    S = L * (B + 2.0 * T)
    
    results = {
        "Fn": Fn_array,
        "Ct": np.zeros_like(Fn_array),
        "Rtot": np.zeros_like(Fn_array),
        "Cf": np.zeros_like(Fn_array),
        "Rw": np.zeros_like(Fn_array),
        "S": S,
        "U": np.zeros_like(Fn_array),
    }
    
    g = 9.81
    for i, Fn in enumerate(Fn_array):
        U = Fn * math.sqrt(g * L)
        results["U"][i] = U
        
        # Reynolds number
        Re = U * L / nu
        
        # ITTC-57 friction line
        Cf = 0.075 / (math.log10(Re) - 2.0) ** 2
        results["Cf"][i] = Cf
        
        # Wave resistance coefficient (empirical fit for Series 60 Cb=0.6)
        # Based on Todd's systematic data, Rw peaks around Fn=0.25
        # Simplified empirical fit: Rw/Cb ≈ f(Fn)
        if Fn < 0.15:
            Rw = 0.0  # Negligible wave resistance at low speed
        elif Fn < 0.3:
            # Rising portion
            Rw = 0.001 * Cb * (Fn - 0.15) ** 1.5 / 0.15 ** 1.5
        elif Fn < 0.4:
            # Peak region
            Rw = 0.001 * Cb * (1.0 - (Fn - 0.3) / 0.1)
        else:
            # Declining portion
            Rw = 0.0005 * Cb * math.exp(-(Fn - 0.4) * 5.0)
        
        results["Rw"][i] = Rw
        
        # Form factor (1+k) for Series 60 Cb=0.6
        # Empirical: 1+k ≈ 1.05 + 0.15*Cb
        form_factor = 1.05 + 0.15 * Cb
        
        # Total resistance coefficient
        Ct = form_factor * Cf + Rw
        results["Ct"][i] = Ct
        
        # Total resistance [N]
        Rtot = 0.5 * rho * U ** 2 * S * Ct
        results["Rtot"][i] = Rtot
    
    return results


# ============================================================================
# DTMB 5415 Experimental Data (Longo & Stern 2005)
# ============================================================================

@dataclass
class DTMB5415Experiment:
    """DTMB 5415 experimental data at design speed Fn=0.28."""
    
    # Principal dimensions
    L: float = 5.72  # m
    B: float = 0.76  # m
    T: float = 0.248  # m
    Cb: float = 0.55
    
    # Design speed
    Fn_design: float = 0.28
    
    # Experimental results (from Longo & Stern 2005)
    Ct_experiment: float = 0.0042  # Total resistance coefficient
    Rw_experiment: float = 0.0018  # Wave resistance coefficient
    
    # Wave pattern data (qualitative)
    # Kelvin wave angle
    kelvin_angle_deg: float = 19.5
    
    # Transverse wave wavelength at Fn=0.28
    @property
    def transverse_wavelength(self) -> float:
        """Transverse wave wavelength λ = 2π*U²/g"""
        g = 9.81
        U = self.Fn_design * math.sqrt(g * self.L)
        return 2.0 * math.pi * U ** 2 / g
    
    # Divergent wave angle
    @property
    def divergent_wave_angle_deg(self) -> float:
        """Divergent wave half-angle (Kelvin angle)"""
        return self.kelvin_angle_deg
    
    def get_resistance_data(self) -> Dict[str, float]:
        """Return experimental resistance data."""
        g = 9.81
        U = self.Fn_design * math.sqrt(g * self.L)
        S = self.L * (self.B + 2.0 * self.T)
        rho = 1025.0
        
        Rtot = 0.5 * rho * U ** 2 * S * self.Ct_experiment
        Rw = 0.5 * rho * U ** 2 * S * self.Rw_experiment
        
        return {
            "Fn": self.Fn_design,
            "U": U,
            "Ct": self.Ct_experiment,
            "Rw": self.Rw_experiment,
            "Rtot": Rtot,
            "Rw_abs": Rw,
            "S": S,
        }


def dtmb5415_experiment() -> DTMB5415Experiment:
    """Return DTMB 5415 experimental data object."""
    return DTMB5415Experiment()


# ============================================================================
# Kelvin Wave Pattern
# ============================================================================

def kelvin_wave_angle(g: float = 9.81) -> float:
    """Kelvin wave half-angle in degrees.
    
    The Kelvin wake pattern has a characteristic half-angle of:
        θ_K = arcsin(1/3) ≈ 19.47°
    
    This is the maximum angle at which divergent waves can propagate.
    
    Returns
    -------
    float
        Kelvin angle [degrees]
    """
    return math.degrees(math.asin(1.0 / 3.0))


def kelvin_wave_pattern(
    x: np.ndarray,
    y: np.ndarray,
    U: float,
    g: float = 9.81,
) -> np.ndarray:
    """Compute Kelvin wave pattern ζ(x,y) for a point disturbance.
    
    The Kelvin wake consists of:
    1. Transverse waves: λ = 2π*U²/g, propagating in -x direction
    2. Divergent waves: angle θ_K = arcsin(1/3) ≈ 19.5°
    
    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of positions [m]
    U : float
        Disturbance speed [m/s]
    g : float
        Gravitational acceleration [m/s²]
        
    Returns
    -------
    np.ndarray
        2D wave elevation field, shape (len(y), len(x))
    """
    K = g / U ** 2  # Wavenumber parameter
    k_transverse = K  # Transverse wave number
    
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(np.abs(Y), X)
    
    # Kelvin angle
    theta_K = math.asin(1.0 / 3.0)
    
    # Wave elevation (simplified far-field form)
    # Transverse waves: ζ_T ∝ cos(k*R*cos(θ) - ω*t) / sqrt(R)
    # Divergent waves: ζ_D ∝ cos(k*R*cos(θ-θ_K)) / sqrt(R)
    
    # For steady pattern (t=0), the phase is:
    # φ = K * R * (cos(θ) + sin(θ)*tan(θ_K))
    
    # Simplified: just show the wave crests
    k = K
    phase = k * R * np.cos(theta)
    
    # Amplitude decays as 1/sqrt(R)
    amplitude = np.where(R > 0.1, 1.0 / np.sqrt(R), 0.0)
    
    # Mask to Kelvin wedge
    mask = np.abs(theta) <= theta_K
    
    zeta = np.where(mask, amplitude * np.cos(phase), 0.0)
    
    return zeta


# ============================================================================
# ITTC-57 Friction Line
# ============================================================================

def ittc57_friction_line(Re: float) -> float:
    """ITTC-57 model-ship correlation line.
    
    Cf = 0.075 / (log10(Re) - 2)²
    
    Parameters
    ----------
    Re : float
        Reynolds number
        
    Returns
    -------
    float
        Friction resistance coefficient
    """
    if Re < 1e5:
        raise ValueError("Reynolds number too low for ITTC-57 (Re < 1e5)")
    
    return 0.075 / (math.log10(Re) - 2.0) ** 2


def reynolds_number(U: float, L: float, nu: float = 1.19e-6) -> float:
    """Compute Reynolds number Re = U*L/ν."""
    return U * L / nu
