"""Michell thin-ship theory for wave resistance and wave elevation.

Michell (1898) derived an analytical theory for the wave resistance and
wave pattern of a slender ship moving at constant speed in calm water.
The theory assumes:
- Slender hull (B/L << 1)
- Linear free-surface boundary condition
- Infinite depth (or finite depth with modified dispersion)
- No viscous effects

The wave elevation along the centerline (y=0) is:
    ζ(x, 0) = Re{ Σ A_n * exp(i * k_n * x) }

where k_n are the wave numbers satisfying the dispersion relation:
    K = k * tanh(k * h)    (finite depth)
    K = k                   (infinite depth, K = g/U²)

and A_n are the Kochin function coefficients determined by the hull geometry.

For the Wigley hull, the Kochin function can be evaluated analytically.

References
----------
Michell, J.H. (1898). "The wave resistance of a ship". Phil. Mag. 45.
Havelock, T.H. (1932). "The wave resistance of a ship". Proc. Roy. Soc.
Noblesse, F. (1981). "The slender ship and the wave resistance".
"""

from __future__ import annotations

import math
import cmath
from typing import Tuple, List, Optional

import numpy as np


def _dispersion_relation(K: float, h: float = float('inf')) -> float:
    """Solve dispersion relation K = k * tanh(k * h) for wave number k.
    
    Parameters
    ----------
    K : float
        Wavenumber parameter K = g / U² [1/m]
    h : float
        Water depth [m]. Use inf for deep water.
        
    Returns
    -------
    float
        Wave number k [1/m]
    """
    if math.isinf(h):
        # Deep water: K = k
        return K
    else:
        # Finite depth: solve K = k * tanh(k*h) by Newton iteration
        k = K  # initial guess (deep water limit)
        for _ in range(50):
            f = k * math.tanh(k * h) - K
            fp = math.tanh(k * h) + k * h / math.cosh(k * h) ** 2
            k_new = k - f / fp
            if abs(k_new - k) < 1e-12:
                break
            k = k_new
        return k


def _kochin_wigley(K: float, L: float, B: float, T: float, 
                   k: float, n_theta: int = 100) -> complex:
    """Kochin function for Wigley parabolic hull.
    
    The Kochin function H(k) relates the hull geometry to the far-field
    wave amplitude. For the Wigley hull with half-breadth:
        y(x,z) = (B/2) * [1 - (2x/L - 1)²] * (z/T)
    
    the Kochin function can be integrated analytically:
        H(k) = ∫∫ (∂y/∂x) * exp(i*k*x) * exp(k*z) dx dz
    
    Parameters
    ----------
    K : float
        Wavenumber parameter g/U²
    L, B, T : float
        Hull dimensions
    k : float
        Wave number
    n_theta : int
        Not used (analytical integration)
        
    Returns
    -------
    complex
        Kochin function value
    """
    # For Wigley hull, the x-integral of the section area derivative gives:
    # ∫₀ᴸ dA/dx * exp(ikx) dx = i*k * ∫₀ᴸ A(x) * exp(ikx) dx
    # where A(x) = (B*T/3) * [1 - (2x/L - 1)²]
    
    # Let u = 2x/L - 1, then x = L(u+1)/2, dx = L/2 du
    # A(u) = (B*T/3) * (1 - u²)
    # ∫₋₁¹ (1-u²) * exp(ik*L*(u+1)/2) du
    
    # This integral can be evaluated:
    # = exp(ikL/2) * ∫₋₁¹ (1-u²) * exp(ikLu/2) du
    # = exp(ikL/2) * [4/(kL)² * (sin(kL/2) - (kL/2)*cos(kL/2))]
    
    kL2 = k * L / 2.0
    if abs(kL2) < 1e-10:
        # Limit as kL → 0
        integral = 4.0 / 3.0
    else:
        sinc = math.sin(kL2) / kL2 if abs(kL2) > 1e-10 else 1.0
        integral = 4.0 / (kL2) ** 2 * (math.sin(kL2) - kL2 * math.cos(kL2))
    
    # Multiply by constants
    A0 = B * T / 3.0  # Maximum section area
    H = 1j * k * A0 * cmath.exp(1j * k * L / 2.0) * integral * L / 2.0
    
    return H


def michell_wave_elevation(
    x: np.ndarray,
    U: float,
    L: float,
    B: float,
    T: float,
    h: float = float('inf'),
    y: float = 0.0,
    g: float = 9.81,
) -> np.ndarray:
    """Compute wave elevation along a cut using Michell theory.
    
    Parameters
    ----------
    x : np.ndarray
        Longitudinal positions [m]
    U : float
        Ship speed [m/s]
    L, B, T : float
        Hull dimensions [m]
    h : float
        Water depth [m] (default: infinite)
    y : float
        Lateral distance from centerline [m] (default: 0)
    g : float
        Gravitational acceleration [m/s²]
        
    Returns
    -------
    np.ndarray
        Wave elevation ζ(x) [m]
    """
    K = g / U ** 2  # Wavenumber parameter
    k = _dispersion_relation(K, h)
    
    # Kochin function
    H = _kochin_wigley(K, L, B, T, k)
    
    # Wave elevation (far-field, divergent wave system)
    # ζ(x,y) = Re{ H(k) / (2π) * exp(-ik*x - k*|y|) / sqrt(k) }
    # For y=0: ζ(x,0) = Re{ H(k) / (2π) * exp(-ik*x) / sqrt(k) }
    
    # Actually, the correct far-field form is:
    # ζ = Re{ -i * K * H / (2π) * exp(i*k*x) }  (divergent waves)
    
    amplitude = abs(H) * K / (2.0 * math.pi)
    phase = cmath.phase(H)
    
    # Include lateral decay
    lateral_decay = math.exp(-k * abs(y))
    
    zeta = amplitude * lateral_decay * np.cos(k * x + phase)
    
    return zeta


def michell_wave_cut(
    x: np.ndarray,
    y: np.ndarray,
    U: float,
    L: float,
    B: float,
    T: float,
    h: float = float('inf'),
    g: float = 9.81,
) -> np.ndarray:
    """Compute 2D wave elevation field ζ(x,y) using Michell theory.
    
    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of longitudinal and lateral positions [m]
    U : float
        Ship speed [m/s]
    L, B, T : float
        Hull dimensions [m]
    h : float
        Water depth [m]
    g : float
        Gravitational acceleration [m/s²]
        
    Returns
    -------
    np.ndarray
        2D wave elevation field ζ(x,y) [m], shape (len(y), len(x))
    """
    K = g / U ** 2
    k = _dispersion_relation(K, h)
    H = _kochin_wigley(K, L, B, T, k)
    
    amplitude = abs(H) * K / (2.0 * math.pi)
    phase = cmath.phase(H)
    
    X, Y = np.meshgrid(x, y)
    zeta = amplitude * np.exp(-k * np.abs(Y)) * np.cos(k * X + phase)
    
    return zeta


def michell_wave_resistance(
    U: float,
    L: float,
    B: float,
    T: float,
    h: float = float('inf'),
    g: float = 9.81,
    rho: float = 1000.0,
    n_theta: int = 100,
) -> float:
    """Compute wave resistance using Michell theory.
    
    The wave resistance is:
        R_w = (4πρg²/U²) * ∫₀^{π/2} |H(k_θ)|² * (1 - 2τ*sin(θ)) * 
              sqrt((1 - 2τ*sin(θ))/cos²(θ)) * sec³(θ) dθ
    
    where τ = g*U/(2*U²) = g/(2U) is a parameter, and k_θ is the wave
    number at angle θ.
    
    For simplicity, we use the deep-water limit and integrate over θ.
    
    Parameters
    ----------
    U : float
        Ship speed [m/s]
    L, B, T : float
        Hull dimensions [m]
    h : float
        Water depth [m]
    g : float
        Gravitational acceleration [m/s²]
    rho : float
        Water density [kg/m³]
    n_theta : int
        Number of integration points
        
    Returns
    -------
    float
        Wave resistance [N]
    """
    K = g / U ** 2
    tau = g / (2.0 * U ** 2)
    
    # Integration over θ from 0 to π/2
    theta = np.linspace(0, math.pi / 2.0 - 1e-6, n_theta)
    dtheta = theta[1] - theta[0]
    
    Rw = 0.0
    for th in theta:
        # Wave number at angle θ
        k_th = K / (1.0 - 2.0 * tau * math.sin(th))
        if k_th < 0:
            continue  # Beyond critical angle
        
        # Kochin function at this wave number
        H_th = _kochin_wigley(K, L, B, T, k_th)
        
        # Integrand
        factor = math.sqrt(abs(1.0 - 2.0 * tau * math.sin(th))) / math.cos(th) ** 2
        integrand = abs(H_th) ** 2 * factor * (1.0 / math.cos(th))
        
        Rw += integrand * dtheta
    
    # Prefactor
    Rw *= 4.0 * math.pi * rho * g ** 2 / U ** 2
    
    return Rw


def froude_number(U: float, L: float, g: float = 9.81) -> float:
    """Compute Froude number Fn = U / sqrt(g*L)."""
    return U / math.sqrt(g * L)


def speed_from_froude(Fn: float, L: float, g: float = 9.81) -> float:
    """Compute ship speed from Froude number."""
    return Fn * math.sqrt(g * L)
