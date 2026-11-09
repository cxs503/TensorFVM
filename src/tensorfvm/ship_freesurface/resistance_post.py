"""Resistance post-processing: ITTC-57, decomposition, wave resistance isolation.

This module provides tools for decomposing total resistance into components
and comparing with experimental data using standard ITTC procedures.

The total resistance is:
    Rt = Rf + Rw + Rapp + Ra
    
where:
- Rf = friction resistance (ITTC-57 line)
- Rw = wave-making resistance
- Rapp = appendage resistance (not modeled here)
- Ra = air resistance (negligible for model scale)

For model-ship extrapolation (ITTC-78):
    Ct_model = Cf_model + Cv_model + Rw_model
    Ct_ship = Cf_ship + Cv_ship + Rw_ship + Ca
    
where Ca is the correlation allowance.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def ittc57_friction(Re: np.ndarray) -> np.ndarray:
    """ITTC-57 model-ship correlation line (vectorized).
    
    Cf = 0.075 / (log10(Re) - 2)²
    
    Parameters
    ----------
    Re : np.ndarray
        Reynolds number array
        
    Returns
    -------
    np.ndarray
        Friction coefficient array
    """
    return 0.075 / (np.log10(Re) - 2.0) ** 2


def schoenherr_friction(Re: np.ndarray) -> np.ndarray:
    """Schoenherr friction line (implicit form).
    
    1/sqrt(Cf) = 4.13 * log10(Cf * Re)
    
    Solved iteratively.
    
    Parameters
    ----------
    Re : np.ndarray
        Reynolds number array
        
    Returns
    -------
    np.ndarray
        Friction coefficient array
    """
    # Initial guess from ITTC-57
    Cf = ittc57_friction(Re)
    
    # Newton iteration
    for _ in range(20):
        lhs = 1.0 / np.sqrt(Cf)
        rhs = 4.13 * np.log10(Cf * Re)
        f = lhs - rhs
        # Derivative
        fp = -0.5 / Cf ** 1.5 - 4.13 / (Cf * np.log(10))
        Cf_new = Cf - f / fp
        if np.all(np.abs(Cf_new - Cf) < 1e-12):
            break
        Cf = Cf_new
    
    return Cf


def total_resistance_decomposition(
    Ct: float,
    Re: float,
    S: float,
    U: float,
    rho: float = 1025.0,
    form_factor: float = 1.1,
) -> Dict[str, float]:
    """Decompose total resistance coefficient into components.
    
    Ct = (1+k)*Cf + Rw
    
    Parameters
    ----------
    Ct : float
        Total resistance coefficient
    Re : float
        Reynolds number
    S : float
        Wetted surface area [m²]
    U : float
        Speed [m/s]
    rho : float
        Water density [kg/m³]
    form_factor : float
        Form factor (1+k)
        
    Returns
    -------
    dict
        Keys: 'Cf', 'Cv', 'Rw', 'Rtot', 'Rf', 'Rv', 'Rw_abs'
    """
    Cf = ittc57_friction(np.array([Re]))[0]
    Cv = form_factor * Cf  # Viscous resistance (friction + form)
    Rw = Ct - Cv  # Wave resistance coefficient
    
    # Absolute resistances
    q = 0.5 * rho * U ** 2
    Rtot = Ct * q * S
    Rf = Cf * q * S
    Rv = Cv * q * S
    Rw_abs = Rw * q * S
    
    return {
        "Cf": Cf,
        "Cv": Cv,
        "Rw": Rw,
        "Ct": Ct,
        "Rtot": Rtot,
        "Rf": Rf,
        "Rv": Rv,
        "Rw_abs": Rw_abs,
        "form_factor": form_factor,
    }


def wave_resistance_isolation(
    Ct_total: np.ndarray,
    Fn_array: np.ndarray,
    L: float,
    S: float,
    rho: float = 1025.0,
    nu: float = 1.19e-6,
    form_factor: float = 1.1,
) -> Dict[str, np.ndarray]:
    """Isolate wave resistance from total resistance data.
    
    Parameters
    ----------
    Ct_total : np.ndarray
        Total resistance coefficient at each Fn
    Fn_array : np.ndarray
        Froude numbers
    L : float
        Ship length [m]
    S : float
        Wetted surface area [m²]
    rho : float
        Water density [kg/m³]
    nu : float
        Kinematic viscosity [m²/s]
    form_factor : float
        Form factor (1+k)
        
    Returns
    -------
    dict
        Keys: 'Fn', 'Ct', 'Cf', 'Cv', 'Rw', 'Rtot', 'Rw_abs'
    """
    g = 9.81
    n = len(Fn_array)
    
    results = {
        "Fn": Fn_array,
        "Ct": Ct_total,
        "Cf": np.zeros(n),
        "Cv": np.zeros(n),
        "Rw": np.zeros(n),
        "Rtot": np.zeros(n),
        "Rw_abs": np.zeros(n),
    }
    
    for i, Fn in enumerate(Fn_array):
        U = Fn * math.sqrt(g * L)
        Re = U * L / nu
        
        Cf = ittc57_friction(np.array([Re]))[0]
        Cv = form_factor * Cf
        Rw = Ct_total[i] - Cv
        
        q = 0.5 * rho * U ** 2
        results["Cf"][i] = Cf
        results["Cv"][i] = Cv
        results["Rw"][i] = Rw
        results["Rtot"][i] = Ct_total[i] * q * S
        results["Rw_abs"][i] = Rw * q * S
    
    return results


def model_ship_extrapolation(
    Ct_model: float,
    Re_model: float,
    Re_ship: float,
    form_factor: float = 1.1,
    Ca: float = 0.0004,
) -> Dict[str, float]:
    """ITTC-78 model-ship extrapolation.
    
    Ct_ship = Cf_ship + (Ct_model - Cf_model)*(1+k) + Ca
    
    Wait, the correct form is:
    Ct_ship = Cf_ship + Cv_model + Rw_model + Ca
    where Cv_model = (1+k)*Cf_model and Rw_model = Ct_model - (1+k)*Cf_model
    
    Parameters
    ----------
    Ct_model : float
        Model total resistance coefficient
    Re_model : float
        Model Reynolds number
    Re_ship : float
        Ship Reynolds number
    form_factor : float
        Form factor (1+k)
    Ca : float
        Correlation allowance
        
    Returns
    -------
    dict
        Keys: 'Ct_ship', 'Cf_model', 'Cf_ship', 'Rw_model', 'Ca'
    """
    Cf_model = ittc57_friction(np.array([Re_model]))[0]
    Cf_ship = ittc57_friction(np.array([Re_ship]))[0]
    
    # Wave resistance (assumed same for model and ship at same Fn)
    Rw_model = Ct_model - form_factor * Cf_model
    
    # Ship total resistance
    Ct_ship = Cf_ship + form_factor * Cf_model + Rw_model + Ca
    
    return {
        "Ct_ship": Ct_ship,
        "Cf_model": Cf_model,
        "Cf_ship": Cf_ship,
        "Rw_model": Rw_model,
        "form_factor": form_factor,
        "Ca": Ca,
    }
