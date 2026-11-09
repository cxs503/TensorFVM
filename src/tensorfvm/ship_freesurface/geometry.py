"""Ship hull geometry definitions and validation.

This module provides parametric geometry for three standard ship hulls
used in CFD verification:

1. **Wigley parabolic hull** — simple mathematical definition, exact Cb=2/3
2. **Series 60** — systematic model series, Cb=0.6 standard
3. **DTMB 5415** — modern naval combatant, Cb≈0.55

All geometries include:
- Principal dimensions (L, B, T)
- Block coefficient Cb
- Displacement volume ∇
- Wetted surface area S
- Half-breadth function y_half(x, z) for mesh generation

Geometry validation checks Cb, L, B, ∇ consistency within 1% tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class HullGeometry:
    """Base class for ship hull geometry."""
    L: float
    B: float
    T: float
    Cb: float
    name: str = "generic"
    
    @property
    def volume(self) -> float:
        """Displacement volume ∇ = Cb * L * B * T [m³]"""
        return self.Cb * self.L * self.B * self.T
    
    @property
    def wetted_surface(self) -> float:
        """Approximate wetted surface area [m²]."""
        return self.L * (self.B + 2.0 * self.T)
    
    def half_breadth(self, x: float, z: float) -> float:
        raise NotImplementedError("Subclass must implement half_breadth")
    
    def station_area(self, x: float) -> float:
        """Cross-sectional area at station x by numerical integration."""
        nz = 40
        z_vals = np.linspace(0, self.T, nz)
        y_vals = np.array([self.half_breadth(x, z) for z in z_vals])
        return 2.0 * np.trapezoid(y_vals, z_vals)
    
    def validate(self, tol: float = 0.01) -> dict:
        """Validate geometry consistency."""
        nx = 60
        x_vals = np.linspace(0, self.L, nx)
        areas = np.array([self.station_area(x) for x in x_vals])
        vol_numerical = np.trapezoid(areas, x_vals)
        vol_analytical = self.volume
        
        rel_error = abs(vol_numerical - vol_analytical) / vol_analytical
        Cb_numerical = vol_numerical / (self.L * self.B * self.T)
        Cb_error = abs(Cb_numerical - self.Cb) / self.Cb
        
        return {
            "valid": rel_error <= tol and Cb_error <= tol,
            "volume_analytical": vol_analytical,
            "volume_numerical": vol_numerical,
            "relative_error": rel_error,
            "Cb_analytical": self.Cb,
            "Cb_numerical": Cb_numerical,
            "Cb_error": Cb_error,
            "L": self.L,
            "B": self.B,
            "T": self.T,
        }


class WigleyHull(HullGeometry):
    """Wigley parabolic hull, Cb = 2/3 exactly.
    
    Geometry: y(x,z) = (B/2) * [1 - (2x/L - 1)²], wall-sided sections.
    """
    
    def __init__(self, L: float = 3.0, B: float = 0.3, T: float = 0.1875):
        super().__init__(L=L, B=B, T=T, Cb=2.0/3.0, name="Wigley")
    
    def half_breadth(self, x: float, z: float) -> float:
        if x < 0 or x > self.L or z < 0 or z > self.T:
            return 0.0
        xi = 2.0 * x / self.L - 1.0
        return (self.B / 2.0) * (1.0 - xi * xi)
    
    @property
    def wetted_surface(self) -> float:
        return self.L * self.B * (1.0 + 2.0 * self.T / self.B)


class Series60Hull(HullGeometry):
    """Series 60 systematic model hull, Cb = 0.6.
    
    Uses a calibrated parametric form matching principal dimensions and Cb.
    The waterline shape y_max(x) = (B/2) * [1 - |2x/L - 1|^n] with n calibrated
    so that the numerically integrated volume matches Cb*L*B*T within 1%.
    Sections are wall-sided (constant half-breadth in z).
    """
    
    def __init__(self, L: float = 3.048, B: float = 0.4064, 
                 T: float = 0.254, Cb: float = 0.6):
        super().__init__(L=L, B=B, T=T, Cb=Cb, name="Series60_Cb0.6")
        self._calibrate()
    
    def _calibrate(self):
        """Calibrate waterline exponent n to match target Cb.
        
        The waterline integral I(n) = ∫₀¹ (1-|2ξ-1|^n) dξ increases with n:
        - n=1: I=0.5
        - n=1.5: I=0.6
        - n=2: I=2/3≈0.667
        
        For Cb=0.6, we need n≈1.5.
        """
        n_lo, n_hi = 0.5, 3.0
        for _ in range(40):
            n_mid = (n_lo + n_hi) / 2
            xi = np.linspace(0, 1, 200)
            wl = 1.0 - np.abs(2.0 * xi - 1.0) ** n_mid
            wl_int = np.trapezoid(wl, xi)
            # I(n) increases with n; if I > Cb, need smaller n
            if wl_int > self.Cb:
                n_hi = n_mid
            else:
                n_lo = n_mid
        
        self._wl_exponent = (n_lo + n_hi) / 2
    
    def half_breadth(self, x: float, z: float) -> float:
        """Series 60 half-breadth with calibrated waterline, wall-sided."""
        if x < 0 or x > self.L or z < 0 or z > self.T:
            return 0.0
        
        xi = x / self.L
        waterline = 1.0 - abs(2.0 * xi - 1.0) ** self._wl_exponent
        
        return (self.B / 2.0) * waterline


class DTMB5415Hull(HullGeometry):
    """DTMB 5415 naval combatant hull, Cb ≈ 0.55."""
    
    def __init__(self, L: float = 5.72, B: float = 0.76, 
                 T: float = 0.248, Cb: float = 0.55):
        super().__init__(L=L, B=B, T=T, Cb=Cb, name="DTMB5415")
        self._calibrate()
    
    def _calibrate(self):
        """Calibrate vertical exponent to match Cb."""
        xi = np.linspace(0, 1, 200)
        wl = np.zeros_like(xi)
        for i, xv in enumerate(xi):
            if xv < 0.5:
                wl[i] = math.sin(math.pi * xv ** 0.7)
            else:
                xa = (xv - 0.5) / 0.5
                wl[i] = math.cos(0.5 * math.pi * xa ** 1.5)
        wl_integral = np.trapezoid(wl, xi)
        
        # Cb = wl_integral / (p+1) => p = wl_integral/Cb - 1
        target_p = wl_integral / self.Cb - 1.0
        self._vertical_exp = max(0.0, target_p)
    
    def half_breadth(self, x: float, z: float) -> float:
        """DTMB 5415 half-breadth (simplified parametric)."""
        if x < 0 or x > self.L or z < 0 or z > self.T:
            return 0.0
        
        xi = x / self.L
        zeta = z / self.T
        
        if xi < 0.5:
            waterline = math.sin(math.pi * xi ** 0.7)
        else:
            xi_aft = (xi - 0.5) / 0.5
            waterline = math.cos(0.5 * math.pi * xi_aft ** 1.5)
        
        vertical = zeta ** self._vertical_exp
        
        return (self.B / 2.0) * waterline * vertical


def validate_geometry(hull: HullGeometry, tol: float = 0.01) -> dict:
    """Validate hull geometry consistency."""
    if hull.L <= 0 or hull.B <= 0 or hull.T <= 0:
        return {"valid": False, "error": "Dimensions must be positive"}
    
    if not (0 < hull.Cb < 1):
        return {"valid": False, "error": "Cb must be in (0, 1)"}
    
    result = hull.validate(tol)
    
    nx, nz = 20, 10
    x_vals = np.linspace(0, hull.L, nx)
    z_vals = np.linspace(0, hull.T, nz)
    min_y = float('inf')
    for x in x_vals:
        for z in z_vals:
            y = hull.half_breadth(x, z)
            if y < min_y:
                min_y = y
    
    result["half_breadth_min"] = min_y
    result["half_breadth_valid"] = min_y >= -1e-10
    
    return result
