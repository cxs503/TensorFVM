"""KVLCC2 tanker hull geometry definition.

KVLCC2 (Kriso Very Large Crude Carrier 2) is a standard ship hydrodynamics
benchmark hull. This module provides the hull offset data and geometric
properties needed for CFD simulations.

Reference: SIMMAN 2021 Workshop on Verification and Validation of Ship
Maneuvering Simulation Methods.

Hull principal dimensions:
    L_pp  = 7.000 m  (length between perpendiculars)
    B     = 0.936 m  (molded breadth)
    T     = 0.342 m  (design draft)
    Cb    = 0.708    (block coefficient)
    LCB   = -1.688%  (longitudinal center of buoyancy, fwd of midship)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional


@dataclass(frozen=True)
class KVLCC2PrincipalDimensions:
    """KVLCC2 principal dimensions (model scale, 1:31.6 prototype)."""
    L_pp: float = 7.000       # Length between perpendiculars [m]
    B: float = 0.936          # Molded breadth [m]
    T: float = 0.342          # Design draft [m]
    Cb: float = 0.708         # Block coefficient [-]
    LCB_fraction: float = -0.01688  # LCB as fraction of L_pp (fwd negative)
    
    @property
    def L(self) -> float:
        """Alias for L_pp."""
        return self.L_pp
    
    @property
    def displaced_volume(self) -> float:
        """Displaced volume ∇ = Cb * L * B * T [m³]."""
        return self.Cb * self.L_pp * self.B * self.T
    
    @property
    def wetted_surface_approx(self) -> float:
        """Approximate wetted surface using Holtrop-Mennen formula [m²]."""
        # S ≈ L*(2T + B) * (0.453 + 0.4425*Cb - 0.2862* Cm - 0.003467*B/T + 0.3696*Cwp)
        # Simplified: S ≈ L*(2T + B)*Cb^0.5 for quick estimate
        return self.L_pp * (2*self.T + self.B) * np.sqrt(self.Cb)
    
    @property
    def midship_section_coefficient(self) -> float:
        """Midship section coefficient Cm ≈ 0.985 for KVLCC2."""
        return 0.985
    
    @property
    def waterplane_coefficient(self) -> float:
        """Waterplane area coefficient Cwp ≈ 0.808 for KVLCC2."""
        return 0.808


@dataclass(frozen=True)
class KVLCC2HullOffsets:
    """KVLCC2 hull station offsets (half-breadths).
    
    Stations are given as fractions of L_pp from AP (x=0) to FP (x=L).
    Half-breadths are given as fractions of B/2.
    
    This is a simplified representation; full offset tables have 20+ stations
    with waterline heights. For CFD meshing, use the full table from SIMMAN.
    """
    # Station positions (fraction of L_pp from AP)
    stations: Tuple[float, ...] = (
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    )
    
    # Half-breadths at design waterline (fraction of B/2)
    # Approximate KVLCC2 shape based on published offset data
    half_breadths_wl: Tuple[float, ...] = (
        0.000, 0.120, 0.245, 0.380, 0.510, 0.625, 0.720, 0.800, 0.865, 0.915,
        0.950, 0.970, 0.985, 0.993, 0.998, 1.000, 0.998, 0.990, 0.970, 0.920, 0.000
    )
    
    def get_half_breadth(self, x_fraction: float) -> float:
        """Interpolate half-breadth at given x/L position."""
        return float(np.interp(x_fraction, self.stations, self.half_breadths_wl))
    
    def get_full_offsets(self, n_stations: int = 21, n_waterlines: int = 11) -> np.ndarray:
        """Generate full 2D offset table.
        
        Returns
        -------
        np.ndarray
            Shape (n_waterlines, n_stations) with half-breadth fractions.
        """
        x_frac = np.linspace(0, 1, n_stations)
        z_frac = np.linspace(0, 1, n_waterlines)  # 0=keel, 1=WL
        
        offsets = np.zeros((n_waterlines, n_stations))
        for i, zf in enumerate(z_frac):
            for j, xf in enumerate(x_frac):
                # Simplified: waterline shape scales with draft
                # Near keel, sections are fuller; near WL, finer
                wb = self.get_half_breadth(xf)
                # Bulbous bow effect near stern, flattening near keel
                shape_factor = 0.7 + 0.3 * zf  # fuller at waterline
                offsets[i, j] = wb * shape_factor
        
        # Stern transom: zero breadth at x=0 (AP)
        offsets[:, 0] = 0.0
        # Ensure waterline values match
        offsets[-1, :] = [self.get_half_breadth(xf) for xf in x_frac]
        
        return offsets


def generate_kvlcc2_surface_mesh(
    dims: Optional[KVLCC2PrincipalDimensions] = None,
    n_stations: int = 41,
    n_waterlines: int = 21,
    n_radial: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate KVLCC2 hull surface mesh points.
    
    Parameters
    ----------
    dims : KVLCC2PrincipalDimensions, optional
        Principal dimensions (default: KVLCC2 standard).
    n_stations : int
        Number of longitudinal stations.
    n_waterlines : int
        Number of waterlines from keel to WL.
    n_radial : int
        Number of points around each section (for circular interpolation).
    
    Returns
    -------
    vertices : np.ndarray
        Shape (N, 3) with (x, y, z) coordinates. x=forward, y=port, z=up.
    faces : np.ndarray
        Shape (M, 4) with quad face indices.
    """
    if dims is None:
        dims = KVLCC2PrincipalDimensions()
    
    offsets = KVLCC2HullOffsets()
    
    # Generate points
    x_stations = np.linspace(0, dims.L_pp, n_stations)
    z_levels = np.linspace(-dims.T, 0, n_waterlines)  # keel to WL
    
    vertices = []
    
    for i, x in enumerate(x_stations):
        x_frac = x / dims.L_pp
        for k, z in enumerate(z_levels):
            z_frac = (z + dims.T) / dims.T  # 0=keel, 1=WL
            
            # Get half-breadth at this station and waterline
            wb_base = offsets.get_half_breadth(x_frac)
            
            # Shape factor: fuller near keel
            shape_factor = 0.6 + 0.4 * z_frac
            wb = wb_base * shape_factor * (dims.B / 2)
            
            # Generate points around the section (port and starboard)
            # Use elliptical section shape
            for j in range(n_radial):
                theta = np.pi * j / (n_radial - 1)  # 0 to pi (port side)
                # Elliptical section: y = wb * sin(theta), z_offset from keel
                y = wb * np.sin(theta)
                z_actual = z + (dims.T + z) * (1 - np.cos(theta)) * 0.1  # slight bilge rounding
                
                vertices.append([x, y, z_actual])
                # Mirror for starboard
                vertices.append([x, -y, z_actual])
    
    vertices = np.array(vertices)
    
    # Generate quad faces
    n_points_per_station = n_waterlines * 2 * n_radial  # both sides
    faces = []
    
    for i in range(n_stations - 1):
        for k in range(n_waterlines - 1):
            for j in range(n_radial - 1):
                # Port side quad
                base = i * n_waterlines * 2 * n_radial + k * 2 * n_radial + j * 2
                v0 = base
                v1 = base + 2
                v2 = base + 2 * n_radial + 2
                v3 = base + 2 * n_radial
                faces.append([v0, v1, v2, v3])
    
    faces = np.array(faces) if faces else np.zeros((0, 4), dtype=int)
    
    return vertices, faces


def compute_hull_properties(
    dims: Optional[KVLCC2PrincipalDimensions] = None,
) -> dict:
    """Compute derived hull properties for CFD setup.
    
    Returns
    -------
    dict
        Dictionary with geometric properties needed for simulation setup.
    """
    if dims is None:
        dims = KVLCC2PrincipalDimensions()
    
    return {
        "L_pp_m": dims.L_pp,
        "B_m": dims.B,
        "T_m": dims.T,
        "Cb": dims.Cb,
        "LCB_fraction": dims.LCB_fraction,
        "displaced_volume_m3": dims.displaced_volume,
        "wetted_surface_m2": dims.wetted_surface_approx,
        "midship_Cm": dims.midship_section_coefficient,
        "waterplane_Cwp": dims.waterplane_coefficient,
        "waterplane_area_m2": dims.waterplane_coefficient * dims.L_pp * dims.B,
        "prismatic_coefficient": dims.Cb / dims.midship_section_coefficient,
    }


if __name__ == "__main__":
    import json
    dims = KVLCC2PrincipalDimensions()
    props = compute_hull_properties(dims)
    print(json.dumps(props, indent=2))
