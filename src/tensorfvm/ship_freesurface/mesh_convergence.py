"""Mesh convergence analysis: Richardson extrapolation and order estimation.

This module provides tools for systematic mesh convergence studies following
the ITTC recommended procedures and ASME V&V 20 standards.

Given solutions on three meshes (coarse, medium, fine) with refinement ratio r,
the Richardson extrapolation estimates:
1. The exact solution S_0
2. The discretization error on each mesh
3. The observed order of convergence p

The method requires:
- Three systematically refined meshes (r = h_coarse/h_medium = h_medium/h_fine)
- Solutions in the asymptotic convergence range
- Monotonic convergence (or oscillatory with small amplitude)

References
----------
Roache, P.J. (1998). "Verification and Validation in CFD". Hermosa.
ITTC (2011). "Uncertainty analysis in CFD". 7.5-02-02-02.
ASME V&V 20 (2009). "Standard for Verification and Validation in CFD".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np


@dataclass
class RichardsonResult:
    """Results from Richardson extrapolation.
    
    Attributes
    ----------
    S_fine : float
        Solution on finest mesh
    S_medium : float
        Solution on medium mesh
    S_coarse : float
        Solution on coarsest mesh
    S_extrapolated : float
        Richardson extrapolated value (estimate of exact solution)
    p : float
        Observed order of convergence
    r : float
        Grid refinement ratio
    epsilon_fine : float
        Error estimate on fine mesh
    epsilon_medium : float
        Error estimate on medium mesh
    epsilon_coarse : float
        Error estimate on coarse mesh
    GCI_fine : float
        Grid Convergence Index on fine mesh (safety factor Fs=1.25)
    convergence_type : str
        'monotonic', 'oscillatory', or 'divergent'
    """
    S_fine: float
    S_medium: float
    S_coarse: float
    S_extrapolated: float
    p: float
    r: float
    epsilon_fine: float
    epsilon_medium: float
    epsilon_coarse: float
    GCI_fine: float
    convergence_type: str


def convergence_order(
    S_coarse: float,
    S_medium: float,
    S_fine: float,
    r: float = 1.414,
) -> Tuple[float, str]:
    """Estimate observed order of convergence p.
    
    Uses the formula:
        p = ln((S_coarse - S_medium) / (S_medium - S_fine)) / ln(r)
    
    Parameters
    ----------
    S_coarse, S_medium, S_fine : float
        Solutions on coarse, medium, fine meshes
    r : float
        Grid refinement ratio (h_coarse/h_medium)
        
    Returns
    -------
    tuple
        (p, convergence_type) where convergence_type is one of:
        'monotonic', 'oscillatory', 'divergent'
    """
    eps1 = S_medium - S_coarse
    eps2 = S_fine - S_medium
    
    # Check convergence type
    if eps1 * eps2 > 0:
        # Same sign: monotonic convergence
        convergence_type = "monotonic"
        if abs(eps2) > abs(eps1):
            convergence_type = "divergent"
    else:
        # Opposite sign: oscillatory
        convergence_type = "oscillatory"
    
    # Compute order (only meaningful for monotonic)
    if convergence_type == "monotonic" and abs(eps2) > 1e-15:
        ratio = abs(eps1 / eps2)
        if ratio > 0 and r > 1:
            p = math.log(ratio) / math.log(r)
        else:
            p = float('nan')
    elif convergence_type == "oscillatory":
        # For oscillatory, use approximate formula
        p = math.log(abs(eps1) / max(abs(eps2), 1e-15)) / math.log(r) if abs(eps2) > 1e-15 else float('nan')
    else:
        p = float('nan')
    
    return p, convergence_type


def richardson_extrapolation(
    S_coarse: float,
    S_medium: float,
    S_fine: float,
    r: float = 1.414,
    Fs: float = 1.25,
) -> RichardsonResult:
    """Perform Richardson extrapolation on three mesh solutions.
    
    Parameters
    ----------
    S_coarse, S_medium, S_fine : float
        Solutions on coarse, medium, fine meshes
    r : float
        Grid refinement ratio (default √2 ≈ 1.414)
    Fs : float
        Safety factor for GCI (default 1.25 for 3 grids)
        
    Returns
    -------
    RichardsonResult
        Complete extrapolation results
    """
    eps1 = S_medium - S_coarse
    eps2 = S_fine - S_medium
    
    # Convergence order
    p, convergence_type = convergence_order(S_coarse, S_medium, S_fine, r)
    
    # Richardson extrapolation (for monotonic convergence)
    if convergence_type == "monotonic" and not math.isnan(p) and p > 0:
        S_extrapolated = S_fine + eps2 / (r ** p - 1.0)
    else:
        # Fallback: use fine mesh value
        S_extrapolated = S_fine
    
    # Error estimates
    if convergence_type == "monotonic" and not math.isnan(p) and p > 0:
        epsilon_fine = abs(S_extrapolated - S_fine)
        epsilon_medium = abs(S_extrapolated - S_medium)
        epsilon_coarse = abs(S_extrapolated - S_coarse)
    else:
        epsilon_fine = abs(eps2)
        epsilon_medium = abs(eps1)
        epsilon_coarse = abs(eps1) * r
    
    # Grid Convergence Index (GCI)
    if abs(S_fine) > 1e-15:
        GCI_fine = Fs * epsilon_fine / abs(S_fine)
    else:
        GCI_fine = Fs * epsilon_fine
    
    return RichardsonResult(
        S_fine=S_fine,
        S_medium=S_medium,
        S_coarse=S_coarse,
        S_extrapolated=S_extrapolated,
        p=p,
        r=r,
        epsilon_fine=epsilon_fine,
        epsilon_medium=epsilon_medium,
        epsilon_coarse=epsilon_coarse,
        GCI_fine=GCI_fine,
        convergence_type=convergence_type,
    )


def generate_mesh_levels(
    base_resolution: Tuple[int, int],
    n_levels: int = 3,
    r: float = 1.414,
) -> List[Tuple[int, int]]:
    """Generate mesh resolution levels for convergence study.
    
    Parameters
    ----------
    base_resolution : tuple
        (nx, ny) of the finest mesh
    n_levels : int
        Number of mesh levels (default 3: coarse, medium, fine)
    r : float
        Refinement ratio between consecutive levels
        
    Returns
    -------
    list
        List of (nx, ny) tuples from coarsest to finest
    """
    nx_fine, ny_fine = base_resolution
    levels = []
    
    for i in range(n_levels):
        factor = r ** (n_levels - 1 - i)
        nx = max(10, int(nx_fine / factor))
        ny = max(10, int(ny_fine / factor))
        levels.append((nx, ny))
    
    return levels


def grid_convergence_study(
    solutions: List[float],
    refinement_ratios: List[float],
) -> List[RichardsonResult]:
    """Analyze convergence for multiple quantities.
    
    Parameters
    ----------
    solutions : list
        List of solution values (one per quantity of interest)
    refinement_ratios : list
        Refinement ratios for each quantity
        
    Returns
    -------
    list
        List of RichardsonResult for each quantity
    """
    results = []
    for S, r in zip(solutions, refinement_ratios):
        # This is a simplified interface; typically you'd pass S_coarse, S_medium, S_fine
        # For now, return empty results
        pass
    return results


def check_asymptotic_range(
    result: RichardsonResult,
    p_expected: float = 2.0,
    tol: float = 0.3,
) -> bool:
    """Check if solution is in asymptotic convergence range.
    
    The solution is in the asymptotic range if the observed order p
    is close to the expected order (e.g., p≈2 for second-order schemes).
    
    Parameters
    ----------
    result : RichardsonResult
        Richardson extrapolation result
    p_expected : float
        Expected order of convergence (default 2.0)
    tol : float
        Tolerance for order comparison (default 0.3)
        
    Returns
    -------
    bool
        True if in asymptotic range
    """
    if math.isnan(result.p):
        return False
    
    return abs(result.p - p_expected) <= tol
