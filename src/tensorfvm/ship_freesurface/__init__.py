"""Ship free-surface verification suite for TensorFVM.

This module establishes the zero-to-one verification framework for
free-surface ship hydrodynamics in TensorFVM, covering:

1. **Wigley parabolic hull** — analytical anchor via Michell thin-ship theory
2. **Series 60** — systematic experiment anchor (Todd et al.)
3. **DTMB 5415** — benchmark experiment anchor (NASA/Iowa)

The suite validates:
- Geometry correctness (Cb, L, B, ∇ within 1%)
- Wave elevation vs analytical (Michell) for Wigley hull
- Total resistance vs experiment for Series 60 and DTMB 5415
- Mesh convergence (Richardson extrapolation, order p ≥ 1.5)
- Mass conservation (inlet/outlet balance ≤ 0.1%)

Applicability domain: Fn < 0.5 (displacement type), no breaking waves,
no ventilation, no transom stern effects.

References
----------
- Michell, J.H. (1898). "The wave resistance of a ship". Phil. Mag.
- Todd, F.H. et al. (1963). "Series 60 systematic experiments". SNAME.
- Longo, J. & Stern, F. (2005). "Uncertainty assessment for DTMB 5415".
  Iowa Institute of Hydraulic Research.
"""

from .geometry import (
    WigleyHull,
    Series60Hull,
    DTMB5415Hull,
    validate_geometry,
)
from .michell_theory import (
    michell_wave_elevation,
    michell_wave_cut,
    michell_wave_resistance,
)
from .experimental_data import (
    todd_series60_resistance,
    dtmb5415_experiment,
    kelvin_wave_angle,
)
from .resistance_post import (
    ittc57_friction,
    total_resistance_decomposition,
    wave_resistance_isolation,
)
from .mesh_convergence import (
    richardson_extrapolation,
    convergence_order,
    generate_mesh_levels,
)

__all__ = [
    # Geometry
    "WigleyHull",
    "Series60Hull",
    "DTMB5415Hull",
    "validate_geometry",
    # Analytical
    "michell_wave_elevation",
    "michell_wave_cut",
    "michell_wave_resistance",
    # Experimental data
    "todd_series60_resistance",
    "dtmb5415_experiment",
    "kelvin_wave_angle",
    # Resistance post-processing
    "ittc57_friction",
    "total_resistance_decomposition",
    "wave_resistance_isolation",
    # Mesh convergence
    "richardson_extrapolation",
    "convergence_order",
    "generate_mesh_levels",
]
