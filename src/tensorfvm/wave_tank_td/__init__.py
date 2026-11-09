"""Time-domain numerical wave tank (NWT) for TensorFVM -- sigma-grid solver.

Domain directory added by campaign node ``P106W2-N05``.  It is deliberately
separate from :mod:`tensorfvm.wave_tank` (the VOF tank of ``p106w1-n03``, whose
wave-propagation accuracy was recorded as an honest FAIL): this tank puts the
free surface *on* a coordinate line, so the interface schemes that damped the
VOF tank do not exist here.

Modules
-------
``wave_target``  Stokes-II target wave (elevation + orbital velocity) used by
                 the relaxation zone; first order reuses
                 :mod:`tensorfvm.wave_tank.theory`.
``sigma_tank``   the tank: sigma (surface-fitted) grid, exact algebraic pressure
                 projection, relaxation-zone generation, absorbing beach,
                 cut-cell (area-fraction) floating section with heave/pitch.
``potential2d``  independent 2-D frequency-domain potential-flow reference
                 (added mass, damping, excitation, RAO) used to cross-check the
                 seakeeping results.
``cases``        reproducible case drivers writing evidence JSON + figures.
"""
from .sigma_tank import SigmaWaveTank, TankConfig, points_in_polygon, polygon_properties
from .wave_target import WaveTarget, stokes2_eta, stokes2_potential_velocities

__all__ = [
    "SigmaWaveTank", "TankConfig", "WaveTarget",
    "stokes2_eta", "stokes2_potential_velocities",
    "points_in_polygon", "polygon_properties",
]