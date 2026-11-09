"""Time-domain numerical wave tank (NWT) for ship seakeeping.

Public surface
--------------
``theory``      closed-form wave targets and analysis (dispersion, Stokes II,
                Goda two-point reflection, three-probe wavenumber, added
                resistance, RAO).
``geometry``    hull forms plus the Cb / L/B / B/T / nabla geometry pre-gate.
``solver``      the 2-D MAC / free-surface tank itself (relaxation-zone
                generation, damping-zone absorption, heave+pitch body).
``bem2d``       2-D frequency-domain potential-flow reference (added mass,
                damping, excitation, drift force) used for cross-validation.
"""
from .geometry import BargeSection, KCSSimplifiedHull, WigleyHull, geometry_gate, polygon_mask
from .solver import RigidBody2D, TankConfig, WaveTank

__all__ = [
    "WaveTank", "TankConfig", "RigidBody2D",
    "WigleyHull", "KCSSimplifiedHull", "BargeSection", "geometry_gate", "polygon_mask",
]
