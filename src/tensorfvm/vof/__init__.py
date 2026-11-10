"""VOF ice-water two-phase free-surface capability.

Public surface:

* :mod:`tensorfvm.vof.thinc` -- THINC interface reconstruction / flux transport.
* :mod:`tensorfvm.vof.ice_water` -- variable-density projection solver with a VOF
  free surface, an immiscible ``ice`` phase (Brinkman-rigid) and analytic
  free-surface anchors (Ritter dam break, standing wave).
"""

from . import ice_water, thinc

__all__ = ["ice_water", "thinc"]