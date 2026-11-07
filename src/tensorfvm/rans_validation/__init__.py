"""RANS turbulence-model validation matrix (k-epsilon / k-omega / SST-2003).

The package answers one question for the three models shipped or reachable from
TensorFVM: on a wall-bounded equilibrium channel, what law-of-the-wall does the
model's own constant set imply, and what deviation does the converged model
profile show against the reference wall laws?  It contains

* :mod:`.models`   -- published constant sets, pointwise equilibrium closures and
  the closed-form log-law slope ``kappa`` each set implies;
* :mod:`.channel`  -- 1-D wall-normal finite-volume half-channel solve (momentum
  integrated analytically, two transport equations solved conservatively);
* :mod:`.reference` -- reference wall laws and the log-region deviation metric;
* :mod:`.sensitivity` -- model-constant sweep of the friction coefficient;
* :mod:`.matrix`   -- the graded validation matrix and its machine-readable output.

Nothing here is a 3-D production solver and no result in this package is
evidence that the 3-D RANS solvers are validated.
"""
from .models import DEFAULT_CONSTANTS, MODELS, WALL_TREATMENT, constants_for, equilibrium_closure, implied_kappa
from .channel import solve_channel
from .reference import log_law, log_region_deviation, reichardt
from .sensitivity import constant_sweep

__all__ = ['DEFAULT_CONSTANTS', 'MODELS', 'WALL_TREATMENT', 'constants_for',
           'equilibrium_closure', 'implied_kappa', 'solve_channel', 'log_law',
           'log_region_deviation', 'reichardt', 'constant_sweep']