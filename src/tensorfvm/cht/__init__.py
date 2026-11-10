"""Conjugate heat transfer (CHT) verification surface for TensorFVM.

The package adds the thermal side of the finite-volume stack: a fluid energy
equation coupled to solid conduction through an *explicit* interface flux
condition.  Two independent discretisations are provided on purpose, so that the
interface-conservation evidence does not rest on a single code path:

``flatplate``
    Single temperature field on a Cartesian orthogonal grid that straddles the
    fluid/solid interface.  The face conductance is the distance-weighted
    harmonic mean of the two cell diffusivities, so the interface flux is one and
    the same number on both sides -- continuity holds by construction, and the
    audit recomputes the two one-sided fluxes from the reconstructed interface
    temperature to prove it numerically.

``polar``
    Body-fitted polar O-grid around a circular cylinder.  The flow is solved with
    a streamfunction-vorticity finite-volume scheme (``PolarVorticitySolver``);
    the thermal problem is solved as *two* domains (solid core, fluid annulus)
    coupled by a Dirichlet-Neumann iteration, with a conservative area-weighted
    transfer across the interface.  Interface continuity is therefore an
    iterative property that has to be *earned*, not assumed, and the non-conformal
    variant (different angular resolutions) exercises the conservative mapping.

Every case writes a JSON ledger with the heat-flux residuals, so a reader can
check ``q_fluid - q_solid <= 1e-6`` and the global energy balance without
re-running the solver.
"""

from .ice_steel import (ICE, STEEL, LayeredWall, LayeredWall2D, Material,
                        TwoDomainCHT, analytic_series_field, interface_audit)
from .icing import (IcingEnvironment, IcingState, icing_state_adiabatic,
                    icing_state_with_conduction, load_w4n08_reference)
from .phase_change import (EnthalpySlab1D, NeumannSolution, PhaseChangeMaterial,
                           RadialBoundary, stefan_lambda)
from .similarity import BlasiusSolution, pohlhausen_nusselt, solve_blasius, solve_pohlhausen
from .flatplate import FlatPlateCHT, FlatPlateResult
from .polar import ConjugateCylinder, CylinderCHTResult, PolarVorticitySolver

__all__ = [
    "BlasiusSolution",
    "ConjugateCylinder",
    "CylinderCHTResult",
    "FlatPlateCHT",
    "FlatPlateResult",
    "PolarVorticitySolver",
    "EnthalpySlab1D",
    "IcingEnvironment",
    "IcingState",
    "LayeredWall",
    "LayeredWall2D",
    "Material",
    "NeumannSolution",
    "PhaseChangeMaterial",
    "RadialBoundary",
    "STEEL",
    "ICE",
    "TwoDomainCHT",
    "analytic_series_field",
    "icing_state_adiabatic",
    "icing_state_with_conduction",
    "interface_audit",
    "load_w4n08_reference",
    "stefan_lambda",
    "pohlhausen_nusselt",
    "solve_blasius",
    "solve_pohlhausen",
]