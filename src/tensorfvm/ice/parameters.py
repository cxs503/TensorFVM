"""Ice parameter definitions with physically validated bounds.

All bounds are derived from peer-reviewed literature and international standards.
Values are in SI units unless otherwise noted.

Bound types
-----------
- **hard_min / hard_max**: Outside the validity domain of ship–ice models.
  Inputs beyond these are rejected outright: either physically impossible
  (e.g. negative thickness, ``nan``) or three orders of magnitude past the
  largest hull/ice case the empirical formulas were fitted on.
- **soft_min / soft_max**: Outside the calibration domain of empirical models.
  Inputs beyond these trigger a warning but are accepted (extrapolation risk).
- **typical_min / typical_max**: Common operating range for Arctic shipping.

References for each parameter are documented inline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IceParameterBounds:
    """Physical bounds for a single ice parameter."""
    name: str
    symbol: str
    unit: str
    hard_min: float
    hard_max: float
    soft_min: float
    soft_max: float
    typical_min: float
    typical_max: float
    reference: str
    description: str

    def is_hard_violation(self, value: float) -> bool:
        """Return True if value is physically impossible.

        Non-finite values are always violations: ``nan < lo`` and ``nan > hi``
        are both False, so a bare comparison would report NaN as admissible
        (the m22209 defect this module repairs).
        """
        if not math.isfinite(value):
            return True
        return value < self.hard_min or value > self.hard_max

    def is_soft_violation(self, value: float) -> bool:
        """Return True if value is outside the calibration domain."""
        return value < self.soft_min or value > self.soft_max

    def is_typical(self, value: float) -> bool:
        """Return True if value is in the typical operating range."""
        return self.typical_min <= value <= self.typical_max

    def clamp(self, value: float) -> float:
        """Clamp value to hard bounds."""
        return max(self.hard_min, min(self.hard_max, value))


@dataclass(frozen=True)
class IceParameter:
    """Named ice parameter with bounds and validation."""
    bounds: IceParameterBounds

    def validate(self, value: float) -> tuple[bool, Optional[str]]:
        """Validate a value. Returns (is_valid, error_message)."""
        if self.bounds.is_hard_violation(value):
            return False, (
                f"{self.bounds.name} ({self.bounds.symbol}) = {value} {self.bounds.unit} "
                f"is outside physical bounds "
                f"[{self.bounds.hard_min}, {self.bounds.hard_max}] {self.bounds.unit}"
            )
        return True, None

    def check_extrapolation(self, value: float) -> tuple[bool, Optional[str]]:
        """Check if value requires extrapolation. Returns (needs_extrapolation, warning)."""
        if self.bounds.is_soft_violation(value):
            return True, (
                f"{self.bounds.name} ({self.bounds.symbol}) = {value} {self.bounds.unit} "
                f"is outside calibration domain "
                f"[{self.bounds.soft_min}, {self.bounds.soft_max}] {self.bounds.unit} "
                f"— results may be unreliable"
            )
        return False, None


# =========================================================================== #
# Ice thickness (h)
# =========================================================================== #
# Level ice: 0.1–2 m (first-year), 1–3 m (multi-year)
# Ridged ice: up to 5 m (keel depth can reach 20–30 m but consolidated layer < 5 m)
# Maximum observed: ~5 m consolidated level ice (Amundsen Gulf)
# 50 m is physically impossible — no natural process produces such thick level ice
#
# Ref: ISO 19906:2019 Annex C; Timco & Frederking (2012)
ICE_THICKNESS = IceParameter(IceParameterBounds(
    name="ice_thickness",
    symbol="h",
    unit="m",
    hard_min=0.01,       # 1 cm — thin ice still has mechanical effect
    hard_max=10.0,       # 10 m — ship-level-ice models' applicability limit, NOT a
                         # physical impossibility: ridge keels and glacier tongues far exceed it
    soft_min=0.05,       # 5 cm — below this, ice effects are negligible
    soft_max=5.0,        # 5 m — beyond this is extremely rare ridged ice
    typical_min=0.3,     # 30 cm — typical first-year ice
    typical_max=2.5,     # 2.5 m — typical multi-year ice
    reference="ISO 19906:2019 Annex C; Timco & Frederking (2012)",
    description="Ice thickness for level ice. Ridged ice equivalent thickness rarely exceeds 5 m.",
))

# =========================================================================== #
# Ice compressive strength (σ_c)
# =========================================================================== #
# First-year ice: 0.5–5 MPa (warm ice, -2°C to -5°C)
# Multi-year ice: 2–10 MPa (cold ice, -10°C to -30°C)
# Laboratory tests: up to ~20 MPa for very cold, saline-free ice
# 100 MPa is unrealistic — exceeds the strength of many structural steels
#
# Ref: Timco & Weeks (2010) "A review of the strength of sea ice", Cold Regions Sci. Tech.
ICE_STRENGTH = IceParameter(IceParameterBounds(
    name="ice_strength",
    symbol="σ_c",
    unit="MPa",
    hard_min=0.1,        # 0.1 MPa — very weak ice (slush)
    hard_max=30.0,       # 30 MPa — absolute upper bound for natural ice
    soft_min=0.3,        # 0.3 MPa — below this, ice is essentially slush
    soft_max=15.0,       # 15 MPa — beyond this is rare laboratory-grade ice
    typical_min=1.0,     # 1 MPa — typical first-year ice
    typical_max=5.0,     # 5 MPa — typical multi-year ice
    reference="Timco & Weeks (2010); ISO 19906:2019",
    description="Uniaxial compressive strength of sea ice. Strongly temperature-dependent.",
))

# =========================================================================== #
# Ice-ship friction coefficient (μ)
# =========================================================================== #
# Steel hull on ice: 0.05–0.20 (lubricated by meltwater)
# Dry ice contact: up to 0.30
# Coated hulls: as low as 0.02
# Values above 0.5 are unrealistic for ship-ice interaction
#
# Ref: Riska et al. (1997); ITTC Ice Committee (2017)
ICE_FRICTION = IceParameter(IceParameterBounds(
    name="ice_friction",
    symbol="μ",
    unit="",
    hard_min=0.0,        # Frictionless (theoretical limit)
    hard_max=1.0,        # Absolute upper bound (Coulomb friction)
    soft_min=0.01,       # Below this, friction is negligible
    soft_max=0.4,        # Beyond this is unrealistic for ship-ice
    typical_min=0.05,    # Typical lubricated steel-ice
    typical_max=0.20,    # Typical dry steel-ice
    reference="Riska et al. (1997) VTT Publ. 319; ITTC Ice Committee (2017)",
    description="Kinetic friction coefficient between ship hull and ice.",
))

# =========================================================================== #
# Ship speed in ice (v)
# =========================================================================== #
# Level ice transit: 0.5–3 m/s (1–6 knots)
# Ridge breaking: 0.2–1 m/s
# Maximum practical: ~5 m/s (10 knots) in thin ice
# Higher speeds cause excessive ice loads and structural risk
#
# Ref: ITTC Ice Committee procedures; Riska (1997)
ICE_SPEED = IceParameter(IceParameterBounds(
    name="ice_speed",
    symbol="v",
    unit="m/s",
    hard_min=0.0,        # Stationary (ice crushing under own weight)
    hard_max=15.0,       # 15 m/s (~30 knots) — absolute upper bound
    soft_min=0.1,        # Below this, quasi-static analysis applies
    soft_max=8.0,        # Beyond this, dynamic effects dominate
    typical_min=0.5,     # 0.5 m/s (~1 knot) — slow ice transit
    typical_max=5.0,     # 5 m/s (~10 knots) — fast ice transit
    reference="ITTC Ice Committee (2017); Riska et al. (1997)",
    description="Ship speed through ice. Higher speeds increase ice resistance nonlinearly.",
))

# =========================================================================== #
# Ice density (ρ_ice)
# =========================================================================== #
# Pure ice: 917 kg/m³ at 0°C
# Sea ice: 850–920 kg/m³ (depends on salinity and porosity)
# Multi-year ice: ~900 kg/m³ (desalinated)
#
# Ref: Timco & Frederking (2012)
ICE_DENSITY = IceParameter(IceParameterBounds(
    name="ice_density",
    symbol="ρ_ice",
    unit="kg/m³",
    hard_min=800.0,      # Very porous ice
    hard_max=920.0,      # Pure ice at 0°C
    soft_min=850.0,      # Typical sea ice lower bound
    soft_max=918.0,      # Pure ice upper bound
    typical_min=880.0,   # Typical first-year sea ice
    typical_max=910.0,   # Typical multi-year ice
    reference="Timco & Frederking (2012); ISO 19906:2019",
    description="Density of sea ice. Depends on salinity, temperature, and porosity.",
))

# =========================================================================== #
# Ice elastic modulus (E)
# =========================================================================== #
# Sea ice: 1–10 GPa (depends on temperature, salinity, loading rate)
# Freshwater ice: 8–12 GPa
# 100 GPa is unrealistic — exceeds aluminum alloys
#
# Ref: Timco & Weeks (2010); Schulson & Duval (2009)
ICE_ELASTIC_MODULUS = IceParameter(IceParameterBounds(
    name="ice_elastic_modulus",
    symbol="E",
    unit="GPa",
    hard_min=0.1,        # Very weak ice
    hard_max=20.0,       # Absolute upper bound for ice
    soft_min=1.0,        # Below this, ice behaves plastically
    soft_max=12.0,       # Beyond this is freshwater ice territory
    typical_min=3.0,     # Typical sea ice
    typical_max=8.0,     # Typical cold sea ice
    reference="Timco & Weeks (2010); Schulson & Duval (2009)",
    description="Young's modulus of ice. Strongly temperature and rate dependent.",
))

# =========================================================================== #
# Ice Poisson's ratio (ν)
# =========================================================================== #
# Ice: 0.30–0.35 (narrow range)
#
# Ref: Schulson & Duval (2009) "Creep and Fracture of Ice"
ICE_POISSON_RATIO = IceParameter(IceParameterBounds(
    name="ice_poisson_ratio",
    symbol="ν",
    unit="",
    hard_min=0.0,        # Theoretical lower bound
    hard_max=0.5,        # Theoretical upper bound (incompressible)
    soft_min=0.25,       # Below this is unrealistic for ice
    soft_max=0.40,       # Above this is unrealistic for ice
    typical_min=0.30,    # Typical lower bound
    typical_max=0.35,    # Typical upper bound
    reference="Schulson & Duval (2009)",
    description="Poisson's ratio of ice. Relatively constant across ice types.",
))

# =========================================================================== #
# Ice temperature (T)
# =========================================================================== #
# Sea ice surface: -2°C to -40°C (Arctic winter)
# Sea ice bottom: ~-1.8°C (freezing point of seawater)
# Below -50°C is extremely rare; above 0°C ice melts
#
# Ref: Timco & Frederking (2012)
ICE_TEMPERATURE = IceParameter(IceParameterBounds(
    name="ice_temperature",
    symbol="T",
    unit="°C",
    hard_min=-60.0,      # Extreme Arctic record
    hard_max=0.0,        # Melting point
    soft_min=-45.0,      # Below this is extremely rare
    soft_max=-1.0,       # Above this, ice is near melting
    typical_min=-30.0,   # Typical Arctic winter
    typical_max=-5.0,    # Typical shoulder season
    reference="Timco & Frederking (2012); WMO Sea Ice Nomenclature",
    description="Ice temperature. Controls strength, modulus, and creep behavior.",
))

# =========================================================================== #
# Ship geometry parameters (the remaining entries of the 12-parameter
# level-ice resistance input set).  The ice parameters above are only half of
# what an ice-resistance formula consumes; h=50 m and sigma_c=100 MPa were two
# of seven silently extrapolated inputs recorded by the JOUR-ICE-STRUCT-GUARD
# audit, so the hull variables are guarded by the same three-tier mechanism.
# Bounds are the ordinary displacement/icebreaker envelope of naval
# architecture (L/B in 4-9, T/L in 0.02-0.08, Cb in 0.50-0.90); they are
# deliberately wide — the guard is not a hull-design checker.
# =========================================================================== #
SHIP_LENGTH = IceParameter(IceParameterBounds(
    name="ship_length",
    symbol="L",
    unit="m",
    hard_min=10.0,       # 10 m — smallest autonomous craft in ice trials
    hard_max=500.0,      # 500 m — beyond any built hull
    soft_min=30.0,       # below this, model-scale / small craft
    soft_max=400.0,      # largest ice-capable hulls
    typical_min=50.0,    # small icebreaker / research vessel
    typical_max=350.0,   # Arctic tanker / heavy icebreaker
    reference="Lewis (1988) Principles of Naval Architecture; Riska et al. (1997)",
    description="Length between perpendiculars. Scales crushing/submersion terms linearly.",
))

SHIP_BEAM = IceParameter(IceParameterBounds(
    name="ship_beam",
    symbol="B",
    unit="m",
    hard_min=2.0,
    hard_max=80.0,       # wider than any built ice-going hull
    soft_min=8.0,
    soft_max=60.0,
    typical_min=15.0,
    typical_max=50.0,
    reference="Lewis (1988) Principles of Naval Architecture",
    description="Beam at waterline. Enters the bending and submersion terms directly.",
))

SHIP_DRAFT = IceParameter(IceParameterBounds(
    name="ship_draft",
    symbol="T",
    unit="m",
    hard_min=0.3,
    hard_max=20.0,       # deeper than any ice-going hull
    soft_min=2.0,
    soft_max=16.0,
    typical_min=5.0,
    typical_max=14.0,
    reference="Lewis (1988); Riska et al. (1997)",
    description="Draft at the ice belt. Governs the displaced volume and submersion term.",
))

SHIP_BLOCK_COEFFICIENT = IceParameter(IceParameterBounds(
    name="ship_block_coefficient",
    symbol="C_B",
    unit="",
    hard_min=0.30,       # finer than a fast craft
    hard_max=0.90,       # fuller than any practical hull
    soft_min=0.45,       # icebreakers run fine (0.50-0.70)
    soft_max=0.88,
    typical_min=0.55,
    typical_max=0.85,
    reference="Lewis (1988) Principles of Naval Architecture",
    description="Block coefficient; consistency check with L, B, T and displacement.",
))

# Registry of all guarded ice-domain parameters
ALL_ICE_PARAMETERS: dict[str, IceParameter] = {
    "ice_thickness": ICE_THICKNESS,
    "ice_strength": ICE_STRENGTH,
    "ice_friction": ICE_FRICTION,
    "ice_speed": ICE_SPEED,
    "ice_density": ICE_DENSITY,
    "ice_elastic_modulus": ICE_ELASTIC_MODULUS,
    "ice_poisson_ratio": ICE_POISSON_RATIO,
    "ice_temperature": ICE_TEMPERATURE,
    "ship_length": SHIP_LENGTH,
    "ship_beam": SHIP_BEAM,
    "ship_draft": SHIP_DRAFT,
    "ship_block_coefficient": SHIP_BLOCK_COEFFICIENT,
}

#: The eight ice-property parameters (the m22209 delivery).
ICE_ONLY_PARAMETERS: tuple[str, ...] = (
    "ice_thickness",
    "ice_strength",
    "ice_friction",
    "ice_speed",
    "ice_density",
    "ice_elastic_modulus",
    "ice_poisson_ratio",
    "ice_temperature",
)

#: The four hull parameters needed by a level-ice resistance calculation.
SHIP_PARAMETERS: tuple[str, ...] = (
    "ship_length",
    "ship_beam",
    "ship_draft",
    "ship_block_coefficient",
)

#: The full 12-parameter level-ice resistance input set.
ICE_RESISTANCE_INPUTS: tuple[str, ...] = ICE_ONLY_PARAMETERS + SHIP_PARAMETERS
