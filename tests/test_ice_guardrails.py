"""Tests for the ice-domain guardrail system.

Covers:
- Parameter bounds validation (hard/soft/typical)
- Non-finite value rejection (NaN, inf)
- Clamping behavior
- Guarded resistance calculation
- Benchmark rows (G1-G6)
"""

import math
import pytest

from tensorfvm.ice import (
    ICE_THICKNESS,
    ICE_STRENGTH,
    ICE_FRICTION,
    ICE_SPEED,
    ICE_DENSITY,
    ICE_ELASTIC_MODULUS,
    ICE_POISSON_RATIO,
    ICE_TEMPERATURE,
    SHIP_LENGTH,
    SHIP_BEAM,
    SHIP_DRAFT,
    SHIP_BLOCK_COEFFICIENT,
    ALL_ICE_PARAMETERS,
    ICE_RESISTANCE_INPUTS,
    ICE_ONLY_PARAMETERS,
    SHIP_PARAMETERS,
    GuardrailStatus,
    IceGuardrailResult,
    IceParameterDomainError,
    validate_ice_parameters,
    require_ice_parameters,
    clamp_ice_parameters,
    check_extrapolation_risk,
    get_parameter_report,
    guarded_level_ice_resistance,
    level_ice_resistance_kN,
    lindqvist_terms,
    IceShipGeometry,
    IceCondition,
    require_guarded_level_ice_resistance,
)


# =========================================================================== #
# 1. Parameter bounds validation
# =========================================================================== #

class TestParameterBounds:
    """Test individual parameter bounds."""

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_hard_bounds_reject_outside(self, param):
        """Values outside hard bounds must be rejected."""
        is_violation = param.bounds.is_hard_violation(param.bounds.hard_min - 1.0)
        assert is_violation
        is_violation = param.bounds.is_hard_violation(param.bounds.hard_max + 1.0)
        assert is_violation

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_hard_bounds_accept_inside(self, param):
        """Values inside hard bounds must be accepted."""
        is_violation = param.bounds.is_hard_violation(param.bounds.hard_min)
        assert not is_violation
        is_violation = param.bounds.is_hard_violation(param.bounds.hard_max)
        assert not is_violation
        mid = (param.bounds.hard_min + param.bounds.hard_max) / 2
        is_violation = param.bounds.is_hard_violation(mid)
        assert not is_violation

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_soft_bounds(self, param):
        """Soft bounds define the calibration domain."""
        is_soft = param.bounds.is_soft_violation(param.bounds.soft_min - 1.0)
        assert is_soft
        is_soft = param.bounds.is_soft_violation(param.bounds.soft_max + 1.0)
        assert is_soft
        is_soft = param.bounds.is_soft_violation(param.bounds.soft_min)
        assert not is_soft
        is_soft = param.bounds.is_soft_violation(param.bounds.soft_max)
        assert not is_soft

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_typical_bounds(self, param):
        """Typical bounds define the normal operating range."""
        is_typical = param.bounds.is_typical(param.bounds.typical_min)
        assert is_typical
        is_typical = param.bounds.is_typical(param.bounds.typical_max)
        assert is_typical
        is_typical = param.bounds.is_typical(param.bounds.typical_min - 1.0)
        assert not is_typical
        is_typical = param.bounds.is_typical(param.bounds.typical_max + 1.0)
        assert not is_typical

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_clamp(self, param):
        """Clamp must return values within hard bounds."""
        clamped = param.bounds.clamp(param.bounds.hard_min - 100.0)
        assert clamped == param.bounds.hard_min
        clamped = param.bounds.clamp(param.bounds.hard_max + 100.0)
        assert clamped == param.bounds.hard_max
        mid = (param.bounds.hard_min + param.bounds.hard_max) / 2
        clamped = param.bounds.clamp(mid)
        assert clamped == mid


# =========================================================================== #
# 2. Non-finite value rejection
# =========================================================================== #

class TestNonFiniteRejection:
    """Test that NaN and inf are always rejected."""

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_nan_rejected(self, param):
        """NaN must be rejected."""
        is_violation = param.bounds.is_hard_violation(float("nan"))
        assert is_violation

    @pytest.mark.parametrize("param", [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        SHIP_LENGTH, SHIP_BEAM, SHIP_DRAFT, SHIP_BLOCK_COEFFICIENT,
    ])
    def test_inf_rejected(self, param):
        """+inf and -inf must be rejected."""
        is_violation = param.bounds.is_hard_violation(float("inf"))
        assert is_violation
        is_violation = param.bounds.is_hard_violation(float("-inf"))
        assert is_violation

    def test_nan_validation_rejected(self):
        """validate_ice_parameters must reject NaN."""
        result = validate_ice_parameters({"ice_thickness": float("nan")})
        assert not result.is_valid
        assert result.status == GuardrailStatus.FAIL
        assert len(result.errors) > 0

    def test_inf_validation_rejected(self):
        """validate_ice_parameters must reject inf."""
        result = validate_ice_parameters({"ice_thickness": float("inf")})
        assert not result.is_valid
        assert result.status == GuardrailStatus.FAIL

    def test_bool_rejected(self):
        """Booleans must be rejected (not treated as 0/1)."""
        result = validate_ice_parameters({"ice_thickness": True})
        assert not result.is_valid
        assert result.status == GuardrailStatus.FAIL


# =========================================================================== #
# 3. Clamping behavior
# =========================================================================== #

class TestClamping:
    """Test clamp_ice_parameters behavior."""

    def test_clamp_hard_bounds(self):
        """Clamping must respect hard bounds."""
        result = clamp_ice_parameters({"ice_thickness": 50.0})
        assert result.clamped_params["ice_thickness"] == 10.0
        assert len(result.warnings) > 0

    def test_clamp_below_hard_bound(self):
        """Clamping must respect hard bounds below."""
        result = clamp_ice_parameters({"ice_thickness": 0.001})
        assert result.clamped_params["ice_thickness"] == 0.01

    def test_clamp_nan_rejected(self):
        """NaN cannot be clamped and must be rejected."""
        result = clamp_ice_parameters({"ice_thickness": float("nan")})
        assert "ice_thickness" not in result.clamped_params
        assert result.status == GuardrailStatus.FAIL

    def test_clamp_inf_clamped_to_bound(self):
        """Inf must be clamped to the nearest bound."""
        result = clamp_ice_parameters({"ice_thickness": float("inf")})
        assert result.clamped_params["ice_thickness"] == 10.0
        assert result.status == GuardrailStatus.WARN

    def test_clamp_typical_values_unchanged(self):
        """Typical values must not be clamped."""
        result = clamp_ice_parameters({"ice_thickness": 1.5, "ice_strength": 3.0})
        assert result.clamped_params["ice_thickness"] == 1.5
        assert result.clamped_params["ice_strength"] == 3.0
        assert len(result.warnings) == 0


# =========================================================================== #
# 4. Guarded resistance calculation
# =========================================================================== #

class TestGuardedResistance:
    """Test guarded_level_ice_resistance."""

    def test_typical_params_pass(self):
        """Typical parameters must pass validation."""
        params = {
            "ice_thickness": 1.5, "ice_strength": 3.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        result = guarded_level_ice_resistance(params, mode="reject")
        assert result.guard.is_valid
        assert result.resistance_kN is not None
        assert result.resistance_kN > 0

    def test_extreme_thickness_rejected(self):
        """Extreme ice thickness must be rejected."""
        params = {
            "ice_thickness": 50.0, "ice_strength": 3.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        result = guarded_level_ice_resistance(params, mode="reject")
        assert not result.guard.is_valid
        assert result.resistance_kN is None

    def test_extreme_strength_rejected(self):
        """Extreme ice strength must be rejected."""
        params = {
            "ice_thickness": 1.5, "ice_strength": 100.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        result = guarded_level_ice_resistance(params, mode="reject")
        assert not result.guard.is_valid
        assert result.resistance_kN is None

    def test_clamp_mode_extreme_thickness(self):
        """Clamp mode must clamp extreme thickness and compute resistance."""
        params = {
            "ice_thickness": 50.0, "ice_strength": 3.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        result = guarded_level_ice_resistance(params, mode="clamp")
        assert result.resistance_kN is not None
        assert result.resistance_kN > 0
        assert result.used_params.get("ice_thickness") == 10.0

    def test_clamp_mode_extreme_strength(self):
        """Clamp mode must clamp extreme strength and compute resistance."""
        params = {
            "ice_thickness": 1.5, "ice_strength": 100.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        result = guarded_level_ice_resistance(params, mode="clamp")
        assert result.resistance_kN is not None
        assert result.resistance_kN > 0
        assert result.used_params.get("ice_strength") == 30.0

    def test_unguarded_resistance_audit(self):
        """Unguarded resistance must be recorded for audit."""
        params = {
            "ice_thickness": 50.0, "ice_strength": 3.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        result = guarded_level_ice_resistance(params, mode="reject")
        assert result.unguarded_resistance_kN is not None
        assert result.unguarded_resistance_kN > 0
        # Unguarded resistance for h=50m should be much larger than guarded (clamped to 10m)
        clamp_result = guarded_level_ice_resistance(params, mode="clamp")
        assert result.unguarded_resistance_kN > clamp_result.resistance_kN


# =========================================================================== #
# 5. Lindqvist terms
# =========================================================================== #

class TestLindqvistTerms:
    """Test lindqvist_terms component calculation."""

    def test_typical_terms_positive(self):
        """All terms must be positive for typical inputs."""
        geometry = IceShipGeometry(length_m=150.0, beam_m=25.0, draft_m=8.0,
                                   block_coefficient=0.75, bow_angle_deg=30.0)
        ice = IceCondition(thickness_m=1.5, strength_mpa=3.0,
                           friction_coeff=0.1, speed_ms=2.5)
        terms = lindqvist_terms(geometry, ice)
        for key in ["crushing", "bending", "submersion", "speed"]:
            assert terms[key] > 0

    def test_velocity_multiplier(self):
        """Velocity multiplier must be > 1 for V > 0."""
        geometry = IceShipGeometry(length_m=150.0, beam_m=25.0, draft_m=8.0,
                                   block_coefficient=0.75, bow_angle_deg=30.0)
        ice = IceCondition(thickness_m=1.5, strength_mpa=3.0,
                           friction_coeff=0.1, speed_ms=2.5)
        terms = lindqvist_terms(geometry, ice)
        assert terms["velocity_multiplier"] > 1.0


# =========================================================================== #
# 6. Benchmark rows (G1-G6)
# =========================================================================== #

class TestBenchmarkRows:
    """Test benchmark rows G1-G6."""

    def test_g1_decision_grid(self):
        """G1: Decision conformance vs independent spec oracle."""
        from tensorfvm.ice.benchmark import row_g1_decision_grid
        row, detail = row_g1_decision_grid()
        assert row.passed
        assert row.error <= 0.03

    def test_g2_nan_containment(self):
        """G2: NaN containment rate."""
        from tensorfvm.ice.benchmark import row_g2_nan
        row = row_g2_nan()
        assert row.passed
        assert row.got == 1.0

    def test_g3_inf_containment(self):
        """G3: +-inf containment rate."""
        from tensorfvm.ice.benchmark import row_g3_inf
        row = row_g3_inf()
        assert row.passed
        assert row.got == 1.0

    def test_g4_load_path_closure(self):
        """G4: Out-of-domain inputs blocked before load arithmetic."""
        from tensorfvm.ice.benchmark import row_g4_load_path_closure
        row, detail = row_g4_load_path_closure()
        assert row.passed
        assert row.got == 1.0

    def test_g5_pass_through(self):
        """G5: In-domain pass-through (no load change)."""
        from tensorfvm.ice.benchmark import row_g5_passthrough
        row = row_g5_passthrough()
        assert row.passed
        assert row.error is not None and row.error < 1e-10

    def test_g6_legacy_anchor(self):
        """G6: Legacy anchor magnitudes (WITHHELD)."""
        from tensorfvm.ice.benchmark import row_g6_legacy
        row, detail = row_g6_legacy()
        # This row is expected to be WITHHELD, not PASS
        assert row.passed is False or row.note != ""


# =========================================================================== #
# 7. Utility functions
# =========================================================================== #

class TestUtilities:
    """Test utility functions."""

    def test_check_extrapolation_risk(self):
        """check_extrapolation_risk must identify extrapolation."""
        risks = check_extrapolation_risk({"ice_thickness": 6.0, "ice_strength": 3.0})
        assert risks["ice_thickness"][0] is True
        assert risks["ice_strength"][0] is False

    def test_get_parameter_report(self):
        """get_parameter_report must produce a string."""
        report = get_parameter_report({"ice_thickness": 1.5, "ice_strength": 3.0})
        assert isinstance(report, str)
        assert "ice_thickness" in report
        assert "ice_strength" in report

    def test_require_ice_parameters(self):
        """require_ice_parameters must raise on invalid input."""
        with pytest.raises(IceParameterDomainError):
            require_ice_parameters({"ice_thickness": 50.0})

    def test_require_ice_parameters_valid(self):
        """require_ice_parameters must return params on valid input."""
        params = require_ice_parameters({"ice_thickness": 1.5, "ice_strength": 3.0})
        assert params["ice_thickness"] == 1.5
        assert params["ice_strength"] == 3.0

    def test_require_guarded_level_ice_resistance(self):
        """require_guarded_level_ice_resistance must raise on invalid input."""
        params = {
            "ice_thickness": 50.0, "ice_strength": 3.0, "ice_friction": 0.1,
            "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
            "ship_length": 150.0, "ship_beam": 25.0, "ship_draft": 8.0,
            "ship_block_coefficient": 0.75,
        }
        with pytest.raises(IceParameterDomainError):
            require_guarded_level_ice_resistance(params)

    def test_known_parameters(self):
        """known_parameters must return a dict with all parameters."""
        from tensorfvm.ice.load import known_parameters
        params = known_parameters()
        assert len(params) == 12
        assert "ice_thickness" in params
        assert "ship_length" in params


# =========================================================================== #
# 8. Registry and constants
# =========================================================================== #

class TestRegistry:
    """Test registry and constants."""

    def test_all_ice_parameters(self):
        """ALL_ICE_PARAMETERS must contain all 12 parameters."""
        assert len(ALL_ICE_PARAMETERS) == 12

    def test_ice_resistance_inputs(self):
        """ICE_RESISTANCE_INPUTS must contain all 12 parameters."""
        assert len(ICE_RESISTANCE_INPUTS) == 12

    def test_ice_only_parameters(self):
        """ICE_ONLY_PARAMETERS must contain 8 ice parameters."""
        assert len(ICE_ONLY_PARAMETERS) == 8

    def test_ship_parameters(self):
        """SHIP_PARAMETERS must contain 4 ship parameters."""
        assert len(SHIP_PARAMETERS) == 4

    def test_parameter_names(self):
        """parameter_names must return all 12 parameter names."""
        from tensorfvm.ice.load import parameter_names
        names = parameter_names()
        assert len(names) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
