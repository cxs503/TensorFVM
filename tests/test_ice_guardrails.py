"""Tests for ice-domain guardrail validation.

Covers:
- Hard bound rejection (physically impossible values)
- Soft bound warnings (extrapolation risk)
- Typical range acceptance
- Clamping behavior
- Parameter report generation
- Integration with the full parameter set

References for test values:
- ISO 19906:2019 Annex C (ice thickness bounds)
- Timco & Weeks (2010) (ice strength bounds)
- Riska et al. (1997) (friction coefficient bounds)
"""

from __future__ import annotations


from tensorfvm.ice import (
    ICE_THICKNESS,
    ICE_STRENGTH,
    ICE_FRICTION,
    ICE_SPEED,
    ICE_DENSITY,
    ICE_ELASTIC_MODULUS,
    ICE_POISSON_RATIO,
    ICE_TEMPERATURE,
    ALL_ICE_PARAMETERS,
    GuardrailStatus,
    validate_ice_parameters,
    clamp_ice_parameters,
    check_extrapolation_risk,
    get_parameter_report,
)


# =========================================================================== #
# Parameter bounds tests
# =========================================================================== #

class TestIceParameterBounds:
    """Test individual parameter bound logic."""

    def test_thickness_hard_bounds(self):
        """Ice thickness: 0.01–10 m hard bounds."""
        assert not ICE_THICKNESS.bounds.is_hard_violation(0.01)
        assert not ICE_THICKNESS.bounds.is_hard_violation(10.0)
        assert ICE_THICKNESS.bounds.is_hard_violation(0.005)  # too thin
        assert ICE_THICKNESS.bounds.is_hard_violation(50.0)   # impossible

    def test_thickness_soft_bounds(self):
        """Ice thickness: 0.05–5 m soft bounds."""
        assert not ICE_THICKNESS.bounds.is_soft_violation(0.05)
        assert not ICE_THICKNESS.bounds.is_soft_violation(5.0)
        assert ICE_THICKNESS.bounds.is_soft_violation(0.03)   # below soft min
        assert ICE_THICKNESS.bounds.is_soft_violation(6.0)    # above soft max

    def test_thickness_typical(self):
        """Ice thickness: 0.3–2.5 m typical."""
        assert ICE_THICKNESS.bounds.is_typical(0.3)
        assert ICE_THICKNESS.bounds.is_typical(2.5)
        assert ICE_THICKNESS.bounds.is_typical(1.5)
        assert not ICE_THICKNESS.bounds.is_typical(0.1)
        assert not ICE_THICKNESS.bounds.is_typical(4.0)

    def test_strength_hard_bounds(self):
        """Ice strength: 0.1–30 MPa hard bounds."""
        assert not ICE_STRENGTH.bounds.is_hard_violation(0.1)
        assert not ICE_STRENGTH.bounds.is_hard_violation(30.0)
        assert ICE_STRENGTH.bounds.is_hard_violation(0.05)
        assert ICE_STRENGTH.bounds.is_hard_violation(100.0)  # unrealistic

    def test_friction_bounds(self):
        """Ice friction: 0–1 hard, 0.01–0.4 soft."""
        assert not ICE_FRICTION.bounds.is_hard_violation(0.0)
        assert not ICE_FRICTION.bounds.is_hard_violation(1.0)
        assert ICE_FRICTION.bounds.is_hard_violation(-0.1)
        assert ICE_FRICTION.bounds.is_hard_violation(1.5)
        assert ICE_FRICTION.bounds.is_soft_violation(0.005)
        assert ICE_FRICTION.bounds.is_soft_violation(0.5)

    def test_speed_bounds(self):
        """Ice speed: 0–15 m/s hard, 0.1–8 m/s soft."""
        assert not ICE_SPEED.bounds.is_hard_violation(0.0)
        assert not ICE_SPEED.bounds.is_hard_violation(15.0)
        assert ICE_SPEED.bounds.is_hard_violation(-1.0)
        assert ICE_SPEED.bounds.is_hard_violation(20.0)

    def test_density_bounds(self):
        """Ice density: 800–920 kg/m³ hard."""
        assert not ICE_DENSITY.bounds.is_hard_violation(800.0)
        assert not ICE_DENSITY.bounds.is_hard_violation(920.0)
        assert ICE_DENSITY.bounds.is_hard_violation(700.0)
        assert ICE_DENSITY.bounds.is_hard_violation(1000.0)

    def test_elastic_modulus_bounds(self):
        """Ice elastic modulus: 0.1–20 GPa hard."""
        assert not ICE_ELASTIC_MODULUS.bounds.is_hard_violation(0.1)
        assert not ICE_ELASTIC_MODULUS.bounds.is_hard_violation(20.0)
        assert ICE_ELASTIC_MODULUS.bounds.is_hard_violation(0.05)
        assert ICE_ELASTIC_MODULUS.bounds.is_hard_violation(100.0)

    def test_poisson_ratio_bounds(self):
        """Ice Poisson's ratio: 0–0.5 hard, 0.25–0.40 soft."""
        assert not ICE_POISSON_RATIO.bounds.is_hard_violation(0.0)
        assert not ICE_POISSON_RATIO.bounds.is_hard_violation(0.5)
        assert ICE_POISSON_RATIO.bounds.is_hard_violation(-0.1)
        assert ICE_POISSON_RATIO.bounds.is_hard_violation(0.6)

    def test_temperature_bounds(self):
        """Ice temperature: -60 to 0°C hard."""
        assert not ICE_TEMPERATURE.bounds.is_hard_violation(-60.0)
        assert not ICE_TEMPERATURE.bounds.is_hard_violation(0.0)
        assert ICE_TEMPERATURE.bounds.is_hard_violation(-70.0)
        assert ICE_TEMPERATURE.bounds.is_hard_violation(5.0)  # above melting

    def test_clamp(self):
        """Clamping to hard bounds."""
        assert ICE_THICKNESS.bounds.clamp(50.0) == 10.0
        assert ICE_THICKNESS.bounds.clamp(0.001) == 0.01
        assert ICE_THICKNESS.bounds.clamp(2.0) == 2.0


# =========================================================================== #
# Validation tests
# =========================================================================== #

class TestValidateIceParameters:
    """Test the validate_ice_parameters function."""

    def test_valid_typical_params(self):
        """Typical Arctic ice parameters should pass."""
        params = {
            "ice_thickness": 1.5,
            "ice_strength": 3.0,
            "ice_friction": 0.1,
            "ice_speed": 2.5,
        }
        result = validate_ice_parameters(params)
        assert result.is_valid
        assert result.status == GuardrailStatus.PASS
        assert not result.errors
        assert not result.warnings

    def test_impossible_thickness_rejected(self):
        """h=50 m is physically impossible — must be rejected."""
        params = {"ice_thickness": 50.0}
        result = validate_ice_parameters(params)
        assert not result.is_valid
        assert result.status == GuardrailStatus.FAIL
        assert len(result.errors) == 1
        assert "50.0" in result.errors[0]
        assert "physical bounds" in result.errors[0]

    def test_impossible_strength_rejected(self):
        """σ_c=100 MPa is unrealistic — must be rejected."""
        params = {"ice_strength": 100.0}
        result = validate_ice_parameters(params)
        assert not result.is_valid
        assert result.status == GuardrailStatus.FAIL

    def test_extrapolation_risk_without_allow(self):
        """h=6 m is outside calibration — should fail by default."""
        params = {"ice_thickness": 6.0}
        result = validate_ice_parameters(params, allow_extrapolation=False)
        assert not result.is_valid
        assert result.status == GuardrailStatus.FAIL
        assert result.extrapolation_risk

    def test_extrapolation_risk_with_allow(self):
        """h=6 m with allow_extrapolation=True should warn but pass."""
        params = {"ice_thickness": 6.0}
        result = validate_ice_parameters(params, allow_extrapolation=True)
        assert result.is_valid
        assert result.status == GuardrailStatus.WARN
        assert result.extrapolation_risk
        assert len(result.warnings) == 1

    def test_unknown_parameter(self):
        """Unknown parameter names should cause an error."""
        params = {"ice_unknown_param": 1.0}
        result = validate_ice_parameters(params)
        assert not result.is_valid
        assert any("Unknown" in e for e in result.errors)

    def test_non_numeric_value(self):
        """Non-numeric values should cause an error."""
        params = {"ice_thickness": "thick"}
        result = validate_ice_parameters(params)
        assert not result.is_valid
        assert any("numeric" in e.lower() for e in result.errors)

    def test_required_params_missing(self):
        """Missing required parameters should cause an error."""
        result = validate_ice_parameters(
            {"ice_thickness": 1.5},
            required={"ice_thickness", "ice_strength"},
        )
        assert not result.is_valid
        assert any("missing" in e.lower() for e in result.errors)

    def test_multiple_violations(self):
        """Multiple violations should all be reported."""
        params = {
            "ice_thickness": 50.0,
            "ice_strength": 100.0,
            "ice_friction": 2.0,
        }
        result = validate_ice_parameters(params)
        assert not result.is_valid
        assert len(result.errors) >= 3

    def test_validated_params_only_valid(self):
        """validated_params should only contain valid parameters."""
        params = {
            "ice_thickness": 1.5,   # valid
            "ice_strength": 100.0,  # invalid
        }
        result = validate_ice_parameters(params)
        assert "ice_thickness" in result.validated_params
        assert "ice_strength" not in result.validated_params


# =========================================================================== #
# Clamping tests
# =========================================================================== #

class TestClampIceParameters:
    """Test the clamp_ice_parameters function."""

    def test_clamp_extreme_thickness(self):
        """h=50 m should be clamped to 10 m (hard max)."""
        result = clamp_ice_parameters({"ice_thickness": 50.0})
        assert result.is_valid
        assert result.clamped_params["ice_thickness"] == 10.0
        assert len(result.warnings) == 1

    def test_clamp_extreme_strength(self):
        """σ_c=100 MPa should be clamped to 30 MPa (hard max)."""
        result = clamp_ice_parameters({"ice_strength": 100.0})
        assert result.is_valid
        assert result.clamped_params["ice_strength"] == 30.0

    def test_clamp_below_min(self):
        """Negative thickness should be clamped to hard min."""
        result = clamp_ice_parameters({"ice_thickness": -5.0})
        assert result.is_valid
        assert result.clamped_params["ice_thickness"] == 0.01

    def test_clamp_to_soft(self):
        """Clamping to soft bounds should use calibration domain."""
        result = clamp_ice_parameters({"ice_thickness": 6.0}, clamp_to_soft=True)
        assert result.clamped_params["ice_thickness"] == 5.0  # soft max

    def test_no_clamp_needed(self):
        """Values within bounds should not be clamped."""
        result = clamp_ice_parameters({"ice_thickness": 2.0})
        assert result.clamped_params["ice_thickness"] == 2.0
        assert not result.warnings

    def test_clamp_multiple(self):
        """Multiple out-of-bounds values should all be clamped."""
        params = {
            "ice_thickness": 50.0,
            "ice_strength": 100.0,
            "ice_friction": 2.0,
        }
        result = clamp_ice_parameters(params)
        assert result.clamped_params["ice_thickness"] == 10.0
        assert result.clamped_params["ice_strength"] == 30.0
        assert result.clamped_params["ice_friction"] == 1.0
        assert len(result.warnings) == 3


# =========================================================================== #
# Extrapolation risk tests
# =========================================================================== #

class TestExtrapolationRisk:
    """Test the check_extrapolation_risk function."""

    def test_typical_no_risk(self):
        """Typical values should have no extrapolation risk."""
        risks = check_extrapolation_risk({
            "ice_thickness": 1.5,
            "ice_strength": 3.0,
        })
        assert not risks["ice_thickness"][0]
        assert not risks["ice_strength"][0]

    def test_outside_calibration_risk(self):
        """Values outside calibration domain should have risk."""
        risks = check_extrapolation_risk({"ice_thickness": 6.0})
        assert risks["ice_thickness"][0]
        assert risks["ice_thickness"][1] is not None

    def test_unknown_param_risk(self):
        """Unknown parameters should be flagged."""
        risks = check_extrapolation_risk({"ice_unknown": 1.0})
        assert risks["ice_unknown"][0]


# =========================================================================== #
# Report generation tests
# =========================================================================== #

class TestParameterReport:
    """Test the get_parameter_report function."""

    def test_report_contains_params(self):
        """Report should contain all provided parameters."""
        report = get_parameter_report({
            "ice_thickness": 1.5,
            "ice_strength": 3.0,
        })
        assert "ice_thickness" in report
        assert "ice_strength" in report

    def test_report_shows_status(self):
        """Report should show status for each parameter."""
        report = get_parameter_report({"ice_thickness": 1.5})
        assert "TYPICAL" in report

    def test_report_shows_extrapolation(self):
        """Report should show extrapolation risk."""
        report = get_parameter_report({"ice_thickness": 6.0})
        assert "EXTRAPOLATION" in report

    def test_report_shows_violation(self):
        """Report should show physical bound violations."""
        report = get_parameter_report({"ice_thickness": 50.0})
        assert "OUT OF PHYSICAL BOUNDS" in report


# =========================================================================== #
# Integration tests — the specific failure cases from the task
# =========================================================================== #

class TestTaskFailureCases:
    """Test the specific failure cases mentioned in the task."""

    def test_h_50m_rejected(self):
        """h=50 m → 2450 MN: this input must be rejected."""
        result = validate_ice_parameters({"ice_thickness": 50.0})
        assert not result.is_valid, "h=50 m must be rejected as physically impossible"

    def test_sigma_c_100mpa_rejected(self):
        """σ_c=100 MPa → 143.9 MN: this input must be rejected."""
        result = validate_ice_parameters({"ice_strength": 100.0})
        assert not result.is_valid, "σ_c=100 MPa must be rejected as unrealistic"

    def test_combined_extreme_inputs(self):
        """Combined extreme inputs should all be caught."""
        params = {
            "ice_thickness": 50.0,
            "ice_strength": 100.0,
            "ice_speed": 20.0,
            "ice_friction": 0.8,
        }
        result = validate_ice_parameters(params)
        assert not result.is_valid
        # At least thickness and strength should be caught
        error_text = " ".join(result.errors)
        assert "50.0" in error_text or "thickness" in error_text.lower()
        assert "100.0" in error_text or "strength" in error_text.lower()

    def test_clamped_extreme_inputs(self):
        """Extreme inputs should be clamped to safe values."""
        params = {
            "ice_thickness": 50.0,
            "ice_strength": 100.0,
        }
        result = clamp_ice_parameters(params)
        assert result.clamped_params["ice_thickness"] <= 10.0
        assert result.clamped_params["ice_strength"] <= 30.0


# =========================================================================== #
# Registry completeness test
# =========================================================================== #

class TestRegistryCompleteness:
    """Test that the parameter registry is complete and consistent."""

    def test_all_parameters_registered(self):
        """All defined parameters should be in the registry.

        m22342 widened the registry from the 8 ice properties to the full
        12-parameter level-ice resistance input set (8 ice + 4 hull) that the
        anchor audit found partially extrapolating.
        """
        expected = {
            "ice_thickness", "ice_strength", "ice_friction", "ice_speed",
            "ice_density", "ice_elastic_modulus", "ice_poisson_ratio",
            "ice_temperature",
            "ship_length", "ship_beam", "ship_draft", "ship_block_coefficient",
        }
        assert set(ALL_ICE_PARAMETERS.keys()) == expected

    def test_all_parameters_have_bounds(self):
        """All registered parameters should have valid bounds."""
        for name, param in ALL_ICE_PARAMETERS.items():
            assert param.bounds.hard_min < param.bounds.hard_max
            assert param.bounds.soft_min <= param.bounds.soft_max
            assert param.bounds.typical_min <= param.bounds.typical_max
            assert param.bounds.soft_min >= param.bounds.hard_min
            assert param.bounds.soft_max <= param.bounds.hard_max
