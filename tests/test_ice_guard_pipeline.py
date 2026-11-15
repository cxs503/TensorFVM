"""Tests for the ice-domain guardrail repair (m22342).

Coverage map
------------
* non-finite containment: ``nan`` / ``+inf`` / ``-inf`` for all 12 guarded
  parameters, in ``validate`` (reject) and ``clamp`` (finite fallback, NaN not
  echoed back) — the defect that failed the m22209 delivery;
* the raising gate ``require_ice_parameters``;
* the guarded physics entry point: out-of-domain inputs must not reach the
  Lindqvist arithmetic, in-domain inputs must not be perturbed;
* the benchmark rows (ref vs got) and the CLI entry point that writes the
  run artifacts;
* one integration caller: ``tensorfvm.ice_channel.run_all`` refuses an
  impossible ice lid before solving.
"""

from __future__ import annotations

import json
import math

import pytest

import tensorfvm
from tensorfvm.ice import (
    ALL_ICE_PARAMETERS,
    ICE_RESISTANCE_INPUTS,
    GuardrailStatus,
    IceParameterDomainError,
    clamp_ice_parameters,
    get_parameter_report,
    guarded_level_ice_resistance,
    require_ice_parameters,
    require_guarded_level_ice_resistance,
    validate_ice_parameters,
)
from tensorfvm.ice.benchmark import NOMINAL_CASES, run_benchmark, write_artifacts
from tensorfvm.ice.load import level_ice_resistance_kN, IceCondition, IceShipGeometry
from tensorfvm.ice.parameters import IceParameterBounds

PARAMS = list(ICE_RESISTANCE_INPUTS)
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def nominal() -> dict[str, float]:
    return dict(NOMINAL_CASES["MV_Arctic"])


class TestNonFiniteContainment:
    """A guardrail whose job is to stop silent garbage must stop NaN."""

    @pytest.mark.parametrize("name", PARAMS)
    @pytest.mark.parametrize("value", NON_FINITE)
    def test_validate_rejects_non_finite(self, name, value):
        res = validate_ice_parameters({name: value})
        assert not res.is_valid, f"{name}={value!r} was accepted"
        assert res.status is GuardrailStatus.FAIL
        assert name not in res.validated_params
        assert name in res.non_finite_params

    @pytest.mark.parametrize("name", PARAMS)
    def test_validate_rejects_non_finite_even_when_extrapolation_allowed(self, name):
        res = validate_ice_parameters({name: float("nan")}, allow_extrapolation=True)
        assert not res.is_valid
        assert name not in res.validated_params

    @pytest.mark.parametrize("name", PARAMS)
    def test_clamp_does_not_echo_nan(self, name):
        res = clamp_ice_parameters({name: float("nan")})
        assert not res.is_valid
        assert name not in res.clamped_params, "NaN must not be returned"
        assert all(math.isfinite(v) for v in res.clamped_params.values())

    @pytest.mark.parametrize("name", PARAMS)
    @pytest.mark.parametrize("value,sign", [(float("inf"), "hi"), (float("-inf"), "lo")])
    def test_clamp_inf_lands_on_bound(self, name, value, sign):
        res = clamp_ice_parameters({name: value})
        clamped = res.clamped_params[name]
        assert math.isfinite(clamped)
        bound = ALL_ICE_PARAMETERS[name].bounds
        expected = bound.hard_max if sign == "hi" else bound.hard_min
        assert clamped == expected
        assert res.extrapolation_risk

    def test_report_never_labels_nan_typical(self):
        text = get_parameter_report({"ice_thickness": float("nan")})
        assert "NON-FINITE" in text
        assert "TYPICAL" not in text

    def test_bool_is_not_a_physical_quantity(self):
        res = validate_ice_parameters({"ice_thickness": True})
        assert not res.is_valid
        assert "bool" in " ".join(res.errors)
        assert "ice_thickness" not in res.validated_params

    def test_is_hard_violation_flags_nan(self):
        b = IceParameterBounds(
            name="x", symbol="x", unit="", hard_min=0.0, hard_max=1.0,
            soft_min=0.1, soft_max=0.9, typical_min=0.3, typical_max=0.7,
            reference="test", description="test",
        )
        assert b.is_hard_violation(float("nan"))
        assert b.is_hard_violation(float("inf"))
        assert not b.is_hard_violation(0.5)


class TestRaisingGate:
    def test_require_returns_validated(self):
        out = require_ice_parameters({"ice_thickness": 1.2})
        assert out["ice_thickness"] == 1.2

    def test_require_raises_on_out_of_domain(self):
        with pytest.raises(IceParameterDomainError) as exc:
            require_ice_parameters({"ice_thickness": 50.0})
        assert "ice_thickness" in str(exc.value)
        assert exc.value.result.errors

    def test_require_raises_on_missing_required(self):
        with pytest.raises(IceParameterDomainError):
            require_ice_parameters({"ice_thickness": 1.0},
                                   required=set(ICE_RESISTANCE_INPUTS))


class TestParameterTable:
    def test_registry_covers_the_12_input_set(self):
        assert len(ALL_ICE_PARAMETERS) == 12
        assert set(ALL_ICE_PARAMETERS) == set(ICE_RESISTANCE_INPUTS)

    @pytest.mark.parametrize("name", PARAMS)
    def test_three_tier_nesting(self, name):
        b = ALL_ICE_PARAMETERS[name].bounds
        assert b.hard_min <= b.soft_min <= b.typical_min
        assert b.typical_max <= b.soft_max <= b.hard_max

    def test_documented_strength_band(self):
        b = ALL_ICE_PARAMETERS["ice_strength"].bounds
        assert (b.soft_min, b.soft_max, b.typical_min, b.typical_max) == (
            0.3, 15.0, 1.0, 5.0)


class TestGuardedLoadPath:
    def test_in_domain_matches_unguarded(self):
        params = nominal()
        res = guarded_level_ice_resistance(params, mode="reject")
        raw = level_ice_resistance_kN(
            IceShipGeometry(222.5, 32.2, 13.5, 0.82),
            IceCondition(1.0, 1.5, 0.15, 3.0),
        )
        assert res.resistance_kN == pytest.approx(raw, rel=1e-15)
        assert res.guard.status is GuardrailStatus.PASS
        assert not res.extrapolation_blocked

    @pytest.mark.parametrize("override", [
        {"ice_thickness": 50.0},
        {"ice_strength": 100.0},
        {"ice_thickness": float("nan")},
        {"ship_length": 5000.0},
        {"ship_block_coefficient": 2.0},
        {"ice_density": -5.0},
    ])
    def test_out_of_domain_never_reaches_arithmetic(self, override):
        params = nominal()
        params.update(override)
        res = guarded_level_ice_resistance(params, mode="reject")
        assert res.resistance_kN is None
        assert res.guard.errors
        assert res.extrapolation_blocked

    def test_legacy_failure_magnitude_is_visible(self):
        """The anchor's h=50 m case: blocked, and the 10^3 MN blow-up recorded."""
        params = nominal()
        params["ice_thickness"] = 50.0
        res = guarded_level_ice_resistance(params, mode="reject")
        assert res.resistance_kN is None
        assert res.unguarded_resistance_kN is not None
        assert res.unguarded_resistance_kN > 1.0e6      # > 1000 MN

    def test_clamp_mode_computes_and_flags(self):
        params = nominal()
        params["ice_thickness"] = 50.0
        res = guarded_level_ice_resistance(params, mode="clamp")
        assert res.resistance_kN is not None
        assert res.used_params["ice_thickness"] == 10.0
        assert res.extrapolation_blocked and res.guard.warnings

    def test_require_raises(self):
        params = nominal()
        params["ice_thickness"] = 50.0
        with pytest.raises(IceParameterDomainError):
            require_guarded_level_ice_resistance(params)

    def test_unknown_parameter_rejected(self):
        params = nominal()
        params["ice_salinity"] = 3.0
        res = guarded_level_ice_resistance(params, mode="reject")
        assert res.resistance_kN is None
        assert any("Unknown" in e for e in res.guard.errors)


class TestBenchmarkAndCli:
    def test_all_graded_rows_pass(self):
        bundle = run_benchmark()
        graded = [r for r in bundle["rows"] if r["passed"] is not None]
        assert graded, "benchmark produced no graded rows"
        assert all(r["passed"] for r in graded), [
            r for r in graded if not r["passed"]
        ]
        # every row must carry a reference and a measured value
        for row in bundle["rows"]:
            assert row["ref"] and row["got"] is not None

    def test_decision_conformance_is_exact(self):
        bundle = run_benchmark()
        g1 = bundle["rows"][0]
        assert g1["error"] == 0.0

    def test_legacy_row_is_withheld_not_passed(self):
        bundle = run_benchmark()
        g6 = bundle["rows"][-1]
        assert g6["passed"] is None
        assert "WITHHELD" in g6["note"]

    def test_cli_writes_artifacts(self, tmp_path):
        from tensorfvm.ice.__main__ import main
        code = main(["--out", str(tmp_path)])
        assert code == 0
        for name in ("benchmark.json", "summary.json", "decision_grid.json",
                     "load_path_closure.json", "legacy_anchor.json",
                     "parameter_report.txt"):
            assert (tmp_path / name).is_file(), name
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["all_graded_rows_pass"] is True
        assert summary["repro_cmd"].startswith("PYTHONPATH=src python -m tensorfvm.ice")

    def test_cli_case_mode(self, tmp_path):
        from tensorfvm.ice.__main__ import main
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"ice": {
            "ice_thickness": 1.0, "ice_strength": 1.5, "ice_friction": 0.15,
            "ice_speed": 3.0, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
            "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
        }, "ship": {"ship_length": 222.5, "ship_beam": 32.2,
                    "ship_draft": 13.5, "ship_block_coefficient": 0.82}}))
        assert main(["--case", str(good), "--out", str(tmp_path)]) == 0

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"ice_thickness": 50.0}))
        assert main(["--case", str(bad), "--out", str(tmp_path)]) == 1

    def test_write_artifacts_returns_repro_cmd(self, tmp_path):
        summary = write_artifacts(tmp_path / "art")
        assert summary["all_graded_rows_pass"] is True
        assert (tmp_path / "art" / "benchmark.json").is_file()


class TestIntegrationCallers:
    def test_package_root_exports_ice(self):
        assert hasattr(tensorfvm, "ice")
        assert tensorfvm.ice.validate_ice_parameters is validate_ice_parameters

    def test_channel_solver_guards_ice_lid(self):
        from tensorfvm.ice_channel import IceChannelConfig, ice_lid_guard
        cfg = IceChannelConfig(top_bc="slip")
        assert ice_lid_guard(cfg).is_valid

    def test_channel_solver_refuses_impossible_lid(self):
        from tensorfvm.ice_channel import IceChannelConfig, ice_lid_guard
        cfg = IceChannelConfig(ice_lid_thickness_m=50.0)
        with pytest.raises(IceParameterDomainError):
            ice_lid_guard(cfg)

    def test_channel_run_records_guard(self, tmp_path):
        from tensorfvm.ice_channel import run_all
        summary = run_all(output_dir=tmp_path)
        assert "ice_lid_guard" in summary
        assert summary["all_passed"]