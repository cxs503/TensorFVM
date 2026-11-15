"""Verify that the 2-D dam-break API is accessible from the package namespace.

These are *import-only* tests: the functional tests live in
``test_shallow_water_2d_dam_break.py`` and
``test_shallow_water_2d_dam_break_oblique.py``.  This module guards against
the regression where the sub-modules exist but their public symbols are not
re-exported through ``tensorfvm.shallow_water.__init__``.
"""

import pytest


# ------------------------------------------------------------------ #
# axis-aligned 2-D dam break (dam_break2d)
# ------------------------------------------------------------------ #
class TestDamBreak2dImports:
    """All public symbols from dam_break2d must be importable from the package."""

    @pytest.fixture(autouse=True)
    def _import_pkg(self):
        from tensorfvm import shallow_water
        self.pkg = shallow_water

    def test_constants(self):
        assert self.pkg.DEFAULT_NY == 8
        assert self.pkg.TRANSVERSE_ROUNDOFF == 1e-12
        assert self.pkg.MODULATION_AMPLITUDE == 1e-2
        assert self.pkg.TRANSVERSE_GROWTH_MAX == 2.0
        assert self.pkg.FAN_RESOLVED_CELLS == 4.0

    def test_run_2d_case(self):
        assert callable(self.pkg.run_2d_case)

    def test_channel_profile(self):
        assert callable(self.pkg.channel_profile)

    def test_sample_2d(self):
        assert callable(self.pkg.sample_2d)

    def test_metrics_2d(self):
        assert callable(self.pkg.metrics_2d)

    def test_verdict_2d(self):
        assert callable(self.pkg.verdict_2d)

    def test_convergence_2d(self):
        assert callable(self.pkg.convergence_2d)

    def test_compare_with_1d(self):
        assert callable(self.pkg.compare_with_1d)

    def test_transverse_defect(self):
        assert callable(self.pkg.transverse_defect)

    def test_transverse_modulation_study(self):
        assert callable(self.pkg.transverse_modulation_study)

    def test_data_driven_series_metrics(self):
        assert callable(self.pkg.data_driven_series_metrics)

    def test_write_artifacts_2d(self):
        assert callable(self.pkg.write_artifacts_2d)

    def test_modulation_contamination_max(self):
        from tensorfvm.shallow_water import TOLERANCE
        assert self.pkg.MODULATION_CONTAMINATION_MAX == 0.5 * TOLERANCE


# ------------------------------------------------------------------ #
# oblique 2-D dam break (dam_break2d_oblique)
# ------------------------------------------------------------------ #
class TestDamBreak2dObliqueImports:
    """All public symbols from dam_break2d_oblique must be importable."""

    @pytest.fixture(autouse=True)
    def _import_pkg(self):
        from tensorfvm import shallow_water
        self.pkg = shallow_water

    def test_angles_deg(self):
        assert 0 in self.pkg.ANGLES_DEG  # axis-aligned control must be present

    def test_oblique_cases(self):
        assert isinstance(self.pkg.OBLIQUE_CASES, dict)

    def test_oblique_agreement_max(self):
        assert self.pkg.OBLIQUE_AGREEMENT_MAX > 0

    def test_strip_half_width(self):
        assert self.pkg.STRIP_HALF_WIDTH > 0

    def test_domain(self):
        assert isinstance(self.pkg.DOMAIN, float)
        assert self.pkg.DOMAIN > 0

    def test_run_oblique_case(self):
        assert callable(self.pkg.run_oblique_case)

    def test_verdict_oblique(self):
        assert callable(self.pkg.verdict_oblique)

    def test_dam_normal(self):
        assert callable(self.pkg.dam_normal)

    def test_normal_coordinate(self):
        assert callable(self.pkg.normal_coordinate)

    def test_normal_strip_profile(self):
        assert callable(self.pkg.normal_strip_profile)

    def test_dam_initial_depth(self):
        assert callable(self.pkg.dam_initial_depth)

    def test_grade_profile(self):
        assert callable(self.pkg.grade_profile)

    def test_estimator_zero_bias(self):
        assert callable(self.pkg.estimator_zero_bias)

    def test_isotropy_study(self):
        assert callable(self.pkg.isotropy_study)

    def test_isotropy_rows(self):
        assert callable(self.pkg.isotropy_rows)

    def test_write_artifacts_oblique(self):
        assert callable(self.pkg.write_artifacts_oblique)
