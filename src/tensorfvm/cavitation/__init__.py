"""Rudder cavitation inception and erosion-risk assessment (TensorFVM).

What this package is
--------------------
A self-contained ``cavitation`` face for rudder/hydrofoil sections, built only on
physics that can be verified inside this workspace:

======================  =====================================================
module                  role
======================  =====================================================
:mod:`.section`         NACA 4-digit geometry and a constant-source /
                        constant-vortex panel method (:math:`C_p`, :math:`C_l`);
                        the source of ``Cp_min``.
:mod:`.conformal`       exact conformal-mapping solution for a family of thick
                        symmetric sections -- the independent anchor for the
                        panel solver.
:mod:`.inception`       cavitation number, minimum-pressure inception
                        ``sigma_i = -Cp_min``, Blake/nuclei tensile correction.
:mod:`.erosion`         Rayleigh-Plesset single-bubble dynamics, water-hammer
                        impact pressure, erosion index ``E ~ (p_impact-p_v)^n``.
:mod:`.reference`       reference bands with explicit provenance status.
:mod:`.report`          the KVLCC2 rudder case: criteria, JSON, figures.
======================  =====================================================

Evidence grading used in the node report
----------------------------------------
* ``REAL``        computed here by a solver: ``Cp`` (panel method, verified
                  against the exact conformal solution to <1 % at the suction
                  peak), Rayleigh collapse (verified against ``0.915 R sqrt(rho/dp)``).
* ``CRITERION``   ``sigma_i = -Cp_min`` -- the ITTC minimum-pressure inception
                  criterion (the procedure's *definition*, not a table).
* ``QUALITATIVE`` the erosion trend bands (peak ``sigma_i/sigma``, exponent ``n``,
                  cavity-length constant, shedding Strouhal).
* ``WITHHELD``    absolute experimental erosion-depth values: no digitised
                  dataset exists in this workspace.

Quick start::

    from tensorfvm.cavitation import report
    summary = report.run_case()          # writes artifacts/cavitation-rudder/
    summary["operating_point"]["sigma"], summary["operating_point"]["onset_angle_deg"]
"""
from __future__ import annotations

from . import conformal, erosion, inception, reference, report, section  # noqa: F401
from .conformal import JoukowskiSection, joukowski_for_thickness  # noqa: F401
from .erosion import (  # noqa: F401
    RayleighCollapse, cavity_length, collapse_velocity, erosion_rate,
    impact_pressure, integrate_rayleigh_plesset, rayleigh_collapse_time,
)
from .inception import (  # noqa: F401
    CavitationState, WaterState, blake_critical_pressure, cavitation_number,
    sigma_inception, sigma_inception_curve, sigma_inception_with_nuclei,
    water_vapour_pressure,
)
from .section import PanelSolution, naca4_contour, panel_solve  # noqa: F401

__all__ = [
    "section", "conformal", "inception", "erosion", "reference", "report",
    "PanelSolution", "naca4_contour", "panel_solve",
    "JoukowskiSection", "joukowski_for_thickness",
    "WaterState", "CavitationState", "cavitation_number",
    "water_vapour_pressure", "sigma_inception", "sigma_inception_curve",
    "sigma_inception_with_nuclei", "blake_critical_pressure",
    "RayleighCollapse", "rayleigh_collapse_time", "collapse_velocity",
    "integrate_rayleigh_plesset", "impact_pressure", "cavity_length",
    "erosion_rate",
]