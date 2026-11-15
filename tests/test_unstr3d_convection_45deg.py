"""45-degree cosine convection boundedness gate (FEAT-UNSTR3D-MESH).

Inflow enters through x=0 (``inlet``) and y=0 (``wall``) of the unit cube with
velocity (1,1,0)/sqrt(2); the exact solution is constant along x=y, so a pure
convection solve has a closed-form reference and any new extremum is a scheme
artefact, not a physical feature.
"""
import numpy as np
import pytest

from tensorfvm.unstructured_fvm3d import UnstructuredFVM3D
from tensorfvm.unstructured_mesh3d import cube_tet_msh, tet_mesh_from_msh

WAVELENGTH = 0.5
VELOCITY = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)


def transverse(point):
    return (point[:, 1] - point[:, 0]) / np.sqrt(2.0)


def exact(point):
    return np.cos(2.0 * np.pi * transverse(point) / WAVELENGTH)


def run(n=8, scheme="limited"):
    mesh = tet_mesh_from_msh(text=cube_tet_msh(n, warp=0.2))
    operator = UnstructuredFVM3D(mesh)
    inflow = exact(operator.face_centers[operator.boundary])
    flux = operator.face_flux(VELOCITY)
    # z-faces are streamwise-parallel: no flux, so they need no data.
    zero_flux = np.abs(flux) < 1e-12
    assert zero_flux.any()
    result = operator.advection(VELOCITY, inflow, scheme=scheme)
    values = result["values"]
    reference = exact(np.asarray(mesh.cell_centers))
    amplitude = 1.0
    overshoot = max(values.max() - 1.0, -1.0 - values.min()) / amplitude
    error = np.sqrt(np.mean((values - reference) ** 2))
    return dict(values=values, reference=reference, overshoot=overshoot, error=error,
                flux=flux, result=result)


def test_limited_scheme_stays_within_two_percent_and_is_monotone():
    outcome = run(scheme="limited")
    assert outcome["overshoot"] <= 0.02, f"overshoot {outcome['overshoot']:.4f}"
    values = outcome["values"]
    reference = outcome["reference"]
    # Monotone: no new extremum anywhere, i.e. the solution stays inside the
    # range of the inflow data (a bounded scheme) and does not oscillate.
    assert np.all(values <= 1.0 + 0.02) and np.all(values >= -1.0 - 0.02)
    residual = values - reference
    assert np.max(np.abs(residual)) < 0.35


def test_upwind_is_bounded_but_first_order():
    outcome = run(scheme="upwind")
    assert outcome["overshoot"] <= 0.02
    assert outcome["error"] > run(scheme="limited")["error"] * 0.5


def test_central_scheme_overshoots_so_the_limiter_is_needed():
    outcome = run(scheme="central")
    limited = run(scheme="limited")
    assert outcome["overshoot"] > limited["overshoot"]
