"""Added mass of the half-submerged sphere via the antisymmetric (odd) extension.

Idea
----
At infinite frequency the free surface is the Dirichlet plane ``phi = 0``.  A body
cut by that plane at ``z = 0`` can be solved **without** any free-surface image:
extend the unknown potential to the mirror half by ``Phi(x, y, -z) = -Phi(x, y, z)``.
``Phi`` is harmonic on the whole exterior of the reflected body (the original
potential is harmonic below, its odd continuation above) and it vanishes on
``z = 0`` by construction, so the lower half of ``Phi`` *is* the free-surface
solution.  The price is a Neumann data set that is no longer rigid: on the mirror
half the prescribed normal velocity is ``-psi`` of the mirrored point
(:func:`antisymmetric_normal_velocity`), and the added mass is the surface
integral over the **lower half only** (the physical wetted surface).

For a sphere centred on the waterplane the reflected body is the full sphere, so
the whole computation runs on an ordinary closed-sphere panel set with the plain
``IMAGE_SPECS["body"]`` kernel -- no image sign, no half-body mesh, and the same
influence matrix serves every mode of motion.

Checks and references
---------------------
* heave: the odd extension of the heave data is the *rigid* heave data of the full
  sphere (``n_z`` is itself odd in ``z``), so the half-submerged heave added mass
  must equal exactly half of the full sphere's -- ``(1/3) pi rho a^3``.  That
  gives an exact structural identity on the code path
  (:func:`half_over_full_ratio`) and an exact analytic anchor.
  The same conclusion is why the wall (double-body) image, which keeps the *full*
  sphere, over-predicts the heave added mass by a factor of two.
* sway/surge: the data set is genuinely non-rigid and has no closed form; the
  reference is the odd-harmonic series of
  :mod:`tensorfvm.halfsub_sphere_reference` (global spectral approximation,
  independent of any panelisation).

Comparison with the image route
-------------------------------
``tensorfvm.panel_potential`` also offers a free-surface *image* branch; the two
formulations are mathematically equivalent.  This module exists because the odd
extension is the only route that needs no new image machinery, and because it puts
the half-submerged problem on a closed-body solver that is already anchored.
"""
from __future__ import annotations

import numpy as np

from .panel_added_mass import added_mass_matrix, source_potential, surface_quadrature
from .panel_potential import IMAGE_SPECS, solve_sources
from .twin_hull_geometry import PanelSet, sphere_panels

__all__ = [
    "antisymmetric_normal_velocity",
    "antisymmetric_added_mass",
    "half_over_full_ratio",
    "sphere_waterplane_panels",
]


def sphere_waterplane_panels(radius: float = 1.0, n_theta: int = 24,
                             n_phi: int = 18) -> PanelSet:
    """Full sphere whose equator ring lies *exactly* on the waterplane ``z = 0``.

    ``n_phi`` must be even so that ``theta = pi/2`` is a node ring; the odd
    extension then has a well defined lower/upper half.
    """
    if int(n_phi) % 2 != 0:
        raise ValueError("n_phi must be even: the equator must be a node ring")
    panels = sphere_panels(radius, n_theta, n_phi)
    return panels


def waterplane_mask(panels: PanelSet) -> np.ndarray:
    """Panels below the waterplane (the physical wetted surface)."""
    return panels.centroids[:, 2] < 0.0


def antisymmetric_normal_velocity(panels: PanelSet, axis: int) -> np.ndarray:
    """Neumann data of the odd extension for a unit translation along ``axis``.

    Lower half: ``n_axis`` (the physical body motion).  Upper half: ``+n_axis`` for
    the vertical axis (``n_z`` is already odd in ``z``, i.e. the extension of heave
    is rigid heave) and ``-n_axis`` for a horizontal axis (``n_x``, ``n_y`` are even
    in ``z``, so the extension reverses them).
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    e = np.zeros(3)
    e[axis] = 1.0
    vn = panels.normals @ e
    sign = np.where(waterplane_mask(panels), 1.0, 1.0 if axis == 2 else -1.0)
    return vn * sign


def antisymmetric_added_mass(panels: PanelSet, *, rho: float = 1.0, order: int = 3,
                             quadrature: str = "gauss",
                             axes: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    """Translational added-mass matrix of the half-submerged body ``z < 0``.

    Rows/columns outside ``axes`` are left zero.  ``M_ij = -rho * integral_lower
    phi_i n_j dS``, with ``phi_i`` solving the odd-extension Neumann problem of
    mode ``i``; the integral runs over the lower-half nodes only.
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    n_all = panels.n_panels
    nodes, weights, generators = surface_quadrature(panels, order=order,
                                                    quadrature=quadrature)
    n_nodes = nodes.shape[0]
    if n_nodes % n_all:
        raise ValueError("quadrature nodes must come in whole panels")
    panel_index = np.arange(n_nodes) % n_all
    lower = waterplane_mask(panels)[panel_index]
    w_low = np.where(lower, weights, 0.0)

    sigmas = []
    for axis in axes:
        psi = antisymmetric_normal_velocity(panels, axis)
        sigmas.append(solve_sources(panels, np.zeros(3), order=order,
                                    images=IMAGE_SPECS["body"], normal_velocity=psi))
    phi = source_potential(nodes, np.array(sigmas), panels, order=order,
                           images=IMAGE_SPECS["body"])
    out = np.zeros((3, 3))
    for row, i in enumerate(axes):
        for j in axes:
            out[i, j] = -rho * float(np.sum(phi[row] * generators[:, j] * w_low))
    return out


def half_over_full_ratio(panels: PanelSet, *, rho: float = 1.0, order: int = 3) -> float:
    """``M33(half, odd extension) / M33(full sphere)`` -- must be exactly ``1/2``.

    The odd extension of the heave mode is the rigid heave mode of the full
    sphere, so the two potentials coincide and the half-body integral is half of
    the closed-body one; any deviation is a bug in the extension or the surface
    integral (the closed-body value is :func:`panel_added_mass.added_mass_matrix`).
    """
    half = antisymmetric_added_mass(panels, rho=rho, order=order,
                                    quadrature="gauss", axes=(2,))[2, 2]
    full = added_mass_matrix(panels, rho=rho, order=order,
                             images=IMAGE_SPECS["body"], quadrature="gauss",
                             directions=(2,))[2, 2]
    return float(half / full)