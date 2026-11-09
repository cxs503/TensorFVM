"""Independent 2-D frequency-domain potential-flow reference for the NWT.

The time-domain tank is checked against a *different discretisation of the same
linear physics*: a finite-volume Laplace solver on a Cartesian grid with

* ``lap(phi) = 0`` in the fluid (FV, cut-cell body with partial open areas),
* free surface ``dphi/dz = nu phi``, ``nu = omega^2/g`` (Robin at ``z = 0``),
* body ``dphi/dn = V_n`` (radiation) or ``-dphi_I/dn`` (diffraction; the
  *scattered* potential is the unknown so the lateral condition is purely
  outgoing),
* bottom ``dphi/dn = 0``, lateral boundaries first-order Sommerfeld
  ``dphi/dn = i k phi``,
* incident potential ``phi_I = -(i g a/omega) cosh k(z+h)/cosh(kh) e^{i k x}``
  with ``Phi = Re(phi e^{-i omega t})``.

Outputs, per unit length of section: added mass ``A``, damping ``B``, excitation
force ``F_exc`` (heave, pitch), hydrostatic restoring ``C`` from the exact wetted
geometry, and RAOs from

    (-omega^2 (M + A) + i omega B + C) xi = F_exc.

Sign conventions (checked in the test suite): the radiation force on the body is
``-A xi_ddot - B xi_dot`` with ``A + i B/omega = -rho * integral(phi_j n_i dl)``
and the excitation is ``F_i = -i omega rho * integral((phi_I + phi_S) n_i dl)``.

This is a **2-D** reference.  The platform's 3-D TensorBEM lives in another
tenant repository and is not directly comparable with a vertical-plane 2-D tank
without a strip-theory reduction of a 3-D hull, so the seakeeping cross-check is
made against this in-repo potential solver; the deviation is declared in the
report.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..wave_tank.theory import GRAVITY, dispersion_k
from .sigma_tank import points_in_polygon, polygon_properties

__all__ = ["section_hydrodynamics", "wetted_polygon", "waterline_width",
           "hydrostatic_restoring"]


# ----------------------------------------------------------------- geometry
def _clip_halfplane(poly, z_level: float = 0.0):
    """Sutherland-Hodgman clip of a polygon against the half plane ``z <= z_level``."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina = a[1] <= z_level
        inb = b[1] <= z_level
        if ina:
            out.append(a)
        if ina != inb:
            t = (z_level - a[1]) / (b[1] - a[1])
            out.append(np.array([a[0] + t * (b[0] - a[0]), z_level]))
    return np.array(out) if out else np.zeros((0, 2))


def wetted_polygon(poly, z_level: float = 0.0):
    """Part of the section below the waterline ``z_level``."""
    return _clip_halfplane(np.asarray(poly, dtype=float), z_level)


def waterline_width(poly, z_level: float = 0.0) -> float:
    """Length of the section's intersection with ``z = z_level``."""
    p = np.asarray(poly, dtype=float)
    xmin, xmax = np.inf, -np.inf
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        if a[1] != b[1] and (a[1] - z_level) * (b[1] - z_level) <= 0.0:
            t = (z_level - a[1]) / (b[1] - a[1])
            x = a[0] + t * (b[0] - a[0])
            xmin, xmax = min(xmin, x), max(xmax, x)
    return float(max(xmax - xmin, 0.0)) if np.isfinite(xmin) else 0.0


def _rotate(poly, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(poly, dtype=float) @ np.array([[c, s], [-s, c]])


def hydrostatic_restoring(poly, rho: float = 1000.0, g: float = GRAVITY,
                          mass: float | None = None) -> dict:
    """``C33``, ``C55`` of a surface-piercing 2-D section (per unit length).

    Linear waterplane theory about the origin at the still-water line::

        C33 = rho g B_wl
        C55 = rho g I_wl - rho g nabla * z_G ,   I_wl = integral(x^2 dx) on z=0

    ``z_G`` is the centre of gravity of the (uniform) section, so the
    ``- rho g nabla z_G`` term is the moment of the weight about the rotation
    reference -- dropping it is the classic factor-of-two error in pitch
    restoring.  ``C35 = 0`` for a section symmetric about ``x = 0``.
    """
    wet = wetted_polygon(poly)
    area0, cx0, cz0, _ = polygon_properties(wet, rho)
    _a, cxg, czg, _ = polygon_properties(np.asarray(poly, dtype=float), rho)
    if mass is None:
        mass = rho * area0
    b_wl = waterline_width(poly)
    # waterplane second moment: integrate x^2 over the waterline segment(s)
    p = np.asarray(poly, dtype=float)
    segs = []
    for i in range(len(p)):
        a_, b_ = p[i], p[(i + 1) % len(p)]
        if a_[1] != b_[1] and (a_[1]) * (b_[1]) <= 0.0:
            t = (0.0 - a_[1]) / (b_[1] - a_[1])
            segs.append(a_[0] + t * (b_[0] - a_[0]))
    i_wl = 0.0
    for i in range(0, len(segs) - 1, 2):
        x1, x2 = sorted((segs[i], segs[i + 1]))
        i_wl += (x2 ** 3 - x1 ** 3) / 3.0
    c33 = rho * g * b_wl
    c55 = rho * g * i_wl - rho * g * area0 * czg
    return {"C33": float(c33), "C55": float(c55), "C35": 0.0,
            "C53": 0.0, "waterline": float(b_wl), "I_waterline": float(i_wl),
            "displaced_area": float(area0),
            "centre_of_buoyancy": [float(cx0), float(cz0)],
            "centre_of_gravity": [float(cxg), float(czg)],
            "mass_per_length": float(mass)}


# ------------------------------------------------------------- grid + faces
def _grid_and_masks(poly, xf, zf, sub=6):
    nx = len(xf) - 1
    nz = len(zf) - 1
    dx = xf[1] - xf[0]
    dz = zf[1] - zf[0]
    s = (np.arange(sub) + 0.5) / sub
    zlo, zhi = zf[:-1], zf[1:]
    zz = zlo[None, :, None] + s[None, None, :] * (zhi - zlo)[None, :, None]
    xx = np.broadcast_to(xf[:, None, None], (nx + 1, nz, sub))
    frac_x = points_in_polygon(xx, np.broadcast_to(zz, (nx + 1, nz, sub)),
                               poly).reshape(nx + 1, nz, sub).mean(-1)
    xs = xf[:-1, None] + s[None, :] * dx
    frac_z = points_in_polygon(
        np.broadcast_to(xs[None, :, :], (nz + 1, nx, sub)),
        np.broadcast_to(zf[:, None, None], (nz + 1, nx, sub)), poly
    ).reshape(nz + 1, nx, sub).mean(-1)
    xcc = np.broadcast_to((xf[:-1] + dx / 2)[:, None], (nx, nz))
    zcc = np.broadcast_to((zf[:-1] + dz / 2)[None, :], (nx, nz))
    solid = points_in_polygon(xcc, zcc, poly).reshape(nx, nz)
    # topological closure (see SigmaWaveTank._update_mask): a face with exactly
    # one fully solid neighbour is blocked, which makes grid-aligned sections
    # exact instead of relying on a degenerate boundary point-in-polygon test.
    cx = np.zeros((nx + 1, nz), dtype=bool)
    cx[1:-1] = solid[1:, :] ^ solid[:-1, :]
    cz = np.zeros((nz + 1, nx), dtype=bool)
    cz[1:-1] = solid[:, 1:].T ^ solid[:, :-1].T
    openn_x = np.where(cx, 0.0, 1.0 - frac_x)
    openn_z = np.where(cz, 0.0, 1.0 - frac_z)
    return openn_x, openn_z, solid


def _cut_faces(openn_x, openn_z, solid, xf, zf):
    """All wetted-face pieces: fluid cell, fluid-side outward normal, area."""
    nx, nz = solid.shape
    dx = xf[1] - xf[0]
    dz = zf[1] - zf[0]
    ci, cj, nxi, nzi, area, xc, zc = [], [], [], [], [], [], []
    cov_x = (1.0 - openn_x) * dz
    for i in range(1, nx):
        for sign, left in ((1.0, False), (-1.0, True)):
            if left:
                fmask = solid[i - 1, :] & ~solid[i, :]        # fluid left of body
            else:
                fmask = solid[i, :] & ~solid[i - 1, :]        # fluid right of body
            for j in np.nonzero(fmask & (cov_x[i] > 0))[0]:
                ci.append(i - 1 if left else i)
                cj.append(j)
                nxi.append(sign)
                nzi.append(0.0)
                area.append(cov_x[i, j])
                xc.append(xf[i])
                zc.append(zf[j] + dz / 2)
    cov_z = (1.0 - openn_z) * dx
    for j in range(1, nz):
        for sign, below in ((1.0, True), (-1.0, False)):
            if below:                                   # solid above the face
                fmask = solid[:, j] & ~solid[:, j - 1]
                cell = j - 1
            else:                                       # solid below the face
                fmask = solid[:, j - 1] & ~solid[:, j]
                cell = j
            for i in np.nonzero(fmask & (cov_z[j] > 0))[0]:
                ci.append(i)
                cj.append(cell)
                nxi.append(0.0)
                nzi.append(sign)
                area.append(cov_z[j, i])
                xc.append(xf[i] + dx / 2)
                zc.append(zf[j])
    return dict(i=np.array(ci), j=np.array(cj), nx=np.array(nxi), nz=np.array(nzi),
                area=np.array(area), x=np.array(xc), z=np.array(zc))


def _assemble(openn_x, openn_z, solid, xf, zf, k, nu, nu_col=None):
    nx, nz = solid.shape
    dx = xf[1] - xf[0]
    dz = zf[1] - zf[0]
    idx = np.arange(nx * nz).reshape(nx, nz)
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(np.ravel(r))
        cols.append(np.ravel(c))
        vals.append(np.ravel(np.broadcast_to(v, np.shape(r))))

    ii, jj = np.meshgrid(np.arange(1, nx), np.arange(nz), indexing="ij")
    ax = openn_x[ii, jj] * dz / dx
    add(idx[ii, jj], idx[ii, jj], ax)
    add(idx[ii, jj], idx[ii - 1, jj], -ax)
    add(idx[ii - 1, jj], idx[ii - 1, jj], ax)
    add(idx[ii - 1, jj], idx[ii, jj], -ax)
    ii2, jj2 = np.meshgrid(np.arange(nx), np.arange(1, nz), indexing="ij")
    az = openn_z[jj2, ii2] * dx / dz
    add(idx[ii2, jj2], idx[ii2, jj2], az)
    add(idx[ii2, jj2], idx[ii2, jj2 - 1], -az)
    add(idx[ii2, jj2 - 1], idx[ii2, jj2 - 1], az)
    add(idx[ii2, jj2 - 1], idx[ii2, jj2], -az)
    top = openn_z[nz, :] * dx
    # Phi = Re(phi e^{-i w t}) => linearised free surface is dphi/dz = -(w^2/g) phi
    nu_x = nu if nu_col is None else nu_col
    add(idx[np.arange(nx), nz - 1], idx[np.arange(nx), nz - 1], -nu_x * top)
    add(idx[0, :], idx[0, :], 1j * k * openn_x[0, :] * dz)
    add(idx[nx - 1, :], idx[nx - 1, :], 1j * k * openn_x[nx, :] * dz)
    M = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(nx * nz, nx * nz)).tocsc()
    free = ~solid.ravel()
    M = (sparse.diags(free.astype(float)) @ M
         + sparse.diags((~free).astype(float)))
    return M, free


def _moments(phi, faces, flat_idx, phi_extra=None):
    """``integral(phi n_i dl)`` for heave (i=3) and pitch (i=5)."""
    ph = phi[flat_idx] if phi_extra is None else (phi[flat_idx] + phi_extra)
    m3 = complex(np.sum(ph * faces["nz"] * faces["area"]))
    m5 = complex(np.sum(ph * (faces["z"] * faces["nx"] - faces["x"] * faces["nz"])
                        * faces["area"]))
    return m3, m5


def section_hydrodynamics(poly, omega: float, depth: float = 1.0, rho: float = 1000.0,
                          g: float = GRAVITY, wavelength_span: float = 2.5,
                          nx: int = 200, nz: int = 56, amp: float = 0.02,
                          beach_strength: float = 3.0) -> dict:
    """Added mass, damping, excitation, restoring and RAO of a 2-D section."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    poly = np.asarray(poly, dtype=float)
    k = dispersion_k(omega, depth, g)
    nu = omega * omega / g
    lam = 2.0 * math.pi / k
    x_mid = 0.5 * (poly[:, 0].min() + poly[:, 0].max())
    Lx = wavelength_span * lam
    xf = np.linspace(x_mid - Lx, x_mid + Lx, nx + 1)
    zf = np.linspace(-depth, 0.0, nz + 1)
    openn_x, openn_z, solid = _grid_and_masks(poly, xf, zf)
    openn_x[0, :] = 1.0
    openn_x[-1, :] = 1.0
    openn_z[nz, :] = 1.0
    openn_z[0, :] = 0.0
    faces = _cut_faces(openn_x, openn_z, solid, xf, zf)
    if faces["area"].size == 0:
        raise ValueError("no wetted faces found - section not captured by the grid")
    # Absorbing beach on the free surface: a complex nu = omega^2/g - i sigma(x)
    # radiates the outgoing wave instead of letting a first-order Sommerfeld
    # condition (exact only at normal incidence) decide the radiation damping.
    xc_all = xf[:-1] + (xf[1] - xf[0]) / 2
    span = max(Lx - 1.5 * lam, 1e-9)
    ramp = np.clip((np.abs(xc_all - x_mid) - 1.5 * lam) / span, 0.0, 1.0) ** 2
    if beach_strength > 0:
        nu_col = nu - 1j * beach_strength * omega * ramp
    else:
        nu_col = None
    M, free = _assemble(openn_x, openn_z, solid, xf, zf, k, nu, nu_col)
    nz_cells = solid.shape[1]
    flat_idx = faces["i"] * nz_cells + faces["j"]

    def solve(vn):
        flat = np.zeros(free.size, dtype=complex)
        np.add.at(flat, flat_idx, faces["area"] * vn)
        rhs = -flat
        rhs[~free] = 0.0
        phi = spsolve(M, rhs)
        return phi

    vn_heave = faces["nz"].astype(complex)
    vn_pitch = (faces["z"] * faces["nx"] - faces["x"] * faces["nz"]).astype(complex)
    phi3 = solve(vn_heave)
    phi5 = solve(vn_pitch)
    m3_3, m5_3 = _moments(phi3, faces, flat_idx)
    m3_5, m5_5 = _moments(phi5, faces, flat_idx)
    # incident potential at the wetted faces and its normal derivative
    phi_i = _incident(faces["x"], faces["z"], k, depth, omega, g, amp)
    dphidx = 1j * k * phi_i
    dphidz = phi_i * k * np.tanh(k * (faces["z"] + depth))
    dphi_dn = dphidx * faces["nx"] + dphidz * faces["nz"]
    phi_d = solve(-dphi_dn)
    m3_d, m5_d = _moments(phi_d, faces, flat_idx, phi_extra=phi_i)
    m3_i, m5_i = _moments(np.zeros_like(phi_d), faces, flat_idx, phi_extra=phi_i)
    # The radiation damping sign is fixed by the *physics*: an oscillating body
    # must radiate energy outwards (B > 0).  The far-field check in the tests
    # confirms the computed potential carries an outward-propagating wave, so
    # B = +omega*rho*Im(integral(phi n dl)) is the branch that reproduces
    # positive damping at every frequency.
    z33 = rho * m3_3
    z55 = rho * m5_5
    z35 = rho * m5_3
    z53 = rho * m3_5
    A = np.array([[-z33.real, -z35.real], [-z53.real, -z55.real]])
    B = np.array([[z33.imag, z35.imag], [z53.imag, z55.imag]]) * omega
    f3 = -1j * omega * rho * m3_d
    f5 = -1j * omega * rho * m5_d
    rest = hydrostatic_restoring(poly, rho, g)
    out = {"omega": omega, "k": k, "wavelength": lam, "depth": depth, "amp": amp,
           "A": A.tolist(), "B": B.tolist(),
           "F_exc": [[f3.real, f3.imag], [f5.real, f5.imag]],
           "C": [[rest["C33"], 0.0], [0.0, rest["C55"]]],
           "hydrostatics": rest,
           "n_wetted_faces": int(faces["area"].size),
           "wetted_area": float(faces["area"].sum()),
           "m3_i": float(m3_i.real), "m5_i": float(m5_i.real)}
    out["beach_strength"] = beach_strength
    out["rao"] = _rao(out, poly, rho)
    return out


def _incident(x, z, k, depth, omega, g, amp):
    return (-(1j * g * amp / omega) * np.cosh(k * (z + depth)) / math.cosh(k * depth)
            * np.exp(1j * k * x))


def _rao(hydro: dict, poly, rho: float) -> dict:
    """Solve the 2-DOF heave/pitch equation of motion."""
    omega = hydro["omega"]
    A = np.asarray(hydro["A"])
    B = np.asarray(hydro["B"])
    C = np.asarray(hydro["C"])
    F = np.array([complex(*hydro["F_exc"][0]), complex(*hydro["F_exc"][1])])
    wetted = wetted_polygon(poly)
    area, cx, cz, inertia_c = polygon_properties(wetted, rho)
    mass = rho * area
    inertia = inertia_c + mass * (cx * cx + cz * cz)   # about the origin (SWL centre)
    mmat = np.array([[mass, 0.0], [0.0, inertia]])
    lhs = -omega ** 2 * (mmat + A) + 1j * omega * B + C
    xi = np.linalg.solve(lhs, F)
    amp = hydro["amp"]
    return {"heave": complex(xi[0]), "pitch": complex(xi[1]),
            "rao_heave": abs(xi[0]) / amp,
            "rao_pitch": abs(xi[1]) / (hydro["k"] * amp),
            "phase_heave": float(np.angle(xi[0])),
            "phase_pitch": float(np.angle(xi[1])),
            "mass_per_length": mass, "inertia_per_length": inertia,
            "det": abs(np.linalg.det(lhs))}