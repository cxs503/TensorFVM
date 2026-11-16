"""3-D periodic-hill case (ERCOFTAC UFR 3-30, Breuer et al. 2009) -- mesh.

Round 33 (mission 23249, TensorFVM).  The 2-D periodic-hill case
(``periodic_hill``) has been exhaustively explored and is an honest FAIL at
every Reynolds number tried (Re = 700/1400/2800): the 2-D mean recirculation
bubble is systematically the wrong length (too long in 2-D laminar, too short
in 2-D RANS), so the 2-D tool cannot meet the anchor's <10 % reattachment
criterion.  The path to a passing result is a 3-D simulation, which is what
this module provides.

Geometry
--------
The 2-D terrain-following cross-section of ``periodic_hill`` (a channel
periodic in ``x`` with a sinusoidal hill on the bottom wall) is extruded in the
spanwise ``z`` direction.  The 3-D domain is therefore periodic in both ``x``
(streamwise) and ``z`` (spanwise), with a no-slip hill surface on the bottom
and a no-slip flat wall on top.  There is **no** inlet/outlet pair (unlike the
cylinder case): the streamwise periodicity replaces it, and a body force
(integral controller) holds the bulk velocity above the crest at the target
value, exactly as the 2-D solver does.

The mesh is a structured hexahedral grid: ``nx`` cells in ``x`` (periodic),
``ny`` in ``y`` (eta levels, terrain following), ``nz`` in ``z`` (periodic).
The lateral faces are the 2-D faces extruded over one ``dz``; the spanwise
faces close the periodic ``z`` direction.

This is a CPU reference implementation.  It is *not* a qualified Re=2800 LES:
the SGS model, the upwind fraction and the time step all need qualification on
a refined grid before any number from it is graded against the DNS reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .periodic_hill import PeriodicHillConfig


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class PeriodicHill3DConfig(PeriodicHillConfig):
    """3-D periodic-hill parameters.

    Extends :class:`PeriodicHillConfig` with the spanwise resolution.  ``nz``
    is the number of spanwise cells and ``span_m`` the spanwise period.  The
    ERCOFTAC reference DNS uses a span of ``4H`` with the flow statistically
    homogeneous in ``z``; we default to ``span_m = 4.0`` (i.e. ``4H``) and a
    modest ``nz`` for the feasibility runs.
    """

    nz: int = 8
    span_m: float = 4.0

    def __post_init__(self):
        super().__post_init__()
        if self.nz < 2:
            raise ValueError("need nz >= 2")
        if not math.isfinite(self.span_m) or self.span_m <= 0.0:
            raise ValueError("invalid span_m: need finite > 0")

    @property
    def dz(self):
        return self.span_m / self.nz


# --------------------------------------------------------------------------- #
# 2-D cross-section (extruder-compatible)
# --------------------------------------------------------------------------- #
def build_hill_cross_section(config):
    """Build the 2-D terrain-following hill cross-section.

    Returns a namespace with the attributes the 3-D extruder consumes:
    ``owner``, ``neighbor``, ``boundary``, ``face_vertices`` (F, 2, 2),
    ``face_area_vectors`` (F, 2), ``centers`` (N, 2), ``volumes`` (N,),
    ``boundary_labels``.  Face ordering is: all east faces (row-major over
    (j, i)), then all eta (north/south) faces from the bottom (j=0) to the top
    (j=ny).  Area vectors point from the owner cell towards the neighbour
    (or out of the domain on a wall).
    """
    c = config
    nx, ny = c.nx, c.ny
    L = c.length_m

    # node grid: x = xi (periodic), y = y_b(x) + eta*(top - y_b(x))
    xv = np.linspace(0.0, L, nx + 1)
    etap = np.linspace(0.0, 1.0, ny + 1)
    yb = np.array([c.hill(x)[0] for x in xv])
    top = c.channel_top_m
    yv = yb[None, :] + etap[:, None] * (top - yb[None, :])

    # cell centres (average of the four quad corners)
    xc = np.broadcast_to(0.5 * (xv[:-1] + xv[1:])[None, :], (ny, nx)).copy()
    yc = 0.25 * (yv[:-1, :-1] + yv[:-1, 1:] + yv[1:, 1:] + yv[1:, :-1])
    # trapezoid area: vertical sides at x0, x1
    h_left = yv[1:, :-1] - yv[:-1, :-1]
    h_right = yv[1:, 1:] - yv[:-1, 1:]
    vol = 0.5 * (h_left + h_right) * (xv[1:] - xv[:-1])

    centers = torch.stack(
        (torch.from_numpy(xc).reshape(-1), torch.from_numpy(yc).reshape(-1)), -1
    ).double()
    volumes = torch.from_numpy(vol.reshape(-1)).double()
    if not bool((volumes > 0).all()):
        raise ValueError("cross-section has non-positive cell volumes")

    cell_id = np.arange(nx * ny).reshape(ny, nx)

    owners, neighbors, endpoints, area_vecs, labels = [], [], [], [], []

    def add_face(o, n, v0, v1, avx, avy, label):
        owners.append(o)
        neighbors.append(n)
        endpoints.append((v0, v1))
        area_vecs.append((avx, avy))
        labels.append(label)

    # east faces: between (j, i) and (j, i+1 mod nx); vertical, normal (+1, 0)
    for j in range(ny):
        for i in range(nx):
            i_e = (i + 1) % nx
            x_f = xv[i + 1]
            y_lo, y_hi = yv[j, i + 1], yv[j + 1, i + 1]
            add_face(
                int(cell_id[j, i]), int(cell_id[j, i_e]),
                (x_f, y_lo), (x_f, y_hi), y_hi - y_lo, 0.0, "interior",
            )

    # eta faces: level j = 0..ny, column i.  The area vector must point from
    # the owner cell towards the face (and out of the domain on a wall); the
    # sign is fixed by the owner-centre -> face-centre displacement so the
    # bottom, interior and top walls all come out consistently.
    for j in range(ny + 1):
        for i in range(nx):
            x_lo, x_hi = xv[i], xv[i + 1]
            y_lo, y_hi = yv[j, i], yv[j, i + 1]
            ex, ey = x_hi - x_lo, y_hi - y_lo
            if j == 0:
                o, n, lab = int(cell_id[0, i]), -1, "hill"
            elif j == ny:
                o, n, lab = int(cell_id[ny - 1, i]), -1, "top"
            else:
                o, n, lab = int(cell_id[j - 1, i]), int(cell_id[j, i]), "interior"
            fcx, fcy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
            j_own = 0 if j == 0 else (ny - 1 if j == ny else j - 1)
            dxo, dyo = fcx - xc[j_own, i], fcy - yc[j_own, i]
            nx_, ny_ = ey, -ex
            if dxo * nx_ + dyo * ny_ < 0.0:
                nx_, ny_ = -nx_, -ny_
            add_face(o, n, (x_lo, y_lo), (x_hi, y_hi), nx_, ny_, lab)

    owner = torch.tensor(owners, dtype=torch.long)
    neighbor = torch.tensor(neighbors, dtype=torch.long)
    face_vertices = torch.stack([
        torch.tensor([list(a), list(b)], dtype=torch.float64)
        for a, b in endpoints
    ])
    face_area_vectors = torch.tensor(area_vecs, dtype=torch.float64)

    return type("HillCrossSection", (), {
        "owner": owner,
        "neighbor": neighbor,
        "boundary": neighbor < 0,
        "face_vertices": face_vertices,
        "face_area_vectors": face_area_vectors,
        "face_centers": face_vertices.mean(1),
        "centers": centers,
        "volumes": volumes,
        "boundary_labels": tuple(labels),
        "nx": nx,
        "ny": ny,
    })()


# --------------------------------------------------------------------------- #
# 3-D mesh
# --------------------------------------------------------------------------- #
@dataclass
class BodyFittedHillMesh3D:
    """Periodic-extruded terrain-following hill mesh (x and z periodic)."""

    nx: int
    ny: int
    nz: int
    length: float
    height: float
    span: float
    channel_top: float
    mesh_type: str
    device: torch.device

    centers: torch.Tensor
    cell_volumes: torch.Tensor
    lateral_owner: torch.Tensor
    lateral_neighbor: torch.Tensor
    lateral_face_area_vectors: torch.Tensor
    lateral_face_centers: torch.Tensor
    lateral_labels: list
    spanwise_face_areas: torch.Tensor
    dz: float

    @classmethod
    def build(cls, config, device: torch.device):
        c = config
        cross = build_hill_cross_section(c)
        nx, ny, nz = c.nx, c.ny, c.nz
        dz = c.span_m / c.nz
        z_nodes = torch.linspace(0.0, c.span_m, nz + 1, dtype=torch.float64,
                                 device=device)
        face_count = len(cross.owner)
        opts = {"dtype": torch.float64, "device": device}

        # extrude every 2-D face into one oriented quad per z layer
        endpoints = cross.face_vertices
        lower = torch.empty((nz, face_count, 2, 3), **opts)
        upper = torch.empty_like(lower)
        lower[..., :2] = endpoints[None]
        upper[..., :2] = endpoints[None]
        lower[..., 2] = z_nodes[:-1, None, None]
        upper[..., 2] = z_nodes[1:, None, None]
        vertices = torch.stack((lower[:, :, 0], lower[:, :, 1],
                                upper[:, :, 1], upper[:, :, 0]), dim=2)
        centers = vertices.mean(dim=2)
        area_vectors = torch.zeros((nz, face_count, 3), **opts)
        area_vectors[..., :2] = cross.face_area_vectors[None] * dz
        cell_count = nx * ny
        offsets = torch.arange(nz, device=device)[:, None] * cell_count
        owner = cross.owner[None] + offsets
        neighbor = torch.full((nz, face_count), -1, dtype=torch.long,
                              device=device)
        interior = cross.neighbor >= 0
        neighbor[:, interior] = cross.neighbor[None, interior] + offsets
        labels = list(cross.boundary_labels)

        # cell centres / volumes: cross-section extruded
        cz = 0.5 * (z_nodes[:-1] + z_nodes[1:])
        centers3 = torch.empty((nz, cell_count, 3), **opts)
        centers3[..., :2] = cross.centers[None]
        centers3[..., 2] = cz[:, None]
        volumes = cross.volumes[None].expand(nz, -1).clone() * dz

        return cls(
            nx=nx, ny=ny, nz=nz, length=c.length_m, height=c.height_m,
            span=c.span_m, channel_top=c.channel_top_m,
            mesh_type="body-fitted", device=device,
            centers=centers3, cell_volumes=volumes,
            lateral_owner=owner, lateral_neighbor=neighbor,
            lateral_face_area_vectors=area_vectors,
            lateral_face_centers=centers,
            lateral_labels=labels,
            spanwise_face_areas=cross.volumes[None].expand(nz + 1, -1).clone(),
            dz=dz,
        )

    @property
    def local_nz(self):
        return self.nz

    @property
    def cell_count(self):
        return self.nz * self.ny * self.nx


__all__ = [
    "PeriodicHill3DConfig",
    "build_hill_cross_section",
    "BodyFittedHillMesh3D",
]
