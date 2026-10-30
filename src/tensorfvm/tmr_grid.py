"""Structured multi-zone PLOT3D grids as collocated finite-volume meshes.

The NASA Turbulence Modeling Resource (TMR) publishes its validation and
verification grids as multi-zone structured PLOT3D files.  This module turns
those files into the face-list mesh contract used by
:class:`tensorfvm.body_fitted.BodyFittedSolver`, so TMR cases can be run on the
official grids instead of on hand-built substitutes.

Face neighbours are recovered by matching shared cell edges, which is valid for
one-to-one abutting zones (every TMR grid family).  Boundary patches are
labelled from geometry alone: the extreme-minimum-x patch is the inflow, the
extreme-maximum-x patch the outflow, the extreme-maximum-y patch the far field,
and every remaining boundary face a no-slip viscous wall.
"""

import numpy as np
import torch


def _fortran_floats(text):
    """Yield every Fortran real in *text*, tolerating omitted exponent markers."""
    import re

    fixed = re.sub(r"(?<=\d)([+-]\d{3}\b)", r"E\1", text)
    return np.array([float(token) for token in fixed.split()], dtype=np.float64)


def read_plot3d_2d(path):
    """Read a formatted, multi-block, 2-D PLOT3D grid.

    Returns a list of ``(x, y)`` arrays, each with shape ``(jdim, idim)``.
    """
    text = open(path, "r").read()
    values = _fortran_floats(text)
    position = 0
    nblocks = int(values[position])
    position += 1
    dims = [(int(values[position + 2 * b]), int(values[position + 2 * b + 1]))
            for b in range(nblocks)]
    position += 2 * nblocks
    blocks = []
    for idim, jdim in dims:
        count = idim * jdim
        x = values[position:position + count].reshape(jdim, idim)
        position += count
        y = values[position:position + count].reshape(jdim, idim)
        position += count
        blocks.append((x, y))
    if position != values.size:
        raise ValueError(f"{path}: {values.size - position} unconsumed values")
    return blocks


class TmrGridMesh:
    """Face-list quad mesh built from a structured multi-zone PLOT3D grid."""

    def __init__(self, config):
        if config.mesh_type != "tmr-grid":
            raise ValueError("TmrGridMesh requires mesh_type='tmr-grid'")
        if not config.mesh_file:
            raise ValueError("TmrGridMesh requires mesh_file pointing at a PLOT3D grid")
        blocks = read_plot3d_2d(config.mesh_file)
        dtype = torch.float64
        # --- vertex deduplication across zone interfaces -------------------
        keys = {}
        coordinates = []
        quads = []
        for x, y in blocks:
            index = np.empty(x.shape, dtype=np.int64)
            for j in range(x.shape[0]):
                for i in range(x.shape[1]):
                    key = (round(float(x[j, i]), 9), round(float(y[j, i]), 9))
                    existing = keys.get(key)
                    if existing is None:
                        existing = len(coordinates)
                        keys[key] = existing
                        coordinates.append((float(x[j, i]), float(y[j, i])))
                    index[j, i] = existing
            jdim, idim = x.shape
            quads.append(np.stack((index[:-1, :-1], index[:-1, 1:],
                                   index[1:, 1:], index[1:, :-1]), axis=-1).reshape(-1, 4))
        quads = np.concatenate(quads)
        vertices = torch.tensor(np.array(coordinates), dtype=dtype)
        quad = vertices[torch.tensor(quads)]
        # --- cell geometry (consistent with the other FVM meshes) ----------
        following = quad.roll(-1, -2)
        cross = quad[..., 0] * following[..., 1] - following[..., 0] * quad[..., 1]
        volumes = cross.sum(-1) / 2
        if not bool((volumes > 0).all()):
            raise ValueError("PLOT3D grid contains nonpositive cell volumes")
        centers = ((quad + following) * cross[..., None]).sum(-2) / (6 * volumes[..., None])
        # --- faces by shared-edge matching ---------------------------------
        owner, neighbor, endpoint_index = [], [], []
        seen = {}
        for cell in range(quads.shape[0]):
            corners = quads[cell]
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                key = (int(min(corners[a], corners[b])), int(max(corners[a], corners[b])))
                if key in seen:
                    neighbor[seen[key]] = cell
                    continue
                seen[key] = len(owner)
                owner.append(cell)
                neighbor.append(-1)
                # Keep the owner-oriented winding: the outward normal of a
                # counter-clockwise quad uses (edge.y, -edge.x), so the stored
                # endpoint order must follow the owner's traversal.
                endpoint_index.append((int(corners[a]), int(corners[b])))
        face_vertices = vertices[torch.tensor(np.array(endpoint_index),
                                              dtype=torch.long)].reshape(-1, 2, 2)
        face_centers = face_vertices.mean(1)
        inside = torch.tensor(np.array(neighbor)) >= 0
        labels = np.array(["interior"] * len(owner), dtype=object)
        boundary = ~inside.numpy()
        bx = face_centers[~inside, 0].numpy()
        by = face_centers[~inside, 1].numpy()
        xmin, xmax = float(face_centers[:, 0].min()), float(face_centers[:, 0].max())
        ymax = float(face_centers[:, 1].max())
        span = max(xmax - xmin, ymax - float(face_centers[:, 1].min()))
        tolerance = 1e-9 * span
        patch = np.full(bx.shape, "wall", dtype=object)
        patch[np.abs(bx - xmin) <= tolerance] = "inlet"
        patch[np.abs(bx - xmax) <= tolerance] = "outlet"
        patch[np.abs(by - ymax) <= tolerance] = config.grid_top_patch
        labels[boundary] = patch
        edge = face_vertices[:, 1] - face_vertices[:, 0]
        face_area_vectors = torch.stack((edge[:, 1], -edge[:, 0]), -1)
        face_lengths = torch.linalg.vector_norm(edge, dim=-1)
        self.vertices = vertices
        self.quads = quad
        self.centers = centers
        self.volumes = volumes
        self.field_shape = (1, int(volumes.numel()))
        self.owner = torch.tensor(owner, dtype=torch.long)
        self.neighbor = torch.tensor(neighbor, dtype=torch.long)
        self.face_vertices = face_vertices
        self.face_centers = face_centers
        self.face_area_vectors = face_area_vectors
        self.face_lengths = face_lengths
        self.face_normals = face_area_vectors / face_lengths[:, None]
        self.interior = self.neighbor >= 0
        self.boundary = ~self.interior
        self.boundary_labels = tuple(labels.tolist())
        self.masks = {name: torch.tensor(np.equal(labels, name), dtype=torch.bool)
                      for name in ("inlet", "outlet", "wall", "far-field",
                                   "cylinder", "airfoil")}
        self.extent = (xmin, xmax, float(face_centers[:, 1].min()), ymax)
        self.zone_dims = tuple((int(x.shape[1]), int(x.shape[0]))
                               for x, _ in blocks)