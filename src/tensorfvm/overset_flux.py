"""Conservative transfer on a common planar Cartesian interface.

This maps *integrated* oriented fluxes, not point values. It requires matching
geometric coverage. It does not turn arbitrary overset receiver interpolation
into a conservative Navier--Stokes coupling.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


def _boxes(value):
    a = np.asarray(value, dtype=float)
    if a.ndim != 3 or a.shape[1] != 2 or a.shape[2] not in (1, 2):
        raise ValueError('Expected (faces, lower/upper, tangential dimensions)')
    if not len(a) or not np.isfinite(a).all() or np.any(a[:, 1] <= a[:, 0]):
        raise ValueError('Interface patches must have finite positive extent')
    return a.copy()


@dataclass(frozen=True)
class PlanarFluxTransfer:
    """Common-refinement mortar for line segments or planar rectangles.

    Coordinates must be in the SAME tangential frame and physical units.
    Normal sign +1 denotes matching normal directions, -1 opposite outward
    normals. Flux is assumed uniform within each donor face; varying flux needs
    finer faces or a higher-order integration scheme.
    """
    donor_bounds: np.ndarray
    receiver_bounds: np.ndarray
    donor_measure: np.ndarray
    receiver_measure: np.ndarray
    overlap: csr_matrix
    normal_sign: int

    @classmethod
    def build(cls, donor_bounds, receiver_bounds, *, normal_sign=-1, tolerance=1e-11):
        d, r = _boxes(donor_bounds), _boxes(receiver_bounds)
        if d.shape[2] != r.shape[2] or normal_sign not in (-1, 1):
            raise ValueError('Tangential dimension or normal orientation mismatch')
        if not np.isfinite(tolerance) or not 0 < tolerance < 1e-3:
            raise ValueError('Invalid relative coverage tolerance')
        dm, rm = np.prod(d[:, 1]-d[:, 0], axis=1), np.prod(r[:, 1]-r[:, 0], axis=1)
        # A partition cannot overlap itself. Pairwise checks also reject duplicate
        # patches even when their union has the correct external extent.
        for boxes in (d, r):
            for i, box in enumerate(boxes):
                extent = np.minimum(box[1], boxes[i+1:, 1])-np.maximum(box[0], boxes[i+1:, 0])
                if np.any(np.all(extent > 0, axis=1)):
                    raise ValueError('Overlapping interface patches')
        rows, cols, areas = [], [], []
        for i, box in enumerate(r):
            extent = np.maximum(0, np.minimum(box[1], d[:, 1])-np.maximum(box[0], d[:, 0]))
            measure = np.prod(extent, axis=1)
            ids = np.flatnonzero(measure > 0)
            rows.extend([i]*len(ids)); cols.extend(ids); areas.extend(measure[ids])
        overlap = csr_matrix((areas, (rows, cols)), shape=(len(r), len(d)))
        if (np.any(np.abs(np.asarray(overlap.sum(axis=0)).ravel()-dm) > tolerance*dm)
                or np.any(np.abs(np.asarray(overlap.sum(axis=1)).ravel()-rm) > tolerance*rm)):
            raise ValueError('Interface coverage mismatch: gaps or unequal domains')
        return cls(d, r, dm, rm, overlap, normal_sign)

    def transfer(self, integrated_flux):
        """Transfer scalar or component fluxes while preserving oriented totals."""
        q = np.asarray(integrated_flux, dtype=float)
        if q.ndim < 1 or q.shape[0] != len(self.donor_measure) or not np.isfinite(q).all():
            raise ValueError('Invalid finite donor integrated flux')
        tail = q.shape[1:]
        density = q.reshape(len(q), -1)/self.donor_measure[:, None]
        return (self.normal_sign*(self.overlap @ density)).reshape((len(self.receiver_measure),)+tail)

    def balance(self, donor_flux, receiver_flux):
        d, r = np.asarray(donor_flux, dtype=float), np.asarray(receiver_flux, dtype=float)
        if (d.ndim < 1 or r.ndim < 1 or d.shape[0] != len(self.donor_measure)
                or r.shape[0] != len(self.receiver_measure) or d.shape[1:] != r.shape[1:]
                or not np.isfinite(d).all() or not np.isfinite(r).all()):
            raise ValueError('Invalid interface flux pair')
        return r.sum(axis=0)-self.normal_sign*d.sum(axis=0)
