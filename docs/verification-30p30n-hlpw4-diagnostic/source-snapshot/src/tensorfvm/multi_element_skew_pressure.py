"""Conservative face-pressure reconstruction on stretched, skewed meshes.

Least-squares gradients supply a geometric skewness correction. The pressure
force is then integrated over faces, preserving equal/opposite internal forces.
Neumann boundary reconstruction removes its wall-normal gradient component.
The assembled pressure response uses precisely the same reconstruction.
"""
import numpy as np
import torch
from scipy import sparse
from .body_fitted import BodyFittedSolver
from .sparse_simple import SparseBodyFittedSolver, host
from .conservative_pressure import ConservativePressureGradient
from .multi_element_flow import MultiElementFlowSolver
from .multi_element_coupled import CoupledMultiElementSolver


class SkewCorrectedPressure:
    pressure_gradient_scheme = 'gauss-skew-corrected'

    def _pressure_skew_displacement(self):
        centers = self.mesh.centers.reshape(-1, 2)
        delta = self.mesh.face_centers - self._interpolate(centers)
        boundary = self.mesh.boundary
        normal = self.mesh.face_normals[boundary]
        delta[boundary] -= (delta[boundary]*normal).sum(1)[:, None]*normal
        delta[self.mesh.masks['outlet']] = 0
        return delta

    def reconstructed_pressure_faces(self, pressure):
        gradient = BodyFittedSolver._gradient(self, pressure, pressure=True)
        face = self._interpolate(pressure)
        face += (self._pressure_skew_displacement()*self._interpolate(gradient)).sum(1)
        face[self.mesh.masks['outlet']] = 0
        return face

    def _gradient(self, field, pressure=False, boundary_values=None):
        if not pressure:
            return BodyFittedSolver._gradient(self, field, boundary_values=boundary_values)
        return self._sum(self.reconstructed_pressure_faces(field)[:, None]*self.S)/self.volume[:, None]

    def _geometry(self):
        if self._geometry_operators is not None:
            return self._geometry_operators
        # SparseBody builds the secondary LS operator from the independently
        # cached LS weights used above. Retain projected interpolation for all
        # velocity/diffusivity/Rhie--Chow operations.
        H, _, secondary = SparseBodyFittedSolver._geometry(self)
        self._geometry_operators = None
        _, I, _ = ConservativePressureGradient._geometry(self)
        delta = host(self._pressure_skew_displacement())
        P = I + sum((sparse.diags(delta[:, a])@I@secondary[a] for a in (0, 1)), start=sparse.csr_matrix(I.shape))
        P = sparse.diags((~host(self.mesh.masks['outlet'])).astype(float))@P
        inverse = sparse.diags(1/host(self.volume))
        G = [inverse@H@sparse.diags(host(self.S)[:, a])@P for a in (0, 1)]
        self._geometry_operators = H, I, G
        return self._geometry_operators

    def _surface_force(self, mask):
        force = super()._surface_force(mask)
        pressure = self.reconstructed_pressure_faces(self.p)
        return force + ((pressure[mask]-self.p[self.o[mask]])[:, None]*self.S[mask]).sum(0)


class SkewPressureMultiElementSolver(SkewCorrectedPressure, MultiElementFlowSolver):
    pass


class SkewPressureCoupledMultiElementSolver(SkewCorrectedPressure, CoupledMultiElementSolver):
    pass
