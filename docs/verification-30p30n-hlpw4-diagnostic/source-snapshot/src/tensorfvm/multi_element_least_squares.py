"""Verified shared least-squares pressure gradient on strongly stretched grids.

Uses BodyFittedSolver's existing production gradient and geometric projected
face interpolation. The complete pressure matrix retains the same gradient in
momentum, velocity correction and Rhie--Chow face flux; probe checks stay active.
"""
from .body_fitted import BodyFittedSolver
from .sparse_simple import SparseBodyFittedSolver
from .conservative_pressure import ConservativePressureGradient
from .multi_element_flow import MultiElementFlowSolver
from .multi_element_coupled import CoupledMultiElementSolver


class LeastSquaresPressure:
    pressure_gradient_scheme='least-squares'

    def _gradient(self,field,pressure=False,boundary_values=None):
        return BodyFittedSolver._gradient(self,field,pressure=pressure,boundary_values=boundary_values)

    def _geometry(self):
        if self._geometry_operators is not None:return self._geometry_operators
        H,_,gradient=SparseBodyFittedSolver._geometry(self)
        self._geometry_operators=None
        _,interpolation,_=ConservativePressureGradient._geometry(self)
        self._geometry_operators=H,interpolation,gradient
        return self._geometry_operators


class LeastSquaresMultiElementSolver(LeastSquaresPressure,MultiElementFlowSolver):
    pass


class LeastSquaresCoupledMultiElementSolver(LeastSquaresPressure,CoupledMultiElementSolver):
    pass
