"""Pressure-open NACA far boundaries with prescribed freestream on backflow.

The paper specifies open boundaries without an exact FV formula. This explicit
pressure-outlet/inletOutlet treatment is a documented numerical choice.
"""
import math
import torch
from .backend_registry import SolverBackend,register_backend,registered_backends
from .gmsh_external import GmshExternalMesh
from .external_high_order import ConservativeLinearUpwindSolver


class OpenBoundaryNacaSolver(ConservativeLinearUpwindSolver):
    def __init__(self,config):
        if config.mesh_type!='gmsh-airfoil-open' or config.device!='cpu':
            raise ValueError('Verified NACA pressure-open path requires gmsh-airfoil-open on CPU')
        super().__init__(config)
        self.mesh.masks['outlet'] |= self.mesh.masks['open-boundary']
        angle=math.radians(config.angle_of_attack)
        velocity=torch.tensor([config.inlet_velocity*math.cos(angle),config.inlet_velocity*math.sin(angle)],dtype=self.p.dtype,device=self.p.device)
        self.boundary_velocity[self.mesh.masks['outlet']]=velocity

    def _momentum(self,velocity,pressure,flux):
        diagonal,ao,an,source=super()._momentum(velocity,pressure,flux)
        backflow=self.mesh.masks['outlet'] & (flux<0)
        # Replace zero-gradient backflow convection by a known freestream value.
        diagonal.index_add_(0,self.o[backflow],-flux[backflow])
        source+=self._sum(torch.where(backflow[:,None],-flux[:,None]*self.boundary_velocity,0))
        return diagonal,ao,an,source


if 'gmsh-airfoil-open' not in registered_backends():
    register_backend(SolverBackend(name='gmsh-airfoil-open',solver_factory=OpenBoundaryNacaSolver,mesh_factory=GmshExternalMesh,structured=False,requires_mesh_file=True,reference_length='airfoil_chord',residual_length='airfoil_chord',required_mesh_masks=('inlet','outlet','wall','far-field','airfoil','open-boundary')))
