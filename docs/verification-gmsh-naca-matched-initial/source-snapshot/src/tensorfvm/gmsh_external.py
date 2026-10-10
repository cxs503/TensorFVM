"""Gmsh linear triangle/quad meshes for body-fitted external-flow benchmarks.

Gmsh is needed only during generation. Reading and solving require no Gmsh.
The existing Mesh2D interface and conservative face operators are retained.
"""
from pathlib import Path
import shlex
import math
from dataclasses import replace
import numpy as np
import torch
from .backend_registry import SolverBackend, register_backend, registered_backends
from .multi_element import ThreeElementMesh
from .external_anderson import AndersonExternalFlowSolver
from .external_coupled import CoupledExternalFlowSolver
from .external_linear import ExternalFlowSolver


class GmshExternalMesh:
    def __init__(self, config):
        path = Path(config.mesh_file)
        sections = ThreeElementMesh._read_sections(path)
        fmt = sections.get('MeshFormat', [''])[0].split()
        if len(fmt)<2 or not fmt[0].startswith('2.') or fmt[1]!='0':
            raise ValueError('External mesh requires ASCII Gmsh v2')
        names = {}
        allowed = {'inlet', 'outlet', 'wall', 'far-field', 'cylinder', 'airfoil', 'open-boundary', 'fluid'}
        for line in sections.get('PhysicalNames', [])[1:]:
            dim, tag, name = shlex.split(line)
            if name not in allowed:
                raise ValueError('Unknown external mesh physical name: '+name)
            names[int(dim), int(tag)] = name
        ids, coordinates = ThreeElementMesh._nodes(sections.get('Nodes', []))
        line_groups, cells = ThreeElementMesh._elements(sections.get('Elements', []), names, {node:i for i,node in enumerate(ids)})
        body = 'cylinder' if config.mesh_type=='gmsh-cylinder' else 'airfoil'
        if not {'inlet', 'outlet', body}.issubset(set(line_groups.values())):
            raise ValueError('Missing body/inlet/outlet physical boundary groups')
        if not cells:
            raise ValueError('External mesh contains no fluid cells')
        centers, volumes, owner, neighbor, endpoints, labels, known = [], [], [], [], [], [], {}
        for ci, cell in enumerate(cells):
            points = [coordinates[i] for i in cell]
            if abs(ThreeElementMesh._signed_area(points))<=1e-18:
                raise ValueError('Zero-area external mesh cell')
            if ThreeElementMesh._signed_area(points)<0:
                cell.reverse();points.reverse()
            center, area = ThreeElementMesh._centroid(points)
            if not np.isfinite(area) or area<=1e-18:
                raise ValueError('Invalid external mesh cell')
            centers.append(center);volumes.append(area)
            for a,b in zip(cell,cell[1:]+cell[:1]):
                key = tuple(sorted((a,b)))
                if key in known:
                    fi = known[key]
                    if neighbor[fi]!=-1 or endpoints[fi]!=(b,a):
                        raise ValueError('Non-manifold or inconsistent cell edge orientation')
                    neighbor[fi]=ci;labels[fi]='interior'
                else:
                    known[key]=len(owner);owner.append(ci);neighbor.append(-1);endpoints.append((a,b));labels.append(line_groups.get(key))
        if any(label is None for label in labels):
            raise ValueError('Untagged external mesh boundary')
        opts = dict(dtype=torch.float64,device=config.device)
        self.vertices = torch.tensor(coordinates,**opts)
        self.cell_nodes = tuple(tuple(cell) for cell in cells)
        self.cells = self.cell_nodes
        self.centers = torch.tensor(centers,**opts)[None]
        self.volumes = torch.tensor(volumes,**opts)[None]
        self.field_shape = (1,len(cells))
        self.owner = torch.tensor(owner,dtype=torch.long,device=config.device)
        self.neighbor = torch.tensor(neighbor,dtype=torch.long,device=config.device)
        self.face_vertices = self.vertices[torch.tensor(endpoints,dtype=torch.long,device=config.device)]
        self.face_centers = self.face_vertices.mean(1)
        edge = self.face_vertices[:,1]-self.face_vertices[:,0]
        self.face_area_vectors = torch.stack((edge[:,1],-edge[:,0]),-1)
        self.face_lengths = torch.linalg.vector_norm(edge,dim=-1)
        self.face_normals = self.face_area_vectors/self.face_lengths[:,None]
        self.interior = self.neighbor>=0
        self.boundary = ~self.interior
        self.boundary_labels = tuple(labels)
        self.body_label = body
        self.masks = {name:torch.tensor([label==name for label in labels],dtype=torch.bool,device=config.device) for name in allowed-{'fluid'}}


class GmshForceResult:
    def _aerodynamic_coefficients(self):
        force=self._surface_force(self.mesh.masks[self.mesh.body_label])
        c=self.config;a=math.radians(c.angle_of_attack)
        scale=.5*c.density*c.inlet_velocity**2*c.reference_length
        return dict(drag=float((force[0]*math.cos(a)+force[1]*math.sin(a))/scale),lift=float((-force[0]*math.sin(a)+force[1]*math.cos(a))/scale))

    def solve(self):
        result=super().solve()
        body=self.mesh.body_label
        force=self._surface_force(self.mesh.masks[body])
        return replace(result,aerodynamic_coefficients=self._aerodynamic_coefficients(),surface_forces={body:dict(x=float(force[0]),y=float(force[1]))})


class GmshExternalSolver(GmshForceResult,AndersonExternalFlowSolver):
    """Current-equation SIMPLE; unstructured geometry has no logical coarsening."""
    def __init__(self, config):
        if config.device!='cpu':
            raise ValueError('Unstructured external GPU preconditioner has not been verified')
        super().__init__(config)


class GmshCoupledSolver(GmshForceResult,CoupledExternalFlowSolver):
    def __init__(self, config):
        if config.mesh_type!='gmsh-cylinder' or config.device!='cpu' or config.time_step is not None or config.turbulence_model!='laminar':
            raise ValueError('Gmsh coupled correction supports steady laminar CPU cylinders')
        ExternalFlowSolver.__init__(self,config)
        self.coupled_history=[];self._coupled_lu=None


for name,body in [('gmsh-cylinder','cylinder'),('gmsh-airfoil','airfoil')]:
    if name not in registered_backends():
        register_backend(SolverBackend(name=name,solver_factory=GmshExternalSolver,mesh_factory=GmshExternalMesh,structured=False,requires_mesh_file=True,supports_parabolic_inlet=body=='cylinder',uses_cylinder=body=='cylinder',requires_cylinder=body=='cylinder',reference_length='cylinder_or_height' if body=='cylinder' else 'airfoil_chord',residual_length='height' if body=='cylinder' else 'airfoil_chord',required_mesh_masks=('inlet','outlet','wall','far-field',body)))
