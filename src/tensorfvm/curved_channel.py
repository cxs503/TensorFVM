"""Conforming sinusoidal channel and nontrivial full Navier--Stokes manufactured flow."""
import math
import torch
from .backend_registry import register_backend,SolverBackend
from .mesh_api import validate_mesh2d
from .sparse_simple import SparseBodyFittedSolver


def shape(x,length,amplitude):
    k=math.pi/length;s=torch.sin(k*x);c=torch.cos(k*x)
    return amplitude*s**4,4*amplitude*k*s**3*c,4*amplitude*k*k*(3*s*s*c*c-s**4),4*amplitude*k**3*(6*s*c**3-10*s**3*c)


def reference(xy,c):
    x,y=xy.unbind(-1);g,g1,g2,g3=shape(x,c.length,.2*c.height);eta=(y-g)/c.height
    w=6*c.inlet_velocity*eta*(1-eta);wy=6*c.inlet_velocity/c.height*(1-2*eta);wyy=torch.full_like(w,-12*c.inlet_velocity/c.height**2)
    G=12*c.viscosity*c.inlet_velocity/c.height**2
    velocity=torch.stack((w,g1*w),-1);pressure=G*(c.length-x)
    lapu=(1+g1*g1)*wyy-g2*wy;lapv=g3*w-3*g1*g2*wy+g1*(1+g1*g1)*wyy
    force=torch.stack((-G-c.viscosity*lapu,c.density*g2*w*w-c.viscosity*lapv),-1)
    return velocity,pressure,force


class CurvedChannelMesh:
    def __init__(self,c):
        nx,ny=c.nx,c.ny;opts=dict(dtype=torch.float64,device=c.device);x=torch.linspace(0,c.length,nx+1,**opts);eta=torch.linspace(0,c.height,ny+1,**opts);y,xx=torch.meshgrid(eta,x,indexing='ij');y=y+shape(xx,c.length,.2*c.height)[0]
        self.vertices=torch.stack((xx,y),-1);self.field_shape=(ny,nx)
        v=self.vertices;pol=torch.stack((v[:-1,:-1],v[:-1,1:],v[1:,1:],v[1:,:-1]),-2);nxt=pol.roll(-1,-2);cross=pol[...,0]*nxt[...,1]-nxt[...,0]*pol[...,1];self.volumes=cross.sum(-1)/2;self.centers=((pol+nxt)*cross[...,None]).sum(-2)/(6*self.volumes[...,None])
        j,i=torch.meshgrid(torch.arange(ny+1,device=c.device),torch.arange(nx,device=c.device),indexing='ij');o=(j.clamp_max(ny-1)*nx+i).ravel();ne=torch.where((j>0)&(j<ny),(j-1)*nx+i,-1).ravel();a=v[:,:-1].reshape(-1,2);b=v[:,1:].reshape(-1,2);top=(j==ny).ravel();ends=torch.stack((torch.where(top[:,None],b,a),torch.where(top[:,None],a,b)),1);wall=((j==0)|(j==ny)).ravel();inlet=torch.zeros_like(wall);outlet=torch.zeros_like(wall)
        jj,ii=torch.meshgrid(torch.arange(ny,device=c.device),torch.arange(nx+1,device=c.device),indexing='ij');ov=(jj*nx+ii.clamp_max(nx-1)).ravel();nv=torch.where((ii>0)&(ii<nx),jj*nx+ii-1,-1).ravel();a=v[:-1].reshape(-1,2);b=v[1:].reshape(-1,2);right=(ii==nx).ravel();ev=torch.stack((torch.where(right[:,None],a,b),torch.where(right[:,None],b,a)),1)
        self.owner=torch.cat((o,ov));self.neighbor=torch.cat((ne,nv));self.face_vertices=torch.cat((ends,ev));self.face_centers=self.face_vertices.mean(1);edge=self.face_vertices[:,1]-self.face_vertices[:,0];self.face_area_vectors=torch.stack((edge[:,1],-edge[:,0]),-1);self.face_lengths=torch.linalg.vector_norm(edge,dim=1);self.face_normals=self.face_area_vectors/self.face_lengths[:,None];self.interior=self.neighbor>=0;self.boundary=~self.interior
        self.masks={'wall':torch.cat((wall,torch.zeros_like(right))),'inlet':torch.cat((inlet,(ii==0).ravel())),'outlet':torch.cat((outlet,right))}
        for name in ['far-field','cylinder','airfoil']:self.masks[name]=torch.zeros_like(self.boundary)
        labels=['interior']*len(self.owner)
        for name in ['wall','inlet','outlet']:
            for index in self.masks[name].nonzero().flatten().cpu().tolist():labels[index]=name
        self.boundary_labels=tuple(labels);validate_mesh2d(self,required_masks=('inlet','outlet','wall','far-field'))


register_backend(SolverBackend(name='curved-channel',solver_factory=SparseBodyFittedSolver,mesh_factory=CurvedChannelMesh,structured=True,min_nx=8,min_ny=4,forbids_cylinder=True,supports_parabolic_inlet=True,reference_length='height'))
