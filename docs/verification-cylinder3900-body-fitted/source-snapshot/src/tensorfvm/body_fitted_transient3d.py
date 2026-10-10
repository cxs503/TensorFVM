"""Shared face FV kernel for periodic extruded body-fitted 3-D meshes.

CPU reference implementation, not a qualified Re3900 LES. Face projection,
full symmetric stress, LS reconstruction and SGS all consume actual mesh
metrics. It deliberately does not inherit the Cartesian mask solver.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .body_fitted3d import BodyFittedCylinderMesh3D

@dataclass
class TransientSettings:
    upwind_fraction: float = .1
    sgs: str = 'wale'
    wale_constant: float = .325
    pressure_tolerance: float = 1e-8
    nonorthogonal_iterations: int = 80
    initial_perturbation: float = .001

class BodyFittedTransient3D:
    def __init__(self, config, settings=None):
        self.config=config;self.settings=settings or TransientSettings()
        if config.device!='cpu':raise NotImplementedError('CPU reference pressure factorization; GPU qualification pending')
        if config.mesh_type!='body-fitted':raise ValueError('actual body-fitted mesh required')
        if not 0<=self.settings.upwind_fraction<=1:raise ValueError('upwind fraction must lie in [0,1]')
        if self.settings.sgs not in ('none','wale'):raise ValueError('supported SGS: none, wale')
        self.mesh=BodyFittedCylinderMesh3D.build(config,torch.device('cpu'))
        m=self.mesh;self.centers=m.centers.reshape(-1,3);self.volumes=m.cell_volumes.flatten();self.N=len(self.volumes)
        count=config.nx*config.ny
        zo=torch.arange(self.N);zn=(zo+count)%self.N
        zs=torch.zeros((self.N,3),dtype=torch.float64);zs[:,2]=m.spanwise_face_areas[1:].flatten()
        zc=self.centers.clone();zc[:,2]+=m.dz/2
        self.owner=torch.cat((m.lateral_owner.flatten(),zo));self.neighbor=torch.cat((m.lateral_neighbor.flatten(),zn))
        self.area=torch.cat((m.lateral_face_area_vectors.reshape(-1,3),zs));self.fc=torch.cat((m.lateral_face_centers.reshape(-1,3),zc))
        labs=list(m.lateral_labels)*m.local_nz+['interior']*self.N
        self.interior=self.neighbor>=0;self.boundary=~self.interior
        self.wall=torch.tensor([x=='cylinder' for x in labs]);self.outlet=torch.tensor([x=='outlet' for x in labs]);self.prescribed=self.boundary&~self.wall&~self.outlet
        self.safe=self.neighbor.clamp_min(0)
        self.d=self.centers[self.safe]-self.centers[self.owner]
        self.d[self.boundary]=self.fc[self.boundary]-self.centers[self.owner[self.boundary]]
        self.d[-self.N:,2]=m.dz
        self.owner_offset=self.fc-self.centers[self.owner]
        self.neighbor_offset=self.fc-self.centers[self.safe]
        self.neighbor_offset[-count:,2]=-m.dz/2
        projection=(self.area*self.d).sum(1)
        if not bool((projection>0).all()):raise ValueError('nonpositive owner-neighbor face projection')
        self.coeff=(self.area*self.area).sum(1)/projection
        self.tangent=self.area-self.coeff[:,None]*self.d
        self.area_mag=torch.linalg.vector_norm(self.area,dim=1)
        self.distance=torch.linalg.vector_norm(self.d,dim=1)
        self.velocity=torch.zeros((self.N,3),dtype=torch.float64);self.velocity[:,0]=config.inlet_velocity
        if self.settings.initial_perturbation:
            xyz=self.centers
            x=(xyz[:,0]-config.cylinder_x)/config.diameter
            y=(xyz[:,1]-config.cylinder_y)/config.diameter
            envelope=torch.exp(-((x-1).square()+y.square())/4)
            phase=2*torch.pi*xyz[:,2]/config.span
            self.velocity[:,1]+=self.settings.initial_perturbation*config.inlet_velocity*envelope*torch.sin(phase)
            self.velocity[:,2]+=self.settings.initial_perturbation*config.inlet_velocity*envelope*torch.cos(phase)
        self.pressure=torch.zeros(self.N,dtype=torch.float64)
        self.time=0.;self.history=[];self.forces=[]
        self._build_ls();self._build_pressure()
        self.flux=self.predict_flux(self.velocity)

    def integrate(self, face_values):
        shape=(self.N,)+face_values.shape[1:]
        out=torch.zeros(shape,dtype=face_values.dtype)
        out.index_add_(0,self.owner,face_values)
        out.index_add_(0,self.neighbor[self.interior],-face_values[self.interior])
        return out

    def _build_ls(self):
        # Weighted neighbors and physical Dirichlet faces. Pressure uses outlet
        # Dirichlet plus zero normal derivative at remaining physical faces.
        w=1/self.distance.square();outer=self.d[:,:,None]*self.d[:,None,:]*w[:,None,None]
        normal=torch.zeros((self.N,3,3),dtype=torch.float64)
        normal.index_add_(0,self.owner,outer)
        normal.index_add_(0,self.neighbor[self.interior],outer[self.interior])
        self.ls_inverse=torch.linalg.inv(normal);self.ls_weight=w

    def gradient(self, field, pressure=False):
        scalar=field.ndim==1
        f=field[:,None] if scalar else field
        delta=f[self.safe]-f[self.owner]
        if pressure:
            delta[self.boundary]=0.;delta[self.outlet]=-f[self.owner[self.outlet]]
        else:
            target=torch.zeros((int(self.boundary.sum()),f.shape[1]),dtype=f.dtype)
            if f.shape[1]==3:target[:,0]=self.config.inlet_velocity
            target[self.wall[self.boundary]]=0
            delta[self.boundary]=target-f[self.owner[self.boundary]]
            delta[self.outlet]=0
        rhs=torch.zeros((self.N,f.shape[1],3),dtype=f.dtype)
        val=delta[:,:,None]*self.d[:,None,:]*self.ls_weight[:,None,None]
        rhs.index_add_(0,self.owner,val)
        rhs.index_add_(0,self.neighbor[self.interior],val[self.interior])
        g=torch.einsum('nij,ncj->nci',self.ls_inverse,rhs)
        return g[:,0] if scalar else g

    def _build_pressure(self):
        o=self.owner.numpy();n=self.neighbor.numpy();a=self.coeff.numpy();i=self.interior.numpy();b=self.outlet.numpy()
        row=np.r_[o[i],n[i],o[i],n[i],o[b]];col=np.r_[o[i],n[i],n[i],o[i],o[b]]
        val=np.r_[a[i],a[i],-a[i],-a[i],a[b]]
        self.pressure_matrix=coo_matrix((val,(row,col)),shape=(self.N,self.N)).tocsc()
        self.pressure_lu=splu(self.pressure_matrix)

    def face_values(self, velocity):
        values=.5*(velocity[self.owner]+velocity[self.safe])
        values[self.wall]=0
        values[self.prescribed]=0;values[self.prescribed,0]=self.config.inlet_velocity
        values[self.outlet]=velocity[self.owner[self.outlet]]
        return values

    def predict_flux(self, velocity):
        return (self.face_values(velocity)*self.area).sum(1)

    def pressure_flux_gradient(self, p):
        g=self.gradient(p,pressure=True);gf=.5*(g[self.owner]+g[self.safe]);gf[self.boundary]=g[self.owner[self.boundary]]
        diff=p[self.safe]-p[self.owner]
        diff[self.boundary]=0;diff[self.outlet]=-p[self.owner[self.outlet]]
        q=self.coeff*diff+(gf*self.tangent).sum(1)
        q[self.boundary&~self.outlet]=0
        return q

    def project(self, tentative_flux):
        dt=self.config.time_step/self.config.density
        rhs=-self.integrate(tentative_flux)/dt;p=self.pressure.clone()
        before=None
        for k in range(self.settings.nonorthogonal_iterations):
            nonorth=self.pressure_flux_gradient(p)
            diff=p[self.safe]-p[self.owner];diff[self.boundary]=0;diff[self.outlet]=-p[self.owner[self.outlet]]
            nonorth-=self.coeff*diff
            nonorth[self.boundary&~self.outlet]=0
            p=torch.from_numpy(self.pressure_lu.solve((rhs+self.integrate(nonorth)).numpy()))
            q=tentative_flux-dt*self.pressure_flux_gradient(p)
            residual=float((self.integrate(q)/self.volumes).abs().max())
            if before is None:before=residual
            if residual<self.settings.pressure_tolerance:return p,q,k+1,residual
            if not np.isfinite(residual):break
        raise RuntimeError(f'nonorthogonal pressure projection failed: max divergence {residual:.6g}, first {before:.6g}')

    def eddy_viscosity(self, gradient):
        if self.settings.sgs=='none':return torch.zeros(self.N,dtype=torch.float64)
        s=.5*(gradient+gradient.transpose(1,2));g2=gradient@gradient
        sd=.5*(g2+g2.transpose(1,2));sd-=torch.eye(3,dtype=torch.float64)[None]*sd.diagonal(dim1=1,dim2=2).sum(1)[:,None,None]/3
        ss=s.square().sum((1,2));dd=sd.square().sum((1,2))
        numerator=dd.pow(1.5);denominator=ss.pow(2.5)+dd.pow(1.25)
        return (self.settings.wale_constant*self.volumes.pow(1/3)).square()*numerator/denominator.clamp_min(1e-60)

    def transport(self, velocity, flux):
        c=self.config;g=self.gradient(velocity);nu=c.kinematic_viscosity+self.eddy_viscosity(g)
        gf=.5*(g[self.owner]+g[self.safe]);gf[self.boundary]=g[self.owner[self.boundary]];mu=c.density*.5*(nu[self.owner]+nu[self.safe])
        mu[self.wall]=c.density*c.kinematic_viscosity
        # Enforce two-point normal derivative while retaining LS tangential derivatives.
        delta=velocity[self.safe]-velocity[self.owner]
        delta[self.wall]=-velocity[self.owner[self.wall]]
        delta[self.prescribed]=-velocity[self.owner[self.prescribed]];delta[self.prescribed,0]+=c.inlet_velocity
        delta[self.outlet]=0
        defect=delta-torch.einsum('fcj,fj->fc',gf,self.d)
        gf+=defect[:,:,None]*self.d[:,None,:]/self.distance.square()[:,None,None]
        stress=mu[:,None,None]*(gf+gf.transpose(1,2)-2/3*torch.eye(3,dtype=torch.float64)[None]*gf.diagonal(dim1=1,dim2=2).sum(1)[:,None,None])
        viscous=torch.einsum('fij,fj->fi',stress,self.area);viscous[self.outlet]=0
        up=velocity[self.owner].clone();reverse=flux<0
        up[reverse&self.interior]=velocity[self.neighbor[reverse&self.interior]]
        up[self.prescribed]=0;up[self.prescribed,0]=c.inlet_velocity
        up[self.outlet&reverse]=0;up[self.outlet&reverse,0]=c.inlet_velocity
        central=self.face_values(velocity)
        transported=(1-self.settings.upwind_fraction)*central+self.settings.upwind_fraction*up
        convection=c.density*flux[:,None]*transported
        return convection,viscous,nu

    @torch.no_grad()
    def step(self):
        c=self.config;u0=self.velocity.clone();oldmomentum=c.density*(u0*self.volumes[:,None]).sum(0)
        conv,visc,nu=self.transport(u0,self.flux)
        tentative=u0+c.time_step/c.density*self.integrate(visc-conv)/self.volumes[:,None]
        p,q,it,div=self.project(self.predict_flux(tentative))
        facep=.5*(p[self.owner]+p[self.safe]);facep[self.boundary]=p[self.owner[self.boundary]];facep[self.outlet]=0
        pressure_traction=facep[:,None]*self.area
        u=tentative-c.time_step/c.density*self.integrate(pressure_traction)/self.volumes[:,None]
        momentum=c.density*(u*self.volumes[:,None]).sum(0)
        ledger=momentum-oldmomentum-c.time_step*(visc-conv-pressure_traction)[self.boundary].sum(0)
        wallforce=(pressure_traction-visc)[self.wall].sum(0)
        force_scale=.5*c.density*c.inlet_velocity**2*c.diameter*c.span
        total=torch.zeros(self.N,dtype=torch.float64);total.index_add_(0,self.owner,q.abs());total.index_add_(0,self.neighbor[self.interior],q[self.interior].abs())
        cfl=float((c.time_step*total/(2*self.volumes)).max())
        if not bool(torch.isfinite(u).all()) or cfl>=1:raise RuntimeError(f'nonfinite field or CFL >= 1: {cfl}')
        self.velocity=u;self.pressure=p;self.flux=q;self.time+=c.time_step
        row=dict(step=len(self.history)+1,time=self.time,continuity=div,projection_iterations=it,cfl=cfl,
                 mass_imbalance=float(q[self.boundary].sum()),momentum_ledger=float(ledger.abs().max()),
                 max_speed=float(torch.linalg.vector_norm(u,dim=1).max()),max_eddy_viscosity=float((nu-c.kinematic_viscosity).max()))
        self.history.append(row);self.forces.append(dict(time=self.time,Cd=float(wallforce[0]/force_scale),Cl=float(wallforce[1]/force_scale),Cz=float(wallforce[2]/force_scale)))
        return row
