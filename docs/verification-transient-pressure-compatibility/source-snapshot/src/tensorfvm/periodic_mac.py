"""Single-rank periodic MAC dual-control-volume momentum/energy prototype.

All momentum components are face unknowns. No center reconstruction overwrites
momentum. This bounded periodic research kernel is not a cylinder/FSI backend.
"""
from dataclasses import dataclass
import math
import torch
from .runtime import DistributedRuntime


@dataclass(frozen=True)
class PeriodicMACConfig:
    nx: int = 16
    ny: int = 16
    nz: int = 8
    lengths_m: tuple[float,float,float] = (2*math.pi,2*math.pi,2*math.pi)
    density_kg_m3: float = 1.
    viscosity_pa_s: float = .01
    time_step_s: float = .01
    nonlinear_tolerance: float = 1e-12
    nonlinear_iterations: int = 100
    device: str = 'cpu'

    def __post_init__(self):
        for name in ('nx','ny','nz','nonlinear_iterations'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<4:raise ValueError(f'invalid {name}')
        if len(self.lengths_m)!=3 or any(not math.isfinite(x) or x<=0 for x in self.lengths_m):raise ValueError('invalid periodic lengths')
        for name in ('density_kg_m3','viscosity_pa_s','time_step_s','nonlinear_tolerance'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(f'invalid {name}')


class PeriodicMACSolver:
    distributed_supported=False

    def __init__(self,config:PeriodicMACConfig,runtime=None):
        self.config=config;self.runtime=runtime or DistributedRuntime.discover(config.device)
        if self.runtime.world_size!=1:raise NotImplementedError('periodic MAC prototype supports single rank only')
        self.device=self.runtime.device
        self.shape=(config.nz,config.ny,config.nx)
        self.spacing=tuple(L/n for L,n in zip(config.lengths_m,(config.nx,config.ny,config.nz)))
        self.volume=math.prod(self.spacing)
        self.velocity=torch.zeros((3,*self.shape),dtype=torch.float64,device=self.device)
        self.pressure=torch.zeros(self.shape,dtype=torch.float64,device=self.device)
        self.dynamic_viscosity=torch.full_like(self.pressure,config.viscosity_pa_s)
        wave=[2*math.pi*torch.fft.fftfreq(n,d=1.,device=self.device,dtype=torch.float64)
              for n in (config.nx,config.ny,config.nz)]
        kz,ky,kx=torch.meshgrid(wave[2],wave[1],wave[0],indexing='ij')
        self.symbol=4*torch.sin(kx/2)**2/self.spacing[0]**2+4*torch.sin(ky/2)**2/self.spacing[1]**2+4*torch.sin(kz/2)**2/self.spacing[2]**2
        self.history=[];self.time=0.

    @staticmethod
    def next(field,axis):return torch.roll(field,-1,dims=-(axis+1))
    @staticmethod
    def previous(field,axis):return torch.roll(field,1,dims=-(axis+1))

    def centers(self):
        nz,ny,nx=self.shape;dx,dy,dz=self.spacing
        z,y,x=torch.meshgrid((torch.arange(nz,device=self.device,dtype=torch.float64)+.5)*dz,
                            (torch.arange(ny,device=self.device,dtype=torch.float64)+.5)*dy,
                            (torch.arange(nx,device=self.device,dtype=torch.float64)+.5)*dx,indexing='ij')
        return x,y,z

    def face_coordinates(self,axis):
        xyz=list(self.centers());xyz[axis]=xyz[axis]+self.spacing[axis]/2
        return tuple(xyz)

    def divergence(self,velocity):
        return sum((velocity[a]-self.previous(velocity[a],a))/self.spacing[a] for a in range(3))

    def gradient(self,pressure):
        return torch.stack([(self.next(pressure,a)-pressure)/self.spacing[a] for a in range(3)])

    def kinetic_energy(self,velocity):
        return .5*self.config.density_kg_m3*self.volume*velocity.square().sum()

    def momentum(self,velocity):
        return self.config.density_kg_m3*self.volume*velocity.sum((1,2,3))

    def project(self,tentative,dt=None):
        dt=self.config.time_step_s if dt is None else dt;rho=self.config.density_kg_m3
        rhs=-rho/dt*self.divergence(tentative);rhs=rhs-rhs.mean()
        rhs_hat=torch.fft.fftn(rhs)
        divisor=self.symbol.clone();divisor[0,0,0]=1
        phat=rhs_hat/divisor;phat[0,0,0]=0
        pressure=torch.fft.ifftn(phat).real
        corrected=tentative-dt/rho*self.gradient(pressure)
        true_residual=rhs+self.divergence(self.gradient(pressure))
        residual=float(torch.linalg.vector_norm(true_residual));target=1e-11*max(1.,float(torch.linalg.vector_norm(rhs)))
        record=dict(pressure_method='periodic-FFT-discrete-symbol',pressure_residual=residual,
                    pressure_target=target,pressure_converged=residual<=target,
                    tentative_energy_J=float(self.kinetic_energy(tentative)),projected_energy_J=float(self.kinetic_energy(corrected)),
                    momentum_change_Ns=(self.momentum(corrected)-self.momentum(tentative)).tolist(),
                    max_divergence_s_inv=float(self.divergence(corrected).abs().max()))
        return corrected,pressure,record

    def convective_fluxes(self,velocity):
        """Shared centered flux on each face of each component's dual CV."""
        rows=[]
        for a in range(3):
            flux=[]
            for b in range(3):
                normal=.5*(velocity[b]+self.next(velocity[b],a))
                transported=.5*(velocity[a]+self.next(velocity[a],b))
                flux.append(self.config.density_kg_m3*normal*transported)
            rows.append(torch.stack(flux))
        return torch.stack(rows)

    def convective_divergence(self,flux):
        return torch.stack([sum((flux[a,b]-self.previous(flux[a,b],b))/self.spacing[b]
                               for b in range(3)) for a in range(3)])

    def full_stress(self,velocity,mu=None):
        mu=self.dynamic_viscosity if mu is None else mu
        if mu.shape!=self.shape or not torch.isfinite(mu).all() or torch.any(mu<=0):raise ValueError('mu must be positive finite cell field')
        normal=torch.stack([2*mu*(velocity[a]-self.previous(velocity[a],a))/self.spacing[a] for a in range(3)])
        shear=torch.zeros((3,3,*self.shape),dtype=velocity.dtype,device=velocity.device)
        edge_mu=torch.zeros_like(shear)
        for a in range(3):
            for b in range(a+1,3):
                me=.25*(mu+self.next(mu,a)+self.next(mu,b)+self.next(self.next(mu,a),b))
                cross=(self.next(velocity[a],b)-velocity[a])/self.spacing[b]+(self.next(velocity[b],a)-velocity[b])/self.spacing[a]
                shear[a,b]=shear[b,a]=me*cross;edge_mu[a,b]=edge_mu[b,a]=me
        return normal,shear,edge_mu

    def stress_divergence(self,stress):
        normal,shear,_=stress
        return torch.stack([(self.next(normal[a],a)-normal[a])/self.spacing[a]
                            +sum((shear[a,b]-self.previous(shear[a,b],b))/self.spacing[b] for b in range(3) if b!=a)
                            for a in range(3)])

    def step(self):
        c=self.config;old=self.velocity.clone()
        if not torch.isfinite(old).all():raise ValueError('initial face velocity must be finite')
        if float(self.divergence(old).abs().max())>1e-9:raise ValueError('initial face velocity must be divergence-free')
        guess=old.clone();iterations=[];converged=False
        for iteration in range(1,c.nonlinear_iterations+1):
            midpoint=.5*(old+guess)
            conv=self.convective_fluxes(midpoint);stress=self.full_stress(midpoint)
            acceleration=(-self.convective_divergence(conv)+self.stress_divergence(stress))/c.density_kg_m3
            tentative=old+c.time_step_s*acceleration
            updated,pressure,projection=self.project(tentative)
            residual=float((updated-guess).abs().max())
            iterations.append(dict(iteration=iteration,fixed_point_residual_m_s=residual,projection=projection))
            guess=updated
            if residual<=c.nonlinear_tolerance:
                converged=True;break
        # Reconstruct the true nonlinear residual at the accepted candidate.
        midpoint=.5*(old+guess);conv=self.convective_fluxes(midpoint);stress=self.full_stress(midpoint)
        accel=(-self.convective_divergence(conv)+self.stress_divergence(stress))/c.density_kg_m3
        tentative=old+c.time_step_s*accel
        check,pressure,projection=self.project(tentative)
        nonlinear_residual=float((check-guess).abs().max())
        converged=converged and nonlinear_residual<=c.nonlinear_tolerance and projection['pressure_converged']
        k0=float(self.kinetic_energy(old));k1=float(self.kinetic_energy(guess))
        conv_power=float(-self.volume*(midpoint*self.convective_divergence(conv)).sum())
        visc_power=float(self.volume*(midpoint*self.stress_divergence(stress)).sum())
        pressure_power=float(-self.volume*(midpoint*self.gradient(pressure)).sum())
        energy_defect=k1-k0-c.time_step_s*(conv_power+visc_power+pressure_power)
        record=dict(step=len(self.history)+1,time_s=self.time+c.time_step_s,nonlinear_converged=converged,
            nonlinear_iterations=iteration,nonlinear_residual_m_s=nonlinear_residual,
            kinetic_before_J=k0,kinetic_after_J=k1,convective_power_W=conv_power,viscous_power_W=visc_power,
            pressure_power_W=pressure_power,
            energy_balance_defect_J=energy_defect,max_divergence_s_inv=float(self.divergence(guess).abs().max()),
            momentum_change_Ns=(self.momentum(guess)-self.momentum(old)).tolist(),projection=projection)
        self.last_raw=dict(old_velocity=old,candidate_velocity=guess,midpoint_velocity=midpoint,tentative_velocity=tentative,
            check_velocity=check,pressure=pressure,dynamic_viscosity=self.dynamic_viscosity.clone(),
            convective_fluxes=conv,normal_stress=stress[0],shear_stress=stress[1],edge_dynamic_viscosity=stress[2],
            iterations=iterations,record=record)
        if not converged:
            # Retain the candidate as failure evidence, without advancing state.
            self.last_raw['accepted']=False
            raise RuntimeError(f'MAC midpoint nonlinear solve failed: residual {nonlinear_residual:.3g}')
        self.last_raw['accepted']=True;self.velocity=guess;self.pressure=pressure;self.time+=c.time_step_s
        self.history.append(record);return record
