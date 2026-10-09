"""Opt-in conservative cell momentum with compatible face mass projection.

This is a hybrid method, not a staggered MAC momentum solver or a validated LES.
The published solver3d baseline and its source-bound evidence remain unchanged.
"""
import torch
from .solver3d import Cylinder3DSolver


class ConservativeCylinder3DSolver(Cylinder3DSolver):
    def _pairs(self, field, axis):
        if axis == 0:
            return (torch.cat((field[..., :1, :], field), -2),
                    torch.cat((field, field[..., -1:, :]), -2))
        if axis == 1:
            return (torch.cat((field[:, :1], field), 1),
                    torch.cat((field, field[:, -1:]), 1))
        following, _ = self._z_neighbors(field)
        return field, following

    def _fluid_pairs(self, axis):
        f=self.mesh.fluid[...,None]
        left,right=self._pairs(f,axis)
        if axis==0:
            left=left.clone();right=right.clone();left[...,0,:]=False;right[...,-1,:]=False
        if axis==1:
            left=left.clone();right=right.clone();left[:,0]=False;right[:,-1]=False
        return left[...,0],right[...,0]

    def _face_values(self, field, physical_velocity=False):
        """Shared arithmetic values; wall values zero for velocity, owner for pressure."""
        result=[]
        for axis in range(3):
            left,right=self._pairs(field,axis);fl,fr=self._fluid_pairs(axis)
            value=.5*(left+right)
            if physical_velocity:
                value=torch.where((fl&fr)[...,None],value,torch.zeros_like(value))
            else:
                value=torch.where(fl[...,None],left,right)
                value=torch.where((fl&fr)[...,None],.5*(left+right),value)
            if axis==0:
                value=value.clone()
                if physical_velocity:
                    value[...,0,:]=0;value[...,0,0]=self.config.inlet_velocity
                    value[...,-1,:]=left[...,-1,:]  # zero normal gradient outlet
                else:
                    value[...,-1,:]=0  # physical zero-pressure outlet
            if axis==1:
                value=value.clone()
                if physical_velocity:
                    value[:,0]=0;value[:,0,:,0]=self.config.inlet_velocity
                    value[:,-1]=0;value[:,-1,:,0]=self.config.inlet_velocity
            result.append(value)
        return tuple(result)

    def vector_divergence(self, fluxes):
        x,y,z=fluxes;_,previous=self._z_neighbors(z)
        return ((x[...,1:,:]-x[...,:-1,:])/self.mesh.dx
                +(y[:,1:]-y[:,:-1])/self.mesh.dy
                +(z-previous)/self.mesh.dz).masked_fill(~self.mesh.fluid[...,None],0)

    def _cell_gradient(self, velocity):
        x,y,z=self._face_values(velocity,physical_velocity=True)
        _,previous=self._z_neighbors(z)
        return torch.stack(((x[...,1:,:]-x[...,:-1,:])/self.mesh.dx,
                            (y[:,1:]-y[:,:-1])/self.mesh.dy,
                            (z-previous)/self.mesh.dz),-1)

    def stress_fluxes(self, velocity, dynamic_viscosity):
        """Full symmetric Newtonian stress columns on shared Cartesian faces.

        The independent viscosity argument supports variable-mu manufactured
        solutions. Physical solid faces use molecular mu (SGS wall value zero).
        """
        grad=self._cell_gradient(velocity);fluxes=[];mus=[];gradients=[]
        for axis,h in enumerate((self.mesh.dx,self.mesh.dy,self.mesh.dz)):
            left,right=self._pairs(velocity,axis);fl,fr=self._fluid_pairs(axis)
            # Flatten component/derivative axes while pairing cells.
            gl,gr=self._pairs(grad.flatten(-2),axis)
            g=.5*(gl+gr);g=g.reshape(*g.shape[:-1],3,3)
            normal=(right-left)/h
            wall=fl^fr
            normal=torch.where((fl&~fr)[...,None],-2*left/h,normal)
            normal=torch.where((~fl&fr)[...,None],2*right/h,normal)
            g=torch.where((fl&fr)[...,None,None],g,torch.zeros_like(g))
            # At the stationary wall velocity is constant along its surface.
            g[...,axis,:]=torch.where(wall[...,None],torch.zeros_like(g[...,axis,:]),g[...,axis,:])
            if axis==0:
                target=torch.zeros_like(left[...,0,:]);target[...,0]=self.config.inlet_velocity
                normal[...,0,:]=2*(right[...,0,:]-target)/h
                normal[...,-1,:]=0
                g[...,0,:,:]=0;g[...,-1,:,:]=gr.reshape(*gr.shape[:-1],3,3)[...,-1,:,:]
            if axis==1:
                target=torch.zeros_like(left[:,0]);target[...,0]=self.config.inlet_velocity
                normal[:,0]=2*(right[:,0]-target)/h
                normal[:,-1]=2*(target-left[:,-1])/h
                g[:,0]=0;g[:,-1]=0
            g[..., :,axis]=normal
            ml,mr=self._pairs(dynamic_viscosity[...,None],axis)
            mu=.5*(ml[...,0]+mr[...,0])
            internal_wall=wall.clone()
            if axis==0:internal_wall[...,0]=False;internal_wall[...,-1]=False
            if axis==1:internal_wall[:,0]=False;internal_wall[:,-1]=False
            mu=torch.where(internal_wall,self.config.density*self.config.kinematic_viscosity,mu)
            tau=mu[...,None]*(g[..., :,axis]+g[...,axis,:])
            tau=tau.masked_fill((~(fl|fr))[...,None],0)
            fluxes.append(tau);mus.append(mu);gradients.append(g)
        self.last_face_dynamic_viscosity=tuple(mus)
        self.last_face_gradient=tuple(gradients)
        return tuple(fluxes)

    def convective_fluxes(self, velocity, faces):
        flux=[]
        for axis,q in enumerate(faces):
            left,right=self._pairs(velocity,axis)
            if axis==0:
                left=left.clone();right=right.clone();left[...,0,:]=0;left[...,0,0]=self.config.inlet_velocity
                right[...,-1,:]=0;right[...,-1,0]=self.config.inlet_velocity
            if axis==1:
                left=left.clone();right=right.clone();left[:,0]=0;left[:,0,:,0]=self.config.inlet_velocity
                right[:,-1]=0;right[:,-1,:,0]=self.config.inlet_velocity
            upwind=torch.where((q>=0)[...,None],left,right)
            flux.append(self.config.density*q[...,None]*upwind)
        return tuple(flux)

    def pressure_fluxes(self, pressure):
        values=self._face_values(pressure[...,None]);fluxes=[]
        for axis,p in enumerate(values):
            f=torch.zeros((*p.shape[:-1],3),dtype=p.dtype,device=p.device)
            f[...,axis]=p[...,0];fluxes.append(f)
        return tuple(fluxes)

    def external_flux_integral(self, fluxes):
        """Outward physical-domain flux; periodic z contributions cancel globally."""
        x,y,_=fluxes
        local=((x[...,-1,:]-x[...,0,:]).sum((0,1))*self.mesh.dy*self.mesh.dz
               +(y[:,-1]-y[:,0]).sum((0,1))*self.mesh.dx*self.mesh.dz)
        return self.runtime.global_sum(local)

    def _global_vector_integral(self, field):
        return self.runtime.global_sum(field.sum(dim=(0,1,2))*self.mesh.cell_volume)

    def wall_wrench(self, pressure_flux, viscous_flux):
        """Force on body: p*n_fluid - tau*n_fluid, with actual staircase facets."""
        c,m=self.config,self.mesh;records=[];force=torch.zeros(3,dtype=self.velocity.dtype,device=self.velocity.device)
        moment=force.clone()
        for axis,area in enumerate((m.dy*m.dz,m.dx*m.dz,m.dx*m.dy)):
            fl,fr=self._fluid_pairs(axis);sign=fl.to(force.dtype)-fr.to(force.dtype)
            mask=fl^fr
            if axis==0:mask=mask.clone();mask[...,0]=False;mask[...,-1]=False
            if axis==1:mask=mask.clone();mask[:,0]=False;mask[:,-1]=False
            shape=fl.shape
            kk,jj,ii=torch.meshgrid(torch.arange(shape[0],device=force.device),torch.arange(shape[1],device=force.device),torch.arange(shape[2],device=force.device),indexing='ij')
            position=torch.stack(((ii+(0 if axis==0 else .5))*m.dx,
                                  (jj+(0 if axis==1 else .5))*m.dy,
                                  (kk+self.partition.start+(1 if axis==2 else .5))*m.dz),-1).to(force.dtype)
            normal=torch.zeros_like(position);normal[...,axis]=sign
            fp=pressure_flux[axis]*sign[...,None]*area
            fv=-viscous_flux[axis]*sign[...,None]*area
            total=(fp+fv)[mask];points=position[mask]
            force+=total.sum(0);moment+=torch.linalg.cross(points,total).sum(0)
            records.append(dict(axis=axis,points_m=points,normal_fluid=normal[mask],area_m2=area,
                                pressure_pa=pressure_flux[axis][...,axis][mask],
                                viscous_traction_on_body_pa=(-viscous_flux[axis]*sign[...,None])[mask],
                                pressure_force_on_body_N=fp[mask],viscous_force_on_body_N=fv[mask],
                                total_force_on_body_N=total))
        return self.runtime.global_sum(force),self.runtime.global_sum(moment),records

    @torch.no_grad()
    def step(self):
        c,m=self.config,self.mesh;old=self.velocity.clone();p0=self._global_vector_integral(c.density*old)
        advecting_faces=tuple(u.clone() for u in self.face_velocity)
        mu=c.density*(c.kinematic_viscosity+self._eddy_viscosity(old))
        conv=self.convective_fluxes(old,self.face_velocity);visc=self.stress_fluxes(old,mu)
        conv_force=-self._global_vector_integral(self.vector_divergence(conv))
        visc_force=self._global_vector_integral(self.vector_divergence(visc))
        tentative=old+c.time_step/c.density*(-self.vector_divergence(conv)+self.vector_divergence(visc))
        predicted=self._predict_faces(tentative)
        info=self._pressure_projection(c.density/c.time_step*self._face_divergence(predicted))
        gradient=self._face_gradient(self.pressure)
        self.last_tentative_faces=tuple(u.clone() for u in predicted)
        self.face_velocity=tuple(u-c.time_step/c.density*g for u,g in zip(predicted,gradient))
        pf=self.pressure_fluxes(self.pressure)
        pressure_force=-self._global_vector_integral(self.vector_divergence(pf))
        self.velocity=tentative-c.time_step/c.density*self.vector_divergence(pf)
        self.velocity.masked_fill_(~m.fluid[...,None],0)
        p1=self._global_vector_integral(c.density*self.velocity)
        residual=p1-p0-c.time_step*(conv_force+visc_force+pressure_force)
        reconstructed=self._reconstruct_velocity(self.face_velocity)
        reconstruction_defect=self._global_vector_integral(c.density*(reconstructed-self.velocity))
        body_force,body_moment,wall=self.wall_wrench(pf,visc)
        wall_pressure=self.runtime.global_sum(sum((r['pressure_force_on_body_N'].sum(0) for r in wall),torch.zeros_like(body_force)))
        wall_viscous=self.runtime.global_sum(sum((r['viscous_force_on_body_N'].sum(0) for r in wall),torch.zeros_like(body_force)))
        exterior_conv=-self.external_flux_integral(conv)
        exterior_visc=self.external_flux_integral(visc)
        exterior_pressure=-self.external_flux_integral(pf)
        boundary_residual=p1-p0-c.time_step*(exterior_conv+exterior_visc+exterior_pressure-body_force)
        self.last_transport=dict(old_velocity=old,tentative_velocity=tentative,dynamic_viscosity=mu,
            advecting_face_velocity=advecting_faces,
            convective_flux=conv,viscous_flux=visc,pressure_flux=pf,wall_facets=wall,
            momentum_before=p0,momentum_after=p1,convective_force_on_fluid=conv_force,
            viscous_force_on_fluid=visc_force,pressure_force_on_fluid=pressure_force,
            momentum_residual=residual,hypothetical_reconstruction_defect=reconstruction_defect,
            body_force=body_force,body_moment=body_moment,
            exterior_convective_force_on_fluid=exterior_conv,
            exterior_viscous_force_on_fluid=exterior_visc,
            exterior_pressure_force_on_fluid=exterior_pressure,
            wall_pressure_force_on_body=wall_pressure,wall_viscous_force_on_body=wall_viscous,
            boundary_momentum_residual=boundary_residual)
        self.time+=c.time_step
        div=self.runtime.global_max(self._face_divergence(self.face_velocity).abs().max())
        cfl=self.runtime.global_max((c.time_step*(self.velocity[...,0].abs()/m.dx+self.velocity[...,1].abs()/m.dy+self.velocity[...,2].abs()/m.dz)).max())
        imbalance=abs(self.runtime.global_sum((self.face_velocity[0][...,-1]-self.face_velocity[0][...,0]).sum()*m.dy*m.dz))
        metric=dict(step=len(self.history)+1,time=self.time,continuity=float(div)*c.diameter/c.inlet_velocity,
            boundary_mass_imbalance=float(imbalance)/(c.inlet_velocity*c.height*c.span),cfl=float(cfl),
            center_velocity_divergence=float(self.runtime.global_max(self._divergence(self.velocity).abs().max()))*c.diameter/c.inlet_velocity,
            pressure_iterations=info.iterations,pressure_initial_residual=info.initial_residual,
            pressure_residual=info.final_residual,pressure_target_residual=info.target_residual,
            pressure_converged=int(info.converged),momentum_ledger_residual_Ns=float(residual.abs().max()),
            hypothetical_reconstruction_defect_Ns=float(reconstruction_defect.abs().max()),
            boundary_momentum_residual_Ns=float(boundary_residual.abs().max()))
        if not all(torch.isfinite(torch.tensor(v)) for v in metric.values()):raise RuntimeError('nonfinite conservative step')
        self.history.append(metric)
        scale=.5*c.density*c.inlet_velocity**2*c.diameter*c.span
        self.force_history.append(dict(step=metric['step'],time=self.time,drag=float(body_force[0])/scale,lift=float(body_force[1])/scale))
        return metric
