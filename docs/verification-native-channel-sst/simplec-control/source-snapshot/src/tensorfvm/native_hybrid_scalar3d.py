"""Conservative cell/face scalar transport for the native hybrid backend.

Solves d(phi)/dt + div(U phi) - div(kappa grad(phi)) + sink*phi = source.
Coefficients and shared volume flux are frozen during each solve. First-order
cell upwind is used for advection; diffusion uses the hybrid energy operator.
No scalar clipping or general discrete maximum principle is claimed.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import copy
from .native_scalar_mmp3d import NativeScalarMMP3D, ScalarMMPConvergenceError
from .native_scalar_mmp_newton3d import solve_mmp_newton
from .native_hybrid_viscous3d import NativeHybridViscous3D


class NativeHybridScalar3D:
    def __init__(self, mesh, diffusivity=1., *, dirichlet_mask=None, stabilization=1., correction='none', mmp_solver='continuation_newton'):
        self.mesh = mesh
        self.diffusion = NativeHybridViscous3D(mesh, diffusivity, components=1,
            dirichlet_mask=dirichlet_mask, stabilization=stabilization)
        if correction not in ('none','nonlinear_mmp'):raise ValueError('unknown scalar diffusion correction')
        if mmp_solver not in ('picard','continuation_newton'):raise ValueError('unknown scalar nonlinear solver')
        self.mmp_solver=mmp_solver
        self.mmp=NativeScalarMMP3D(self.diffusion) if correction=='nonlinear_mmp' else None
        self.N, self.F = self.diffusion.N, self.diffusion.F
        self.boundary = mesh.neighbor < 0
        ids = np.arange(self.F); interior = ~self.boundary
        self.D = sparse.coo_matrix((np.r_[np.ones(self.F), -np.ones(interior.sum())],
            (np.r_[mesh.owner, mesh.neighbor[interior]], np.r_[ids, ids[interior]])),
            shape=(self.N, self.F)).tocsr()

    @staticmethod
    def _array(value, size, name):
        answer = np.broadcast_to(np.asarray(value, float), (size,)).copy()
        if not np.isfinite(answer).all():
            raise ValueError(f'finite {name} required')
        return answer

    def gradient(self, face_values):
        values = self._array(face_values, self.F, 'face scalar')
        return np.column_stack([self.D @ (values*self.mesh.area_vectors[:, j])
                                for j in range(3)])/self.mesh.cell_volumes[:, None]

    def solve(self, volume_flux=0., *, source=0., sink=0., boundary_values=0.,
              boundary_diffusive_flux=0., inflow_values=None, old_values=None, dt=None):
        """One linear steady solve or backward Euler step, with an explicit ledger.

        boundary_diffusive_flux is outward kappa*grad(phi).n per unit area
        on Neumann boundaries. Inflow values are mandatory wherever q<0,
        including Neumann diffusion boundaries. sink must be nonnegative.
        Both dt and old_values must be supplied for a transient step.
        """
        if self.mmp is not None:
            return self._solve_mmp(volume_flux,source=source,sink=sink,boundary_values=boundary_values,
                boundary_diffusive_flux=boundary_diffusive_flux,inflow_values=inflow_values,old_values=old_values,dt=dt)
        q = self._array(volume_flux, self.F, 'shared volume flux')
        source = self._array(source, self.N, 'source')
        sink = self._array(sink, self.N, 'sink')
        if np.any(sink < 0):
            raise ValueError('nonnegative sink required')
        transient = dt is not None
        if transient != (old_values is not None):
            raise ValueError('dt and old_values must be supplied together')
        if transient and (not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0):
            raise ValueError('positive finite scalar dt required')
        old = self._array(old_values, self.N, 'old scalar') if transient else np.zeros(self.N)
        inverse_dt = 1./dt if transient else 0.
        bd = np.broadcast_to(np.asarray(boundary_values, float), (self.F,))
        gn = np.broadcast_to(np.asarray(boundary_diffusive_flux, float), (self.F,))
        incoming = self.boundary & (q < 0)
        if incoming.any() and inflow_values is None:
            raise ValueError('explicit scalar inflow/backflow values required')
        inflow = np.zeros(self.F) if inflow_values is None else np.broadcast_to(np.asarray(inflow_values, float), (self.F,))
        if not np.isfinite(inflow[incoming]).all():
            raise ValueError('finite incoming scalar required')
        active = ~incoming
        ids = np.flatnonzero(active)
        upstream = np.where(q >= 0, self.mesh.owner, self.mesh.neighbor)
        transport = sparse.coo_matrix((q[ids], (ids, upstream[ids])),
                                      shape=(self.F, self.N)).tocsr()
        face_source = np.zeros(self.F)
        face_source[incoming] = q[incoming]*inflow[incoming]
        adv_matrix = self.D @ transport
        volume = self.mesh.cell_volumes
        cell_matrix = adv_matrix+sparse.diags(volume*(sink+inverse_dt))
        extra = sparse.block_diag((cell_matrix, sparse.csr_matrix((self.F,self.F))), format='csr')
        operator = self.diffusion.full_matrix+extra
        state = self.diffusion._prescribed(bd[:,None])
        forcing = self.diffusion._traction_load(gn[:,None])
        forcing[:self.N] += volume*(source+old*inverse_dt)-self.D @ face_source
        free = self.diffusion.free
        rhs = (forcing-operator @ state)[free]
        matrix = operator[free][:,free].tocsc()
        answer = splu(matrix).solve(rhs)
        relative = float(np.linalg.norm(matrix @ answer-rhs)/max(np.linalg.norm(rhs),1e-30))
        if not np.isfinite(answer).all() or relative > 1e-9:
            raise RuntimeError(f'hybrid scalar true linear residual failed: {relative:g}')
        state[free] = answer
        return self._solution_result(state,q,source,sink,old,inverse_dt,inflow,relative)

    def _solution_result(self,state,q,source,sink,old,inverse_dt,inflow_values,linear_residual):
        volume=self.mesh.cell_volumes
        diffusion = self.diffusion.diagnostics(state)
        cells, faces = state[:self.N], state[self.N:]
        upstream=np.where(q>=0,self.mesh.owner,self.mesh.neighbor)
        advective_flux=q*cells[upstream.clip(min=0)]
        incoming=self.boundary & (q<0)
        if incoming.any():advective_flux[incoming]=q[incoming]*np.broadcast_to(np.asarray(inflow_values,float),(self.F,))[incoming]
        diffusive_flux = diffusion['face_flux'][:,0]
        total_flux = advective_flux-diffusive_flux
        storage = volume*inverse_dt*(cells-old)
        destruction = volume*sink*cells
        production = volume*source
        cell_balance = storage+self.D @ total_flux+destruction-production
        global_balance = float(storage.sum()+total_flux[self.boundary].sum()+destruction.sum()-production.sum())
        # Independently reconstructed neighbor-side diffusion, no sign reuse.
        neighbor_total = -advective_flux-diffusion['neighbor_face_flux'][:,0]
        interface = total_flux[~self.boundary]+neighbor_total[~self.boundary]
        scalar_work = float(cells @ (storage+self.D @ advective_flux+destruction-production))
        direct_work_balance = scalar_work+diffusion['energy_form']-diffusion['boundary_work']
        work_balance,cell_work,interface_work,cell_extended,face_extended=self._stable_work_balance(state,storage,advective_flux,destruction,production)
        scale = max(float(np.abs(storage).sum()+np.abs(total_flux[self.boundary]).sum()
                          +np.abs(destruction).sum()+np.abs(production).sum()),1e-30)
        return dict(cell_values=cells.copy(), face_values=faces.copy(),
            gradient=self.gradient(faces), advective_flux=advective_flux,
            diffusive_flux=diffusive_flux, total_flux=total_flux,
            neighbor_total_flux=neighbor_total, storage=storage,
            destruction=destruction, production=production, cell_balance=cell_balance,
            cell_balance_max=float(np.max(np.abs(cell_balance))),
            interface_flux_defect_max=float(np.max(np.abs(interface),initial=0.)),
            global_balance=global_balance, global_relative_balance=abs(global_balance)/scale,
            diffusion_energy=diffusion['energy_form'], boundary_diffusive_work=diffusion['boundary_work'],
            work_balance=float(work_balance),work_balance_direct_float64=float(direct_work_balance),
            work_balance_cell_terms=cell_work,work_balance_interface_terms=interface_work,
            cell_equation_extended=cell_extended,face_diffusion_reaction_extended=face_extended,
            work_balance_precision_bits=int(np.finfo(np.longdouble).nmant+1),
            linear_relative_residual=linear_residual,
            cell_min=float(cells.min()), face_min=float(faces.min()))

    def _stable_work_balance(self,state,storage,advective_flux,destruction,production):
        """Same work identity without subtracting large total energies.

        Constant-preserving local differences evaluate the intended zero-row-sum
        diffusion stencil. Extended accumulation retains small equilibrium
        defects when omega boundary work is many orders larger than the defect.
        The direct float64 total-energy subtraction is separately retained.
        """
        dtype=np.longdouble;u=np.asarray(state,dtype=dtype)
        cell=np.zeros(self.N,dtype=dtype);owner=np.zeros(self.F,dtype=dtype);neighbor=owner.copy()
        for c,(faces,sign,ids,matrix,*_) in enumerate(self.diffusion.local):
            local=matrix.astype(dtype);np.fill_diagonal(local,0.)
            phi=u[ids];reaction=np.sum(local*(phi[None,:]-phi[:,None]),axis=1)
            cell[c]=reaction[0]
            owner[faces[sign>0]]=reaction[1:][sign>0];neighbor[faces[sign<0]]=reaction[1:][sign<0]
        remainder=np.asarray(storage,dtype=dtype)+np.asarray(destruction,dtype=dtype)-np.asarray(production,dtype=dtype)
        internal=~self.boundary
        np.add.at(remainder,self.mesh.owner,np.asarray(advective_flux,dtype=dtype))
        np.add.at(remainder,self.mesh.neighbor[internal],-np.asarray(advective_flux[internal],dtype=dtype))
        cell_work=u[:self.N]*(remainder+cell)
        interface_work=u[self.N:][internal]*(owner[internal]+neighbor[internal])
        balance=np.sum(cell_work,dtype=dtype)+np.sum(interface_work,dtype=dtype)
        return (float(balance),np.asarray(cell_work,float),np.asarray(interface_work,float),
                remainder+cell,owner+neighbor)

    def steady_residual(self, cell_values, face_values, volume_flux=0., *, source=0.,
                        sink=0., boundary_values=0., boundary_diffusive_flux=0., inflow_values=None):
        """Evaluate CURRENT equations, including unequilibrated face unknowns.

        No time term, frozen solve or state repair is used. All cell equations
        and free face equations enter the normalized original-system residual.
        """
        if self.mmp is not None:
            gn=np.broadcast_to(np.asarray(boundary_diffusive_flux,float),(self.F,))
            if np.any(gn[self.diffusion.traction_boundary]!=0):
                raise ValueError('nonlinear scalar correction currently requires homogeneous Neumann data')
            op=copy.copy(self);op.mmp=None
            cells=self._array(cell_values,self.N,'cell scalar');faces=self._array(face_values,self.F,'face scalar')
            op.diffusion,info=self.mmp.operator(np.r_[cells,faces])
            result=op.steady_residual(cell_values,face_values,volume_flux,source=source,sink=sink,
                boundary_values=boundary_values,boundary_diffusive_flux=boundary_diffusive_flux,inflow_values=inflow_values)
            result.update(info);return result
        cells=self._array(cell_values,self.N,'cell scalar')
        faces=self._array(face_values,self.F,'face scalar')
        q=self._array(volume_flux,self.F,'volume flux')
        src=self._array(source,self.N,'source');snk=self._array(sink,self.N,'sink')
        bd=np.broadcast_to(np.asarray(boundary_values,float),(self.F,))
        gn=np.broadcast_to(np.asarray(boundary_diffusive_flux,float),(self.F,))
        incoming=self.boundary & (q<0)
        if incoming.any() and inflow_values is None:
            raise ValueError('explicit scalar inflow/backflow values required')
        inflow=np.zeros(self.F) if inflow_values is None else np.broadcast_to(np.asarray(inflow_values,float),(self.F,))
        if not np.isfinite(inflow[incoming]).all():raise ValueError('finite incoming scalar required')
        upstream=np.where(q>=0,self.mesh.owner,self.mesh.neighbor)
        adv=q*cells[upstream.clip(min=0)];adv[incoming]=q[incoming]*inflow[incoming]
        state=np.r_[cells,faces];diff=self.diffusion.diagnostics(state)
        reaction=self.diffusion.full_matrix @ state
        load=self.diffusion._traction_load(gn[:,None])
        production=self.mesh.cell_volumes*src;destruction=self.mesh.cell_volumes*snk*cells
        adv_div=self.D @ adv
        residual=reaction-load
        residual[:self.N]+=adv_div+destruction-production
        free=self.diffusion.free
        prescribed=self.diffusion._prescribed(bd[:,None])
        prescribed_load=self.diffusion.full_matrix @ prescribed
        scale=max(np.linalg.norm(prescribed_load[free])+np.linalg.norm(load[free])+
                  np.linalg.norm(production)+np.linalg.norm(destruction)+np.linalg.norm(adv_div),1e-30)
        mask=self.diffusion.dirichlet
        bc_error=float(np.max(np.abs(faces[mask]-bd[mask]),initial=0.))
        total=adv-diff['face_flux'][:,0]
        # Shared owner flux ledger is separate: it alone misses free-face residuals.
        local=self.D @ total+destruction-production
        return dict(relative_residual=float(np.linalg.norm(residual[free])/scale),
            residual_scale=float(scale),free_residual=residual[free],
            cell_equation=residual[:self.N],face_equation=residual[self.N:],
            dirichlet_error=bc_error,cell_equation_max=float(np.max(np.abs(residual[:self.N]))),
            free_face_equation_max=float(np.max(np.abs(residual[self.N:][~mask]),initial=0.)),
            interface_flux_defect_max=diff['interface_flux_defect_max'],
            total_flux=total,local_shared_flux_balance=local,
            global_balance=float(total[self.boundary].sum()+destruction.sum()-production.sum()))

    def _solve_mmp(self,volume_flux,**kwargs):
        gn=np.broadcast_to(np.asarray(kwargs['boundary_diffusive_flux'],float),(self.F,))
        if np.any(gn[self.diffusion.traction_boundary]!=0):
            raise ValueError('nonlinear scalar correction currently requires homogeneous Neumann data')
        if self.mmp_solver=='continuation_newton':return solve_mmp_newton(self,volume_flux,**kwargs)
        linear=copy.copy(self);linear.mmp=None
        initial=linear.solve(volume_flux,**kwargs)
        state=np.r_[initial['cell_values'],initial['face_values']]
        q=self._array(volume_flux,self.F,'volume flux')
        source=self._array(kwargs['source'],self.N,'source');sink=self._array(kwargs['sink'],self.N,'sink')
        dt=kwargs['dt'];invdt=0. if dt is None else 1./dt
        old=np.zeros(self.N) if dt is None else self._array(kwargs['old_values'],self.N,'old scalar')
        residual_kwargs={key:value for key,value in kwargs.items() if key not in ('old_values','dt')}
        history=[]
        for iteration in range(600):
            linear.diffusion,_=self.mmp.operator(state)
            solved=linear.solve(q,**kwargs)
            candidate=np.r_[solved['cell_values'],solved['face_values']]
            change=float(np.linalg.norm(candidate-state)/max(np.linalg.norm(candidate),1e-30))
            state=.5*(state+candidate)
            current,info=self.mmp.operator(state);linear.diffusion=current
            audit=linear.steady_residual(state[:self.N],state[self.N:],q,**residual_kwargs)
            residual=audit['free_residual'].copy()
            residual[:self.N]+=self.mesh.cell_volumes*invdt*(state[:self.N]-old)
            scale=audit['residual_scale']+np.linalg.norm(self.mesh.cell_volumes*invdt*old)
            error=float(np.linalg.norm(residual)/max(scale,1e-30))
            history.append(dict(iteration=iteration+1,current_nonlinear_residual=error,change=change))
            if max(error,change)<1e-12:
                result=linear._solution_result(state,q,source,sink,old,invdt,kwargs['inflow_values'],solved['linear_relative_residual'])
                ledger=max(result['cell_balance_max'],result['interface_flux_defect_max'],abs(result['global_balance']),abs(result['work_balance']))
                if ledger>1e-8:continue
                result.update(info,nonlinear_history=history,current_nonlinear_residual=error,
                              scalar_diffusion_correction='nonlinear_mmp')
                return result
        result=linear._solution_result(state,q,source,sink,old,invdt,kwargs['inflow_values'],solved['linear_relative_residual'])
        result.update(info,nonlinear_history=history,current_nonlinear_residual=error,converged=False)
        raise ScalarMMPConvergenceError(result)
