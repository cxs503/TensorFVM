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
    def __init__(self, mesh, diffusivity=1., *, dirichlet_mask=None, stabilization=1., correction='none', mmp_solver='continuation_newton', work_balance_policy='absolute', cell_constraints=None):
        self.mesh = mesh
        self.diffusion = NativeHybridViscous3D(mesh, diffusivity, components=1,
            dirichlet_mask=dirichlet_mask, stabilization=stabilization)
        if correction not in ('none','nonlinear_mmp'):raise ValueError('unknown scalar diffusion correction')
        if mmp_solver not in ('picard','continuation_newton'):raise ValueError('unknown scalar nonlinear solver')
        if work_balance_policy not in ('absolute','relative'):raise ValueError('unknown scalar work balance policy')
        self.work_balance_policy=work_balance_policy
        self.mmp_solver=mmp_solver
        self.N, self.F = self.diffusion.N, self.diffusion.F
        self.constraint_cells=np.empty(0,dtype=int);self.constraint_values=np.empty(0)
        if cell_constraints is not None:
            cells,values=cell_constraints;cells=np.asarray(cells)
            if (cells.ndim!=1 or cells.dtype.kind not in 'iu' or np.any(cells<0)
                    or np.any(cells>=self.N) or len(np.unique(cells))!=len(cells)):
                raise ValueError('distinct valid integer constrained cell indices required')
            self.constraint_cells=cells.astype(int,copy=True)
            self.constraint_values=self._array(values,len(cells),'cell constraint values')
        self.free=np.setdiff1d(self.diffusion.free,self.constraint_cells)
        self.free_cells=self.free[self.free<self.N]
        self.mmp=NativeScalarMMP3D(self.diffusion,free_dofs=self.free) if correction=='nonlinear_mmp' else None
        self.boundary = mesh.neighbor < 0
        ids = np.arange(self.F); interior = ~self.boundary
        self.D = sparse.coo_matrix((np.r_[np.ones(self.F), -np.ones(interior.sum())],
            (np.r_[mesh.owner, mesh.neighbor[interior]], np.r_[ids, ids[interior]])),
            shape=(self.N, self.F)).tocsr()

    @staticmethod
    def _array(value, size, name, *, preserve_precision=False):
        value=np.asarray(value) if preserve_precision else np.asarray(value,float)
        if value.dtype.kind!='f':value=value.astype(float)
        answer = np.broadcast_to(value, (size,)).copy()
        if not np.isfinite(answer).all():
            raise ValueError(f'finite {name} required')
        return answer

    def gradient(self, face_values):
        values = self._array(face_values, self.F, 'face scalar')
        return np.column_stack([self.D @ (values*self.mesh.area_vectors[:, j])
                                for j in range(3)])/self.mesh.cell_volumes[:, None]

    def _prescribed(self,boundary_values):
        state=self.diffusion._prescribed(boundary_values)
        state[self.constraint_cells]=self.constraint_values
        return state

    def _constraint_reaction(self,cell_equation):
        reaction=np.zeros_like(cell_equation)
        reaction[self.constraint_cells]=cell_equation[self.constraint_cells]
        return reaction

    def _constraint_error(self,cells):
        return float(np.max(abs(cells[self.constraint_cells]-self.constraint_values),initial=0.))

    def solve(self, volume_flux=0., *, source=0., sink=0., boundary_values=0.,
              boundary_diffusive_flux=0., inflow_values=None, old_values=None, dt=None):
        """One linear steady solve or backward Euler step, with an explicit ledger.

        boundary_diffusive_flux is outward kappa*grad(phi).n per unit area
        on Neumann boundaries. Inflow values are mandatory wherever q<0,
        including Neumann diffusion boundaries. sink must be nonnegative.
        Both dt and old_values must be supplied for a transient step.
        Constructor cell_constraints=(indices,values) eliminate specified cell
        unknowns. Their original-equation reactions are explicit source terms
        in the returned local/global flux and work ledgers.
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
        state = self._prescribed(bd[:,None])
        forcing = self.diffusion._traction_load(gn[:,None])
        forcing[:self.N] += volume*(source+old*inverse_dt)-self.D @ face_source
        free = self.free
        rhs = (forcing-operator @ state)[free]
        matrix = operator[free][:,free].tocsc()
        answer = splu(matrix).solve(rhs) if len(free) else np.empty(0)
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
        cell_reaction,owner_reaction,neighbor_reaction=self._local_diffusion_reactions(state)
        diffusive_flux = owner_reaction
        total_flux = advective_flux-diffusive_flux
        storage = volume*inverse_dt*(cells-old)
        destruction = volume*sink*cells
        production = volume*source
        raw_cell_balance = storage+self.D @ total_flux+destruction-production
        constraint=self._constraint_reaction(self._cell_equation(storage,advective_flux,destruction,production,cell_reaction))
        cell_balance=raw_cell_balance-constraint
        raw_global=storage.sum()+total_flux[self.boundary].sum()+destruction.sum()-production.sum(dtype=np.longdouble)
        global_balance = float(raw_global-constraint.sum(dtype=np.longdouble))
        # Independently reconstructed neighbor-side diffusion, no sign reuse.
        neighbor_total = -advective_flux-neighbor_reaction
        interface = total_flux[~self.boundary]+neighbor_total[~self.boundary]
        scalar_work = float(cells @ (storage+self.D @ advective_flux+destruction-production))
        constraint_work=float(np.sum(cells*constraint,dtype=np.longdouble))
        direct_work_balance = scalar_work+diffusion['energy_form']-diffusion['boundary_work']-constraint_work
        work_balance,cell_work,interface_work,cell_extended,face_extended,work_scale=self._stable_work_balance(state,storage,advective_flux,destruction,production,
            reactions=(cell_reaction,owner_reaction,neighbor_reaction),constraint=constraint)
        scale = max(float(np.abs(storage).sum()+np.abs(total_flux[self.boundary]).sum()
                          +np.abs(destruction).sum()+np.abs(production).sum()+np.abs(constraint).sum()),1e-30)
        return dict(cell_values=cells.copy(), face_values=faces.copy(),
            gradient=self.gradient(faces), advective_flux=advective_flux,
            diffusive_flux=diffusive_flux, total_flux=total_flux,
            neighbor_total_flux=neighbor_total, storage=storage,
            destruction=destruction, production=production, cell_balance=cell_balance,
            cell_balance_without_constraint=raw_cell_balance,cell_constraint_reaction=constraint,
            constrained_cells=self.constraint_cells.copy(),prescribed_cell_values=self.constraint_values.copy(),
            cell_constraint_work=constraint_work,cell_constraint_error=self._constraint_error(cells),
            global_balance_without_constraint=float(raw_global),
            cell_balance_max=float(np.max(np.abs(cell_balance))),
            interface_flux_defect_max=float(np.max(np.abs(interface),initial=0.)),
            global_balance=global_balance, global_relative_balance=abs(global_balance)/scale,
            diffusion_energy=diffusion['energy_form'], boundary_diffusive_work=diffusion['boundary_work'],
            work_balance=float(work_balance),work_balance_direct_float64=float(direct_work_balance),
            work_balance_scale=work_scale,relative_work_balance=abs(float(work_balance))/max(work_scale,np.finfo(float).tiny),
            absolute_work_gate_passed=bool(abs(work_balance)<1e-8),work_balance_policy=self.work_balance_policy,
            work_balance_cell_terms=cell_work,work_balance_interface_terms=interface_work,
            cell_equation_extended=cell_extended,face_diffusion_reaction_extended=face_extended,
            work_balance_precision_bits=int(np.finfo(np.longdouble).nmant+1),
            local_flux_precision_bits=int(np.finfo(np.longdouble).nmant+1),
            linear_relative_residual=linear_residual,
            cell_min=float(cells.min()), face_min=float(faces.min()))

    def _local_diffusion_reactions(self,state):
        """Preserve the constant nullspace and the scalar state's precision.

        The assembled float64 matrix remains the sparse solve approximation;
        authoritative local fluxes use the same coefficients as the nonlinear
        residual, without downcasting the converged extended state.
        """
        dtype=np.longdouble;u=np.asarray(state,dtype=dtype)
        cell=np.zeros(self.N,dtype=dtype);owner=np.zeros(self.F,dtype=dtype);neighbor=owner.copy()
        for c,(faces,sign,ids,matrix,*_) in enumerate(self.diffusion.local):
            local=matrix.astype(dtype);np.fill_diagonal(local,0.)
            phi=u[ids];reaction=np.sum(local*(phi[None,:]-phi[:,None]),axis=1)
            cell[c]=reaction[0]
            owner[faces[sign>0]]=reaction[1:][sign>0];neighbor[faces[sign<0]]=reaction[1:][sign<0]
        return cell,owner,neighbor

    def _cell_equation(self,storage,advective_flux,destruction,production,cell):
        dtype=np.longdouble
        remainder=np.asarray(storage,dtype=dtype)+np.asarray(destruction,dtype=dtype)-np.asarray(production,dtype=dtype)
        internal=~self.boundary
        np.add.at(remainder,self.mesh.owner,np.asarray(advective_flux,dtype=dtype))
        np.add.at(remainder,self.mesh.neighbor[internal],-np.asarray(advective_flux[internal],dtype=dtype))
        return remainder+cell

    def _stable_work_balance(self,state,storage,advective_flux,destruction,production,*,reactions=None,constraint=None):
        """Local work identity with the same precise fluxes as the mass ledger."""
        dtype=np.longdouble;u=np.asarray(state,dtype=dtype);internal=~self.boundary
        cell,owner,neighbor=self._local_diffusion_reactions(state) if reactions is None else reactions
        equation=self._cell_equation(storage,advective_flux,destruction,production,cell)
        if constraint is None:constraint=self._constraint_reaction(equation)
        cell_work=u[:self.N]*(equation-constraint)
        interface_work=u[self.N:][internal]*(owner[internal]+neighbor[internal])
        balance=np.sum(cell_work,dtype=dtype)+np.sum(interface_work,dtype=dtype)
        # This scale consists only of gross terms in the same work identity.
        # Under phi -> a*phi (and source/old/boundary -> a*...), both the work
        # defect and scale transform as a^2. No acceptance policy is changed.
        diffusion_scale=np.longdouble(0.)
        for _,_,ids,matrix,*_ in self.diffusion.local:
            i,j=np.triu_indices(len(ids),1)
            diffusion_scale+=np.sum(abs(matrix[i,j].astype(dtype)*(u[ids[i]]-u[ids[j]])**2),dtype=dtype)
        advection=self.D@np.asarray(advective_flux,dtype=dtype)
        scale=diffusion_scale+np.sum(abs(u[self.N:][self.boundary]*owner[self.boundary]),dtype=dtype)
        for term in (storage,advection,destruction,production):
            scale+=np.sum(abs(u[:self.N]*np.asarray(term,dtype=dtype)),dtype=dtype)
        scale+=np.sum(abs(u[:self.N]*constraint),dtype=dtype)
        return (float(balance),np.asarray(cell_work,float),np.asarray(interface_work,float),
                equation,owner+neighbor,float(scale))

    def steady_residual(self, cell_values, face_values, volume_flux=0., *, source=0.,
                        sink=0., boundary_values=0., boundary_diffusive_flux=0., inflow_values=None):
        """Evaluate CURRENT equations, including unequilibrated face unknowns.

        No time term, frozen solve or state repair is used. Unconstrained cell
        and free face equations enter the normalized original-system residual.
        Prescribed cell errors and original constraint reactions are separate;
        a convergence check must include cell_constraint_error as well.
        """
        if self.mmp is not None:
            gn=np.broadcast_to(np.asarray(boundary_diffusive_flux,float),(self.F,))
            if np.any(gn[self.diffusion.traction_boundary]!=0):
                raise ValueError('nonlinear scalar correction currently requires homogeneous Neumann data')
            op=copy.copy(self);op.mmp=None
            cells=self._array(cell_values,self.N,'cell scalar',preserve_precision=True);faces=self._array(face_values,self.F,'face scalar',preserve_precision=True)
            op.diffusion,info=self.mmp.operator(np.r_[cells,faces])
            result=op.steady_residual(cell_values,face_values,volume_flux,source=source,sink=sink,
                boundary_values=boundary_values,boundary_diffusive_flux=boundary_diffusive_flux,inflow_values=inflow_values)
            result.update(info);return result
        cells=self._array(cell_values,self.N,'cell scalar',preserve_precision=True)
        faces=self._array(face_values,self.F,'face scalar',preserve_precision=True)
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
        cell_reaction,owner_reaction,neighbor_reaction=self._local_diffusion_reactions(state)
        reaction=np.r_[cell_reaction,owner_reaction+neighbor_reaction]
        load=self.diffusion._traction_load(gn[:,None])
        production=self.mesh.cell_volumes*src;destruction=self.mesh.cell_volumes*snk*cells
        adv_div=self.D @ adv
        residual=reaction-load
        residual[:self.N]+=adv_div+destruction-production
        free=self.free
        prescribed=self._prescribed(bd[:,None])
        prescribed_load=self.diffusion.full_matrix @ prescribed
        scale=max(np.linalg.norm(prescribed_load[free])+np.linalg.norm(load[free])+
                  np.linalg.norm(production)+np.linalg.norm(destruction)+np.linalg.norm(adv_div),1e-30)
        mask=self.diffusion.dirichlet
        bc_error=float(np.max(np.abs(faces[mask]-bd[mask]),initial=0.))
        total=adv-owner_reaction
        # Shared owner flux ledger is separate: it alone misses free-face residuals.
        local=self.D @ total+destruction-production
        constraint=self._constraint_reaction(residual[:self.N])
        constraint_error=self._constraint_error(cells)
        return dict(relative_residual=float(np.linalg.norm(residual[free])/scale),
            residual_scale=float(scale),free_residual=residual[free],
            cell_equation=residual[:self.N],face_equation=residual[self.N:],
            cell_constraint_reaction=constraint,cell_constraint_error=constraint_error,
            constrained_cells=self.constraint_cells.copy(),prescribed_cell_values=self.constraint_values.copy(),
            unconstrained_cell_equation_max=float(np.max(abs(residual[self.free_cells]),initial=0.)),
            dirichlet_error=bc_error,cell_equation_max=float(np.max(np.abs(residual[:self.N]))),
            free_face_equation_max=float(np.max(np.abs(residual[self.N:][~mask]),initial=0.)),
            interface_flux_defect_max=float(np.max(abs((owner_reaction+neighbor_reaction)[~self.boundary]),initial=0.)),
            total_flux=total,local_shared_flux_balance=local-constraint,local_shared_flux_balance_without_constraint=local,
            global_balance=float(total[self.boundary].sum()+destruction.sum()-production.sum(dtype=np.longdouble)-constraint.sum(dtype=np.longdouble)))

    def ledger_acceptable(self,result):
        # All local/interface/global flux gates stay absolute and unchanged.
        # Relative quadratic work is an explicit alternative, never an
        # implicit fallback on a failed absolute gate.
        flux=max(result['cell_balance_max'],result['interface_flux_defect_max'],abs(result['global_balance']))
        work=(abs(result['work_balance'])<1e-8 if self.work_balance_policy=='absolute'
              else result['relative_work_balance']<1e-12)
        constraint_error=result.get('cell_constraint_error',0.)
        finite=np.isfinite([result['cell_balance_max'],result['interface_flux_defect_max'],result['global_balance'],
            result['work_balance'],result['work_balance_scale'],result['relative_work_balance'],constraint_error]).all()
        return bool(finite and flux<1e-8 and work and constraint_error<1e-12)

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
            cells=self.free_cells
            residual[:len(cells)]+=self.mesh.cell_volumes[cells]*invdt*(state[cells]-old[cells])
            scale=audit['residual_scale']+np.linalg.norm(self.mesh.cell_volumes*invdt*old)
            error=float(np.linalg.norm(residual)/max(scale,1e-30))
            history.append(dict(iteration=iteration+1,current_nonlinear_residual=error,change=change))
            if max(error,change)<1e-12:
                result=linear._solution_result(state,q,source,sink,old,invdt,kwargs['inflow_values'],solved['linear_relative_residual'])
                if not self.ledger_acceptable(result):continue
                result.update(info,nonlinear_history=history,current_nonlinear_residual=error,
                              scalar_diffusion_correction='nonlinear_mmp')
                return result
        result=linear._solution_result(state,q,source,sink,old,invdt,kwargs['inflow_values'],solved['linear_relative_residual'])
        result.update(info,nonlinear_history=history,current_nonlinear_residual=error,converged=False)
        raise ScalarMMPConvergenceError(result)
