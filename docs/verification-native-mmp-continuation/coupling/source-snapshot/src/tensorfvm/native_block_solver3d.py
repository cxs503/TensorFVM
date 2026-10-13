"""Optional CPU block preconditioner for native hybrid velocity/pressure systems.

Scalar full-gradient energies precondition the symmetric stress operator;
component AMG cycles and a pressure mass approximation accelerate GMRES.
Only the preconditioner is approximate: acceptance checks the CURRENT original
coupled matrix. No artificial compressibility or altered pressure equation.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,gmres


class NativeHybridBlockSolver3D:
    def __init__(self,solver):
        try:
            import pyamg
        except ImportError as exc:
            raise ImportError('block_amg requires the optional tensorfvm[amg] dependency') from exc
        self.solver=solver;self.viscous=solver.viscous;self.mu=self.viscous.mu.copy()
        v=self.viscous;N,F=v.N,v.F;rows=[];cols=[];data=[]
        start=time.perf_counter()
        # A full-gradient scalar norm, not a replacement physical viscosity.
        # The actual symmetric stress and gradient jump remain in the matrix.
        for cell,(_,_,ids,_,g,residual,penalty) in enumerate(v.local):
            nodes=ids[::3]//3
            block=v.mu[cell]*v.mesh.cell_volumes[cell]*(g.T@g)
            block+=residual.T@(penalty[:,None]*residual)
            rows.append(np.repeat(nodes,len(nodes)));cols.append(np.tile(nodes,len(nodes)));data.append(block.ravel())
        scalar=sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),
                                 shape=(N+F,N+F)).tocsr()
        self.groups=[];self.complexities=[]
        for component in range(3):
            positions=np.flatnonzero(v.free%3==component);nodes=v.free[positions]//3
            if len(nodes)==0:continue
            # Constants suffice for the scalar diffusion preconditioner.
            hierarchy=pyamg.smoothed_aggregation_solver(scalar[nodes][:,nodes],max_coarse=30)
            self.groups.append((positions,hierarchy.aspreconditioner()))
            self.complexities.append(float(hierarchy.operator_complexity()))
        self.setup_seconds=time.perf_counter()-start
        self.pyamg_version=pyamg.__version__

    def velocity_inverse(self,rhs):
        result=np.zeros(len(rhs),dtype=float)
        for positions,cycle in self.groups:result[positions]=cycle@rhs[positions]
        return result

    def solve(self,matrix,rhs):
        s=self.solver;nv=len(s.viscous.free);matrix=matrix.tocsr();rhs=np.asarray(rhs,float)
        if matrix.shape!=(nv+s.N,nv+s.N) or rhs.shape!=(nv+s.N,):
            raise ValueError('block solver requires the native coupled velocity/pressure system')
        if self.viscous is not s.viscous or not np.array_equal(self.mu,s.viscous.mu):
            raise RuntimeError('viscosity changed; rebuild block preconditioner')
        start=time.perf_counter();a=matrix[:nv,:nv];d=s.mesh.cell_volumes/s.viscous.mu
        if s.gauge:
            # Eliminate the fixed pressure variable AND its replaced row.
            # Account for nonzero pressure/gauge increments in momentum RHS.
            b=s.B[1:];gauge=rhs[nv]
            reduced_rhs=np.r_[rhs[:nv]+s.B[0].toarray().ravel()*gauge,rhs[nv+1:]]
            mass=d[1:]
        else:
            b=s.B;gauge=None;reduced_rhs=rhs;mass=d
        reduced=sparse.bmat([[a,-b.T],[b,None]],format='csr')
        def precondition(residual):
            u=self.velocity_inverse(residual[:nv])
            pressure_rhs=residual[nv:]-b@u
            pressure=pressure_rhs/mass
            if s.gauge:
                # Inverse of diag(d_rest)-d_rest*d_rest.T/sum(d), retaining
                # the constant-pressure nullspace before gauge elimination.
                pressure+=pressure_rhs.sum()/d[0]
            u+=self.velocity_inverse(b.T@pressure)
            return np.r_[u,pressure]
        history=[]
        answer,info=gmres(reduced,reduced_rhs,
            M=LinearOperator(reduced.shape,matvec=precondition,dtype=float),
            rtol=1e-13,atol=0.,restart=80,maxiter=30,
            callback=lambda residual:history.append(float(residual)),callback_type='pr_norm')
        if s.gauge:answer=np.r_[answer[:nv],gauge,answer[nv:]]
        error=float(np.linalg.norm(matrix@answer-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny))
        record=dict(method='block_amg',gmres_info=int(info),gmres_iterations=len(history),
            true_relative_residual=error,preconditioned_residual_history=history,
            operator_complexities=self.complexities,setup_seconds=self.setup_seconds,
            solve_seconds=time.perf_counter()-start,pyamg_version=self.pyamg_version)
        if info!=0 or not np.isfinite(answer).all() or not np.isfinite(error) or error>1e-9:
            raise RuntimeError(f'block AMG failed CURRENT true residual: info={info}, residual={error:g}')
        return answer,error,record
