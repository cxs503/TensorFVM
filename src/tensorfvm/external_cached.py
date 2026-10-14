"""CPU LU-preconditioned GMRES for external flow; current equations are never lagged.

A cached factor is only a preconditioner for the current assembled matrix.
Refresh on a bounded schedule or failed true residual; retain strict budgets.
"""
import math,time
import numpy as np
import torch
from scipy.sparse.linalg import splu,spilu,gmres,LinearOperator
from .external_linear import ExternalFlowSolver
from .sparse_simple import host


class CachedExternalFlowSolver(ExternalFlowSolver):
    # A cached factor is refreshed every ``lu_refresh_interval`` solves of the
    # same kind, after ``gmres_max_restarts`` restarts of 30 Krylov vectors
    # failed to reach the budget (direct fallback), and on request.
    lu_refresh_interval=20
    gmres_max_restarts=10
    # A factor that needed this many Krylov vectors (or a direct fallback) is
    # refreshed before the next solve of the same key; refreshing early is
    # cheaper than a failed restart cycle followed by the fallback.
    krylov_refresh_threshold=10
    # 'lu': the cached preconditioner is a complete SuperLU factor (exact for
    # the matrix it was built from).  'krylov': incomplete LU (momentum, SA)
    # and Ruge-Stueben AMG (pressure, ILU if pyamg is absent) preconditioners
    # whose setup is O(N); the true-residual budget, the operator probes and
    # the direct fallback are identical, only the preconditioner differs.
    # At 1.3e5 cells a complete factor costs 12 s and ~1 GB each, an ILU/AMG
    # setup well below a second.
    linear_backend='lu'
    krylov_restart=50
    krylov_max_restarts=4
    ilu_drop_tolerance={'pressure':1e-4,'momentum':1e-3}
    ilu_fill_factor={'pressure':10,'momentum':5}

    def __init__(self,config):
        super().__init__(config);self._lu_cache={};self._linear_calls={};self._cache_key_suffix='';self._refresh_next={}

    def _krylov_preconditioner(self,kind,A):
        if kind=='pressure':
            try:
                import pyamg
            except ImportError:pyamg=None
            if pyamg is not None:
                hierarchy=pyamg.ruge_stuben_solver(A.tocsr(),max_coarse=50)
                return hierarchy.aspreconditioner(cycle='V'),'amg-ruge-stuben'
        ilu=spilu(A.tocsc(),drop_tol=self.ilu_drop_tolerance[kind],fill_factor=self.ilu_fill_factor[kind])
        return LinearOperator(A.shape,matvec=ilu.solve),'ilu'

    def _linear(self,diagonal,ao,an,rhs,initial,symmetric=False,operator=None,
                threshold_floor=1e-10,relative_tolerance=1e-9,allow_inexact=False):
        if rhs.device.type!='cpu':return super()._linear(diagonal,ao,an,rhs,initial,symmetric,operator,threshold_floor,relative_tolerance,allow_inexact)
        start=time.perf_counter();pressure=operator is not None;kind='pressure' if pressure else 'momentum';A=self._pressure_csr if pressure else self._matrix(diagonal,ao,an)
        if A is None:raise RuntimeError('Pressure matrix was not assembled')
        defect=None
        if pressure:
            probe=torch.sin(torch.arange(self.count,dtype=rhs.dtype)*.371);target=host(operator(probe));defect=float(np.linalg.norm(A@host(probe)-target)/np.linalg.norm(target))
            if defect>2e-12:raise RuntimeError('Current pressure operator mismatch')
        b=host(rhs);force=self.config.density*self.config.inlet_velocity**2*self.config.reference_length;budget=max(1e-12,min(np.linalg.norm(b)*1e-11,self.config.tolerance*force/(10*math.sqrt(self.count))))
        key=kind+self._cache_key_suffix;count=self._linear_calls.get(key,0);interval=max(1,int(self.lu_refresh_interval));refresh=key not in self._lu_cache or count%interval==0 or bool(self._refresh_next.get(key))
        krylov=self.linear_backend=='krylov';iterations=[0];x0=host(initial)
        def counter(_):iterations[0]+=1
        def solve(preconditioner,restart,cycles):
            iterations[0]=0
            q,info=gmres(A,b,x0=x0,M=preconditioner,atol=budget*.2,rtol=0,restart=restart,maxiter=cycles,callback=counter,callback_type='pr_norm')
            return q,info,float(np.linalg.norm(A@q-b))
        if krylov:
            if refresh:self._lu_cache[key]=self._krylov_preconditioner(kind,A)
            preconditioner,label=self._lu_cache[key]
            q,info,residual=solve(preconditioner,int(self.krylov_restart),max(1,int(self.krylov_max_restarts)));rebuilt=False
            if info!=0 or not np.isfinite(q).all() or residual>budget*1.05:
                # Stale preconditioner: rebuild from the current matrix once.
                self._lu_cache[key]=self._krylov_preconditioner(kind,A);preconditioner,label=self._lu_cache[key];rebuilt=True
                q,info,residual=solve(preconditioner,int(self.krylov_restart),max(1,int(self.krylov_max_restarts)))
            backend='cpu-current-csr-gmres-'+label
        else:
            if refresh:self._lu_cache[key]=splu(A.tocsc())
            lu=self._lu_cache[key];rebuilt=False
            q,info,residual=solve(LinearOperator(A.shape,matvec=lu.solve),30,max(1,int(self.gmres_max_restarts)));backend='cpu-current-csr-gmres-cached-lu'
        fallback=False
        if info!=0 or not np.isfinite(q).all() or residual>budget*1.05:
            lu=splu(A.tocsc());q=lu.solve(b);residual=float(np.linalg.norm(A@q-b));fallback=True
            if not krylov:self._lu_cache[key]=lu
        if not np.isfinite(q).all() or residual>budget*1.05:raise RuntimeError(f'Current-system true residual {residual} exceeds {budget}')
        self._linear_calls[key]=count+1;self._refresh_next[key]=fallback or (not krylov and iterations[0]>=self.krylov_refresh_threshold) or (krylov and iterations[0]>=int(self.krylov_restart));self.linear_history.append(dict(kind=kind,backend=backend,true_residual=residual,target=budget,preconditioner_refreshed=refresh or rebuilt,direct_fallback=fallback,krylov_iterations=iterations[0],assembled_operator_relative_defect=defect,elapsed_s=time.perf_counter()-start))
        return torch.as_tensor(q,dtype=rhs.dtype,device=rhs.device)
