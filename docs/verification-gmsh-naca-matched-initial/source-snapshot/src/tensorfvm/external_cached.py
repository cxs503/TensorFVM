"""CPU LU-preconditioned GMRES for external flow; current equations are never lagged.

A cached factor is only a preconditioner for the current assembled matrix.
Refresh on a bounded schedule or failed true residual; retain strict budgets.
"""
import math,time
import numpy as np
import torch
from scipy.sparse.linalg import splu,gmres,LinearOperator
from .external_linear import ExternalFlowSolver
from .sparse_simple import host


class CachedExternalFlowSolver(ExternalFlowSolver):
    def __init__(self,config):
        super().__init__(config);self._lu_cache={};self._linear_calls={}

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
        count=self._linear_calls.get(kind,0);refresh=kind not in self._lu_cache or count%20==0
        if refresh:self._lu_cache[kind]=splu(A.tocsc())
        lu=self._lu_cache[kind];preconditioner=LinearOperator(A.shape,matvec=lu.solve)
        q,info=gmres(A,b,x0=host(initial),M=preconditioner,atol=budget*.2,rtol=0,restart=30,maxiter=10)
        residual=float(np.linalg.norm(A@q-b));fallback=False
        if info!=0 or not np.isfinite(q).all() or residual>budget*1.05:
            lu=splu(A.tocsc());self._lu_cache[kind]=lu;q=lu.solve(b);residual=float(np.linalg.norm(A@q-b));fallback=True
        if not np.isfinite(q).all() or residual>budget*1.05:raise RuntimeError(f'Current-system true residual {residual} exceeds {budget}')
        self._linear_calls[kind]=count+1;self.linear_history.append(dict(kind=kind,backend='cpu-current-csr-gmres-cached-lu',true_residual=residual,target=budget,preconditioner_refreshed=refresh,direct_fallback=fallback,assembled_operator_relative_defect=defect,elapsed_s=time.perf_counter()-start))
        return torch.as_tensor(q,dtype=rhs.dtype,device=rhs.device)
