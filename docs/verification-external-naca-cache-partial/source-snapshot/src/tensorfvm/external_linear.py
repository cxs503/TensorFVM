"""CPU sparse direct solves for external-flow verification with true residual gates.

Retains the production finite-volume equations and geometry. CUDA uses the
existing Torch Krylov backend; SciPy SuperLU runs only on the CPU path.
"""
import math
import time
import numpy as np
import torch
from scipy.sparse.linalg import splu
from .sparse_simple import SparseBodyFittedSolver, host


class ExternalFlowSolver(SparseBodyFittedSolver):
    def _linear(self,diagonal,ao,an,rhs,initial,symmetric=False,operator=None,
                threshold_floor=1e-10,relative_tolerance=1e-9,allow_inexact=False):
        if rhs.device.type!='cpu':
            return super()._linear(diagonal,ao,an,rhs,initial,symmetric,operator,
                                  threshold_floor,relative_tolerance,allow_inexact)
        start=time.perf_counter();pressure=operator is not None
        A=self._pressure_csr if pressure else self._matrix(diagonal,ao,an)
        if A is None:raise RuntimeError('Pressure matrix was not assembled')
        if pressure:
            probe=torch.sin(torch.arange(self.count,dtype=rhs.dtype)*.371)
            target=host(operator(probe));defect=np.linalg.norm(A@host(probe)-target)/np.linalg.norm(target)
            if defect>2e-12:raise RuntimeError('Assembled pressure operator mismatch')
        q=splu(A.tocsc()).solve(host(rhs));residual=float(np.linalg.norm(A@q-host(rhs)))
        force=self.config.density*self.config.inlet_velocity**2*self.config.reference_length
        budget=max(1e-12,min(np.linalg.norm(host(rhs))*1e-11,self.config.tolerance*force/(10*math.sqrt(self.count))))
        if not np.isfinite(q).all() or residual>budget*1.05:
            raise RuntimeError(f'Sparse direct true residual {residual} exceeds {budget}')
        self.linear_history.append(dict(kind='pressure' if pressure else 'momentum',backend='scipy-superlu-cpu',true_residual=residual,target=budget,elapsed_s=time.perf_counter()-start))
        return torch.as_tensor(q,dtype=rhs.dtype,device=rhs.device)
