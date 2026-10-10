"""Conservative Gauss pressure gradient and geometric linear interpolation.

Pressure forces use the same physical face pressures as the momentum source.
Boundary pressure is owner pressure for Neumann faces and zero at the outlet.
Velocity reconstruction remains least squares. No artificial physical source
or reference coefficient enters the discrete equations.
"""
import numpy as np
import torch
from scipy import sparse
from .sparse_simple import host


class ConservativePressureGradient:
    pressure_gradient_scheme = 'gauss-linear'
    face_interpolation_scheme = 'projected-linear'
    def _face_owner_weights(self):
        if not hasattr(self,'_owner_weights'):
            delta=self.mesh.face_centers[self.f]-self.mesh.centers.reshape(-1,2)[self.oi]
            fraction=(delta*self.S[self.f]).sum(-1)/(self.d[self.f]*self.S[self.f]).sum(-1)
            weights=torch.ones(len(self.o),dtype=self.p.dtype,device=self.p.device)
            weights[self.f]=1-fraction
            if not torch.all((weights[self.f]>=0)&(weights[self.f]<=1)):
                raise ValueError('Face interpolation requires centroid projections bracketing the face')
            self._owner_weights=weights
        return self._owner_weights

    def _interpolate(self, field):
        values=field[self.o].clone()
        weights=self._face_owner_weights()[self.f]
        shape=(-1,)+(1,)*(field.ndim-1)
        values[self.f]=weights.reshape(shape)*field[self.oi]+(1-weights).reshape(shape)*field[self.ni]
        return values

    def _gradient(self,field,pressure=False,boundary_values=None):
        if not pressure:
            return super()._gradient(field,pressure=False,boundary_values=boundary_values)
        face=self._interpolate(field)
        face[self.mesh.masks['outlet']]=0
        return self._sum(face[:,None]*self.S)/self.volume[:,None]

    def _geometry(self):
        if self._geometry_operators is not None:
            return self._geometry_operators
        o,ni,f=host(self.o),host(self.ni),host(self.f);oi=o[f]
        nf=len(o);index=np.arange(nf);w=host(self._face_owner_weights())
        H=sparse.coo_matrix((np.r_[np.ones(nf),-np.ones(len(ni))],(np.r_[o,ni],np.r_[index,index[f]])),shape=(self.count,nf)).tocsr()
        I=sparse.coo_matrix((np.r_[w,1-w[f]],(np.r_[index,index[f]],np.r_[o,ni])),shape=(nf,self.count)).tocsr()
        active=(~host(self.mesh.masks['outlet'])).astype(float)
        P=sparse.diags(active)@I
        inverse=sparse.diags(1/host(self.volume))
        gradients=[inverse@H@sparse.diags(host(self.S)[:,a])@P for a in range(2)]
        self._geometry_operators=H,I,gradients
        return self._geometry_operators
