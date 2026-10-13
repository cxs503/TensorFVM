"""Opt-in nonlinear symmetric graph correction of hybrid scalar diffusion.

Based on the correction principle reviewed by Droniou (2014), section 6.3,
https://arxiv.org/abs/1407.1567. This is a project cell/face implementation;
no unconditional convergence/order guarantee is claimed. Local symmetric
positive graph energies retain coercivity and conservative face reactions.
"""
import copy
import numpy as np
from scipy import sparse


class ScalarMMPConvergenceError(RuntimeError):
    """Unaccepted original-system candidate and full nonlinear iteration trace."""
    def __init__(self,result):
        self.scalar_result=result
        self.scalar_name='scalar'
        super().__init__(f'nonlinear scalar correction failed current residual/change: {result["nonlinear_history"][-1]}')


class NativeScalarMMP3D:
    def __init__(self, diffusion, margin=1.01, regularization=.01,*,free_dofs=None):
        if diffusion.components!=1:raise ValueError('scalar diffusion required')
        if not np.isfinite(margin) or margin<=1:raise ValueError('margin must exceed one')
        if not np.isfinite(regularization) or regularization<=0:raise ValueError('positive graph regularization required')
        self.base=diffusion;self.margin=margin
        self.free=diffusion.free if free_dofs is None else np.asarray(free_dofs,int)
        self.regularization=regularization
        self.size=diffusion.N+diffusion.F
        self.edges=[];pairs=[]
        for _,_,ids,*_ in diffusion.local:
            i,j=np.triu_indices(len(ids),1);self.edges.append((i,j))
            pairs.append(np.sort(np.column_stack((ids[i],ids[j])),axis=1))
        self.pairs,self.inverse,self.counts=np.unique(np.vstack(pairs),axis=0,return_inverse=True,return_counts=True)
        # Assemble scalar reactions from local coefficients before rounding the
        # global matrix. Extended-state residuals must match local flux ledgers.
        rows=[];cols=[];data=[]
        for _,_,ids,matrix,*_ in diffusion.local:
            rows.append(np.repeat(ids,len(ids)));cols.append(np.tile(ids,len(ids)))
            data.append(matrix.ravel().astype(np.longdouble))
        a=sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),
            shape=diffusion.full_matrix.shape).tocsr().tocoo();off=a.row!=a.col
        self.ai,self.aj,self.av=a.row[off],a.col[off],a.data[off]
        i,j=self.pairs.T;e=np.arange(len(i))
        self.E=sparse.coo_matrix((np.r_[np.ones(len(i)),-np.ones(len(i))],
            (np.r_[e,e],np.r_[i,j])),shape=(len(i),self.size)).tocsr()
        self.H=abs(self.E)
        self.reaction_matrix=sparse.coo_matrix((self.av.astype(float),(self.ai,self.aj)),shape=(self.size,self.size)).tocsr()
        self.reaction_matrix-=sparse.diags(np.asarray(self.reaction_matrix.sum(axis=1)).ravel())
        # Strict positive edge correction is essential even when A*u=0:
        # an uncorrected discrete harmonic solution may already be negative.
        # O((h/L)^2) scaling makes this regularization vanish under refinement.
        h=diffusion.mesh.cell_volumes**(1/3)
        node_h=np.r_[h,h[diffusion.mesh.owner]].copy()
        internal=diffusion.mesh.neighbor>=0
        node_h[diffusion.N+np.flatnonzero(internal)]=np.maximum(
            h[diffusion.mesh.owner[internal]],h[diffusion.mesh.neighbor[internal]])
        length=float(np.max(np.ptp(diffusion.mesh.vertices,axis=0)))
        degree=np.bincount(self.pairs.ravel(),minlength=self.size)
        self.strict=regularization*(node_h/length)**2*diffusion.full_matrix.diagonal()/degree
        self.strict[np.setdiff1d(np.arange(self.size),self.free)]=0.

    def operator(self,state):
        u=np.asarray(state)
        if u.dtype.kind!='f':u=u.astype(float)
        i,j,_,_,_,_,_,_,_,_,weight=self._coefficients(u,0.,1.)
        local_weights=weight[self.inverse]/self.counts[self.inverse]
        rows=[];cols=[];data=[];local=[];offset=0;extra_energy=0.
        for record,(ii,jj) in zip(self.base.local,self.edges):
            faces,sign,ids,original,g,residual,penalty=record
            w=local_weights[offset:offset+len(ii)];offset+=len(ii)
            correction=np.zeros_like(original,dtype=u.dtype)
            np.add.at(correction,(ii,ii),w);np.add.at(correction,(jj,jj),w)
            np.add.at(correction,(ii,jj),-w);np.add.at(correction,(jj,ii),-w)
            extra_energy+=float(np.sum(w*(u[ids[ii]]-u[ids[jj]])**2))
            matrix=original+correction
            local.append((faces,sign,ids,matrix,g,residual,penalty))
            rows.append(np.repeat(ids,len(ids)));cols.append(np.tile(ids,len(ids)));data.append(matrix.ravel())
        op=copy.copy(self.base);op.local=local
        op.full_matrix=sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=self.base.full_matrix.shape).tocsr().astype(float)
        op.matrix=op.full_matrix[op.free][:,op.free].tocsc();op._face_factor=None
        return op,dict(correction_energy=extra_energy,maximum_edge_weight=float(weight.max(initial=0)),
                       active_edges=int(np.count_nonzero(weight)),regularization=self.regularization)

    def nonlinear_system(self,state,*,smoothing=0.,state_scale=1.,jacobian=False):
        """Smooth continuation system; smoothing=0 is the original graph law.

        state_scale supplies scalar units. Smoothing is dimensionless; it is
        removed before accepting any result. The Jacobian differentiates both
        graph coefficients and their multiplication by scalar jumps.
        """
        i,j,du,reaction,ad,ar,den,active,gap,ag,weight=self._coefficients(state,smoothing,state_scale)
        du,reaction,ad,ar,den,gap,ag,weight=[np.asarray(a,float) for a in (du,reaction,ad,ar,den,gap,ag,weight)]
        matrix=self.base.full_matrix+self.E.T@sparse.diags(weight)@self.E
        if not jacobian:return matrix
        dre=np.divide(reaction,ar,out=np.zeros_like(reaction),where=ar>0)
        ddu=np.divide(du,ad,out=np.zeros_like(du),where=ad>0)
        f1=np.zeros(self.size);f2=f1.copy()
        f1[active]=self.margin*dre[active]/den[active]
        f2[active]=self.margin*ar[active]/den[active]**2
        db=sparse.diags(f1)@self.reaction_matrix-sparse.diags(f2)@(self.H.T@sparse.diags(ddu)@self.E)
        sg=np.divide(gap,ag,out=np.zeros_like(gap),where=ag>0);e=np.arange(len(i))
        pick=sparse.coo_matrix((np.r_[.5*(1+sg),.5*(1-sg)],
            (np.r_[e,e],np.r_[i,j])),shape=self.E.shape).tocsr()
        tangent=matrix+self.E.T@sparse.diags(du)@pick@db
        return matrix,tangent

    def _coefficients(self,state,smoothing,state_scale):
        u=np.asarray(state)
        if u.dtype.kind!='f':u=u.astype(float)
        if u.shape!=(self.size,) or not np.isfinite(u).all():raise ValueError('finite hybrid scalar state required')
        if not np.isfinite(smoothing) or smoothing<0 or not np.isfinite(state_scale) or state_scale<=0:
            raise ValueError('nonnegative smoothing and positive state scale required')
        i,j=self.pairs.T;du=u[i]-u[j];diagonal=self.base.full_matrix.diagonal()
        reaction=np.zeros(self.size,dtype=u.dtype)
        np.add.at(reaction,self.ai,self.av*(u[self.aj]-u[self.ai]))
        ad=np.hypot(du,smoothing*state_scale)
        ar=np.hypot(reaction,diagonal*smoothing*state_scale)
        den=np.zeros(self.size,dtype=u.dtype)
        np.add.at(den,i,ad);np.add.at(den,j,ad)
        active=self.free[den[self.free]>0]
        beta=np.zeros(self.size,dtype=u.dtype);beta[active]=self.margin*ar[active]/den[active];beta+=self.strict
        gap=beta[i]-beta[j];ag=np.hypot(gap,smoothing*np.maximum(diagonal[i],diagonal[j]))
        weight=.5*(beta[i]+beta[j]+ag)
        return i,j,du,reaction,ad,ar,den,active,gap,ag,weight

    def nonlinear_reaction(self,state,*,smoothing=0.,state_scale=1.):
        """Constant-preserving equation evaluation with the state's precision."""
        i,j,du,reaction,*_,weight=self._coefficients(state,smoothing,state_scale)
        answer=reaction.copy()
        np.add.at(answer,i,weight*du);np.add.at(answer,j,-weight*du)
        return answer
