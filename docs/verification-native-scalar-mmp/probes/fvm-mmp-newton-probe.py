import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
p,c,t,_,_,_=structured_layers(axial=12,azimuthal=8,layers=2,first_height=.001,growth=1.2)
m=NativeMesh3D.from_cells(p,c,t,triangulate_faces=True);g=np.array([.1,-.2,.3]);bd=1+m.face_centers@g
op=NativeHybridScalar3D(m,correction='nonlinear_mmp');mm=op.mmp;free=op.diffusion.free
r=NativeHybridScalar3D(m).solve(boundary_values=bd);u=np.r_[r['cell_values'],r['face_values']]
i,j=mm.pairs.T;e=np.arange(len(i));E=sparse.coo_matrix((np.r_[np.ones(len(i)),-np.ones(len(i))],(np.r_[e,e],np.r_[i,j])),shape=(len(i),len(u))).tocsr();H=abs(E)
A0=sparse.coo_matrix((mm.av,(mm.ai,mm.aj)),shape=(len(u),len(u))).tocsr();A0-=sparse.diags(np.asarray(A0.sum(1)).ravel())
scale=max(np.linalg.norm((op.diffusion.full_matrix@op.diffusion._prescribed(bd[:,None]))[free]),1.)

def evaluate(u):
 a,info=mm.operator(u);return a.full_matrix@u,a

def jacobian(u):
 du=E@u;den=H.T@abs(du);re=A0@u
 dden=H.T@sparse.diags(np.sign(du))@E
 f1=np.zeros(len(u));f2=f1.copy();positive=free[den[free]>0]
 f1[positive]=mm.margin*np.sign(re[positive])/den[positive]
 f2[positive]=mm.margin*abs(re[positive])/den[positive]**2
 db=sparse.diags(f1)@A0-sparse.diags(f2)@dden
 beta=np.zeros(len(u));beta[positive]=mm.margin*abs(re[positive])/den[positive];beta+=mm.strict
 picked=np.where(beta[i]>=beta[j],i,j)
 P=sparse.coo_matrix((np.ones(len(i)),(e,picked)),shape=E.shape).tocsr()
 _,a=evaluate(u)
 return a.full_matrix+E.T@sparse.diags(du)@P@db
# Independent finite difference away from nonsmooth zero switches.
rng=np.random.default_rng(49);z=u+rng.normal(size=len(u))*.01;v=rng.normal(size=len(u));v[np.setdiff1d(np.arange(len(u)),free)]=0
fd=(evaluate(z+1e-7*v)[0]-evaluate(z-1e-7*v)[0])/2e-7
print('JAC',np.linalg.norm(fd-jacobian(z)@v)/np.linalg.norm(fd),flush=True)
for it in range(100):
 residual,_=evaluate(u);merit=np.linalg.norm(residual[free])/scale
 J=jacobian(u)[free][:,free].tocsc();dx=splu(J).solve(-residual[free]);du=np.linalg.norm(dx)/max(np.linalg.norm(u[free]),1)
 print('ITER',it,merit,du,flush=True)
 if merit<1e-12 and du<1e-12:break
 for back in range(30):
  trial=u.copy();trial[free]+=2.**-back*dx
  rr,_=evaluate(trial)
  if np.linalg.norm(rr[free])/scale <= (1-1e-4*2.**-back)*merit:
   u=trial;break
 else:print('LINESEARCHFAIL',flush=True);break
print('FINAL',it,merit,du,'error',np.max(np.abs(u[:op.N]-(1+m.cell_centers@g))),flush=True)
