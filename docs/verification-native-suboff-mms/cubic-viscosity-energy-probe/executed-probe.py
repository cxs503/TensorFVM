import os,sys,time,json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
repo=Path('/home/jsyc/tensor-suite-development/TensorFVM');os.chdir(repo);sys.path.insert(0,str(repo/'src'))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
from tensorfvm.native_quadratic_system3d import prepare_geometry
from tensorfvm.native_cubic_gradient3d import NativeCubicFaceGradient3D
out=repo/'docs/verification-native-suboff-mms/cubic-viscosity-energy-probe';out.mkdir()
meshfile=repo/'docs/verification-native-suboff-mms/tail-transition-quadratic/input.msh'
m=NativeMesh3D.from_gmsh(meshfile,triangulate_faces=True)
s=NativeQuadraticSteadySolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),settings=NativeSteadySettings(viscosity=.1,convection=False),quadratic_linear_solver='assembled')
result={};vectors={}
for kind in ['quadratic','cubic']:
 start=time.perf_counter();s.viscous_reconstruction=kind
 if kind=='cubic':s.viscous_gradient=NativeCubicFaceGradient3D(m,m.neighbor<0)
 A=prepare_geometry(s)['diffusion'];scale=np.maximum(abs(A.diagonal()),1e-30)**-.5
 S=sparse.diags(scale)@((A+A.T)*.5)@sparse.diags(scale)
 vals,vecs=eigsh(S,k=3,which='SA',v0=np.random.default_rng(421).normal(size=S.shape[0]),tol=1e-9,maxiter=3000)
 residuals=[float(np.linalg.norm(S@vecs[:,k]-vals[k]*vecs[:,k])) for k in range(3)]
 worst=np.linalg.norm((scale*vecs[:,0]).reshape(-1,3),axis=1).argmax()
 result[kind]=dict(eigenvalues=vals.tolist(),eigenpair_residuals=residuals,negative_diagonal_count=int((A.diagonal()<0).sum()),largest_mode_cell=int(worst),largest_mode_center=m.cell_centers[worst].tolist(),seconds=time.perf_counter()-start,nnz=A.nnz)
 vectors[kind+'_scaled_eigenvectors']=vecs;vectors[kind+'_diagonal_inverse_sqrt']=scale
 print(json.dumps({kind:result[kind]}),flush=True)
result['scope']='Symmetric part of integrated homogeneous velocity diffusion, scaled by sqrt(abs(diagonal)); negative energy does not by itself prove a pressure-coupling instability.'
result['mesh_sha256']=hashlib.sha256(meshfile.read_bytes()).hexdigest()
result['sources']={str(Path(mod.__file__).relative_to(repo)):hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() for name,mod in sys.modules.items() if name=='tensorfvm' or name.startswith('tensorfvm.')}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'eigenvectors.npz',**vectors);shutil.copy2(__file__,out/'executed-probe.py')
(out/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
