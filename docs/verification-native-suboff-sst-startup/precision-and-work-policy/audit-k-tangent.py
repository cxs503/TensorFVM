import json
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
root=Path('results/suboff-sst-precise-relative-three-steps')
f=np.load(root/'rejected-scalar-candidate.npz');u=np.r_[f['cell_values'],f['face_values']]
m=NativeMesh3D.from_gmsh(root/'initial-input.msh',triangulate_faces=True)
a=NativeHybridScalar3D(m,f['diffusivity'],dirichlet_mask=(m.neighbor<0)&(m.patch_labels!='outlet'),correction='nonlinear_mmp')
meta=json.loads((root/'rejected-scalar-candidate.json').read_text())['diagnostics'];scale=meta['continuation_state_scale'];free=a.diffusion.free
rng=np.random.default_rng(15);d=np.zeros(len(u));d[free]=rng.normal(size=len(free))*scale
out=[]
for smoothing in (1e-3,0.):
 _,j=a.mmp.nonlinear_system(u,smoothing=smoothing,state_scale=scale,jacobian=True)
 exact=j@d
 for h in (1e-4,1e-5,1e-6,1e-7,1e-8):
  fd=(a.mmp.nonlinear_reaction(u+h*d,smoothing=smoothing,state_scale=scale)-a.mmp.nonlinear_reaction(u-h*d,smoothing=smoothing,state_scale=scale))/(2*h)
  out.append(dict(smoothing=smoothing,relative_step=h,relative_direction_error=float(np.linalg.norm((fd-exact)[free])/np.linalg.norm(exact[free]))))
print(json.dumps(dict(scope='directional Jacobian diagnostic at rejected third-step k candidate; not solver convergence',state_scale=scale,checks=out),indent=2))
