"""Evaluate two frozen producers on identical archived triangular mesh fields."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--producer',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
config=json.loads((args.producer/'config.json').read_text())
sys.path.insert(0,str((args.producer/'source-snapshot/src').resolve()))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
from tensorfvm.native_quadratic_system3d import current_matrix
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
bindings={}
for name,module in list(sys.modules.items()):
 if name=='tensorfvm' or name.startswith('tensorfvm.'):
  path=Path(module.__file__).resolve()
  relative=str(path.relative_to((args.producer/'source-snapshot').resolve()))
  if config['source_sha256'].get(relative)!=sha(path):raise ValueError('unbound numerical import: '+relative)
  bindings[relative]=sha(path)
m=NativeMesh3D.from_gmsh(args.run/'input.msh',triangulate_faces=True)
settings=NativeSteadySettings(**config['settings']);field=lambda x:abc_fields(x,nu=settings.viscosity)
exact=field(m.cell_centers);boundary=m.neighbor<0
solver=NativeQuadraticSteadySolver3D(m,boundary,field(m.face_centers)['velocity'],settings=settings,body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,field),gauge_value=exact['pressure'][0],boundary_velocity_means=exact_boundary_velocity_means(m,field),boundary_velocity_gradients=field(m.face_centers)['velocity_gradient'],quadratic_linear_solver='assembled')
with np.load(args.run/'fields.npz') as data:u=data['velocity'].copy();pressure=data['pressure'].copy();flux=data['flux'].copy()
block,rhs,a,correction,source,merit,metrics=solver._assess_state(u,pressure,flux)
operator=current_matrix(solver,flux,a)
vector=np.random.default_rng(251011).normal(size=4*len(m.cells))
np.savez_compressed(args.output,pressure_face_means=solver.quadratic_pressure_faces(),velocity_face_means=solver.quadratic_velocity_faces(),traction=solver.quadratic_traction(),current_equation_residual=block@np.r_[u.ravel(),pressure]-rhs,current_consistent_flux=solver.velocity_flux_matrix@u.ravel()+correction@pressure+source,full_matrix_random_action=operator@vector)
args.output.with_suffix('.json').write_text(json.dumps(dict(producer=str(args.producer),input_sha256={name:sha(args.run/name) for name in ('input.msh','fields.npz')},frozen_import_sha256=bindings,current_equation_metrics=metrics,matrix_nnz=operator.nnz,output_sha256=sha(args.output),scope='read-only operator/source compatibility; no new solve'),indent=2)+'\n')
print(json.dumps(metrics),flush=True)
