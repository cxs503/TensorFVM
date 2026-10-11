"""Rebuild archived FVM equations and audit actual accepted raw fields."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
run=Path(sys.argv[1]).resolve()
config=json.loads((run/'config.json').read_text())
bindings=json.loads((run/f"n-{config['grids'][0]}/result.json").read_text())['source_sha256']
for name,expected in bindings.items():
    assert hashlib.sha256((run/'source-snapshot'/name).read_bytes()).hexdigest()==expected,name
sys.path.insert(0,str(run/'source-snapshot/src'))
spec=importlib.util.spec_from_file_location('archived_driver',run/'source-snapshot/scripts/verify_native_abc_convergence3d.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
rows=[]
for n in config['grids']:
    mesh=driver.skew_box(n);exact=driver.abc_fields(mesh.cell_centers)
    settings=driver.NativeSteadySettings(viscosity=.1,density=1.,convection=True,convection_order=2,bounded_convection=False,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8)
    solver=driver.NativeQuadraticSteadySolver3D(mesh,mesh.neighbor<0,driver.abc_fields(mesh.face_centers)['velocity'],boundary_velocity_means=driver.exact_boundary_velocity_means(mesh,driver.abc_fields),boundary_velocity_gradients=driver.abc_fields(mesh.face_centers)['velocity_gradient'],body_force=exact['acceleration'],boundary_flux=driver.exact_boundary_flux(mesh,driver.abc_fields),gauge_value=exact['pressure'][0],settings=settings)
    path=run/f'n-{n}/fields.npz';raw=np.load(path)
    solver.u,solver.p,solver.flux=raw['velocity'],raw['pressure'],raw['flux']
    if hasattr(solver,'_assess_state'):
        metrics=solver._assess_state(solver.u,solver.p,solver.flux)[-1]
    else:
        block,rhs,a,correction,source=solver._system(solver.flux)
        consistent=solver.velocity_flux_matrix@solver.u.ravel()+correction@solver.p+source
        residual=block@np.r_[solver.u.ravel(),solver.p]-rhs
        scale=max(np.linalg.norm(solver.force_integrated)+np.linalg.norm(solver.diffusion_source)+np.linalg.norm(solver.p_force_source),1)
        metrics=dict(momentum_residual=float(np.linalg.norm(residual[:3*solver.N])/scale),native_divergence_max=float(abs(solver.viscous.op.integrate(solver.flux)/mesh.cell_volumes).max()),current_reconstructed_divergence_max=float(abs(solver.viscous.op.integrate(consistent)/mesh.cell_volumes).max()),face_flux_consistency_max=float(abs((solver.flux-consistent)/np.linalg.norm(mesh.area_vectors,axis=1)).max()))
    rows.append(dict(n=n,raw_fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics,all_current_equation_gates_pass=all(v<1e-9 for v in metrics.values())))
result=dict(producer_sources_verified=True,replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows,passed=all(r['all_current_equation_gates_pass'] for r in rows))
print(json.dumps(result,indent=2))
assert result['passed']
