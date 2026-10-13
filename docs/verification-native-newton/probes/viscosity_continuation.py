from tensorfvm.verification.native_hybrid_ns import potential_solver
from tensorfvm.native_manufactured3d import volume_weighted_error
state=pressure=None
for mu in (.01,.005,.002,.001):
 m,s,e=potential_solver(6,mu)
 if state is None:s.initialize_stokes()
 else:s.state=state.copy();s.p=pressure.copy()
 r=s.solve_newton(max_iterations=40,tolerance=1e-8,callback=lambda row:print('ITER',mu,row['iteration'],row['step_length'],row['current_momentum_residual'],flush=True))
 print('RESULT',mu,r['numerically_converged'],r['newton_failure'],volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes),volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True),flush=True)
 if not r['numerically_converged']:break
 state=s.state;pressure=s.p
