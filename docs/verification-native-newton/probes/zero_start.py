from tensorfvm.verification.native_hybrid_ns import potential_solver
from tensorfvm.native_manufactured3d import volume_weighted_error
for mu in (.1,.01,.001):
 m,s,e=potential_solver(4,mu)
 def progress(row):print('ITER',mu,{k:v for k,v in row.items() if k!='trials'},flush=True)
 r=s.solve_newton(max_iterations=50,tolerance=1e-8,callback=progress)
 print('RESULT',mu,r['numerically_converged'],r['newton_failure'],volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes),volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True),r['energy_balance_error'],r['global_momentum_balance_max'],flush=True)
