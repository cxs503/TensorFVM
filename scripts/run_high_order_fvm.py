#!/usr/bin/env python3
"""Heavy convergence and conservation tests for high-order FVM."""
import sys, math, time
sys.path.insert(0, "src")

import torch
import numpy as np
from tensorfvm.high_order_reconstruction import *

results = {}

# 1. Uniform flow preservation (criterion a)
print("=== Uniform Flow Preservation ===")
mesh = make_triangular_mesh_uniform(8, 8, lx=1.0, ly=1.0)
q0 = uniform_flow(rho=1.0, u=1.5, v=0.5, p=1.0, ncell=mesh.ncell)

for name, recon, flux in [("MUSCL+Roe", "muscl", "roe"), ("WENO3+Roe", "weno", "roe"), ("MUSCL+HLLC", "muscl", "hllc")]:
    solver = EulerSolver(mesh, reconstruction=recon, weno_order=3, flux_scheme=flux, limiter="none", time_method="rk3")
    t0 = time.perf_counter()
    result = solver.run(q0, n_steps=1000, cfl=0.5, report_interval=1000)
    dt_cpu = time.perf_counter() - t0
    max_change = float((result["q"] - q0).abs().max().item())
    results[f"uniform_{name}"] = {"max_change": max_change, "cpu_time": dt_cpu, "passed": max_change < 1e-10}
    print(f"  {name}: max_change={max_change:.2e}, cpu={dt_cpu:.1f}s, passed={max_change < 1e-10}")

# 2. Conservation (criterion e)
print("\n=== Conservation Check ===")
for name, recon, flux in [("MUSCL+Roe", "muscl", "roe"), ("WENO3+Roe", "weno", "roe"), ("MUSCL+HLLC", "muscl", "hllc"), ("MUSCL+AUSM+", "muscl", "ausm_plus")]:
    solver = EulerSolver(mesh, reconstruction=recon, weno_order=3, flux_scheme=flux, limiter="none", time_method="rk3")
    result = solver.run(q0, n_steps=100, cfl=0.5, report_interval=100)
    res = conservation_check(q0, result["q"], mesh, tol=1e-12)
    results[f"conservation_{name}"] = res
    print(f"  {name}: max_residual={res['max_residual']:.2e}, passed={res['passed']}")

# 3. Isentropic vortex convergence (criterion b, c)
print("\n=== Isentropic Vortex Convergence ===")
for recon_name, recon, weno_order in [("MUSCL", "muscl", 3), ("WENO-3", "weno", 3), ("WENO-5", "weno", 5)]:
    errors = []
    for nx in [4, 8, 16]:
        m = make_triangular_mesh_uniform(nx, nx, lx=10.0, ly=10.0)
        q0_v = isentropic_vortex_2d(m, M=0.5, beta=5.0)
        solver = EulerSolver(m, reconstruction=recon, weno_order=weno_order, flux_scheme="roe", time_method="rk3")
        result = solver.run(q0_v, n_steps=50, cfl=0.4, report_interval=50)
        err = float(torch.sqrt(((result["q"] - q0_v)**2).mean()).item())
        errors.append(err)
    order = math.log(errors[0] / errors[1]) / math.log(2) if errors[1] > 0 else 0
    results[f"vortex_{recon_name}"] = {"errors": errors, "order": order}
    print(f"  {recon_name}: errors={[f'{e:.4f}' for e in errors]}, order={order:.2f}")

# 4. Sod shock tube (criterion d)
print("\n=== Sod Shock Tube ===")
for recon_name, recon, weno_order in [("MUSCL", "muscl", 3), ("WENO-5", "weno", 5)]:
    m = make_triangular_mesh_uniform(32, 1, lx=1.0, ly=0.01)
    q0_s = sod_shocktube_1d(64)
    q0_s[:, 2] = 0.0
    solver = EulerSolver(m, reconstruction=recon, weno_order=weno_order, flux_scheme="roe", time_method="rk2")
    result = solver.run(q0_s, n_steps=100, cfl=0.4, report_interval=100)
    rho = result["q"][:, 0]
    results[f"sod_{recon_name}"] = {"rho_max": float(rho.max().item()), "rho_min": float(rho.min().item())}
    print(f"  {recon_name}: rho_max={rho.max():.3f}, rho_min={rho.min():.3f}")

# 5. Efficiency (criterion f)
print("\n=== Efficiency Comparison ===")
m = make_triangular_mesh_uniform(16, 16, lx=1.0, ly=1.0)
q0_e = isentropic_vortex_2d(m, M=0.5, beta=5.0)

solver_m = EulerSolver(m, reconstruction="muscl", flux_scheme="roe", limiter="venkatakrishnan", time_method="rk3")
result_m = solver_m.run(q0_e, n_steps=50, cfl=0.4, report_interval=50)

solver_w = EulerSolver(m, reconstruction="weno", weno_order=5, flux_scheme="roe", time_method="rk3")
result_w = solver_w.run(q0_e, n_steps=50, cfl=0.4, report_interval=50)

ratio = result_w["cpu_time"] / result_m["cpu_time"] if result_m["cpu_time"] > 0 else 0
results["efficiency"] = {"muscl_cpu": result_m["cpu_time"], "weno5_cpu": result_w["cpu_time"], "ratio": ratio, "passed": ratio < 3}
print(f"  MUSCL: {result_m['cpu_time']:.2f}s, WENO-5: {result_w['cpu_time']:.2f}s, ratio: {ratio:.2f}, passed: {ratio < 3}")

# Save results
import json
with open("runs/high_order_fvm_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to runs/high_order_fvm_results.json")
