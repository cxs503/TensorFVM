import matplotlib
matplotlib.use("Agg")
import time
import lid_driven_cavity_fvm as M

if __name__ == "__main__":
    t0 = time.time()
    # 标准方案: 掩码投影 + Neumann 压力 + 内部面投影 (模块设计/稳健基准)
    u, v, p, psi, omega, h, ke, solid, stats, forces = M.solve_cylinder(
        Nx=120, Ny=48, Re=100.0, nsteps=9000)
    print(f"[solve_cylinder 标准方案] 完成 {time.time()-t0:.0f}s")
    stats = M.postprocess_cylinder(u, v, p, psi, omega, h, solid, stats, forces,
                                   100.0, 0.2, 2.5, 1.0)
    print("FINAL_STATS", stats)
