#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方腔流 (lid-driven cavity flow) 有限体积法 (FVM) 求解器  ——  基于 torch 线性求解器调研

数值方法
--------
- 交错网格 (MAC grid)：u 定义在垂直面 (i+1/2,j)，v 在水平面 (i,j+1/2)，p 在单元中心。
- 时间推进：Chorin 投影法 (一阶)
    1) 动量显式更新 (迎风对流 + 中心扩散)  -> 中间速度 u*, v*
    2) 解压力泊松方程  ∇²p = (1/dt)·∇·u*   (Neumann 边界, 固定一点去奇异)
    3) 投影  u = u* - dt·∇p ,  v = v* - dt·∇p
- 后处理：流函数 ψ 由 ∇²ψ = -ω 解出 (Dirichlet 边界)，用于画流线与定位主涡。

与《torch 线性求解器调研》的对应
----------------------------------
- 压力泊松 / 流函数方程的系数矩阵在网格固定时【只构造、只 LU 分解一次】
  (torch.linalg.lu_factor_ex -> cuSOLVER getrf, 见文档第一节)，
  每步仅用 torch.linalg.lu_solve (-> getrs, 文档第一/二节) 做 O(n²) 回代。
- 可选 use_sparse=True 走 torch.sparse.spsolve (-> cuDSS, 文档第三节) 的稀疏 CSR 路径；
  无 CUDA/cuDSS 编译时自动回退到稠密 LU (torch.linalg.solve, 文档第二节)。

案例
----
- `cavity`  (默认): 方腔流, 单位正方形域, 顶盖驱动。
- `cylinder`: 圆柱绕流, 矩形渠道 (长 Lx, 高 Ly), 左侧均匀来流 U 入口, 右侧零梯度出口,
  上下滑移壁面, 内部一个无滑移圆柱障碍 (solid mask)。圆柱作为零速度障碍区, 复用与 cavity
  同源的全域压力泊松 (Neumann) 求解器 (固定一点去奇异), 投影后把固体面速度钳为 0 —— 比
  「掩码降维系统」更稳健且代码路径与已校验的 cavity 完全一致。显式格式条件稳定, 未指定
  --dt 时按 CFL 自适应选取 (近圆柱峰值速度 ≈ 2U), 定量输出涡脱落 Strouhal 数 St 与升/阻力
  系数 (基于阶梯化固体边界的压力积分近似)。

运行:
    python3 lid_driven_cavity_fvm.py --case cavity  --N 64 --Re 100 --nsteps 6000
    python3 lid_driven_cavity_fvm.py --case cylinder --Nx 120 --Ny 48 --Re 100 \
           --nsteps 9000          # dt 自适应 (≈0.003); 也可 --dt 0.003 显式指定
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------
# 1. 系数矩阵构造
# ----------------------------------------------------------------------------
def build_pressure_system(Nx, Ny, h):
    """压力泊松系数矩阵 (Neumann 边界, 固定 p[0,0]=0 去奇异)。对称正定 M-矩阵。
    支持矩形域 (Nx×Ny); 方阵时 Nx=Ny=N。向量化构造 (避免 Python 三重循环)。"""
    n = Nx * Ny
    A = torch.zeros(n, n, dtype=torch.float64)
    idx = lambda i, j: i * Ny + j
    I = torch.arange(Nx); J = torch.arange(Ny)
    ig, jg = torch.meshgrid(I, J, indexing="ij")
    i_f = ig.reshape(-1); j_f = jg.reshape(-1)
    r = (i_f * Ny + j_f).long()
    inv_h2 = -1.0 / h**2
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni = i_f + di; nj = j_f + dj
        valid = (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
        ni_v = ni[valid].long(); nj_v = nj[valid].long()
        A[r[valid], ni_v * Ny + nj_v] = inv_h2
    deg = ((i_f >= 1).long() + (i_f < Nx - 1).long() +
           (j_f >= 1).long() + (j_f < Ny - 1).long()).double()
    A[r, r] = deg / h**2   # 对角 = 邻居数 / h²  (Neumann 镜像已并入对角)
    fix = idx(0, 0)
    A[fix, :] = 0.0
    A[fix, fix] = 1.0
    return A


def build_dirichlet_system(Nx, Ny, h):
    """流函数泊松系数矩阵 (Dirichlet ψ=0 边界)。内部 4 邻居, 边界行设单位。
    支持矩形域 (Nx×Ny); 方阵时 Nx=Ny=N。向量化构造 (避免 Python 三重循环)。"""
    n = Nx * Ny
    A = torch.zeros(n, n, dtype=torch.float64)
    I = torch.arange(Nx); J = torch.arange(Ny)
    ig, jg = torch.meshgrid(I, J, indexing="ij")
    i_f = ig.reshape(-1); j_f = jg.reshape(-1)
    r = (i_f * Ny + j_f).long()
    boundary = (i_f == 0) | (i_f == Nx - 1) | (j_f == 0) | (j_f == Ny - 1)
    inv_h2 = -1.0 / h**2
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni = i_f + di; nj = j_f + dj
        valid = (ni >= 0) & (ni < Nx) & (nj >= 0) & (nj < Ny)
        ni_v = ni[valid].long(); nj_v = nj[valid].long()
        A[r[valid], ni_v * Ny + nj_v] = inv_h2
    deg = ((i_f >= 1).long() + (i_f < Nx - 1).long() +
           (j_f >= 1).long() + (j_f < Ny - 1).long()).double()
    A[r, r] = deg / h**2
    bidx = r[boundary]
    A[bidx, :] = 0.0
    A[bidx, bidx] = 1.0   # 边界 Dirichlet ψ=0 (整行清零, 仅对角=1)
    return A


def to_csr(A):
    """稠密矩阵 -> CSR (用于 torch.sparse.spsolve, 文档第三节 cuDSS 路径)。"""
    n = A.shape[0]
    rows, cols, vals = [], [], []
    for r in range(n):
        for c in range(n):
            if A[r, c] != 0:
                rows.append(r); cols.append(c); vals.append(A[r, c])
    crow = torch.zeros(n + 1, dtype=torch.int64)
    for r in range(n):
        crow[r + 1] = crow[r] + int((A[r, :] != 0).sum())
    return (crow, torch.tensor(cols, dtype=torch.int64),
            torch.tensor(vals, dtype=torch.float64), (n, n))


# ----------------------------------------------------------------------------
# 2. 动量显式更新 (迎风对流 + 中心扩散)
# ----------------------------------------------------------------------------
def momentum(u, v, nu, h, dt, U):
    u_new = u.clone()
    v_new = v.clone()
    N = u.shape[0] - 1
    # 用壁面值补齐, 使最上/最外邻居索引合法 (顶盖 u=U, 顶壁 v=0)
    uP = torch.cat([u, torch.full((N + 1, 1), U, dtype=u.dtype)], dim=1)   # (N+1, N+1)
    vP = torch.cat([v, torch.zeros(1, N + 1, dtype=v.dtype)], dim=0)      # (N+1, N+1)

    # --- u 方程: 内部竖直面 i=1..N-1, j=1..N-2 ---
    uu = u[1:N, 1:N - 1]
    dudx = torch.where(uu >= 0,
                       (u[1:N, 1:N - 1] - u[0:N - 1, 1:N - 1]) / h,
                       (u[2:N + 1, 1:N - 1] - u[1:N, 1:N - 1]) / h)
    dudy = (u[1:N, 2:N] - u[1:N, 0:N - 2]) / (2 * h)
    vcoef = 0.25 * (vP[1:N, 1:N - 1] + vP[2:N + 1, 1:N - 1] +
                    vP[1:N, 2:N] + vP[2:N + 1, 2:N])
    lapu = ((u[2:N + 1, 1:N - 1] - 2 * u[1:N, 1:N - 1] + u[0:N - 1, 1:N - 1]) +
            (u[1:N, 2:N] - 2 * u[1:N, 1:N - 1] + u[1:N, 0:N - 2])) / h**2
    u_new[1:N, 1:N - 1] = u[1:N, 1:N - 1] + dt * (-(uu * dudx + vcoef * dudy) + nu * lapu)

    # --- v 方程: 内部水平面 i=1..N-2, j=1..N-1 ---
    vv = v[1:N - 1, 1:N]
    dvdy = torch.where(vv >= 0,
                       (v[1:N - 1, 1:N] - v[1:N - 1, 0:N - 1]) / h,
                       (v[1:N - 1, 2:N + 1] - v[1:N - 1, 1:N]) / h)
    dvdx = (v[2:N, 1:N] - v[0:N - 2, 1:N]) / (2 * h)
    ucoef = 0.25 * (uP[1:N - 1, 1:N] + uP[2:N, 1:N] +
                    uP[1:N - 1, 2:N + 1] + uP[2:N, 2:N + 1])
    lapv = ((v[2:N, 1:N] - 2 * v[1:N - 1, 1:N] + v[0:N - 2, 1:N]) +
            (v[1:N - 1, 2:N + 1] - 2 * v[1:N - 1, 1:N] + v[1:N - 1, 0:N - 1])) / h**2
    v_new[1:N - 1, 1:N] = v[1:N - 1, 1:N] + dt * (-(ucoef * dvdx + vv * dvdy) + nu * lapv)

    # 边界条件
    lid = lid_profile(N, h, U)
    u_new[:, 0] = 0.0;  u_new[:, N - 1] = lid   # 底=0, 顶盖=U(平滑)
    u_new[0, :] = 0.0;  u_new[N, :] = 0.0       # 左/右壁=0
    v_new[:, 0] = 0.0;  v_new[:, N] = 0.0
    v_new[0, :] = 0.0;  v_new[N - 1, :] = 0.0
    return u_new, v_new


def lid_profile(N, h, U, ramp=0.08):
    """顶盖速度沿 x 的剖面: 主体均匀 U, 两端平滑过渡到 0 (消除角点奇异性)。"""
    x = (torch.arange(N + 1, dtype=torch.float64) + 0.5) * h
    prof = torch.ones_like(x)
    prof = torch.where(x < ramp, x / ramp, prof)
    prof = torch.where(x > 1 - ramp, (1 - x) / ramp, prof)
    return U * prof


def enforce_bc(u, v, U, h=None, N=None):
    if N is None:
        N = u.shape[0] - 1
    if h is None:
        h = 1.0 / N
    lid = lid_profile(N, h, U)
    u[:, 0] = 0.0;  u[:, N - 1] = lid
    u[0, :] = 0.0;  u[N, :] = 0.0
    v[:, 0] = 0.0;  v[:, N] = 0.0
    v[0, :] = 0.0;  v[N - 1, :] = 0.0
    return u, v


# ----------------------------------------------------------------------------
# 3. 主求解循环
# ----------------------------------------------------------------------------
def solve_cavity(N=64, Re=100.0, U=1.0, dt=0.004, nsteps=6000, use_sparse=False):
    h = 1.0 / N
    nu = U * 1.0 / Re
    n = N * N
    idx = lambda i, j: i * N + j

    u = torch.zeros(N + 1, N, dtype=torch.float64)
    v = torch.zeros(N, N + 1, dtype=torch.float64)
    p = torch.zeros(N, N, dtype=torch.float64)

    # 系数矩阵: 只构造、只分解一次
    A_p = build_pressure_system(N, N, h)
    A_psi = build_dirichlet_system(N, N, h)

    if use_sparse:
        crow, col, val, shape = to_csr(A_p)
        A_csr = torch.sparse_csr_tensor(crow, col, val, shape)
        LU = pivots = None
        print("[solver] 稀疏路径 torch.sparse.spsolve (cuDSS, 文档第三节)")
    else:
        LU, pivots, _ = torch.linalg.lu_factor_ex(A_p)   # getrf, 文档第一节
        print(f"[solver] 稠密 LU  torch.linalg.lu_factor_ex + lu_solve (文档第一/二节), n={n}")

    ke = []
    for step in range(1, nsteps + 1):
        u_star, v_star = momentum(u, v, nu, h, dt, U)

        # 散度 (单元中心) -> 压力方程右端
        div = ((u_star[1:, :] - u_star[:-1, :]) +
               (v_star[:, 1:] - v_star[:, :-1])) / h
        rhs = div.reshape(-1) / dt
        rhs[idx(0, 0)] = 0.0

        # 解压力 (每步仅回代)
        if use_sparse:
            try:
                p_flat = torch.sparse.spsolve(A_csr, rhs)
            except Exception as e:
                # CPU 无 cuDSS 后端时 spsolve 抛 NotImplementedError -> 回退稠密 LU
                if LU is None:
                    LU, pivots, _ = torch.linalg.lu_factor_ex(A_p)
                p_flat = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(1)).squeeze(1)
        else:
            p_flat = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(1)).squeeze(1)   # getrs, 文档第一/二节
        p = p_flat.reshape(N, N)

        # 投影 (只更新内部面; 本矩阵约定下用 + 号使内部散度归零, p 有界)
        u_star[1:N, 1:N - 1] += dt * (p[1:, :] - p[:-1, :])[:, 1:N - 1] / h
        v_star[1:N - 1, 1:N] += dt * (p[:, 1:] - p[:, :-1])[1:N - 1, :] / h
        u, v = enforce_bc(u_star.clone(), v_star.clone(), U, h, N)

        ke.append(float((u**2).sum() + (v**2).sum()) / 2)
        if step % 1000 == 0 or step == 1:
            print(f"  step {step:5d}  KE={ke[-1]:.6e}  umax={float(u.abs().max()):.4f}")

    # 流函数 (后处理, 复用线性求解器)
    # 单元中心速度 (collocated): MAC 面速度取相邻面平均
    u_cell = 0.5 * (u[:N, :N] + u[1:, :N])
    v_cell = 0.5 * (v[:N, :N] + v[:N, 1:])
    # 涡量 ω = ∂v/∂x - ∂u/∂y (单元中心, 中心差分; 边界仅用于绘图)
    uc, vc = u_cell.numpy(), v_cell.numpy()
    gx_v, _ = np.gradient(vc, h, h)
    _, gy_u = np.gradient(uc, h, h)
    omega = torch.tensor(gx_v - gy_u, dtype=torch.float64)
    # Dirichlet 泊松: 系数矩阵 A_psi = -∇² (diag +4/h², 邻居 -1/h²), 故 A_psi·ψ = ω,
    # 即 ∇²ψ = -ω (标准流函数-涡量关系, 与 ω = ∂v/∂x - ∂u/∂y 自洽)。
    # 边界 ψ=0 已并入矩阵 (整行单位), 右端项边界分量置零。
    rhs_psi = omega.reshape(-1)
    # Dirichlet ψ=0: 边界单元右端项必须置零 (否则 ψ 被钉在 -ω_边界 上)
    idx = lambda i, j: i * N + j
    for j in range(N):
        for i in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                rhs_psi[idx(i, j)] = 0.0
    LU_psi, piv_psi, _ = torch.linalg.lu_factor_ex(A_psi)
    psi = torch.linalg.lu_solve(LU_psi, piv_psi, rhs_psi.unsqueeze(1)).squeeze(1).reshape(N, N)
    return u, v, p, psi, omega, h, ke


# ----------------------------------------------------------------------------
# 4. 后处理与可视化
# ----------------------------------------------------------------------------
def postprocess(u, v, p, psi, omega, h, N, Re):
    # 主涡中心 = ψ 最小点  (psi 展平索引 k = i*N + j, i 为 x 索引, j 为 y 索引)
    k = int(psi.argmin())
    ci, cj = divmod(k, N)
    x_c, y_c = (ci + 0.5) * h, (cj + 0.5) * h
    psi_min = float(psi.min())

    # Ghia et al. (1982) Re=100 基准
    ghia = {"x": 0.6172, "y": 0.7343, "psi": -0.103423}
    print("\n==== 主涡 (Primary vortex) ====")
    print(f"  本程序 : 中心 ({x_c:.4f}, {y_c:.4f})  ψ_min = {psi_min:.6f}")
    if abs(Re - 100) < 1e-6:
        print(f"  Ghia基准: 中心 ({ghia['x']}, {ghia['y']})  ψ_min = {ghia['psi']}")
        print(f"  偏差    : Δx={x_c-ghia['x']:+.4f}  Δy={y_c-ghia['y']:+.4f}")

    # 速度大小 (单元中心, 用相邻面平均)
    u_cell = 0.5 * (u[:N, :N] + u[1:, :N])
    v_cell = 0.5 * (v[:N, :N] + v[:N, 1:])
    speed = torch.sqrt(u_cell**2 + v_cell**2)

    X, Y = np.meshgrid(np.linspace(h / 2, 1 - h / 2, N),
                       np.linspace(h / 2, 1 - h / 2, N), indexing="ij")

    # 图1: 流函数等值线 (流线) + 主涡
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    cs = ax[0].contour(X, Y, psi, levels=30, cmap="RdYlBu_r")
    ax[0].clabel(cs, inline=True, fontsize=6)
    ax[0].plot(x_c, y_c, "k*", ms=12, label=f"Primary vortex ({x_c:.2f},{y_c:.2f})")
    ax[0].set_title(f"Streamfunction ψ (Streamlines), Re={Re:.0f}")
    ax[0].set_aspect("equal"); ax[0].legend(loc="upper right")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

    im = ax[1].contourf(X, Y, speed, levels=40, cmap="viridis")
    fig.colorbar(im, ax=ax[1], label="|U|")
    ax[1].set_title("|Velocity| |U|"); ax[1].set_aspect("equal")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    fig.tight_layout()
    fig.savefig("/workspace/cavity_streamfunction.png", dpi=130)
    plt.close(fig)

    # 图2: 涡量云图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(X, Y, omega, levels=40, cmap="seismic")
    fig.colorbar(im, ax=ax, label="ω = ∂v/∂x − ∂u/∂y")
    ax.set_title(f"Vorticity ω (Re={Re:.0f})")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig("/workspace/cavity_vorticity.png", dpi=130)
    plt.close(fig)

    # 中线速度剖面 (对比 Ghia et al. 1982, Re=100 基准)
    mid_i = N // 2
    u_vert = u[mid_i, :].numpy()          # 沿 x=0.5 竖线的 u
    mid_j = N // 2
    v_horiz = v[:, mid_j].numpy()         # 沿 y=0.5 横线的 v
    yv = (np.arange(N) + 0.5) * h      # u 竖直线的 y 坐标 (单元中心)
    xh = (np.arange(N) + 0.5) * h      # v 水平线的 x 坐标
    # Ghia et al. (1982) Table I/II 基准 (Re=100)
    # u 沿竖线 x=0.5: (y, u)
    ghia_yu = [(0.0000,0.00000),(0.0547,-0.03717),(0.0625,-0.04192),(0.0703,-0.04775),
               (0.1016,-0.06434),(0.1719,-0.10150),(0.2813,-0.15662),(0.4531,-0.21090),
               (0.5000,-0.20581),(0.6172,-0.13641),(0.7344,0.00332),(0.8516,0.23151),
               (0.9531,0.68717),(0.9609,0.73722),(0.9688,0.78871),(0.9766,0.84123),
               (1.0000,1.00000)]
    # v 沿横线 y=0.5: (x, v)
    ghia_xv = [(0.0000,0.00000),(0.0625,0.09233),(0.0703,0.10091),(0.0781,0.10890),
               (0.0938,0.12317),(0.1563,0.16077),(0.2266,0.17507),(0.2344,0.17527),
               (0.5000,0.05454),(0.8047,-0.24533),(0.8594,-0.22445),(0.9063,-0.16914),
               (0.9453,-0.10313),(0.9531,-0.08864),(0.9609,-0.07391),(0.9688,-0.05906),
               (1.0000,0.00000)]
    gy, gu = zip(*ghia_yu); gx, gv = zip(*ghia_xv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(u_vert, yv, "b-", lw=2, label="FVM (this work)")
    ax[0].plot(gu, gy, "ko", ms=4, label="Ghia et al. (1982)")
    ax[0].set_xlabel("u"); ax[0].set_ylabel("y"); ax[0].set_title("u profile (x=0.5)")
    ax[0].grid(True); ax[0].legend()
    ax[1].plot(xh, v_horiz, "r-", lw=2, label="FVM (this work)")
    ax[1].plot(gx, gv, "ko", ms=4, label="Ghia et al. (1982)")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("v"); ax[1].set_title("v profile (y=0.5)")
    ax[1].grid(True); ax[1].legend()
    fig.tight_layout()
    fig.savefig("/workspace/cavity_profiles.png", dpi=130)
    plt.close(fig)

    # 定量对比: 在 Ghia 采样点线性插值本程序结果
    gy, gu = np.array(gy), np.array(gu); gx, gv = np.array(gx), np.array(gv)
    u_pred = np.interp(gy, yv, u_vert)
    v_pred = np.interp(gx, xh, v_horiz)
    eu = np.abs(u_pred - gu); ev = np.abs(v_pred - gv)
    print("\n==== 中线速度剖面 vs Ghia et al. (1982), Re=100 ====")
    print(f"  u(x=0.5):  max|Δ|={eu.max():.4f}   RMSE={np.sqrt((eu**2).mean()):.4f}")
    print(f"  v(y=0.5):  max|Δ|={ev.max():.4f}   RMSE={np.sqrt((ev**2).mean()):.4f}")

    return {"x_c": x_c, "y_c": y_c, "psi_min": psi_min}


# ----------------------------------------------------------------------------
# 5. 圆柱绕流 (flow past a circular cylinder) —— 掩码投影法 (masked projection)
# ----------------------------------------------------------------------------
def cylinder_mask(Nx, Ny, h, cx_frac=0.25, cy_frac=0.5, D=0.2, Lx=2.5, Ly=1.0):
    """返回 (solid, fluid) 布尔张量 (Nx,Ny); solid=True 表示圆柱内部。"""
    cx, cy = cx_frac * Lx, cy_frac * Ly
    R = 0.5 * D
    xs = (torch.arange(Nx, dtype=torch.float64) + 0.5) * h
    ys = (torch.arange(Ny, dtype=torch.float64) + 0.5) * h
    Xc, Yc = torch.meshgrid(xs, ys, indexing="ij")
    solid = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= R ** 2
    fluid = ~solid
    return solid, fluid


def build_pressure_system_masked(Nx, Ny, h, fluid):
    """掩码压力泊松系数矩阵 (仅流体单元), 固体邻居按 Neumann 略去 (A=+∇²)。
    返回 dense A (n_f, n_f) 与 fluid 单元列表 flist (n_f,2)。固定一个流体单元去奇异。"""
    flist = fluid.nonzero(as_tuple=False)            # (n_f, 2) in (i,j)
    n_f = flist.shape[0]
    fmap = -torch.ones(Nx, Ny, dtype=torch.long)
    fmap[flist[:, 0], flist[:, 1]] = torch.arange(n_f)
    A = torch.zeros(n_f, n_f, dtype=torch.float64)
    for k in range(n_f):
        i, j = int(flist[k, 0]), int(flist[k, 1])
        diag = 0.0
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and bool(fluid[ni, nj]):
                A[k, int(fmap[ni, nj])] = -1.0 / h ** 2
                diag += 1.0 / h ** 2
        A[k, k] = diag
    # 纯 Neumann -> 常数压力为 null space, 固定一个流体单元
    A[0, :] = 0.0
    A[0, 0] = 1.0
    return A, flist


def build_dirichlet_system_masked(Nx, Ny, h, fluid):
    """掩码流函数泊松系数矩阵: A_psi = -∇² (diag +deg/h², 邻居 -1/h²)。
    与圆柱固体表面及域壁相邻的流体单元设 Dirichlet ψ=0 (整行单位)。"""
    flist = fluid.nonzero(as_tuple=False)
    n_f = flist.shape[0]
    fmap = -torch.ones(Nx, Ny, dtype=torch.long)
    fmap[flist[:, 0], flist[:, 1]] = torch.arange(n_f)
    A = torch.zeros(n_f, n_f, dtype=torch.float64)
    solid = ~fluid
    for k in range(n_f):
        i, j = int(flist[k, 0]), int(flist[k, 1])
        # 该流体单元是否触碰固体或域壁 -> Dirichlet ψ=0
        touches = (i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1)
        if not touches:
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < Nx and 0 <= nj < Ny and bool(fluid[ni, nj]):
                    A[k, int(fmap[ni, nj])] = -1.0 / h ** 2
                    A[k, k] += 1.0 / h ** 2
        if touches or bool(solid[i, j]):
            A[k, :] = 0.0
            A[k, k] = 1.0
    return A, flist


def momentum_cyl(u, v, nu, h, dt, U):
    """圆柱绕流动量显式更新 (迎风对流 + 中心扩散); 仅算内部面, BC 由调用方设置。
    上/下为滑移壁面 (∂u/∂y=0 -> 端部 y 导数置 0); 左入口/右出口对 v 做 x 方向填充。"""
    u_new = u.clone()
    v_new = v.clone()
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1

    # ---------- u 方程: 内部面 i=1..Nx-1, j=0..Ny-1 ----------
    uu = u[1:Nx, :]                                                   # (Nx-1, Ny)
    dudx = torch.where(uu >= 0,
                       (u[1:Nx, :] - u[0:Nx - 1, :]) / h,
                       (u[2:Nx + 1, :] - u[1:Nx, :]) / h)             # (Nx-1, Ny)
    # 中心 y 导数 (仅 j=1..Ny-2 有效, 端部 j=0,Ny-1 滑移置 0)
    dudy_int = (u[1:Nx, 2:Ny] - u[1:Nx, 0:Ny - 2]) / (2 * h)         # (Nx-1, Ny-2)
    dudy = torch.zeros(Nx - 1, Ny, dtype=u.dtype)
    dudy[:, 1:Ny - 1] = dudy_int
    vcoef = 0.25 * (v[0:Nx - 1, 0:Ny] + v[1:Nx, 0:Ny] +
                    v[0:Nx - 1, 1:Ny + 1] + v[1:Nx, 1:Ny + 1])       # (Nx-1, Ny)
    lapx = (u[2:Nx + 1, :] - 2 * u[1:Nx, :] + u[0:Nx - 1, :]) / h ** 2       # (Nx-1, Ny)
    lapy_int = (u[1:Nx, 2:Ny] - 2 * u[1:Nx, 1:Ny - 1] +
                u[1:Nx, 0:Ny - 2]) / h ** 2                          # (Nx-1, Ny-2)
    lapy = torch.zeros(Nx - 1, Ny, dtype=u.dtype)
    lapy[:, 1:Ny - 1] = lapy_int
    u_new[1:Nx, :] = u[1:Nx, :] + dt * (-(uu * dudx + vcoef * dudy) + nu * (lapx + lapy))

    # ---------- v 方程: 内部面 j=1..Ny-1, i=0..Nx-1 ----------
    vv = v[:, 1:Ny]                                                   # (Nx, Ny-1)
    dvdy = torch.where(vv >= 0,
                       (v[:, 1:Ny] - v[:, 0:Ny - 1]) / h,
                       (v[:, 2:Ny + 1] - v[:, 1:Ny]) / h)             # (Nx, Ny-1) 迎风
    # x 方向填充 (入口 v=0 镜像, 出口零梯度), 用于中心 x 导数/拉普拉斯
    v_x = torch.cat([v[0:1, :], v, v[Nx - 1:Nx, :]], dim=0)          # (Nx+2, Ny+1)
    dvdx = (v_x[2:Nx + 2, 1:Ny] - v_x[0:Nx, 1:Ny]) / (2 * h)        # (Nx, Ny-1) 中心
    ucoef = 0.25 * (u[0:Nx, 0:Ny - 1] + u[1:Nx + 1, 0:Ny - 1] +
                    u[0:Nx, 1:Ny] + u[1:Nx + 1, 1:Ny])               # (Nx, Ny-1)
    lapx_v = (v_x[2:Nx + 2, 1:Ny] - 2 * v_x[1:Nx + 1, 1:Ny] +
              v_x[0:Nx, 1:Ny]) / h ** 2                              # (Nx, Ny-1)
    lapy_v = (v[:, 2:Ny + 1] - 2 * v[:, 1:Ny] + v[:, 0:Ny - 1]) / h ** 2   # (Nx, Ny-1)
    v_new[:, 1:Ny] = v[:, 1:Ny] + dt * (-(ucoef * dvdx + vv * dvdy) + nu * (lapx_v + lapy_v))
    return u_new, v_new


def enforce_bc_cyl(u, v, U, solid):
    """圆柱绕流边界条件: 左入口 u=U,v=0; 右出口零梯度; 上下滑移; 固体=0。"""
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    # 入口
    u[0, :] = U
    v[0, :] = 0.0
    # 出口零梯度 (u 出口面 = u[Nx,:]; v 出口列 = v[Nx-1,:])
    u[Nx, :] = u[Nx - 1, :]
    v[Nx - 1, :] = v[Nx - 2, :]
    # 上下滑移壁面: v=0 (无穿透), u 切向零梯度
    v[:, 0] = 0.0
    v[:, Ny] = 0.0
    u[:, 0] = u[:, 1]
    u[:, Ny - 1] = u[:, Ny - 2]
    # 固体 (无滑移): u 多一列面, 用显式掩码
    umask = torch.zeros((Nx + 1, Ny), dtype=torch.bool)
    umask[:Nx] = solid
    u[umask] = 0.0
    vmask = torch.zeros((Nx, Ny + 1), dtype=torch.bool)
    vmask[:, :Ny] = solid
    v[vmask] = 0.0
    return u, v


def solve_cylinder(Nx=120, Ny=48, Re=100.0, U=1.0, dt=None, nsteps=9000,
                  use_sparse=False, cx_frac=0.25, cy_frac=0.5, D=0.2,
                  Lx=2.5, Ly=1.0, ramp=150, probe_every=1):
    h = Ly / Ny
    Nx = int(round(Lx / h))          # 保证 h 各向同性
    nu = U * D / Re                  # 以圆柱直径 D 定义 Re

    # 自适应稳定时间步 (显式一阶迎风 + 中心扩散的条件稳定):
    #   S = dt·( |u|_peak/h + 4·ν/h² ) ≤ S_target  —— 近圆柱峰值速度 ≈ 2U
    # 未显式给定 dt 时按此选取, 使粗细网格都落在稳定区内 (Nx=120/原 dt=0.008 会越界发散)。
    if dt is None:
        S_target = 0.5
        umax_peak = 2.0 * U
        S_num = umax_peak / h + 4.0 * nu / h**2
        dt = S_target / S_num
    solid, fluid = cylinder_mask(Nx, Ny, h, cx_frac, cy_frac, D, Lx, Ly)

    # 从静止起步 (u=0), 入口速度缓启动, 避免初始均匀流与缓启动入口的速度冲击
    u = torch.zeros((Nx + 1, Ny), dtype=torch.float64)
    v = torch.zeros((Nx, Ny + 1), dtype=torch.float64)
    u, v = enforce_bc_cyl(u, v, U * min(1.0, 1.0 / ramp), solid)

    # 掩码投影的压力/流函数系统 (仅流体单元, 固体邻居按 Neumann 略去):
    #   —— 比「全域系统 + 零速度障碍」稳健: 后者把固体当流体求解压力, 在固壁附近
    #      产生虚假压力梯度并泄漏进流体, 细网格下表现为持续能量注入 (umax→非物理)。
    flist = fluid.nonzero(as_tuple=False)            # (n_f, 2) 流体单元坐标
    n_f = flist.shape[0]
    A_p, _ = build_pressure_system_masked(Nx, Ny, h, fluid)       # 仅流体单元 Neumann 泊松
    A_psi, flist_psi = build_dirichlet_system_masked(Nx, Ny, h, fluid)  # 掩码 Dirichlet 流函数

    if use_sparse:
        # 圆柱大稀疏系统 (文档第四节) -> 转 CSR 走 torch.sparse.spsolve (cuDSS)
        # CPU 无 cuDSS 编译时自动回退稠密 LU (torch.linalg.solve, 文档第二节)。
        crow, col, val, shape = to_csr(A_p)
        A_csr = torch.sparse_csr_tensor(crow, col, val, shape)
        LU = pivots = None
        print(f"[solver] 稀疏路径 torch.sparse.spsolve (cuDSS, 文档第三节), n={n_f}")
    else:
        LU, pivots, _ = torch.linalg.lu_factor_ex(A_p)   # getrf, 文档第一节
        print(f"[solver] 稠密 LU torch.linalg.lu_factor_ex + lu_solve (文档第一/二节), n={n_f}")
    LU_psi, piv_psi, _ = torch.linalg.lu_factor_ex(A_psi)  # 流函数一次分解

    # 固体边界面掩码 (异或: 恰一侧为固体) -> 投影后钳 0 (无滑移/无穿透)
    left_f = torch.zeros(Nx + 1, Ny, dtype=torch.bool)
    right_f = torch.zeros(Nx + 1, Ny, dtype=torch.bool)
    left_f[1:Nx, :] = solid[0:Nx - 1, :]
    right_f[1:Nx, :] = solid[1:Nx, :]
    u_solidface = (left_f ^ right_f)
    bot_f = torch.zeros(Nx, Ny + 1, dtype=torch.bool)
    top_f = torch.zeros(Nx, Ny + 1, dtype=torch.bool)
    bot_f[:, 1:Ny] = solid[:, 0:Ny - 1]
    top_f[:, 1:Ny] = solid[:, 1:Ny]
    v_solidface = (bot_f ^ top_f)

    # 升/阻力 (阶梯化边界, 近似) 与尾流横向速度探针
    px = int((cx_frac * Lx + 1.0 * D) / h)   # 圆柱下游约 1D 处
    py = int(cy_frac * Ly / h)
    px = min(max(px, 1), Nx - 1)
    cl_hist, cd_hist, vprobe = [], [], []

    ke = []
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)      # 入口速度缓启动
        u_star, v_star = momentum_cyl(u, v, nu, h, dt, Ueff)
        u_star, v_star = enforce_bc_cyl(u_star, v_star, Ueff, solid)
        # 无穿透约束: 固体边界面速度钳 0 (掩码投影以之为边界值, 不变量)
        u_star[u_solidface] = 0.0
        v_star[v_solidface] = 0.0

        # 散度 (全集, 固体面已钳 0 -> 等价于仅流体单元散度) -> 取流体单元作右端
        div = ((u_star[1:, :] - u_star[:-1, :]) +
               (v_star[:, 1:] - v_star[:, :-1])) / h      # (Nx, Ny)
        rhs = div[flist[:, 0], flist[:, 1]] / dt           # 仅流体单元

        if use_sparse:
            try:
                p_flat = torch.sparse.spsolve(A_csr, rhs)
            except Exception:
                if LU is None:
                    LU, pivots, _ = torch.linalg.lu_factor_ex(A_p)
                p_flat = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(1)).squeeze(1)
        else:
            p_flat = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(1)).squeeze(1)
        p_full = torch.zeros(Nx, Ny, dtype=torch.float64)
        p_full[flist[:, 0], flist[:, 1]] = p_flat

        # 投影 (仅内部面, 与 cavity 同号约定 + ; 固体面随后钳 0)
        Gx = (p_full[1:, :] - p_full[:-1, :]) / h          # (Nx-1, Ny)
        Gy = (p_full[:, 1:] - p_full[:, :-1]) / h          # (Nx, Ny-1)
        u_star[1:Nx, :] += dt * Gx
        v_star[:, 1:Ny] += dt * Gy
        # 固体边界面速度钳 0 (无滑移/无穿透)
        u_star[u_solidface] = 0.0
        v_star[v_solidface] = 0.0
        u, v = enforce_bc_cyl(u_star.clone(), v_star.clone(), Ueff, solid)

        # 升/阻力近似 (压力在阶梯化固体边界积分, n 为圆柱外法线)
        fxp = 0.0
        fyp = 0.0
        for di, dj, ni, nj in ((1, 0, 1, 0), (-1, 0, -1, 0),
                               (0, 1, 0, 1), (0, -1, 0, -1)):
            # 固体单元 (i,j) 的 neighbor (i+di,j+dj) 为流体 -> 面法线指向 (di,dj)
            rolled_fluid = (~solid).roll(-di, 0).roll(-dj, 1)
            s = solid & rolled_fluid
            if bool(s.any()):
                idx = s.nonzero(as_tuple=False)                  # 固体单元坐标
                nf_i = (idx[:, 0] + di).clamp(0, Nx - 1)
                nf_j = (idx[:, 1] + dj).clamp(0, Ny - 1)
                pf = p_full[nf_i, nf_j]                         # 流体邻居处压力
                fxp += float((-pf * ni).sum()) * h
                fyp += float((-pf * nj).sum()) * h
        q = 0.5 * U * U * D
        cd_hist.append(fxp / q if q != 0 else 0.0)
        cl_hist.append(fyp / q if q != 0 else 0.0)
        if step % probe_every == 0:
            vprobe.append(float(v[px, py]))

        ke.append(float((u ** 2).sum() + (v ** 2).sum()) / 2)
        if step % 1000 == 0 or step == 1:
            print(f"  step {step:5d}  KE={ke[-1]:.5e}  umax={float(u.abs().max()):.4f}  "
                  f"Cd={cd_hist[-1]:.3f}  Cl={cl_hist[-1]:+.3f}")

    # ---- 后处理: 流函数 ψ 由 ∇²ψ=-ω 解出 (掩码 Dirichlet, 已在前构造并分解) ----
    u_cell = 0.5 * (u[:Nx, :Ny] + u[1:, :Ny])
    v_cell = 0.5 * (v[:Nx, :Ny] + v[:Nx, 1:])
    uc, vc = u_cell.numpy(), v_cell.numpy()
    gx_v, _ = np.gradient(vc, h, h)
    _, gy_u = np.gradient(uc, h, h)
    omega = torch.tensor(gx_v - gy_u, dtype=torch.float64)
    # 掩码 Dirichlet 流函数: 域壁/固体相邻流体单元 ψ=0 已在矩阵中; 取流体单元右端
    rhs_psi = omega[flist_psi[:, 0], flist_psi[:, 1]].reshape(-1).clone()
    psi_flat = torch.linalg.lu_solve(LU_psi, piv_psi, rhs_psi.unsqueeze(1)).squeeze(1)
    psi = torch.zeros(Nx, Ny, dtype=torch.float64)
    psi[flist_psi[:, 0], flist_psi[:, 1]] = psi_flat
    psi = psi.where(~solid, torch.tensor(float("nan")))   # 固体区不绘制

    stats = estimate_strouhal(vprobe, dt * probe_every, D, U)
    return u, v, p_full, psi, omega, h, ke, solid, stats, (cl_hist, cd_hist)


def estimate_strouhal(vprobe, dt_sample, D, U):
    """由尾流横向速度探针序列估计涡脱落 Strouhal 数 St = f·D/U。"""
    if len(vprobe) < 64:
        return {"St": float("nan"), "f": float("nan"), "T": float("nan")}
    y = np.array(vprobe, dtype=float)
    y = y - np.mean(y)
    # 去掉最初的瞬态 (前 30%)
    cut = int(0.3 * len(y))
    y = y[cut:]
    t = np.arange(len(y)) * dt_sample
    if np.std(y) < 1e-9:
        return {"St": float("nan"), "f": float("nan"), "T": float("nan")}
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=dt_sample)
    peak = np.argmax(np.abs(fft)[1:]) + 1   # 跳过 DC
    f = freqs[peak]
    T = 1.0 / f if f > 0 else float("nan")
    St = f * D / U if U > 0 else float("nan")
    return {"St": St, "f": f, "T": T}


def postprocess_cylinder(u, v, p, psi, omega, h, solid, stats, forces, Re, D, Lx, Ly):
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    u_cell = 0.5 * (u[:Nx, :Ny] + u[1:, :Ny])
    v_cell = 0.5 * (v[:Nx, :Ny] + v[:Nx, 1:])
    speed = torch.sqrt(u_cell ** 2 + v_cell ** 2)
    speed = speed.where(~solid, torch.tensor(float("nan")))

    X, Y = np.meshgrid(np.linspace(h / 2, Lx - h / 2, Nx),
                       np.linspace(h / 2, Ly - h / 2, Ny), indexing="ij")

    # 图1: 涡量云图 (可见涡街)
    om = omega.clone()
    om = om.where(~solid, torch.tensor(float("nan")))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.contourf(X, Y, om, levels=60, cmap="seismic", extend="both")
    fig.colorbar(im, ax=ax, label="ω = ∂v/∂x − ∂u/∂y")
    ax.set_title(f"Vorticity ω (cylinder, Re={Re:.0f})")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig("/workspace/cavity_cylinder_vorticity.png", dpi=130)
    plt.close(fig)

    # 图2: 流线 (ψ 等值线) + 速度大小
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    cs = ax[0].contour(X, Y, psi.numpy(), levels=40, cmap="RdYlBu_r")
    ax[0].contour(X, Y, solid.numpy(), levels=[0.5], colors="k", linewidths=1.5)
    ax[0].set_title(f"Streamfunction ψ (Streamlines), Re={Re:.0f}")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    im2 = ax[1].contourf(X, Y, speed.numpy(), levels=40, cmap="viridis")
    ax[1].contour(X, Y, solid.numpy(), levels=[0.5], colors="k", linewidths=1.5)
    fig.colorbar(im2, ax=ax[1], label="|U|")
    ax[1].set_title("|Velocity| |U|"); ax[1].set_aspect("equal")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    fig.tight_layout()
    fig.savefig("/workspace/cavity_cylinder_stream.png", dpi=130)
    plt.close(fig)

    # 图3: 升/阻力系数时程 + St 标注
    cl_hist, cd_hist = forces
    n = len(cl_hist)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(n), cl_hist, "b-", lw=1, label="C_L")
    ax[0].plot(np.arange(n), cd_hist, "r-", lw=1, label="C_D")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("coefficient")
    ax[0].set_title(f"Lift/Drag (approx., Re={Re:.0f})")
    ax[0].grid(True); ax[0].legend()
    ax[1].plot(np.arange(len(cd_hist))[-min(1500, n):], cd_hist[-min(1500, n):], "r-", lw=1)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("C_D")
    ax[1].set_title(f"Drag, mean C_D≈{np.mean(cd_hist[-min(1500,n):]):.3f}")
    ax[1].grid(True)
    fig.suptitle(f"Strouhal St = {stats['St']:.3f}  (f={stats['f']:.3f}, T={stats['T']:.2f})"
                 if stats["St"] == stats["St"] else "St: n/a")
    fig.tight_layout()
    fig.savefig("/workspace/cavity_cylinder_force.png", dpi=130)
    plt.close(fig)

    print("\n==== 圆柱绕流定量诊断 ====")
    print(f"  Re={Re:.0f} (基于直径 D={D}), 网格 {Nx}x{Ny}, h={h:.4f}")
    if stats["St"] == stats["St"]:
        print(f"  涡脱落 Strouhal St = {stats['St']:.3f}  "
              f"(理论/文献 ~0.16-0.20 @ Re=100)")
        print(f"  脱落频率 f = {stats['f']:.4f},  周期 T = {stats['T']:.2f}")
    else:
        print("  暂未能估计 St (序列过短或尚未起摆)")
    print(f"  平均阻力系数 C_D ≈ {np.mean(cd_hist[-min(1500, n):]):.3f}  "
          f"(文献 ~1.0-1.4 @ Re=100, 阶梯边界近似)")
    print(f"  升力系数 C_L 振幅 ≈ {np.std(cl_hist[-min(1500, n):]):.3f}")
    return stats


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="方腔流 / 圆柱绕流 FVM (torch 线性求解器)")
    ap.add_argument("--case", choices=["cavity", "cylinder"], default="cavity")
    ap.add_argument("--N", type=int, default=64, help="cavity 网格 (Nx=Ny=N)")
    ap.add_argument("--Nx", type=int, default=120, help="cylinder 流向网格")
    ap.add_argument("--Ny", type=int, default=48, help="cylinder 展向网格")
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--D", type=float, default=0.2, help="cylinder 圆柱直径")
    ap.add_argument("--Lx", type=float, default=2.5, help="cylinder 域长")
    ap.add_argument("--Ly", type=float, default=1.0, help="cylinder 域高")
    ap.add_argument("--sparse", action="store_true", help="走 torch.sparse.spsolve (cuDSS) 路径")
    args = ap.parse_args()

    if args.case == "cavity":
        dt = args.dt if args.dt else 0.004
        nsteps = args.nsteps if args.nsteps else 6000
        print(f"== 方腔流 FVM: N={args.N}, Re={args.Re}, dt={dt}, steps={nsteps} ==")
        u, v, p, psi, omega, h, ke = solve_cavity(
            N=args.N, Re=args.Re, U=args.U, dt=dt,
            nsteps=nsteps, use_sparse=args.sparse)
        postprocess(u, v, p, psi, omega, h, args.N, args.Re)
        print("完成。产物: cavity_streamfunction.png / cavity_vorticity.png / cavity_profiles.png")
    else:
        dt = args.dt                       # None -> solve_cylinder 自适应稳定步长
        nsteps = args.nsteps if args.nsteps else 9000
        print(f"== 圆柱绕流 FVM: Nx={args.Nx}, Ny={args.Ny}, Re={args.Re}, "
              f"D={args.D}, dt={'adaptive' if dt is None else dt}, steps={nsteps} ==")
        out = solve_cylinder(Nx=args.Nx, Ny=args.Ny, Re=args.Re, U=args.U, dt=dt,
                             nsteps=nsteps, use_sparse=args.sparse, D=args.D,
                             Lx=args.Lx, Ly=args.Ly)
        u, v, p, psi, omega, h, ke, solid, stats, forces = out
        postprocess_cylinder(u, v, p, psi, omega, h, solid, stats, forces,
                             args.Re, args.D, args.Lx, args.Ly)
        print("完成。产物: cavity_cylinder_vorticity.png / cavity_cylinder_stream.png "
              "/ cavity_cylinder_force.png")


if __name__ == "__main__":
    main()
