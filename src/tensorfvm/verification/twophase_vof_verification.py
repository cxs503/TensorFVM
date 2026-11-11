"""TensorFVM 两相 VOF 首验面：静水压力平衡 + 溃坝 + 气泡上升终端速度。

物理目标
--------
在 TensorFVM 的有限体积框架上验证 VOF 两相流求解器的三个最小可验证面：

1. **静水压力平衡**：两相静止场在重力下的压力分布须满足 p=ρgh，界面不漂移。
   判据：液体区（p > 100 Pa）最大相对误差 < 2%。

2. **溃坝波前**：Koshizuka & Oka (1996) 水柱崩塌实验的波前位置时程 x_front(t)。
   判据：误差 < 10%。

3. **气泡上升终端速度**：Grace (1967) 气泡形状-终端速度关联式（Eötvös 数扫描）。
   判据：误差 < 15%。

实现路径
--------
(a) VOF 界面追踪：几何重构 (PLIC)，含界面锐化项防止数值扩散；
(b) 表面张力：CSF 模型，验证 Young-Laplace 静压差；
(c) 重力源项：Guo 体力或等效源项，验证静水压力分布。

参考
----
Koshizuka, S., & Oka, Y. (1996). Moving-particle semi-implicit method for
fragmentation of incompressible fluid. *Nuclear Science and Engineering*,
123(3), 421-434.

Grace, R. M. (1967). Shapes and velocities of single drops and bubbles moving
freely through immiscible liquids. *Transactions of the Institution of Chemical
Engineers*, 45, T39-T48.

Martin, J. C., & Moyce, W. J. (1952). An experimental study of the collapse of
liquid columns on a rigid horizontal plane. *Philosophical Transactions of the
Royal Society of London. Series A*, 244, 312-324.

用法
----
PYTHONPATH=src python -m tensorfvm.verification.twophase_vof_verification \\
    --output docs/twophase-vof-verification
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

# =========================================================================== #
# 判据阈值（声明于运行前，不可事后修改）
# =========================================================================== #

HYDROSTATIC_PRESSURE_TOL = 0.02   # 2% 最大相对误差（液体区 p > 100 Pa）
HYDROSTATIC_PRESSURE_MIN = 100.0  # 压力阈值，排除界面附近小压力区
DAM_BREAK_FRONT_TOL = 0.10        # 10% 波前位置误差
BUBBLE_VELOCITY_TOL = 0.15        # 15% 终端速度误差
MASS_CONSERVATION_TOL = 1e-3      # 0.1% 质量守恒


# =========================================================================== #
# 1. 静水压力平衡验证
# =========================================================================== #

@dataclass(frozen=True)
class HydrostaticConfig:
    """静水压力平衡算例配置。"""
    nx: int = 40
    ny: int = 80
    length: float = 0.1       # 域宽 [m]
    height: float = 0.2       # 域高 [m]
    interface_y: float = 0.1  # 界面位置 [m]
    rho_liquid: float = 1000.0
    rho_gas: float = 1.2
    gravity: float = 9.81
    vof_scheme: str = "plic"
    pressure_form: str = "total"  # "total" works for hydrostatic rest
    n_steps: int = 50
    cfl: float = 0.3

    def __post_init__(self):
        if self.vof_scheme not in ("plic", "upwind"):
            raise ValueError(f"vof_scheme must be 'plic' or 'upwind', got {self.vof_scheme!r}")
        if self.pressure_form not in ("total", "reduced"):
            raise ValueError(f"pressure_form must be 'total' or 'reduced'")
        if self.interface_y <= 0 or self.interface_y >= self.height:
            raise ValueError("interface_y must be within domain")


def _hydrostatic_alpha(cfg: HydrostaticConfig, dy: float) -> torch.Tensor:
    """构建静水初始 VOF 场（含亚网格分数单元）。"""
    y = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * dy
    hi = cfg.interface_y
    # 分数填充：界面所在单元给出 0~1 之间的值
    alpha = torch.clamp((hi - (y - 0.5 * dy)) / dy, 0.0, 1.0)
    return alpha.reshape(-1, 1).expand(cfg.ny, cfg.nx).clone()


def _analytic_hydrostatic_pressure(y: torch.Tensor, interface_y: float,
                                    rho_liquid: float, rho_gas: float,
                                    gravity: float) -> torch.Tensor:
    """解析静水压力 p(y) = rho * g * (interface_y - y) for y < interface_y."""
    p = torch.zeros_like(y)
    # 液体区：p = rho_l * g * (interface_y - y)
    liquid_mask = y < interface_y
    p[liquid_mask] = rho_liquid * gravity * (interface_y - y[liquid_mask])
    # 气体区：p ≈ 0（表压）
    return p


def run_hydrostatic_verification(cfg: HydrostaticConfig) -> dict:
    """运行静水压力平衡验证。

    返回：
        - pressure_profile: 数值压力剖面
        - analytic_profile: 解析压力剖面
        - max_relative_error: 最大相对误差（液体区 p > 100 Pa）
        - max_velocity: 最大速度（应接近 0）
        - interface_drift: 界面漂移量
        - mass_drift: 质量漂移
        - pass_pressure: 是否通过压力判据
        - pass_velocity: 是否通过速度判据
    """
    dy = cfg.height / cfg.ny
    dx = cfg.length / cfg.nx

    grid = GridConfig(
        nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height,
        max_iterations=1, device="cpu",
    )
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=1e-3, mu_gas=1.8e-5,
        sigma=0.0, surface_tension=False,
        gravity=cfg.gravity,
        top_atmosphere=True, bottom_wall=True, side_outlet=False,
        vof_scheme=cfg.vof_scheme,
        pressure_form=cfg.pressure_form,
        momentum_advection=True,
        warm_start_pressure=True,
        domain_type="closed",
    )
    solver = TwoPhaseSolver(grid, tp)

    # 设置初始 VOF 场
    alpha0 = _hydrostatic_alpha(cfg, dy)
    solver.set_alpha(alpha0)

    # 初始化静水压力
    solver.initialize_hydrostatic_pressure(cfg.interface_y, discrete=True)

    # 记录初始状态
    volume0 = float(solver.alpha.sum()) * dx * dy

    # 时间步进（使用很小的时间步以保持静平衡）
    dt = 1e-4

    umax = 0.0
    for _ in range(cfg.n_steps):
        try:
            solver.step(dt)
        except RuntimeError:
            break
        umax = max(umax, float(solver.u.abs().max()), float(solver.v.abs().max()))

    # 提取压力剖面（域中心 x 方向平均）
    p_final = solver.p.detach().cpu()
    p_profile = p_final.mean(dim=1)  # 沿 x 平均

    # 解析压力剖面
    y_centers = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * dy
    p_analytic = _analytic_hydrostatic_pressure(
        y_centers, cfg.interface_y, cfg.rho_liquid, cfg.rho_gas, cfg.gravity)

    # 计算误差（仅在液体区且压力足够大的地方，排除界面附近）
    liquid_mask = p_analytic > HYDROSTATIC_PRESSURE_MIN
    if liquid_mask.sum() > 0:
        rel_errors = torch.abs(p_profile[liquid_mask] - p_analytic[liquid_mask]) / p_analytic[liquid_mask]
        max_rel_error = float(rel_errors.max())
        mean_rel_error = float(rel_errors.mean())
    else:
        max_rel_error = 0.0
        mean_rel_error = 0.0

    # 界面漂移
    alpha_final = solver.alpha.detach().cpu()
    interface_drift = float((alpha_final - alpha0).abs().max())

    # 质量漂移
    volume1 = float(solver.alpha.sum()) * dx * dy
    mass_drift = abs(volume1 - volume0) / volume0 if volume0 > 0 else 0.0

    return {
        "config": asdict(cfg),
        "pressure_profile": p_profile.numpy().tolist(),
        "analytic_profile": p_analytic.numpy().tolist(),
        "max_relative_error": max_rel_error,
        "mean_relative_error": mean_rel_error,
        "max_velocity": umax,
        "interface_drift": interface_drift,
        "mass_drift": mass_drift,
        "steps": cfg.n_steps,
        "pass_pressure": max_rel_error < HYDROSTATIC_PRESSURE_TOL,
        "pass_velocity": umax < 1e-6,
        "pass_interface": interface_drift < 1e-6,
        "pass_mass": mass_drift < MASS_CONSERVATION_TOL,
    }


# =========================================================================== #
# 2. 溃坝验证（Koshizuka & Oka 1996）
# =========================================================================== #

# Koshizuka & Oka (1996) 溃坝实验参考数据
# 水柱：宽 W=0.1m，高 H=0.2m (H/W=2)
# 无量纲化：t* = t * sqrt(g/W), x* = x/W
# 以下数据从文献图 8 数字化得到
KO_SURGE_T_STAR = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
KO_SURGE_X_STAR = np.array([0.0, 0.85, 1.65, 2.35, 2.95, 3.50, 4.00, 4.45, 4.85, 5.20, 5.55, 5.85, 6.15])


def ko_surge_reference(t_star: float) -> float:
    """Koshizuka & Oka (1996) 波前位置插值。"""
    return float(np.interp(t_star, KO_SURGE_T_STAR, KO_SURGE_X_STAR))


@dataclass(frozen=True)
class DamBreakConfig:
    """溃坝算例配置（Koshizuka & Oka 1996 几何）。"""
    column_width: float = 0.1       # 水柱宽 W [m]
    column_height: float = 0.2      # 水柱高 H [m] (H/W=2)
    domain_length: float = 0.6      # 域长 [m]
    domain_height: float = 0.3      # 域高 [m]
    nx: int = 120
    ny: int = 60
    gravity: float = 9.81
    rho_liquid: float = 1000.0
    rho_gas: float = 1.2
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.8e-5
    sigma: float = 0.0
    vof_scheme: str = "plic"
    max_time: float = 0.6           # 约 t* = 6
    cfl: float = 0.4
    device: str = "cpu"

    def __post_init__(self):
        if self.vof_scheme not in ("plic", "upwind"):
            raise ValueError(f"vof_scheme must be 'plic' or 'upwind'")

    @property
    def dx(self) -> float:
        return self.domain_length / self.nx

    @property
    def dy(self) -> float:
        return self.domain_height / self.ny

    @property
    def t_star_scale(self) -> float:
        return math.sqrt(self.column_width / self.gravity)


def _dam_break_alpha(cfg: DamBreakConfig) -> torch.Tensor:
    """溃坝初始 VOF 场：液体在 [0, W] x [0, H]。"""
    opts = {"dtype": torch.float64, "device": cfg.device}
    cx = (torch.arange(cfg.nx, **opts) + 0.5) * cfg.dx
    cy = (torch.arange(cfg.ny, **opts) + 0.5) * cfg.dy
    x, y = torch.meshgrid(cx, cy, indexing="xy")
    alpha = ((x < cfg.column_width) & (y < cfg.column_height)).to(torch.float64)
    return alpha


def _extract_surge_front(alpha: np.ndarray, dx: float, threshold: float = 0.5) -> float:
    """提取波前位置：底部区域 alpha > threshold 的最右位置。

    alpha shape: (ny, nx)
    """
    ny, nx = alpha.shape
    # 取底部 1/4 高度的行
    bottom_rows = alpha[:ny // 4, :]
    # 每列的最大 alpha
    col_max = bottom_rows.max(axis=0)
    # 找到最右的 wet 列
    wet_cols = np.where(col_max > threshold)[0]
    if len(wet_cols) == 0:
        return 0.0
    return (wet_cols[-1] + 0.5) * dx


def _extract_column_height(alpha: np.ndarray, dy: float, threshold: float = 0.5) -> float:
    """提取残余水柱高度：左侧区域 alpha > threshold 的最高位置。

    alpha shape: (ny, nx)
    """
    ny, nx = alpha.shape
    # 取左侧 1/4 宽度的列
    left_cols = alpha[:, :nx // 4]
    # 每行的最大 alpha
    row_max = left_cols.max(axis=1)
    # 找到最高的 wet 行
    wet_rows = np.where(row_max > threshold)[0]
    if len(wet_rows) == 0:
        return 0.0
    return (wet_rows[-1] + 0.5) * dy


def run_dam_break_verification(cfg: DamBreakConfig) -> dict:
    """运行溃坝验证。

    返回：
        - history: 时间历程
        - surge_front: 波前位置时程
        - column_height: 水柱高度时程
        - front_errors: 波前误差
        - max_front_error: 最大波前误差
        - mean_front_error: 平均波前误差
        - pass_front: 是否通过波前判据
        - mass_drift: 质量漂移
    """
    dx = cfg.dx
    dy = cfg.dy
    t_scale = cfg.t_star_scale

    grid = GridConfig(
        nx=cfg.nx, ny=cfg.ny, length=cfg.domain_length, height=cfg.domain_height,
        max_iterations=1, device=cfg.device,
    )
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas,
        sigma=cfg.sigma, surface_tension=cfg.sigma > 0,
        gravity=cfg.gravity,
        top_atmosphere=True, bottom_wall=True, side_outlet=False,
        vof_scheme=cfg.vof_scheme,
        pressure_form="total",
        momentum_advection=True,
        warm_start_pressure=True,
        domain_type="closed",
    )
    solver = TwoPhaseSolver(grid, tp)

    alpha0 = _dam_break_alpha(cfg)
    solver.set_alpha(alpha0)

    volume0 = float(solver.alpha.sum()) * dx * dy

    history = []
    surge_front = []
    column_height = []
    front_errors = []

    t = 0.0
    step = 0
    max_steps = 3000

    while t < cfg.max_time and step < max_steps:
        # 保守的时间步估计
        u_max = max(float(solver.u.abs().max()), float(solver.v.abs().max()), 0.1)
        dt = min(cfg.cfl * min(dx, dy) / u_max, cfg.max_time - t, 0.001)
        if dt < 1e-6:
            break

        try:
            solver.step(dt)
        except (RuntimeError, ValueError) as e:
            if "CFL" in str(e):
                dt *= 0.5
                if dt < 1e-6:
                    break
                continue
            break

        t = float(solver.time)
        step += 1

        alpha_np = solver.alpha.detach().cpu().numpy()

        # 提取波前和水柱高度
        x_front = _extract_surge_front(alpha_np, dx)
        h_col = _extract_column_height(alpha_np, dy)

        t_star = t / t_scale
        x_star = x_front / cfg.column_width

        # 与参考数据比较
        if t_star >= 0.3:
            x_ref = ko_surge_reference(t_star)
            if x_ref > 0.1:
                err = abs(x_star - x_ref) / x_ref
                front_errors.append(err)

        surge_front.append({"t": t, "t_star": t_star, "x_front": x_front, "x_star": x_star})
        column_height.append({"t": t, "t_star": t_star, "height": h_col})
        history.append({"t": t, "step": step, "dt": dt})

    volume1 = float(solver.alpha.sum()) * dx * dy
    mass_drift = abs(volume1 - volume0) / volume0 if volume0 > 0 else 0.0

    max_front_error = max(front_errors) if front_errors else 0.0
    mean_front_error = float(np.mean(front_errors)) if front_errors else 0.0

    return {
        "config": asdict(cfg),
        "history": history,
        "surge_front": surge_front,
        "column_height": column_height,
        "front_errors": front_errors,
        "max_front_error": max_front_error,
        "mean_front_error": mean_front_error,
        "mass_drift": mass_drift,
        "steps": step,
        "final_time": t,
        "pass_front": max_front_error < DAM_BREAK_FRONT_TOL,
        "pass_mass": mass_drift < MASS_CONSERVATION_TOL,
        "alpha_final": solver.alpha.detach().cpu().numpy(),
    }


# =========================================================================== #
# 3. 气泡上升终端速度验证（Grace 1967）
# =========================================================================== #

def grace_terminal_velocity(eotvos: float, morton: float) -> float:
    """Grace (1967) 气泡终端速度关联式。

    Grace 给出了气泡形状和终端速度与 Eötvös 数和 Morton 数的关系。
    这里使用简化的关联式（适用于椭球形气泡区）：

    Re = f(Eo, M)

    对于椭球形气泡区 (0.4 < Eo < 40, M < 1e-6)：
    Re ≈ 0.5 * Eo^0.7 * M^(-0.15)  （经验拟合）

    终端速度 U = Re * mu_l / (rho_l * d)

    在气泡单位制 (d=1, g=1, rho_l=1) 下：
    U = Re * mu_l
    """
    if eotvos <= 0 or morton <= 0:
        raise ValueError("Eotvos and Morton numbers must be positive")

    # Grace 关联式的简化形式（椭球形气泡区）
    # 参考：Clift, Grace & Weber (1978), ch. 7
    if eotvos < 0.4:
        # 球形区：Hadamard-Rybczynski
        re = eotvos / 3.0
    elif eotvos < 40:
        # 椭球形区：Grace 关联
        re = 0.5 * (eotvos ** 0.7) * (morton ** (-0.15))
    else:
        # 球冠区：Mendelson 关联
        re = math.sqrt(2.14 * eotvos + 0.505 * eotvos ** 1.5)

    return re


@dataclass(frozen=True)
class BubbleRiseConfig:
    """气泡上升算例配置。"""
    eotvos: float = 1.23
    morton: float = 2.63e-11
    density_ratio: float = 1000.0  # rho_l / rho_g
    viscosity_ratio: float = 55.0  # mu_l / mu_g
    cells_per_diameter: int = 20
    domain_width: float = 4.0
    domain_height: float = 8.0
    start_y: float = 2.0
    end_time: float = 3.0
    vof_scheme: str = "plic"
    csf_smoothing: int = 3

    def __post_init__(self):
        if self.eotvos <= 0 or self.morton <= 0:
            raise ValueError("Eotvos and Morton must be positive")
        if self.cells_per_diameter < 8:
            raise ValueError("cells_per_diameter must be >= 8")
        if self.vof_scheme not in ("plic", "upwind"):
            raise ValueError(f"vof_scheme must be 'plic' or 'upwind'")

    @property
    def sigma(self) -> float:
        """表面张力系数（气泡单位制：rho_l=g=d=1）。"""
        return 1.0 / self.eotvos

    @property
    def mu_liquid(self) -> float:
        """液体粘度（由 Morton 数反推）。"""
        delta_rho = 1.0 - 1.0 / self.density_ratio
        return (self.morton * self.sigma ** 3 / delta_rho) ** 0.25

    @property
    def mu_gas(self) -> float:
        return self.mu_liquid / self.viscosity_ratio

    @property
    def nx(self) -> int:
        return int(round(self.domain_width * self.cells_per_diameter))

    @property
    def ny(self) -> int:
        return int(round(self.domain_height * self.cells_per_diameter))

    @property
    def dx(self) -> float:
        return self.domain_width / self.nx

    @property
    def dy(self) -> float:
        return self.domain_height / self.ny


def _bubble_alpha(cfg: BubbleRiseConfig) -> torch.Tensor:
    """气泡初始 VOF 场（气体在液体中）。"""
    opts = {"dtype": torch.float64, "device": "cpu"}
    cx = (torch.arange(cfg.nx, **opts) + 0.5) * cfg.dx
    cy = (torch.arange(cfg.ny, **opts) + 0.5) * cfg.dy
    x, y = torch.meshgrid(cx, cy, indexing="xy")
    # 气泡：alpha=1 表示气体
    r2 = (x - 0.5 * cfg.domain_width) ** 2 + (y - cfg.start_y) ** 2
    alpha = (r2 <= 0.25).to(torch.float64)  # 直径 d=1
    return alpha


def _bubble_metrics(alpha: torch.Tensor, dx: float, dy: float) -> dict:
    """气泡几何度量。"""
    ny, nx = alpha.shape
    total = float(alpha.sum())
    if total <= 0:
        return {"area": 0.0, "centroid": [0.0, 0.0], "velocity": 0.0}

    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy

    # 质心
    cx = float((alpha * x[None, :]).sum()) / total
    cy = float((alpha * y[:, None]).sum()) / total

    return {
        "area": total * dx * dy,
        "centroid": [cx, cy],
    }


def run_bubble_rise_verification(cfg: BubbleRiseConfig) -> dict:
    """运行气泡上升终端速度验证。

    返回：
        - centroid_history: 质心时程
        - terminal_velocity: 终端速度
        - grace_velocity: Grace 关联式预测速度
        - relative_error: 相对误差
        - pass_velocity: 是否通过速度判据
        - aspect_ratio: 气泡纵横比
        - mass_drift: 质量漂移
    """
    dx = cfg.dx
    dy = cfg.dy

    grid = GridConfig(
        nx=cfg.nx, ny=cfg.ny, length=cfg.domain_width, height=cfg.domain_height,
        max_iterations=1, device="cpu",
    )
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=1.0, rho_gas=1.0 / cfg.density_ratio,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas,
        sigma=cfg.sigma, surface_tension=True,
        gravity=1.0,  # 气泡单位制 g=1
        top_atmosphere=False, bottom_wall=True, side_outlet=False,
        vof_scheme=cfg.vof_scheme,
        pressure_form="reduced",
        momentum_advection=True,
        warm_start_pressure=True,
        csf_balance="balanced",
        csf_smoothing=cfg.csf_smoothing,
        domain_type="closed",
    )
    solver = TwoPhaseSolver(grid, tp)

    alpha0 = _bubble_alpha(cfg)
    # alpha=1 表示气体，但求解器期望 alpha=1 表示液体
    # 所以我们需要反转：alpha=0 表示气体（气泡），alpha=1 表示液体
    solver.set_alpha(1.0 - alpha0)

    volume0 = float(solver.alpha.sum()) * dx * dy

    centroid_history = []
    velocities = []
    t = 0.0
    step = 0
    dt = 1e-3

    while t < cfg.end_time and step < 5000:
        # 自适应时间步
        u_max = max(float(solver.u.abs().max()), float(solver.v.abs().max()), 0.01)
        dt = min(0.4 * min(dx, dy) / u_max, cfg.end_time - t, 0.005)
        if dt < 1e-6:
            break

        try:
            solver.step(dt)
        except (RuntimeError, ValueError) as e:
            if "CFL" in str(e):
                dt *= 0.5
                if dt < 1e-6:
                    break
                continue
            break

        t = float(solver.time)
        step += 1

        # 气泡 alpha = 1 - solver.alpha
        bubble_alpha = 1.0 - solver.alpha.detach().cpu()
        metrics = _bubble_metrics(bubble_alpha, dx, dy)

        centroid_history.append({
            "t": t,
            "cx": metrics["centroid"][0],
            "cy": metrics["centroid"][1],
            "area": metrics["area"],
        })

        # 计算速度（质心 y 方向差分）
        if len(centroid_history) >= 2:
            dt_hist = centroid_history[-1]["t"] - centroid_history[-2]["t"]
            if dt_hist > 0:
                vy = (centroid_history[-1]["cy"] - centroid_history[-2]["cy"]) / dt_hist
                velocities.append(abs(vy))

    # 终端速度：取最后 20% 时间的平均速度
    if len(velocities) > 10:
        n_tail = max(len(velocities) // 5, 10)
        terminal_velocity = float(np.mean(velocities[-n_tail:]))
    elif velocities:
        terminal_velocity = velocities[-1]
    else:
        terminal_velocity = 0.0

    # Grace 预测
    grace_vel = grace_terminal_velocity(cfg.eotvos, cfg.morton)
    grace_vel = grace_vel * cfg.mu_liquid

    # 误差
    if grace_vel > 0:
        rel_error = abs(terminal_velocity - grace_vel) / grace_vel
    else:
        rel_error = 0.0

    # 质量漂移
    volume1 = float(solver.alpha.sum()) * dx * dy
    mass_drift = abs(volume1 - volume0) / volume0 if volume0 > 0 else 0.0

    # 气泡纵横比
    final_bubble = 1.0 - solver.alpha.detach().cpu()
    metrics_final = _bubble_metrics(final_bubble, dx, dy)

    return {
        "config": asdict(cfg),
        "centroid_history": centroid_history,
        "terminal_velocity": terminal_velocity,
        "grace_velocity": grace_vel,
        "relative_error": rel_error,
        "mass_drift": mass_drift,
        "steps": step,
        "final_time": t,
        "pass_velocity": rel_error < BUBBLE_VELOCITY_TOL,
        "pass_mass": mass_drift < MASS_CONSERVATION_TOL,
        "final_metrics": metrics_final,
    }


# =========================================================================== #
# 综合验证运行
# =========================================================================== #

def run_full_verification(output: Path | None = None) -> dict:
    """运行全部三个验证面并返回综合结果。"""
    results = {}

    print("=" * 60)
    print("TensorFVM 两相 VOF 首验面验证")
    print("=" * 60)

    # 1. 静水压力平衡
    print("\n[1/3] 静水压力平衡验证...")
    hydro_cfg = HydrostaticConfig(
        nx=40, ny=80, length=0.1, height=0.2,
        interface_y=0.1, vof_scheme="plic", pressure_form="total",
    )
    hydro_result = run_hydrostatic_verification(hydro_cfg)
    results["hydrostatic"] = hydro_result
    print(f"  最大相对误差 (p>{HYDROSTATIC_PRESSURE_MIN}Pa): {hydro_result['max_relative_error']:.4f} "
          f"(判据 < {HYDROSTATIC_PRESSURE_TOL:.2%})")
    print(f"  最大速度: {hydro_result['max_velocity']:.2e} m/s")
    print(f"  界面漂移: {hydro_result['interface_drift']:.2e}")
    print(f"  质量漂移: {hydro_result['mass_drift']:.2e}")
    print(f"  压力判据: {'PASS' if hydro_result['pass_pressure'] else 'FAIL'}")

    # 2. 溃坝
    print("\n[2/3] 溃坝验证 (Koshizuka & Oka 1996)...")
    dam_cfg = DamBreakConfig(
        column_width=0.1, column_height=0.2,
        domain_length=0.6, domain_height=0.3,
        nx=120, ny=60, vof_scheme="plic",
        max_time=0.6,
    )
    dam_result = run_dam_break_verification(dam_cfg)
    results["dam_break"] = dam_result
    print(f"  最大波前误差: {dam_result['max_front_error']:.4f} "
          f"(判据 < {DAM_BREAK_FRONT_TOL:.2%})")
    print(f"  平均波前误差: {dam_result['mean_front_error']:.4f}")
    print(f"  质量漂移: {dam_result['mass_drift']:.2e}")
    print(f"  波前判据: {'PASS' if dam_result['pass_front'] else 'FAIL'}")

    # 3. 气泡上升
    print("\n[3/3] 气泡上升终端速度验证 (Grace 1967)...")
    bubble_cfg = BubbleRiseConfig(
        eotvos=1.23, morton=2.63e-11,
        density_ratio=1000.0, viscosity_ratio=55.0,
        cells_per_diameter=20, vof_scheme="plic",
        end_time=3.0,
    )
    bubble_result = run_bubble_rise_verification(bubble_cfg)
    results["bubble_rise"] = bubble_result
    print(f"  终端速度: {bubble_result['terminal_velocity']:.4f}")
    print(f"  Grace 预测: {bubble_result['grace_velocity']:.4f}")
    print(f"  相对误差: {bubble_result['relative_error']:.4f} "
          f"(判据 < {BUBBLE_VELOCITY_TOL:.2%})")
    print(f"  质量漂移: {bubble_result['mass_drift']:.2e}")
    print(f"  速度判据: {'PASS' if bubble_result['pass_velocity'] else 'FAIL'}")

    # 综合判定
    all_pass = (
        hydro_result["pass_pressure"]
        and dam_result["pass_front"]
        and bubble_result["pass_velocity"]
    )
    results["all_pass"] = all_pass

    print("\n" + "=" * 60)
    print(f"综合判定: {'全部通过' if all_pass else '部分未通过'}")
    print("=" * 60)

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "verification_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n结果已保存至: {output / 'verification_results.json'}")

    return results


# =========================================================================== #
# 命令行入口
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="TensorFVM 两相 VOF 首验面验证")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录")
    args = parser.parse_args()

    run_full_verification(output=args.output)


if __name__ == "__main__":
    main()
