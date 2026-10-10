# 大网格、贴体边界、壁面函数与 GPU 验证

## 本轮交付及验收范围

| 案例 | 规模 | 验收结果 | 范围 |
|---|---:|---|---|
| 贴体环形管轴向 Poiseuille | 16,384 / 65,536 / 262,144 | 全部误差小于 3%；速度二阶收敛 | 三维充分发展流的严格二维截面约化，完整稀疏矩阵；没有一般三维 NS 或数值压力求解 |
| Spalding 壁面牵引 | 141 点独立反解及 65,536 点 CPU/CUDA | 本构反解通过 | 尚未接入 SA 输运壁面边界 |
| NASA Coles 关联曲线 | 固定 κ=0.41、E=9.8；50≤y+≤200 | L2 1.939%、归一化峰值误差 2.530% | 公开关联曲线对照；不等于完整 RANS 验证 |
| SA 平板 Re=100000 | 64×56，1800 次迭代 | **未通过**：Cf 误差 7.223%，残差 6.915e−5；最大局部 y+=2.090 | 首层贴壁生产模型诊断；经验平均 Cf 参考，不是匹配 NASA 工况 |
| 实际 CPU/CUDA 矩阵 | 10 项、分别完整预热及三次重复 | 9 项一致性通过；SA 压力差 0.575% 未通过 | CUDA 无 CPU 求解回退；SA 项只有五次生产迭代 |

[环形管完整报告](verification-engineering/report.md)、[GPU 性能与正确性报告](verification-gpu/report.md)、[公开壁面关联曲线](verification-wall-law-reference/report.md)、[SA 未达标诊断](verification-sa-diagnostic/report.md)。

GPU 为 RTX 3090，float64、单线程 CPU；环形管双方统一 torch CSR/Jacobi CG。计时包括网格、求解、诊断和原场主机导出，排除绘图。用同步计时中位数计算 CPU/CUDA 比；结果只适用于该算法、硬件与规模。小网格没有保证加速。SciPy 正式 CPU 基准与 GPU 性能矩阵的 CPU 后端不同，不能混用时间。完整三维 ABC 案例另行测试。

环形管报告中的压力曲线是给定驱动压力；阶梯边界比较只比较几何误差，不替代 TensorLBM 流动计算。实际 LBM 的 CPU/CUDA 对照来自独立 TensorLBM 代码及原始 BGK 状态。

## 复现

依赖现有项目环境、可用 CUDA Torch、SciPy>=1.10、matplotlib。CPU 可先执行环形管与壁面本构；GPU 命令必须有真实 CUDA，输出目录需为空。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.engineering --output /tmp/fvm-engineering-new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.gpu --output /tmp/fvm-gpu-new --lbm-repo ../TensorLBM
PYTHONPATH=src python scripts/audit_gpu_benchmarks.py
PYTHONPATH=src python -m pytest tests/test_engineering_benchmarks.py -q
```

GPU 审计脚本默认读取仓库中保存的正式输出。manifest 记录源码和原场 SHA256；独立 NumPy 审计重建环形控制体几何/通量、所有 MAC/BGK 接受时间步，并重新计算 CPU/GPU 场差与时间比。SA 五步只核对字段和设备一致性，不证明稳态物理正确。导出缺失非正交通量的初次失败及当时源码保存在 [失败归档](verification-engineering-export-failure/README.md)，修复后从零重算，没有放宽门槛。

## 从易到难的下一阶段

1. SA 收敛诊断：分离动量、SA、连续性残差，核对入口/出口、首层 y+ 与迭代预算；现有失败保留。
2. 匹配 NASA TMR 平板条件及公开速度、摩阻数据，至少三组贴壁网格，所有目标量严格 3% 验收。
3. 一致地接入 Spalding 壁面剪切和湍流输运边界；同工况比较贴壁积分与壁面函数，分别报告 y+ 范围、误差与成本。
4. 曲壁湍流：先弯管/翼型，再分离流或后向台阶。扩大网格到百万级；贴体网格与 LBM 在相同物理精度下比较。
5. SUBOFF 裸体及附体：先固定体流动与公开阻力/压力数据，再考虑自由运动、冰及多求解器耦合。

每一级均增加实际 CPU/CUDA 误差、收敛、显存与至少三次重复计时；未通过物理门的案例单独保留。单卡加速不自动证明多 GPU 扩展性或工程破冰可靠性。

## SA 的 GPU 一致性失败

128×112 五次迭代实测 CPU 118.809 s、CUDA 17.175 s，时间比 6.917；压力相对 L2 差 0.005749、速度相对 L2 差 1.3344e−6，均超过统一 1e−6 一致性门。该项记为失败，整体 GPU correctness_passed=False。不放宽门限；不能把时间比宣称为经过正确性认证的 SA 加速。需要继续定位拉伸网格下压力线性迭代的停止条件、真实残差及 CPU/CUDA 归约敏感性。
