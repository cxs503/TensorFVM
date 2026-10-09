# 独立流体边界证据交叉审计

本报告由 TensorFVM 侧 NumPy 实现独立重建，不导入 TensorLBM 求解器或审计器，也没有运行同边界的 FVM 圆柱算例。现有 FVM SIMPLE/MAC 通道解析资格仍见 [通道验证](../suite-channel/README.rst)。不同边界的通道资格不替代这里的移动圆盘资格。

## 可复现命令

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_external_lbm_boundary.py --lbm-root ../TensorLBM
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_external_boundary_audit.py
```

[report.json](report.json) 绑定 19 份原始场、3 个 study 和其求解器/运行脚本 SHA256，另记录独立审计模块 SHA256。外部原始场留在 TensorLBM 仓库；重跑时源版本必须与各自 study 一致。

## 实際检查

- 从显式 SI 的 dx、dt、rho、挤出厚度、nu 重建 `c_s=dx/(sqrt(3)*dt)`、`tau=0.5+3*nu*dt/dx²`、Mach 和直径 Reynolds 数。
- 不复用模拟器 mask，按中心和半径独立计算排除内部流体的圆盘节点；周期移动模型使用最短周期距离。
- 从九方向最终 populations 重建流体质量、动量及固定模型的最终最大速度/Mach/Re。驱动模型初始 Mach/Re 为零，另列非零最终值，避免把初值当全过程值。
- 固定模型核对 `ΔP_fluid + I_body - I_drive = 0`，并由逐步力积分独立重建固体冲量。
- 移动模型逐步核对墙面冲量、覆盖/暴露节点的移除/添加冲量和全局 reservoir 修正，分别列出 SI kg、N·s。全局 reservoir 是外部数值源。
- 测试验证 SI/EOS 转换、周期 mask、拒绝固体内部流体、拒绝伪造节点转换冲量。

## 实际发现及资格限制

19 份原始场及全局账本审计通过，4 项测试通过。这是独立证据一致性验证；没有解析曲面阻力或移动排水物理验证。

驱动圆盘 32/48/64 网格及 64 时间步减半案例的 1.5–1.75 s 与 1.75–2 s 平均力窗口漂移分别为 7.89035%、7.97035%、8.00266%、7.99581%。原阈值为 1%，四例均未稳态。空间及时间改变后末段力接近，只能称瞬态窗口敏感性，不能称稳态阻力收敛。

移动基准 Re_D=2.4。dt 从 0.01 s 缩至 0.000625 s 时，SI 声速从 0.57735 增至 9.23760 m/s，Mach 从 0.05196 降至 0.0032476。故这组数据仍为时间/压缩性组合敏感性，并非固定 EOS 的纯时间误差。motion-half/quarter 则改变 Re_D 至 1.2/0.6，不能混入同物理问题的时间收敛组。

移动案例全局 reservoir 的累计绝对质量交换约 0.044–0.048 kg。账本闭合没有解决局部 swept-volume 质量转移；报告始终 `local_mass_conservation_qualified=false`、`temporal_convergence_qualified=false`、`physical_accuracy_qualified=false`。

下一步应分别完成曲面阻力独立解析/同条件对照、固定 EOS 的误差识别和局部保守移动边界。FVM 尚未获得移动冰求解器资格。

## 延长至 12 s 的独立双门资格审计

新增 [developed-report.json](developed-report.json) 与独立模块，原 19 份历史报告及其 source/raw SHA 保持不变。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_developed_lbm_boundary.py --lbm-root ../TensorLBM
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_developed_boundary_audit.py
```

按实际排除圆盘内部节点后的流体质量，逐步积分 `m_fluid*a*ramp(t_mid)*dt` 驱动冲量；对完整每步反力序列积分，独立恢复最后 0.5 s 的净流体动量起止值，并核对最终 populations。稳态必须同时满足相邻 0.5 s 力窗漂移 <1% 与反力/驱动力差 <1%。仅漂移通过时不能宣称稳态。

| 实际算例 | 力窗漂移 | 反力/驱动力差 | 最后窗流体净动量率 N | 真正稳态 |
|---|---:|---:|---:|---|
| n32, dt=0.0005 s | 0.641766% | 4.650379% | 0.177477 | 未通过 |
| n48, dt=0.0005 s | 0.649088% | 4.716988% | 0.179426 | 未通过 |
| n64, dt=0.0005 s | 0.656260% | 4.788444% | 0.182139 | 未通过 |
| n64, dt=0.00025 s | 0.654659% | 4.772500% | 0.181532 | 未通过 |

四例原始场审计通过；最大相对质量误差 4.75e-12，质量阈值为 1e-11。最后窗仍存在实际流体加速。五项新增测试包含构造非负原始 populations、平坦力窗但4.5%未平衡的场，确认独立审计拒绝伪报 `steady_qualified=true`。未运行同条件 FVM 圆柱计算，物理精度与纯时间收敛仍未获资格。
