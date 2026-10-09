# 守恒 cell 动量与完整应力通量（实验性独立后端）

`ConservativeCylinder3DSolver` 是新 opt-in 后端；保留上一轮 `solver3d.py`、
`benchmark_cylinder3d.py` 的源码字节和已发布原场，不覆盖历史通过/失败。
历史 main 基线为 `16febbfb9227f73cb94379c873e570055a5bd234`。
`study.json` 绑定本轮所有源与原始数据 SHA256。

## 本轮实现

- 动量是 cell-centered；输运速度使用上一时刻的权威法向 face velocity。
  每个共享面只计算一份 `rho*q*u_upwind`，相邻流体 cell 收到相反贡献。
- 黏性通量使用完整 `mu*(grad(u)+grad(u)^T)` 应力列；可变有效黏度在面插值。
  固壁面用无滑移半格法向梯度、零壁面切向导数、分子黏度（SGS壁面值零）。
  入口/远场规定面值，出口法向零速度梯度。
- 相容 face 质量投影沿用已验证的 `-D_face G_face`；cell 动量的压力作用用
  共享面压力 traction divergence 更新，**不再被 face 平均重建覆盖**。
  每步记录物理外边界 convective/viscous/pressure 冲量以及固壁反力，独立闭合 cell 总动量。
- 固壁输出实际 staircase facet 点、fluid outward normal、面积、压力、黏性 traction、
  压力/黏性/总 body force 和关于原点的力矩。body force=`p*n_fluid - tau*n_fluid`。
  黏性作用取显式步起始 velocity；压力为步末求解结果，时间分别记录。

这是 **cell 动量 / face 质量混合方法**，不是完整 MAC 动量离散。
中心速度没有满足不可压约束；没有以动量账本闭合宣称总体物理正确。

## 真实结果及保留限制

CPU 与 RTX3090 单 GPU，48×32×12，dt=0.005，50 步，PCG预算250。
SI单位为 m/s/kg，但黏度来自合成 Re3900，不能当作校准后的实际海水。

| 项目 | CPU | CUDA |
|---|---:|---:|
| 最大 cell 动量账本残差 (Ns) | 2.06101e-13 | 2.87119e-13 |
| 最大外边界+固壁独立账本残差 (Ns) | 2.06092e-13 | 2.87112e-13 |
| 最大 face 连续性 | 3.08399e-9 | 3.07988e-9 |
| 最大 center 散度指标 | **0.653287** | **0.653287** |
| 假如强制平均重建的最大动量缺口 (Ns) | **0.937472** | **0.937472** |

face 投影、cell 离散动量账本通过；cell 不可压和完整物理模型 **未通过/未认证**。
不额外整体调平或注入冲量来隐藏该缺口。下一阶段需真正相容的面动量推进、
压力作用与中心/面表示，而不能把本轮混合守恒子集当成最终求解器。

完整可变黏度制造解：`u=(xy,x²,0), mu=1+x`，
`div(tau)=(2y,3+6x,0)`，内域误差为零；该跨分量用例能识别 transpose
和 grad(mu) 缺项，旧 `mu*Laplacian(u)` 不会给出这个答案。
另用光滑 sin 速度与变 μ，在固定物理评估域 `10<x<15, 3<y<8` 的
32/64/128 网格应力散度 L2 误差为 **0.995268% / 0.252620% / 0.0629645%**，约二阶。
这验证应力算子子问题，不是圆柱 Cd/St 网格收敛。

6 项定向回归通过：两种制造解、逐步 boundary/cell/wall 守恒、signed upwind、
真实 CUDA 以及非均匀展向扰动两 rank Gloo。通信精确性回归采用更严格
PCG rtol=1e-12、预算300；不能替代默认 rtol=1e-8 的比较。
独立审查中，默认 PCG 下非均匀展向串/并严格1e-9绝对压力/力矩门失败
（压力约7.70e-9、力矩约9.36e-9差），保持为失败；紧化线性求解容差是另一次诊断。
另一次更紧诊断实际采用 rtol=1e-12、atol=1e-13、最大预算1000：
压力差6.706e-13、速度差5.35e-15、面速度差2.55e-15、body力矩差2.27e-12，
通过同一1e-9绝对门。它支持线性求解精度影响串/并比较的诊断，
不能放宽原门或将默认失败改成通过。最终独立报告记录完整原场核查。

## 数据与复跑

- `cpu/transport-raw.json.gz`、`cuda/...`：完整步末原场、步初 velocity/advecting
  faces、mu、face梯度、convective/viscous/pressure通量、50步冲量账本。
- `wall-facets.json`：直接可供保持作用点的力/矩/功映射；并不代表已做双向 FSI。
- `face-fields.json`：权威 face 质量原场；`midspan.csv` 是中心场。
- `manufactured.json`：真实数值应力散度和独立解析值。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/run_conservative_transport.py --output /tmp/conservative-transport
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m pytest -q tests/test_conservative3d.py
```

尚未认证能量稳定性、时间/圆柱几何细化、LES统计、Cd/St、移动体、自由液面、
三维贴体流动和柔性双向 FSI。上风输运和显式Euler仍为低阶。

完整独立审计、默认失败的另一次精确复现和脚本源绑定见 [审核归档](../conservative-transport-audit/README.md)。
