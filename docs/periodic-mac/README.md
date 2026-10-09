# 真正周期 MAC 面动量原型

这是新的独立 `PeriodicMACSolver`，用于证明有限周期域内的面动量、压力、
质量与能量相容。所有分量直接存在各自错位面上；没有 cell/face 平均回写、
动量调平或附加能量修正。上一轮 cylinder 混合方法的失败没有改成通过。
旧 `solver3d.py`、`conservative3d.py`、benchmark 与原场源码哈希保持不变，
历史基线 `db88ce75037845dea79a3accc990581c0087f140`。

## 离散与时间推进

三个分量均采用 `(z,y,x)` 数组；`u_a[n]` 是 cell n 的 +a 面，三个数组代表
不同物理坐标。权威散度为后向面差，压力梯度为前向 cell 差，因此 `G=-D*`。
周期 FFT 解的是这个离散 `-DG` 的精确 Fourier symbol；只消除常数压力模式，
棋盘格具有非零梯度。周期平均压力零，无出口边界或固壁。

每个分量的 dual control volume 共享中心动量通量。方向 b 的 advecting
velocity 是 `.5*(u_b[n]+u_b[n+e_a])`，transported component 是
`.5*(u_a[n]+u_a[n+e_b])`。邻居 dual CV 使用同一面通量相反符号；
无散度速度的离散对流功率为零，总动量守恒。

完整应力法向项在 cell：`tau_aa=2*mu*D_a u_a`；交叉项在 a/b edge：
`tau_ab=mu_edge*(G_b u_a+G_a u_b)`，edge μ为四邻 cell 平均。
它的负功等于全部法向/剪切应变的非负耗散，包含 grad(μ) 与 transpose 项，
没有将变 μ应力替换成 μ乘拉普拉斯。

隐式 midpoint 采用 Picard 求解，迭代增量和最终真实 fixed-point 残差必须
都满足 `1e-12 m/s`，每次候选都用同一相容面投影。失败候选保留但不更新
速度、时间或 accepted history。压力投影在面动能内积中正交，因此不增能。
步能量账本分列对流功率、黏性功率及压力功率，没有吸收未收敛残差。

## 实际验证与明确门槛

每步数值门：非线性真实残差 ≤1e-12、面散度 <1e-10、动量变化 <1e-10 Ns、
能量缺口 <1e-10乘 max(1,初动能)、对流功率 <1e-10 W；黏性功率非正，
投影及整步不增能的浮点容差为1e-12乘 max(1,初动能)。这些是周期数值内核门，
不是 LES/工程误差。解析误差各自采用3%，观察阶门为1.8–2.2。

CPU与RTX3090真实随机3D curl面场、正变 μ场，各10步：

| 指标 | CPU | CUDA |
|---|---:|---:|
| 最大能量账本缺口 (J) | 1.5591e-16 | 1.2382e-16 |
| 最大面散度 (1/s) | 1.6653e-16 | 1.1102e-16 |
| 最大每步动量变化 (Ns) | 1.1953e-16 | 9.5622e-17 |
| 总能量变化 (J) | −0.00555550 | −0.00555550 |

另有非零周期平均速度回归，证明动量保持不限于零均值 curl 初态。

### Taylor–Green 与应力制造解

Taylor–Green 是二维解析场嵌入三维周期网格；随机 curl 测试是实际三维场。
空间比较在固定 t=0.5、dt=0.0125，参考连续解析速度 `exp(-2νt)`。
时间比较在固定16×16×4、t=1，参考该固定网格独立离散拉普拉斯特征值的
精确时间指数，隔离 midpoint 时间误差；不能把它称为新的连续物理误差。

| 空间网格 n×n×4 | TG相对L2误差 |
|---|---:|
| 8 | 0.504854% |
| 16 | 0.127928% |
| 32 | 0.032086% |

空间观察阶1.981、1.995。固定网格 dt=0.1/0.05/0.025 的时间误差
为6.41457e-6、1.60358e-6、4.00890e-7，观察阶约2.000。

全周期变 μ三维完整应力制造解，含非零 w方向应力散度：

| 网格 n³ | 应力散度L2误差 | 3%门 |
|---|---:|---|
| 8 | **5.316805%** | **失败，保留** |
| 16 | 1.354609% | 通过 |
| 32 | 0.340273% | 通过 |

观察阶1.973、1.993。最细网格通过不改变最粗失败，
`mms_all_grids_3_percent_passed=false`。

### 拒绝控制与支持范围

两个4次 Picard预算控制均保留原场且没有推进状态：dt=0.01候选真实残差
1.58e-13已小于门，但上一迭代增量约3.60e-11未过，按双门拒绝；
dt=0.5候选真实残差 **4.98816e-5**，明确未收敛。没有把前者误报成
真实残差失败。9项测试涵盖adjoint/棋盘格、投影正交性、全应力负功、
共享对流功率与动量、真实 midpoint、拒绝事务、硬件对照及细化失败保留。

**仅单 rank CPU/CUDA。** `distributed_supported=false`，任何多 rank runtime
显式抛出 NotImplementedError；未发布 Gloo/NCCL或扩展性能通过。
当前没有固壁、移动边界、自由液面、圆柱/壳体、LES统计或双向FSI。
周期原型的资格不能回填旧 cylinder 的 center divergence/面动量缺口。

## 原场与复现

`study.json` 绑定生产源与13个 gzip JSON原场。正例保存每个accepted step的
完整面速度，因此可由相邻状态独立重建 midpoint、D、KE、P及通量功率；
最后一步还保存每面动量通量、normal/edge stress、μ、压力、投影前后场和
真实非线性迭代历史。失败候选及制造解 analytic/computed 数组均保留。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/run_periodic_mac.py --output /tmp/periodic-mac
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m pytest -q tests/test_periodic_mac.py
```

下一步应将相容面动量拓展到真实固壁 dual CV及边界压力/黏性功，再考虑
圆柱与移动体；不能直接把周期 FFT 边界当成上浮破冰流体后端。
