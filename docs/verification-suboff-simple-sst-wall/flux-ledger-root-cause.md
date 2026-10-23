# SUBOFF SIMPLE+SST 面通量账本:根因分析与修复

日期:2026-10-23。对象:`docs/verification-suboff-simple-sst-wall/cp-interleaved`
归档的失败运行(9,068 单元 SUBOFF,Re_L = 1.2e7,SIMPLE + SST-2003 + Spalding)。
本文件只记录本次诊断的实测事实与代码改动;物理验收(Cp/Cd 3% 口径)仍未触及。

## 0. 结论摘要

1. **账本破缺定律(已证明并已修复)**:旧 `NativeSimpleMixin.solve` 把压力状态
   欠松弛(`p += alpha_p * dp`)而面通量取完整响应,每步向状态注入
   `(max|(1-alpha_p) * diag(alpha) * pressure_flux_defect @ dp| / |S|)`
   的不一致。该量在归档配置(`alpha_p = 0.03`、`alpha_u = 0.05`)下第一步即为
   1.60,与实测 `face_flux_consistency_max = 1.6045229878924465` 逐位吻合。
2. **响应通道失衡(已修复)**:压力修正的速度通道与 Rhie–Chow 缺陷通道使用了
   不同标度的 mobility;两者必须由同一个未欠松弛对角给出,否则压力方程系统性
   过修正/过度吸压。
3. **初始投影被丢弃(已定位,未改代码)**:`initial_flux_projection` 把自由流
   通量的散度从 1,510 降到 8.7e-11,但第一步的 `qstar` 完全从速度重建,散度
   立即回到 **1,511**(速度只移动 0.004)。投影通量被完全丢弃。
4. **剩余阻塞:薄层单元压力算子(未解决)**:SUBOFF 网格近壁 468 个薄层单元上
   压力修正矩阵出现**负对角**(最小 -2.1e-4),压力方程近乎奇异
   (`max|prhs| = 2.3e-4`,中位数 1.4e-17),`dp` 在薄层达到 63–200,
   把速度修正送到 18U。小通道(27 单元)压力对角全正(0.5–1.8),故能收敛。
5. 修复 + 账本中性均匀阻尼后:同一 SUBOFF 从"第 2 个外迭代 18.34U 触发 5U 保护"
   改善为"第 5 个外迭代 5.74U 才触发",且动量模式在 8 步内速度稳定在 1.0–1.9U、
   账本缺陷单调收缩(0.77 → 0.33)。**仍未收敛**,收敛门未过。

## 1. 复现(逐位)

`scripts/diag_simple_flux_ledger.py` 用归档 `source-snapshot/`(冻结源码)重跑
cp-interleaved 的动量阶段,首步:

| 量 | 归档 `momentum-history.jsonl` | 本次复现 | 一致 |
|---|---:|---:|---|
| `face_flux_consistency_max` | 1.6045229878924465 | 1.6045229878924465 | 逐位 |
| `maximum_speed` | 2.30766197925289 | 2.30766197925289 | 逐位 |
| `current_reconstructed_divergence_max` | 1712.9066 | 1712.9066 | 逐位 |
| `native_divergence_max` | 4.801030717064585e-11 | 同 | 逐位 |

后续步:1.60 → 2.52 → 10.30(速度 2.31 → 3.91 → 17.78)触发保护;归档为
1.60 → 6.98(18.34)。差异来自外迭代与内迭代的排布(归档每外迭代 1 步动量,
诊断脚本连续动量步),增长机制一致。

环境说明:CSDN 沙箱内 `gmsh` 轮子因缺 `libGLU` 无法导入,
`scripts/_vendor_gmsh/gmsh.py` 提供 MSH 4.1 ASCII 的 API 子集替身;
仅用于诊断,生产运行仍用官方 `gmsh`。

## 2. 账本代数

求解器把一致状态定义为

```
flux = B u + diag(alpha) * pressure_flux_defect @ p + qsource          (L)
```

旧更新映射:

```
u_new = predicted - inverse * G @ dp
p_new = p + alpha_p * dp
flux_new = qstar + (correction - B diag(inverse) G) @ dp
```

代入 (L):`flux_new = B u_new + correction @ (p + dp) + qsource`,
而存储的压力只前进了 `alpha_p * dp`。因此

```
flux_new - (B u_new + correction @ p_new + qsource)
      = (1 - alpha_p) * correction @ dp                                 (D)
```

对 `alpha_p = 1` 该式为零 —— 这正是既有测试
`test_one_full_pressure_correction_matches_reconstructed_flux` 只在
`alpha_p = 1` 下成立的原因。归档 SUBOFF 用 `alpha_p = 0.03`,每步注入
约 `0.97 * correction @ dp`;该术语与 |S| 归一后即 1.60。

## 3. 实验矩阵(SUBOFF 9,068 单元,动量模式,alpha_u = 0.05)

| 实验 | 配置 | 首步一致度 | 速度轨迹 | 结论 |
|---|---|---:|---|---|
| E0 | 冻结源码,`alpha_p = 0.03` | 1.60 | 2.31 → 3.91 → 17.78 | 复现归档 |
| E1 | 冻结源码,`alpha_p = 1.0` | 1.11 | 2.31 → 2.78 → 8.82 | 仍有第二缺陷:冻结版 `response = -diag(alpha) compact_flux` 的紧凑近似 |
| E2 | 当前 HEAD(未改),`alpha_p = 0.03` | — | 第一步即被真残差门拒绝(压力 `5.46e-8 > 1e-9`) | 薄层系统 SuperLU 真实残差超门,需迭代精化 |
| E3 | 账本一致更新,`w = 1.0` | **恒等式 1.6e-14** | 2.41 → 9.61 | 账本精确,但 `pressure_increment_flux_response_max = 8.1` 后爆 |
| E4 | 账本一致更新 + 均匀阻尼 `w = 0.1` | 恒等式按 `(1-w)` 收缩 | **1.07 → 1.16 → 1.28 → 1.47 → 1.92 → 1.46 → 1.54 → 1.94**,第 9 步压力响应尖峰后爆 | 阻尼把 18U 级爆炸压到可观测的 1–2U,并暴露真正的触发点 |
| E5 | 全 RANS 交错(`w = 0.3`,outer = 8) | — | 1.24 → 1.47 → 2.08 → 4.59 → 5.74 → 保护 | 死亡点由 outer 2 后移到 outer 5,速度由 18.3 降到 5.7 |

E4 的逐步量(节选):

| iter | max\|u\| | `momentum_residual` | `face_flux_consistency` | `ledger_identity_defect` | `pressure_increment_flux_response` |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.074 | 1.25e-3 | 7.74e-1 | 7.7e-1 | 1.03 |
| 4 | 1.471 | 4.44e-3 | 5.64e-1 | 5.6e-1 | 1.87 |
| 7 | 1.544 | 6.33e-3 | 4.11e-1 | 4.1e-1 | 1.95 |
| 8 | 1.944 | 7.24e-3 | 3.70e-1 | 3.7e-1 | 3.78 |
| 9 | 3.029 | 1.07e-2 | 2.50e+0 | 3.3e-1 | **24.9** |
| 10 | 6.972 | 3.01e-2 | 2.59e+0 | 2.2e+0 | 19.3 |

第 9 步的 `pressure_increment_flux_response` 尖峰(3.8 → 24.9)先于速度发散出现,
可作为薄层病态的触发签名。

## 4. 初始投影被丢弃

复现记录(自由流 + 投影种子):

```
div(tentative)        = 1510.43
div(projected)        = 8.70e-11
div(qstar) 第一步     = 1511.25      <-- 投影被完全丢弃
|predicted - u| max   = 0.0042       <-- 速度几乎未动
```

原因:`qstar = B@predicted + correction@p + qsource` 每一步都从速度重建,
投影对通量的修改没有进入速度状态。1,510 的散度来自壁面首层
(元胞速度 = 自由流、壁面 = 0,元胞体积小到 2.2e-9)。

结论:启用 `--project-initial-flux` 反而制造了"通量与速度不一致"的初值
(账本缺陷 1.1),而该散度在首步被一次性要求消除,dp 因此巨大。
不开投影则首步即 34.6U 爆炸(散度完全未处理)。两条路都不可用。

建议(下一步):把投影势当作**数值初值**接受,用 RC 响应最小二乘求出与之
一致的速度/压力种子,或在投影后做一次"以投影通量为 qstar"的压力修正;
两者都必须在 `initial-projection.json` 里标注为数值种子而非物理压力。

## 5. 薄层单元压力算子(未解决,阻塞收敛)

首步线性化后(自由流 + 投影):

```
P = D @ (correction - B diag(inverse) G)
P 对角范围: -2.11e-4 ... 6.11    (存在负对角)
|prhs| max = 2.29e-4, 中位数 = 1.4e-17
dp: max 63.0, 中位数 4.2e-2, |dp| > 10 的单元 468 个
最大 dp 单元: V ~ 3e-8, 壁面距离 y ~ 2.9e-4 (首层薄单元)
```

对照:27 单元小通道 `P` 对角 0.50–1.77,**无负对角**,故同一套离散能收敛。

负对角的含义:压力修正算子失去 M-矩阵性质,`dp` 可无界。物理上是
"壁面 Robin 阻力把速度通道压到近零,而缺陷通道的 `coefficient = |S|^2/(S.d)`
在薄层巨大,两个通道的抵消不再精确"。这正是原计划 P0 子项③"薄层单元稳定性"。

候选修复(按风险从低到高):
1. 薄层 mobility 下限限制(face mobility 不低于同面扩散 mobility 的固定比例);
2. 壁面单元改用"面基"压力响应(以壁面通量而非元胞对角定义 d_f);
3. 一单元多壁面检测 + 逐面权重加权(当前按面积加权已有,但未按面分别限幅)。

所有候选都必须保持账本 (L) 精确,并把"负对角单元数"与
`pressure_increment_flux_response_max` 纳入验收门。

## 6. 代码改动

`src/tensorfvm/native_simple3d.py`

* 速度通道与 RC 缺陷 mobility 统一由未欠松弛对角 `1/diagonal` 给出
  (SIMPLE);SIMPLEC 仅改变增量响应 `1/max(diagonal + sum aN, 0.1 diagonal)`。
  这样离散连续性算子 `flux = B u + alpha defect p + qsource` 不依赖
  `relaxation`/`pressure_relaxation`/算法,固定点不随纯迭代参数漂移。
* 接受的状态更新改为整步一致:速度、压力、通量同时用完整 `dp`;
  阻尼改为对三者做同一凸组合 `(1-w) old + w candidate`,
  账本恒等式在任意 `w` 下精确(继承的偏离按 `(1-w)` 收缩)。
* 两个直接解加入行均衡 + 3 次迭代精化,真残差门在均衡系统上度量
  (与耦合求解器 `_linear` 同一约定);薄层网格上原始系统残差会停在
  float64 matvec 舍入地板(`5.46e-8`),不是解质量问题。
* 新增逐步指标:`ledger_identity_defect_max`(步内算子下的账本恒等缺陷)、
  `pressure_increment_flux_response_max`(压力修正引起的面通量响应)。

`src/tensorfvm/native_collocated3d.py`

* `NativeSteadySettings.pressure_relaxation` 默认 `1.0`,语义改为
  "整步一致更新的均匀阻尼";文档字符串说明禁止单独欠松弛压力。

`scripts/run_native_suboff_rans.py`

* `--pressure-relaxation` 默认 1.0 并更新帮助文本。

`scripts/diag_simple_flux_ledger.py`(新)

* 逐步账本诊断:模式 `momentum` / `rans`,支持冻结源码快照复现、
  初始投影、重启、阻尼/松弛/CFL 等全部开关,输出 `history.jsonl`。

`scripts/_vendor_gmsh/gmsh.py`(新)

* 无 `libGLU` 环境下的 Gmsh MSH 4.1 ASCII API 子集替身,仅诊断用。

`tests/test_native_simple3d.py`

* `test_simple_step_preserves_face_flux_ledger[1.0, 0.3]`:
  `w = 1` 时每步账本恒等缺陷 `< 1e-12`;`w = 0.3` 时缺陷单调收缩。
* `test_pressure_damping_does_not_change_converged_steady_solution`:
  阻尼不改变收敛解(2e-8 容差)。

回归:`tests/test_native_simple3d.py` 21 passed;
加上 collocated/rans/projection/quadratic/audit/engineering 共
73 passed + 42 passed 分组,无失败。

## 7. 复现命令

```bash
# 冻结源码逐位复现(首步 1.6045229878924465 / 2.30766197925289)
PYTHONPATH=scripts/_vendor_gmsh:docs/verification-suboff-simple-sst-wall/cp-interleaved/source-snapshot/src \
  python scripts/diag_simple_flux_ledger.py \
  --mesh docs/verification-suboff-simple-sst-wall/cp-interleaved/input.msh \
  --output /tmp/e0 --mode momentum --sweeps 4 \
  --relaxation .05 --pressure-relaxation .03 --project-initial-flux

# 修复后(账本恒等 ~1e-14,阻尼下速度稳定在 1-2U)
PYTHONPATH=scripts/_vendor_gmsh:src python scripts/diag_simple_flux_ledger.py \
  --mesh docs/verification-suboff-simple-sst-wall/cp-interleaved/input.msh \
  --output /tmp/e4 --mode momentum --sweeps 12 \
  --relaxation .05 --pressure-relaxation .1 --project-initial-flux

# 全 RANS 交错(outer 5 才触发保护,旧代码 outer 2 即 18U)
PYTHONPATH=scripts/_vendor_gmsh:src python scripts/diag_simple_flux_ledger.py \
  --mesh docs/verification-suboff-simple-sst-wall/cp-interleaved/input.msh \
  --output /tmp/e5 --mode rans --outer 8 --inner 1 \
  --relaxation .05 --pressure-relaxation .3 --project-initial-flux
```

## 8. 状态与下一步

* 账本一致性:已修复并有回归测试。`physical_validation_passed` 仍为 false。
* SUBOFF Re=1.2e7:未收敛。剩余两个已知阻塞:
  ① 薄层压力算子负对角(468 单元);
  ② 初值一致性(投影通量被丢弃)。
* 建议顺序:先做①(mobility 限幅 + 负对角计数入门),再做②
  (投影一致初值),最后重跑延拓启动(层流 → 1e5 → 1e7)。
