# CASE-FVM-KOVASZNAY-ORDER — 结案：32/64/128 三网格收敛阶实测

**日期**: 2026-11-14 · **席**: m20033（selfstart, tp-merge-1）· **仓**: TensorFVM
**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature_ledger，本单开工时为 `open`）
**判据（原文）**: L2 收敛阶 ≥ 1.8（名义二阶），log-log 斜率 on 32/64/128 三套网格
**外部基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999

## 结论

**PASS**。三网格（32/64/128，同一离散配置：`zero_gradient` 展向闭合、Re=40、二阶对流、
耦合牛顿）全 3D 速度 L2 最小二乘斜率 = **1.9623**
（R² = 0.999944），判据阈值 1.8 → 达标；
去均值压力斜率 1.8777 亦达标。三套网格全部数值收敛
（动量残差、散度、边界速度误差均在 1e−9 量级或以下）。

## 一、案例介绍

Kovasznay (1948) 二维定常不可压 N–S 精确解（Re=40，单位域 [0,1]²，对称挤出 n×n×1）：

    u = 1 − exp(λx)·cos(2πy),  v = (λ/2π)·exp(λx)·sin(2πy),  w = 0,  p = ½(1 − exp(2λx))

λ = −2(2π)²/(Re + √(Re² + 4(2π)²)) ≈ −0.963741，满足 λ² − Re·λ = 4π²。
非线性项、黏性项、压力梯度均非平凡且 div u ≡ 0，是有限体积求解器精度验证的理想制造解。

| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ / μ | 1 / 0.025 |
| 网格 | n×n×1（n = 32/64/128 → 1024/4096/16384 单元）|
| 展向闭合 | `zero_gradient`（物理对称 du/dz = dv/dz = 0）|
| 对流格式阶 | 2 |
| 算法 | coupled（Picard 外层松弛 0.7，tol 1e−8，线性 reused_lu）|
| 比较量 | 体积加权相对 L2，解析场取单元中心值 |

## 二、怎么做的（方法/流程/工具链）

1. `scripts/verify_kovasznay_order.py` 每次一个网格，写出 `cases.json` + `n<N>-fields.npz`
   + 外/内/线性迭代史，并记录离散守恒恒等式（能量平衡误差、界面牵引缺陷、全局动量不平衡、散度）。
2. `scripts/summarize_kovasznay_32_64_128.py` 从 npz **重算**各层 L2（与 cases.json 不符即硬报错，
   rtol 1e−10），再做 log-log 最小二乘；`validate_levels()` 强制四元配置（Re / 闭合 / 对流阶 / 算法）
   在各层一致，混配序列直接拒绝拟合。
3. n=128 层级在集群 n330 节点运行（8 核；build 约 129 s、
   solve 719.7 s），工件经 tar 取回工作树，零手工转录。

## 三、计算结果

| n | h | cells | vel L2 (full 3D) | vel L2 (u,v) | max &#124;w&#124; | p L2 (mean-free) | p L2 (fixed) | sampling bias (vel) | outer it | converged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 32 | 0.03125 | 1024 | 3.4356154e-03 | 3.4356154e-03 | 1.60e-16 | 2.8851422e-02 | 7.8434934e-02 | 6.55e-04 | 22 | yes |
| 64 | 0.01562 | 4096 | 8.9726140e-04 | 8.9726140e-04 | 2.49e-17 | 7.9583389e-03 | 3.9609045e-02 | 1.64e-04 | 21 | yes |
| 128 | 0.00781 | 16384 | 2.2623250e-04 | 2.2623250e-04 | 4.82e-17 | 2.1363115e-03 | 2.0237665e-02 | 4.10e-05 | 21 | yes |

分段观测阶（两点估计，对噪声更敏感）：

| segment | ratio | velocity order | pressure (mean-free) order |
|---|---:|---:|---:|
| 32→64 | 2 | 1.9370 | 1.8581 |
| 64→128 | 2 | 1.9877 | 1.8973 |

三网格最小二乘斜率：

| quantity | log-log least-squares slope (32/64/128) | R² |
|---|---:|---:|
| velocity L2 (full 3D u,v,w) | 1.9623 | 0.999944 |
| velocity L2 (in-plane u,v) | 1.9623 | 0.999944 |
| spanwise w RMS / |u_exact| (round-off, not a rate) | 1.4563 | 0.829783 |
| pressure L2 (fixed gauge, not a rate) | 0.9772 | 0.999975 |
| pressure L2 (mean-free gauge) | 1.8777 | 0.999964 |

判定：`{"threshold": 1.8, "quantity": "L2 relative error log-log least-squares slope over 32/64/128", "velocity_full_slope": 1.9623452238776438, "velocity_full_pass": true, "velocity_inplane_slope": 1.9623452238776438, "velocity_inplane_pass": true, "pressure_meanfree_slope": 1.8777242570668982, "pressure_meanfree_pass": true, "all_levels_converged": true, "pass": true, "pass_inplane": true}`

## 四、结果图

见本目录 `order.png`：左上为速度/压力 L2 对 h 的 log-log 误差与名义二阶参考线（h²）；
右上为误差预算（面内 u,v 与伪展向 w）；左下为 n=128 速度幅值云图；
右下为 y≈0.5 中心线 u_x 与解析解逐点对比。

## 五、注意点（踩过的坑 / 局限 / 适用范围）

1. **本单真正的阻塞不是算力，而是证据目录语义（已修复）**：`NativeRunEvidence` 原先对任何非空
   输出目录抛 `ValueError('evidence output must be empty')`，而本仓已把 n=128 的
   `environment.json`/`sources.json`/`source-snapshot`（纯溯源元数据）**提交入库**。于是该层级形成死锁：
   每次重跑在构造 evidence 时即抛错，永远写不出 `cases.json`。这解释了先前多单
   （m18408、m18510 与 cj1914/cj1944）"作业 succeeded 但目录仍空"的现象——rc 来自外层 wrapper，
   Python 早已 traceback。修复：只把本类自己写的元数据视为可复用；目录中一旦存在**测得产物**
   （cases.json / *.npz / summary.json）仍默认拒绝，除非显式 `overwrite=True`，且拒绝时不动任何文件。
   回归测试 `tests/test_native_run_evidence_resume.py`（7 passed）。
2. **展向 w 的斜率不是收敛阶**：|w| 已处舍入级（1.60e-16 …
   4.82e-17），其斜率（1.456，
   R² 0.83）是对舍入噪声的拟合，**不构成格式失准**。这正是
   `zero_gradient` 闭合的预期效果（伪展向模态比 `free` 闭合低约 13 个数量级）。
3. **固定规范压力阶不是精度量度**：动量方程只定压力梯度，固定规范下压力含随网格变化的常数偏差，
   斜率 0.9772；有物理意义的是去均值规范 1.8777。
4. **采样偏差已受控**：单元中心点值与单元平均之差逐层比该层 L2 误差小一个数量级以上
   （n=128：4.10e-05 vs 2.26e-04），
   故所测斜率不是比较口径（点值 vs 单元均值）造成的伪影。
5. **适用范围**：结论限于 Re=40、二阶对流、`zero_gradient` 闭合的耦合求解路径；不外推到 `free` 闭合。
6. **未做的部分（honest）**：历史 `free` 闭合序列 `runs-32-64-128/` 仍只有 n=32，故验收门中 8 项
   对这些层级 skip（"anchor artifacts absent"，诚实 not-measured）。该序列**不是判据所需**，且先验分析
   表明 `free` 闭合把全 3D 速度阶压到约 1.68；如需补全，应在**新目录**跑 n=64/128 后单独汇总，
   不可与 `zero_gradient` 混配（门会拒绝）。

## 六、误差判定（对基准逐项，显式回答是否 <3%）

基准 = Kovasznay 1948 解析解本身（误差定义为体积加权相对 L2）：

| 关键量 | n=32 | n=64 | n=128 | 三网格斜率 | 最细网格是否 <3% |
|---|---|---|---|---|---|
| 速度 L2（全 3D） | 3.4356154e-03 | 8.9726140e-04 | 2.2623250e-04 | 1.962 | **是**（0.0226%）|
| 压力 L2（去均值） | 2.8851422e-02 | 7.9583389e-03 | 2.1363115e-03 | 1.878 | **是**（0.2136%）|

判据"斜率 ≥ 1.8"：速度 1.9623 ≥ 1.8 → **通过**。

## 七、复现清单

| 工件 | 路径 |
|---|---|
| n=32/64/128 原始场与 cases | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/{n32,n64,n128}` |
| 汇总（斜率/判定/图） | 同目录 `summary.json` / `RESULTS.md` / `order.png` |
| 证据目录可续跑修复 | `src/tensorfvm/verification/native_run.py`（`METADATA_NAMES`）|
| 其回归测试 | `tests/test_native_run_evidence_resume.py` |
| 验收门（16 passed / 8 skipped） | `tests/test_kovasznay_32_64_128_acceptance.py` |

```bash
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --markdown docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/RESULTS.md \
  --figure docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/order.png \
  --require-grids 32,64,128
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py -q
```
