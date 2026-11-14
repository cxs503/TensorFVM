# CASE-WOODWARD-COLELLA — 第三方代码横向复核（Clawpack/PyClaw）：清账"自证"欠账

**锚点**：`XVAL-FVM-M18737-ROUND2`（feature_ledger，总账标题「TensorFVM 的 m18737-round2 做法」；其对象锚 = `CASE-WOODWARD-COLELLA`，ledger `open`）
**验收**：`【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】`；本单另按 owner 精度标准逐项回答"是否 <3%"
**仓**：TensorFVM —— 本单只动 `docs/`、`tests/`（求解器与指标代码零改动）
**判定**：**第三方代码独立复现 TensorFVM 的数字**（nx=800/1600 峰值 0.40/2.32 %、L1 1.77/1.63 %，对更正后参考）；两码在匹配网格上互差 L1 1.18/0.73 %（nx=800/1600）。既有 <10 % 裁决不再是"本仓自证"。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

一维闭管 [0,1] 内两束相向爆轰波相互作用（Woodward & Colella 1984 标准算例 B3）。两束激波+接触+稀疏在中部对撞（t≈0.028）后形成强干涉波系。

| 参数 | 值 | 来源 |
|---|---|---|
| 域 / 维数 | x∈[0,1]，1D | W&C 1984（经 Dumbser 2019 §5.1.4、Fu 2021 Eq.(37) 两条开源复述逐字核对） |
| 初始 (ρ,u,p) | (1,0,10³) x≤0.1；(1,0,10⁻²) 0.1<x<0.9；(1,0,10²) x≥0.9 | 同上 |
| γ / 终止时刻 | 1.4 / t=0.038 | 同上 |
| 壁面 | 两端反射闭管 | 同上 |
| 被审计格式 | MUSCL-Hancock（MC 限幅）+ HLLC + SSP-RK，CFL 0.4（TensorFVM） | `src/tensorfvm/compressible/` |
| 第三方格式 | Clawpack 5.14.0 Classic 波传播法，MC 限幅，Roe+熵修正 / HLLE，CFL 0.4 | `/usr/lib64/python3.11/site-packages/clawpack`（loongarch64 二进制） |

**为什么本单必要**：此前唯一的"交叉码"证据 `xcode/wc_crosscode.py`（WENO5/Rusanov）住在**同一仓库**、共用网格/ghost/调用点代码，只能排除"离散格式选错"，排不掉仓库级系统错；`docs/sod_xval_report.md` 曾记录 PyClaw 在 loongarch64 上无法编译而**降级为单参照**。现在 Clawpack 5.14.0 提供了 loongarch64 二进制，这条欠账可以还。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. `docs/woodward-colella-double-blast/pyclaw/wc_pyclaw_xval.py`：用 PyClaw `ClawSolver1D`（Classic）在同一初始数据、同一反射壁、同一终止时刻上求解，网格 200/400/800/1600，两套 Riemann 解算器（Roe+熵修正 `euler_with_efix_1D`；HLLE `euler_hlle_1D`，后者不解析接触间断 → 给出"通量模型"宽度）。
2. **指标零改动复用** `tensorfvm.compressible.wc_reference.metrics_from_profiles`：同一套 peak/peak_x/L1/window_mass/low_plateau/faces 定义与阈值，只换参考表（原始 `reference_density.csv` 与更正 `reference_density_calibrated.csv` 两套并列报告，honest）。
3. 三个横向量：第三方 vs 文献参考；第三方 vs TensorFVM（匹配网格，L1 相对范数）；Roe vs HLLE（同码换通量）。
4. 自检门：闭管质量守恒；同码异通量互差带宽；`t` 是否到达 0.038（未到者本单不采信）。
5. 回归门 `tests/test_compressible_wc_pyclaw_xval.py`（7 例，读存档 JSON，不重跑求解器；Clawpack 缺失时 skip）。

**踩坑（已固化在脚本注释里）**：Clawpack 的 Euler 守恒量约定是 q=[ρ, ρu, E_total]，**E 是总能密度而非压力**；按压力写入会得到一个"软"解（峰值 5.99 @x=0.734、L1 vs 参考 107 %），这个错相与"求解器精度差"极易混淆——正是第三方跑之前必须做 setup 自检的原因。另：`solver.cfl` 在 5.14 是对象，要设 `cfl_desired/cfl_max`；`nsteps` 计数不可用（脚本回填 0，报告不引用）。

## ③ 计算结果（数字表，程序化注入 `pyclaw/results.json`）

对**更正后**参考（nx=800 生产档；nx=1600 为最细可信档）：

| 代码 | nx | 峰值 ρ_max 误差 | 峰值位置误差 | L1 剖面误差 | 窗质量误差 | 低平台误差 | 裁决 |
|---|---|---|---|---|---|---|---|
| TensorFVM MUSCL/HLLC | 800 | 2.03 % | 0.0024 | 2.01 % | 1.41 % | 0.18 % | PASS |
| **Clawpack Roe+efix** | 800 | **0.40 %** | 0.0007 | **1.77 %** | 0.02 % | 0.12 % | **PASS** |
| Clawpack HLLE | 800 | 0.34 % | — | 4.83 % | 0.03 % | 0.36 % | PASS |
| TensorFVM MUSCL/HLLC | 1600 | 2.34 % | 0.0020 | 1.86 % | — | 0.31 % | PASS |
| **Clawpack Roe+efix** | 1600 | **2.32 %** | 0.0010 | **1.63 %** | 0.00 % | 0.27 % | **PASS** |
| Clawpack HLLE | 1600 | 43.04 %（见⑤-3） | — | 4.54 % | 0.01 % | 0.07 % | FAIL |

**两码互差（匹配网格，L1 相对范数）**：

| nx | Clawpack-Roe vs TensorFVM | Clawpack-HLLE vs TensorFVM |
|---|---|---|
| 400 | 2.06 % | 5.26 % |
| 800 | **1.18 %** | 4.26 % |
| 1600 | **0.73 %** | 5.38 % |

**收敛（Clawpack-Roe，对参考 L1）**：400 → 6.28 %、800 → 1.77 %、1600 → 1.63 %（单调下降，800 之后进入 ~1.7 % 平台，与 TensorFVM 的 2.0→1.9 % 平台一致）。
**守恒**：闭管质量最大相对漂移 1.2e-3（Classic 壁面 ghost 的求积效应），能量积分守恒同量级。
**Roe vs HLLE 带宽**：3.97–4.93 %（全部网格），即"通量模型"宽度 **大于** 两码互差（0.73–1.18 %）。

## ④ 结果图

`docs/woodward-colella-double-blast/pyclaw/fig_wc_pyclaw_xval.png`（两栏）：
(a) nx=1600 密度剖面：TensorFVM（实线）、Clawpack-Roe（虚线）、Clawpack-HLLE（点线）、更正后文献参考（黑细线），近真空平台区与激波峰放大插入；
(b) 误差-网格：Clawpack vs 参考 L1、TensorFVM vs 参考 L1、两码互差 L1 三条曲线同图，直观看出"两码互差 < 各自与数字化参考之差" → 残余属参考保真度而非任一求解器。
（一维算例无二维场量云图；云图不适用，故以剖面 + 误差曲线代替，理由在此声明。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **Clawpack 守恒量约定**：q[2] 必须是总能密度；写成压力会得到峰值 −3.5 %、峰值位置 −0.042（x=0.734 vs 0.776）的"看起来还行"的错误解。脚本已在注释固化，且 setup 自检（质量守恒 + 同码异通量带宽）能抓住这类错。
2. **HLLE 不可作为本算例的第二成员**：HLLE 不解析接触间断，本算例接触间断极强，nx=1600 峰值误差 43 %（而 nx=800 只有 0.34 %，非单调 → 强间断处的熵/正性处理主导，非收敛量）。HLLE 只用于给"通量模型宽度"上界（4–5 %），不参与裁决。
3. **nx=3200 的 Clawpack 档未取得**：`evolve_to_time` 在 t≈0.0285 提前停止（对撞时刻，疑为 CFL 台账步长拒绝或 Roe 在近真空处的正性失败），误差 342 %。本单**不采信、不引用**该档；已用 `--grids 200 400 800 1600` 限定产物，`fail_stage` 性质写明。要补需改为 HLLC/加正性修正，属下一步。
4. **参考解仍是数值解**：更正后参考 = RPJ@3200 绘图曲线的矢量化数字化；<3 %/<10 % 的语义都是"与已发表数值解一致"。两码互差 0.73 % 说明"求解器误差"已 ≪ "参考保真度"（~1.6–1.8 % 平台）→ **判决精度到 1.6 % 就到底了**，再收紧阈值对任何代码都无意义。
5. **壁面质量漂移 1.2e-3**：Classic 的 wall ghost 实现 + 求积，不是物理泄漏；TensorFVM 同算例漂移 2e-16。跨码比守恒时应比"量级"不比"机器零"。
6. **适用范围**：结论只对 1D 双爆轰波、t=0.038、γ=1.4 成立；三维/AMR 不在射程。

## ⑥ 误差判定（逐项 vs 基准，显式回答"是否 <3%"）

| 关键量（基准=更正后文献参考） | TensorFVM nx=1600 | Clawpack-Roe nx=1600 | <3 %? |
|---|---|---|---|
| 峰值密度 ρ_max | 2.34 % | 2.32 % | **是**（两码） |
| 峰值位置 | 0.0020 | 0.0010 | 是（位置容差 0.02） |
| 剖面 L1 | 1.86 % | 1.63 % | **是**（两码） |
| 窗质量积分 | — | 0.00 % | 是 |
| 低密度平台 | 0.31 % | 0.27 % | 是 |
| 两码互差 L1（独立误差上界） | — | 0.73 % | **是** |

**结论**：<10 % 总账容差在两套独立代码上均满足且余量充足；<3 % 亦满足。残留 ~1.6 % 经"两码互差 0.73 %"识别为**参考保真度地板**，不是求解器缺陷。**诚实申报未完成项**：Clawpack nx=3200 档未取得（提前停止于 t=0.0285，已排除在结论外）。

### 复现命令

```bash
PYTHONPATH=src python docs/woodward-colella-double-blast/pyclaw/wc_pyclaw_xval.py --grids 200 400 800 1600   # ~6 s
PYTHONPATH=src python -m pytest tests/test_compressible_wc_pyclaw_xval.py -q                                   # 7 例
```

### 证据指针

- `docs/woodward-colella-double-blast/pyclaw/results.json`（全部数字，程序化注入）
- `docs/woodward-colella-double-blast/pyclaw/profiles.npz`（10 条剖面：5 网格 × 2 通量，含 p、u）
- `docs/woodward-colella-double-blast/pyclaw/fig_wc_pyclaw_xval.png`
- 前序：`docs/CASE-WOODWARD-COLELLA-final-verdict.md`（自证裁决）、`docs/CASE-WOODWARD-COLELLA-calibration-bias.md`（参考更正）、`docs/CASE-WOODWARD-COLELLA-GCI-report.md`
- 本单对总账的建议：`XVAL-FVM-M18737-ROUND2` 可判 done；`CASE-WOODWARD-COLELLA` 具备结项条件（<10 % 已由两条独立代码满足），若 owner 要求"原文包络"则仍需 1984 原文（闭源，WITHHELD）。