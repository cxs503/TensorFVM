# CASE-FVM-KOVASZNAY-ORDER —— 第 2 轮：调研结论、方法选型与迁移方案

锚点判据（总账原文）：
`【算例池|FVM-KOVASZNAY-ORDER｜ref: Kovasznay 1948 Proc. Camb. Phil. Soc. 44:58 精确解（网格收敛阶验证）
DOI:10.1017/s0305004100023999｜容差: L2 收敛阶 >=1.8 (名义二阶) | 指标: log-log 斜率 on 32/64/128 三套网格】`

本轮作业方式：先调研（开源横向 + 方法谱系），再选型，最后结合实现。落库条目：
`oss_solutions#414 (repo=TensorFVM, feature_slug=CASE-FVM-KOVASZNAY-ORDER)`。

---

## 0. 前轮（m16068）失败归因 —— 排序信号怎么读

`tensor_dev_missions#16068`：`status=failed`，`fail_stage` 空，`result={"stop_condition": "agent_lost"}`。
即**不是数值失败，是作业体丢失**：任务目标要求 32/64/128 三套网格（单元数 1 024 → 16 384 → 65 536，
本仓 `triangulate_faces` 默认把六面体劈成棱柱后为 2 048 / 32 768 / 131 072），而 2D 稀疏 LU 的直接解
代价 ≈ O(n³)：实测 n=16 解算 15 s、n=32 91 s、n=64 ≈ 7-10 min、n=128 ≈ 1-1.7 h。把整条网格梯子放进单次
前台执行，必然撞上 agent 的 wall-clock 预算 → `agent_lost`。

**对策（本轮迁移方案的第 1 条，也是唯一真正改变结果的工程决定）**：网格梯子拆成**每网格一个集群作业**
（`run_cluster submit` + `stage_dir` 直送工作树），本地只做 O(秒) 的合并与斜率拟合。单个作业失败也不会
把整条梯子带走，且 32/64 可以并行。

## 1. 开源横向调研（`~/oss-cache`）

已按任务榜要求先查同类实现源码：

```
$ ls ~/oss-cache
Maarten-vd-Sande_lbm   hschwane(GraSPH)   pylbm
$ grep -ril kovasznay ~/oss-cache | head
(无输出)
$ grep -rli 'observed order|order of accuracy|convergence order' ~/oss-cache --include=*.py | head
(无输出)
$ grep -rli convergence --include=*.py ~/oss-cache | head
pylbm/demo/2D/Poiseuille_vec.py   pylbm/demo/3D/poiseuille.py
```

结论（诚实）：**缓存里没有可直接抄结构的同类实现** —— 3 仓全是 LBM/SPH 系（`pylbm` f9f5d16 2026-11-09、
`Maarten-vd-Sande_lbm` 573a01c、`GraSPH` 无 git），与"不可压 FVM 的 Navier–Stokes 精确解阶验证"不同谱系；
唯一沾边的 `pylbm` Poiseuille demo 只做数值解与解析剖面的定性对比，**没有阶验证/斜率估计**这一层。
因此本项目的方法不采自 OSS，而采自文献与 V&V 标准（下节），并把"为什么抄不到"记进
`oss_solutions.transfer`（供其他 FVM 仓不再重复这次检索）。

## 2. 方法谱系（external_anchor）

| 层 | 出处 | 在本项目的用法 |
|---|---|---|
| 精确解 | Kovasznay, L. I. G. (1948), *Proc. Cambridge Philos. Soc.* **44**:58–62, doi:10.1017/S0305004100023999 | unforced 稳态 NS 精确解：κ=2π，λ=−2κ²/(Re+√(Re²+4κ²))，u=1−e^{λx}cos κy，v=(λ/κ)e^{λx}sin κy，p=½(1−e^{2λx})；恒等式 λ²−Re·λ=4π² 使对流项+压力梯度−黏性项严格为零，**不需要离散算子反算源项**（避免"源项自洽"这种自证陷阱） |
| 阶验证规程 | Roache, *Verification and Validation in Computational Science and Engineering* (1998), ch. 5（GCI / grid refinement） | 用同一离散格式在同一族几何相似网格上取 L2 误差序列，`p = log(e_coarse/e_fine)/log(h_coarse/h_fine)` |
| 报告口径 | Celik et al. (2008), *ASME J. Fluids Eng.* **130**:078001（ASME V&V 20 的 CFD 附录） | 三网格（而非两网格）报告：成对观测阶 + 最小二乘整体斜率 + R²，并显式声明接受阈值（1.8）在**看结果之前**定死 |
| 同族验收纪律 | ITTC 7.5-03-01-01（数值不确定度分析） | 网格加密序列要同时给出守恒残差/收敛证据，收敛不到就不许报阶 |
| 已知坑（文献侧） | Ferziger & Perić, *Computational Methods for Fluid Dynamics*, 3rd ed., §7（同位网格压力） | 全 Dirichlet 速度边界下，**同位/单元中心**格式的压力在近壁层只有局部一阶；因此压力误差必须区分"含常数零空间分量"与"去均值"两种范数分别报告 |

判据口径的选型（**先定后测**，不看结果改口径）：

- **主指标**：速度体积加权相对 L2 误差 `e_u(n)` 对 `h=1/n` 的 log-log 最小二乘斜率，网格 32/64/128，
  阈值 **1.8**。理由：速度是携带 Dirichlet 边界的原始未知量，也是 Kovasznay 阶验证在文献中的标准量。
- **次指标（如实并列，不参与阈值判定）**：压力去均值相对 L2 斜率、压力定标（fixed gauge）相对 L2 斜率。
- **采样一致性控制**：单元中心 FV 未知量是**单元平均**，而比较量是形心处的点值 —— 二者相差 O(h²)。
  同网格上再测一次"点值 vs 单元平均"的采样偏差，只有当该偏差的衰减不低于被测阶时，斜率才可信。

## 3. 本轮迁移方案（怎么把方法落到本仓）

1. `scripts/verify_kovasznay_order.py --n N --output runs/kovasznay-order/nN`：单网格一个进程，
   `kovasznay_solver(n, Re=40, symmetry_extrusion=True, coupled_linear_method='reused_lu')`，
   Picard 外迭代 relaxed 0.7 / tol 1e-8 / max 120 步；落 `cases.json`（误差 + 收敛 + 守恒残差 +
   采样偏差）、`nN-{outer,inner,linear}.json`、`nN-fields.npz`、`sources.json`（源码 sha256，跑完自校验）。
2. `scripts/summarize_kovasznay_order.py --runs n32 n64 n128 --output summary`：成对观测阶 + 三网格
   最小二乘斜率 + 守恒门槛 + `acceptance_passed`（velocity slope ≥ 1.8 且三套网格全部 `numerically_converged`）。
3. 集群分网格递交（见 §0），本地只跑第 2 步。
4. 结果、图、逐项判定由脚本程序化注入 `docs/case-kovasznay-order-order-verification/`（**零手抄**）。

## 4. 边界与局限（写在前面，避免事后找补）

- 该算例只验**层流稳态二阶空间格式**，不含湍流模型（SST/SA）、不含时间推进阶、不含动网格。
- 挤出方向（z）只取 1 层单元、两个对称面，`w` 不恒为零（见报告 §5 的 `max_abs_z_velocity`），
  因此严格说是"平面流的 3D 退化算例"，不是完整三维阶验证。
- 阶斜率 ≥1.8 只说明**离散格式在光滑解上达到名义二阶**，不能用它推断复杂几何/高 Re 的精度。