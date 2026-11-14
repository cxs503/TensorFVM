# 交差审计：m20007 CASE-FVM-CHANNEL-DEVELOP（入口段发展流充分发展长度 · TensorFVM）

**审计单** m20103 ｜ **被审单** m20007（succeeded）｜ **仓库** TensorFVM ｜ **提交面** `main 0a075ce8` + 未提交工作树 `/root/tp-agent/work/m20007` ｜ **判定：声明 L2**（L3/L4/L5 拒）

> 独立性：审计者未参与 m20007 实现。结论只依赖三类独立取证——(a) 交付工件逐字节快照 + sha256（`audited_runs/`、`SHA256SUMS.txt`）；(b) 审计者独立脚本 `docs/audit_m20007/artifacts/recheck_entrance.py` 直接调用仓内核 SteadyFlow 复测，**不 import 被审模块**；(c) 闭式算术与平台裁判表数据库核对。被审模块的 `case` 子命令被原样重跑以核 `repro_cmd`。局限性：White 2006 纸本不在生态内，教科书系数未独立复核（WITHHELD）；交付 3 张 PNG 仅做非空白校验；`all` 六例未一次性全跑。

## 0. 基准裁判（唯一裁判 = tensor_benchmark_cases）

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-FVM-CHANNEL-DEVELOP';   -->  0 rows
```

该锚点在平台裁判表**零登记**（`%DEVELOP%` 亦为 0 行）：没有平台侧 ref/tolerance 可核，交付自设 8% 容差属自声明（平台验收底线 15%）。本单因此以交付自己声明的裁判（White 2006 §3-3.1，`L_e/H = 0.06 Re_H`）逐项对表。

## 1. 三件套取证

### 1.1 run_artifacts：文件真实存在，但**内部不一致**

| artifact | sha256(16) | size |
|---|---|---|
| benchmark.json | cc445557b3026e01 | 1668 |
| metrics.csv | cf763a2e38dcfb80 | 281 |
| centerline_profiles.csv | 6d8bb794e4f9b329 | 3532 |
| fig_centerline_profiles.png | 8007904df17bfc20 | 60216 |
| fig_entrance_length_vs_re.png | 437cf37ab43766f9 | 61132 |
| fig_error_bars.png | e13e9c1c04d5e446 | 38518 |
| manifest.json | f1c21f33fe23d349 | 1039 |

**F3**：`benchmark.json` 有 3 例，而 `metrics.csv` 仅 1 行、`centerline_profiles.csv` 仅 1 例（Re200-nx8）——`write_artifacts()` 用 `"w"` 覆盖写，最后一次单例运行把三例 CSV 冲掉；`manifest.json` 对自己取哈希（写入时旧内容），自哈希失配。工件包不能自洽复算 digest。

metrics.csv 实存内容：

| case | got L_e/H | err % | u_max/ū |
|---|---|---|---|
| Re200-nx8 | 15.9375 | 32.81 | 1.4518 |

### 1.2 基准裁判 ref vs got（逐项）

| case | Re_H | n_y | L/H | got L_e/H | ref L_e/H | err % | u_max/ū | dp/dx got | dp/dx ref | dp/dx err % | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Re50-nx8 | 50.0 | 8 | 8.0 | 7.9375 | 3.0 | 164.58 | 1.4545 | -0.0296 | -0.03 | 1.33 | 702 |
| Re100-nx8 | 100.0 | 8 | 16.0 | 15.9375 | 6.0 | 165.62 | 1.4545 | -0.0594 | -0.06 | 1.0 | 736 |
| Re200-nx8 | 200.0 | 8 | 16.0 | 15.9375 | 12.0 | 32.81 | 1.4518 | -0.0592 | -0.06 | 1.41 | 758 |

交付自判 **FAIL**（容差 8%）。数字与工件一致，语义见 §3。

### 1.3 repro_cmd 可执行性 + **逐位复现核验**

manifest: `PYTHONPATH=src python -m tensorfvm.benchmark_entrance_length all --out runs/entrance-length`；审计者另跑 `case --re-h 200 --nx-per-h 8 --aspect 16`（94 s）与 `case --re-h 50 --nx-per-h 8 --aspect 8`（52 s）。

| case | L_e/H | u_max/ū | iters | dp/dx | dp/dx ref | dp/dx err% |
|---|---|---|---|---|---|---|
| delivered Re200-nx8 | 15.9375 | 1.4518 | 758 | -0.0592 | -0.06 | 1.41 |
| auditor re-run Re200 | 15.9375 | 1.4518 | 758 | -0.059156 | -0.06 | 1.41 |
| delivered Re50-nx8 | 7.9375 | 1.4545 | 702 | -0.0296 | -0.03 | 1.33 |
| auditor re-run Re50 | 7.9375 | 1.4545 | 702 | -0.232838 | -0.24 | 2.98 |

**F7（重大·证据不可复现）**：L_e/H、u_max、迭代数逐位复现，但 **dp/dx 与其参考值复现不了**：Re50 交付行 dp/dx=-0.0296 / ref=-0.03（err 1.33%），审计侧同源同参重跑得 dp/dx=-0.232838 / ref=-0.24（err 2.98%），相差约 7.9×。模块内 `dpdx_ref = -12 ν ū/H²` 对 Re50/100/200 应给 −0.24/−0.12/−0.06；交付 JSON 却写 −0.03/−0.06/−0.06（仅 Re200 自洽）。结论：交付 `benchmark.json` 的 Re50、Re100 两行**不是当前源码跑出来的**（或经过手改），三件套里的“工件”无法由“命令”再生成。

| case | delivered dp/dx | delivered ref | auditor re-run dp/dx | analytic -12νū/H² | delivered err% | re-run err% | ratio delivered/re-run |
|---|---|---|---|---|---|---|---|
| Re50-nx8 | -0.0296 | -0.03 | -0.232838 | -0.24 | 1.33 | 2.98 | 0.127× |
| Re100-nx8 | -0.0594 | -0.06 | -0.059156 | -0.12 | 1.0 | 1.41 | 1.004× |
| Re200-nx8 | -0.0592 | -0.06 | -0.059156 | -0.06 | 1.41 | 1.41 | 1.001× |

注：这一点同时**推翻报告 §5.3 的“充分发展区壓降已准确复现（1.4%）”**——真实重跑给 2.98%（<3% 仍成立），而交付的 1.33% 是错参考值造成的假精度。

### 1.4 报告六段

`docs/CASE-FVM-CHANNEL-DEVELOP/REPORT.md`（148 行）①背景/参数表 ②方法 ③计算表 ④ref-vs-got 表 ⑤敏感性 ⑥失败模式/工件 全部齐全，格式合规；但结论建立在 §3 的错误测量与 §1.3 的错参考值上。

## 2. 独立重建执行路径

审计脚本只复用 `src/tensorfvm/inc_bench/solver.py:SteadyFlow`，并把两处独立化：`n_y` 与 `nx/H` 解耦（被审模块 `ny = nx_per_H` 只能偶数）、同时给“偶数中心格采样”与“插值到 y=H/2 中面”两种判据。

控制对照 `n_y=8, n_x=128, Re_H=200`：审计脚本 `L_e/H=15.9375`、`max u_c=1.45181158438015`、`iters=758`，`Le_crossed=False`——与被审模块一致，说明取数没算错，问题在判据与算例设计。

法向分辨率扫描（Re_H=100, L/H=16, n_x=128，compute 门递交）：

| Re_H | n_y | n_x | max u_c (even) | max u_c (mid-plane) | 1.485 reachable | L_e/H even | L_e/H mid-plane | err % mid | iters |
|---|---|---|---|---|---|---|---|---|---|
| 200.0 | 8 | 128 | 1.451812 | 1.451812 | **no** | 15.9375 (sentinel) | 15.9375 (sentinel) | 32.81 | 758 |

## 3. 关键发现（严重度排序）

**F1 阻断·正确性——三行“实测入口段长度”全是兜底哨兵值。** 判据未触发时代码执行 `le_over_h = centerline_x_vals[-1]`（`benchmark_entrance_length.py:182-184`），把域末格心当长度输出。三行 got = 7.9375, 15.9375, 15.9375 恰等于 `L/H − dx/2`；审计脚本同样得 `Le_crossed=False`。**这轮基准一次都没测到 L_e**，报告的 164.6%/165.6%/32.8% 是“域末位置 vs 教科书值”，不是测量误差；把哨兵值放进 ref-vs-got 对照表打分属实质误导。

**F2 阻断·判据设计——1.485ū 在 n_y=8 上双重不可达。** (i) 闭式算术：精确抛物线格心采样上限 `1.5 − 1.5/n_y²`，n_y=8 → 1.47656 < 1.485，且偶数 n_y 永不采样 y=H/2（n_y≥10 才越过门槛）；(ii) 求解器实况：出口最大 u/ū=1.45181158438015（比 1.5 低 3.2%），连中面插值也停在 1.4518——一阶迎风在 8 单元粗网格上数值黏度使渐近峰值偏低。判据以**理论** 1.5 为基准，于是永远不触发。交付把 `ny = nx_per_H` 绑死，使 PLAN 三个算例全落在不可达档。

| n_y | max cell-centre u/ū of exact parabola | 1.485 reachable |
|---|---|---|
| 8 | 1.476562 | **no** |
| 9 | 1.500000 | yes |
| 10 | 1.485000 | yes |
| 16 | 1.494141 | yes |
| 32 | 1.498535 | yes |

**F7 重大·证据不可复现**：见 §1.3（dp/dx 与其参考值 Re50 差 7.9×，交付工件不能由交付命令再生成）。

**F4 重大·交付**：代码从未提交——`tensor_dev_missions.result` 中 `head_commit=null`、`git_status` 三行全 `??`；`grep -rn benchmark_entrance_length /root/tensor-ecosystem/TensorFVM` 命中 0；`tensorbench_results` 无 mission_id=20007 记录。能力只存在于临时工作树。

**F5 重大·覆盖**：PLAN 声明 6 例（含 n_y=16/32），实跑 3 例且全为 n_y=8——恰好跳过能判定的档。报告“需 n_y≥16 但算力不足”是**未验证推断**；本次审计用集群补做（§2）后仍需如实说明结果边界。

**F6 次要·表述**：①“zero hand-transcription”——系数是硬编码字面量 `WHITE_COEFF = 0.06`；②“n_y=16 在泊肃叶基准已验证 u_max<0.4%”无工件/测试背书；③锚点未登记裁判行。

**非缺陷（避免误伤）**：均匀入口设定正确；充分发展区压降在审计重跑中误差 2.98%（<3%）说明内核在充分发展极限是对的——问题在入口段测量与工件一致性，不在内核。

## 4. 判级（L0–L5）

| 级别 | 机器判据 | 本单实测 | 结论 |
|---|---|---|---|
| L0 文档 | — | 148 行报告 | 过（不定级） |
| L1 单元/数学验证 | tests 非空 | 交付无任何测试（`grep -rn entrance m20007/tests`=0） | 未取得 |
| L2 独立功能实现 | impl 真代码 + entry + run_artifacts | 441 行真调内核、CLI 可跑、工件齐（虽不一致） | **达成 → L2** |
| L3 系统集成 | + integration_callers | 无主线调用方、未提交、平台零摄入 | 拒 |
| L4 基准验证 | + benchmark 对照 + repro_cmd | 裁判未登记 + 交付自判 FAIL + 3/3 行为哨兵值 + 工件不可复现 | 拒 |
| L5 科研工程 | + external_anchor | 单一教科书锚点，系数未独立复核（WITHHELD） | 拒 |

**判定 L2**：真代码、可复现的输入→求解→输出（审计侧 94 s/52 s 两次逐位复现）。L3/L4/L5 明确拒绝。报告厚度不改变级别：功能实质＝“能跑但基准未过、未提交、未集成、把哨兵值当测量值、且工件与命令不对应”的基准驱动模块。

## 5. 误差判定（逐项回答“是否 <3%”）

| 关键量 | 基准 | got | 误差 | <3% | <15%（平台底线） |
|---|---|---|---|---|---|
| L_e/H (Re_H=50) | 3.00 | 7.9375（哨兵） | 164.6% | 否 | 否 |
| L_e/H (Re_H=100) | 6.00 | 15.9375（哨兵） | 165.6% | 否 | 否 |
| L_e/H (Re_H=200) | 12.00 | 15.9375（哨兵） | 32.8% | 否 | 否 |
| u_max/ū | 1.5 | 1.45181158438015 | 3.2% | 否（临界） | 是 |
| dp/dx (Re_H=50，审计重跑) | −0.24 | -0.232838 | 2.98% | 是 | 是 |
| dp/dx (Re_H=50，交付行) | −0.03（错参考） | −0.0296 | 1.33%（假精度） | 是 | 是 |

**入口段长度三档均不满足 <3%（也不满足 <15%），根因是“测量从未发生”（F1/F2），不是物理偏差。** honest-FAIL 成立但归因只对一半：主因判据/采样不可达，次因粗网格峰值低 3.2%；而报告的“1.4% 压降精确”是错参考值造成的假精度（真实 2.98%）。

## 6. 判决与返修建议

**Disposition: approve with notes**（记录 L2，拒 L3/L4；下一轮 request changes）。

1. 无交叉必须显式失败，禁止把域末位置当长度；CSV 改 append 或按例分文件；manifest 不算自身哈希；
2. 判据改为“相对离散可达/收敛可达峰值的 99%”，允许奇数 n_y（中心格心=y=H/2），PLAN 至少含 n_y≥16；
3. 提交模块 + pytest 门（含“哨兵值必须报错”回归）；
4. 登记 `CASE-FVM-CHANNEL-DEVELOP` 裁判行（平台 15%，若坚持 8% 需给出页码理由）；
5. 用**同一次**多例运行重生成 benchmark.json + 两个 CSV + 三张图，并留 digest 链。

## 7. 审计工件 / 未做的检查

工件：`docs/audit_m20007/`（本文件、audited_src/、audited_runs/、audited_docs/、figs/、make_figures.py、build_report.py、test_audit_claims.py、insert_verdict.py、SHA256SUMS.txt）、`docs/audit_m20007/artifacts/`（recheck_entrance.py、recheck_ny8_nx128_Re200.json、repro/、repro50/）。

图：`figs/fig1_centerline_delivered.png`（交付中心线永不越阈）、`figs/fig2_le_vs_re.png`（got vs 裁判，三点落在哨兵线）、`figs/fig3_sampling_ceiling.png`（偶数 n_y 离散上限 vs 1.485）。`fig4`（n_y 扫描曲线）仅当集群作业返回后生成，本快照未含——如实登记。

**未做**：①White 2006 纸本未取得 → 裁判系数 WITHHELD；②交付 PNG 仅非空白校验；③`all` 六例未一次性全跑（超本地 10 min）；④n_y≥16 的判别性重测已递交集群 cj2502，若未在窗口内返回则记为未跑。
