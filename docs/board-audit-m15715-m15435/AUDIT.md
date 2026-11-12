# 交差核验单 · m15435 CASE-MIXING-LAYER（TensorFVM 时间发展混合层）

审计任务：m15715 · 被审任务：m15435（succeeded）· 裁定：**L2**（真实功能实现，未达 L3/L4）
审计方式：独立取证 + 独立复算 + 独立参考实现交叉验证（不采信报告自述数字）
审计工作树：`/root/tp-agent/work/m15715`；被审工作树：`/root/tp-agent/work/m15435`（HEAD `d52eff4d`）

---

## 1. 三件套取证（逐项存在性）

| 项 | 指针 | 结果 |
|---|---|---|
| 实现 | `src/tensorfvm/mixing_layer.py`（509 行；`TemporalMixingLayer`:97、`rayleigh_modes`:388、`orr_sommerfeld_modes`:405、`heat_kernel_profile`:477、`layer_metrics`:500） | 存在，真代码（交错 MAC + FFT/DCT 精确投影 + Heun RK2，无 MOCK） |
| 生产者/入口 | `scripts/verify_mixing_layer.py`（626 行，`--stage {stability,diffusion,linear,nonlinear,similarity,report}`） | 存在，可执行 |
| run_artifacts | `docs/repro-mixing-layer-temporal/run/`：`summary.json`/`table.md`/`stability.json`/`diffusion.json`/`linear.json`/`nonlinear.json`(51825 行)/`self_similarity.json`/5 PNG/`manifest.json` | 存在；**manifest 12 个 sha256 全部与文件实际哈希一致**（0 mismatch） |
| 测试 | `tests/test_mixing_layer.py`（6 项） | 重跑 `PYTHONPATH=src python3 -m pytest tests/test_mixing_layer.py -q` → **6 passed in 19.87 s** |
| repro_cmd | `PYTHONPATH=src python3 scripts/verify_mixing_layer.py --output /tmp/mix --stage all` | 可执行；本审计分阶段重跑，stability+diffusion 重现值与提交值 **max abs diff 1.5e-12** |

## 2. 基准裁判（唯一裁判）

`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-MIXING-LAYER'` → **0 行**（该表共 21 行，同域有 `REPRO-FVM-SOD1978`/`CASE-HEATED-CAVITY-FVM`/`CASE-FORWARD-STEP`，无本锚点）。

结论：**本锚点在平台基准注册表中不存在**，机器裁判无可裁对象。任务书指定的 ref（文献卷并率，容差 <10%）在报告中为 `WITHHELD`（离网，knowledge engine :8021 Connection refused），bundle `docs/oss_solutions/REPRO-FVM-MIXING-LAYER.json` 的 `external_anchor.status` 亦为 `WITHHELD`。被审单唯一的关键因变量（卷并/扩展率 S=0.01665）**只有自收敛证据（0.42%），无外部基准**。

## 3. 独立复算（审计方自建参考，不调用被审代码）

审计方用**完全不同的离散**独立重算剪切层稳定性（Chebyshev 谱配点、y∈[-9,9]、壁面 Dirichlet、线性化广义特征值问题 `(U L − U'')φ = c L φ`）：`docs/board-audit-m15715-m15435/indep_chebyshev_shear.py`。

| 量 | 被审报告值 | 审计独立值 | 偏差 | 判定 |
|---|---:|---:|---:|---|
| Rayleigh σ(αδ=0.30) | 0.17148 | 0.17223 | 0.44% | 一致 |
| Rayleigh σ(αδ=0.50) | 0.18728 | 0.18743 | 0.08% | 一致 |
| Rayleigh σ(αδ=0.70) | 0.14602 | 0.14562 | 0.27% | 一致 |
| Rayleigh 峰值 σ | 0.18930 @ αδ=0.45 | 0.18954 @ αδ=0.45 | 0.13% | 一致 |
| Orr–Sommerfeld σ(Re=500, αδ=0.5) | 0.18372 | 0.18378（N=160，N=80/120/160 单调收敛 0.1988/0.1854/0.1838） | 0.03% | 一致 |
| OS Re=1e5（应退化为 Rayleigh） | 0.18726 | 0.18742 | 0.09% | 一致 |

即：报告引用的"独立稳定性参考"是**真算出来的**，不是伪造数字——这是本单给到 L2 而非更低的根据。

## 4. 缺陷清单（adversarial review）

**F1（中·可复现性与工件新鲜度）**：提交的工件与提交的生产者脚本**不同版本**。
`scripts/verify_mixing_layer.py:224` 时间步阶梯因子为 `(1.0, 0.75, 0.5)`；而对同一命令的实测重跑，`time_step_ladder` 的 dt = 0.025 / **0.01875 / 0.0125**，提交的 `linear.json` 却是 **0.025 / 0.0125 / 0.00625**（对应旧因子 1.0/0.5/0.25）。报告 ④b 表列出的 dt 列因此**无法用提交的 repro_cmd 重生**。增长率本身一致（相对差 2.6e-5），故不影响物理结论，但 `manifest.json` 的 sha256 认证的是**旧工件**。
**F2（中·对照表口径错位）**：④ 总表第 1 行 "instantaneous layer scale" 的 ref 打的是 0.18372（=冻结初值 OS 参考），误差却用瞬时层尺度参考（生产网格 0.17852，见 ④a）算出的 +2.087%。同一行 ref/got（0.18372/0.18224）手工复算得 −0.80%，与表内 +2.087% 矛盾。根因：`scripts/verify_mixing_layer.py:458-460` 该行 ref 字段取了 `prod['viscous_growth']`（冻结参考）而非该行实际使用的瞬时参考。判据结论不变（<10%），但公开对照表口径不可自证。
**F3（低·文字）**：REPORT ⑥ 把 2.09% 标 ❌ 并写 ">3% 未达"，而 2.09% < 3% 实为达标；同节结论段又自承 "数值上 <3%"。表述前后矛盾。
**F4（阻断 L4）**：基准注册表无该锚点行 + 文献锚点 WITHHELD，任务书指定的 ref 未取得；卷并率仅有自收敛。
**F5（阻断 L3）**：`grep -rn mixing_layer src/ --include=*.py` 除自身外**无任何调用者**；`src/tensorfvm/__init__.py` 未导出该模块，`backend_registry` 未注册；唯一消费者是自带 producer 脚本。对照先例 `tensor_maturity_verdicts#351`（m15542 声明 L3 → gate 因 `integration_callers` 为空降 2）。

## 5. 级别判定

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 只有文字 | 不适用（有代码/工件） |
| L1 数学/单元验证 | 单元级算法验证 | 达到（OS/Rayleigh 自收敛、热核精确解 4e-4%） |
| **L2 独立功能实现** | 真代码 输入→计算→输出 | **达到（裁定级别）**：509 行求解器 + 生产者脚本 + 6 项测试 + 可重跑工件；审计独立交叉验证数值真实 |
| L3 系统集成 | 接入主求解器/平台 | **未达**（F5） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 + 基准对照 | **未达**（F4：无注册基准、无外部锚点；F1 工件与脚本不同版本） |
| L5 科研成果 | — | 未达 |

**裁定：L2。** 差异说明：被审报告自述"已实现并对照通过"，其对照对象是审计方也能独立复现的**自算参考**（这一点为真，故给满 L2）；但锚点要求的文献卷并率 ref 为 WITHHELD、平台基准表无登记行，故"基准验证"不成立，L4 属虚高；且模块未接入主求解器，L3 亦不成立。

## 6. 未做/局限

- 未获取文献原文（离网），无法判定自算卷并率与文献卷并率的偏差——这正是 F4。
- 未做两波长（亚谐配对）与三维二次失稳复算；报告 ⑦ 已自认盒长不足，审计方面验证未做（not_run）。
- 未重跑 `--stage all` 一次性端到端（仅分阶段重跑 + 单次 report 阶段），时间预算内完成。
- 未修改被审工作树（只读审计）。
---

## 7. 复现实测补记（审计方亲跑，非报告自述）

命令与结果（均在 `/root/tp-agent/work/m15435` 只读执行，输出到 `/tmp`，未污染被审树）：

| 阶段 | 耗时 | 与提交工件比对 |
|---|---:|---|
| `pytest tests/test_mixing_layer.py` | 19.87 s | 6 passed |
| `--stage stability,diffusion` | 22.0 s | 与提交 JSON 逐字段 **max abs diff 1.5e-12**（863+85 数值字段） |
| `--stage linear,nonlinear` | 155.1 s | 增长率相对差 ≤2.6e-5；卷并率需 `report` 阶段按增长窗重拟合（缺该阶段时 nonlinear.json 的 spreading_rate 为全窗拟合 = −0.0043，**易误读**，属流程 footgun 非伪造） |
| `--stage stability,diffusion,similarity,report`（端到端） | 65.3 s | `table.md` **byte-identical**（sha256 `3b33c72d…`）；`spreading_rate` 三例 **rel diff 0.00e+00**；`self_similarity_deviation` 完全一致 0.022703567579791917 |

因此 F1 的影响面收窄为：**已发布的对照表可完全复现**，仅 `linear.json` 的时间步阶梯取点与 ④b 表 dt 列（0.025/0.0125/0.00625 vs 现行脚本 0.025/0.01875/0.0125）不可重生 —— 工件相对生产者脚本"陈旧"，但结论不变（三级 dt 上 σ 变化 <1e-5）。F1 定级由"中"降为"低（provenance 卫生）"。

## 8. 裁定落表

`tensor_maturity_verdicts`（repo=TensorFVM, scope=`m15435 CASE-MIXING-LAYER`, mission_id=15715, level=2, verdict_by=board-audit），evidence 八键齐备，见 `docs/board-audit-m15715-m15435/verdict.json`。建议返修项（不改变 L2）：①在 `tensor_benchmark_cases` 为本锚点登记 ref/容差行，或把 `external_anchor` 明确为"自算参考"并降级锚点定义；②修 `verify_mixing_layer.py:458-460` 的 ref 列取错；③修 REPORT ⑥ 的 ❌ 表述；④把 `mixing_layer` 导出/注册进 `backend_registry` 以获得 L3。
