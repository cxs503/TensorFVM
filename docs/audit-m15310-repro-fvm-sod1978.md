# 交差审计 · m15310 · REPRO-FVM-SOD1978（TensorFVM）→ 判级 L3

审计任务 m15344（`board-audit`）· 被审任务 m15310（`repro-fvm-sod1978-r2`，status=succeeded，
finished_at 2026-11-11 22:20:40）· 被审仓库 TensorFVM · 审计时仓库 HEAD `dfd36259`
（= m15310 工作树 HEAD）· 唯一裁判 = `tensor_benchmark_cases` 行 id=15（anchor_slug `REPRO-FVM-SOD1978`）。

**结论**：三件套（run_artifacts / 基准 ref-vs-got / repro_cmd）在**第 1 轮提交物**上真实存在且可逐位复现；
但**基准判据 C3（L1 阶 ≥ 1.0，literal，unchanged）实测 FAIL**，且第 2 轮（本轮）交付物**全部未提交**、
其唯一工件是 `--quick` 冒烟配置。故 **L4（基准验证）不予认定，判 L3**；`maturity_gate` 预期 reopened。

---

## ① 案例介绍（被审对象）

Sod (1978) 激波管，外部锚点 Sod, G. A., *J. Comput. Phys.* **27** (1978) 1–31：

| 项 | 值 |
|---|---|
| 方程 | 1-D Euler，γ=1.4，x∈[0,1] |
| 初值 | (ρ,u,p)=(1,0,1) x<0.5；(0.125,0,0.1) x>0.5（膜片 x0=0.5） |
| 判据时刻 | **t=0.14**（论文图 3 为 t=0.2，任务显式声明偏差并另给 t=0.2 运行） |
| 数值 | HLLC + MUSCL-Hancock（order 2）/ donor cell（order 1），limiter=mc，CFL=0.4，outflow |
| 发布参考量 | p\*=0.30313、u\*=0.92745、ρ\*_L=0.42632、ρ\*_R=0.26557；S_head=−1.18322、S_tail=−0.07042、S_contact=0.92745、S_shock=1.75216 |

基准裁判原文（`tensor_benchmark_cases` id=15，status=**pending**）：

> C1 four wave positions; C2 star state; C3 whole-field L1 convergence order; R3 analytic-region order;
> R5 smooth entropy-wave design order — 容差：C1 <2%；C2 <5%；**C3 >=1.0 (literal, unchanged)**；
> R3 >= design order\*0.85；R5 within 3% of 1.0/2.0

## ② 怎么审的（方法）

1. **取证**：读 `/root/tp-agent/work/m15310` 工作树、`git status`/`git ls-files`、`docs/repro-fvm-sod1978*`、
   `src/tensorfvm/verification/sod1978*.py`、`scripts/`、`tests/`；读 `tensor_dev_missions.result`（m15310）与
   `tensor_reports` id=14406。
2. **独立复算**（不复用被审方结论）：在**干净工作树** `/root/tp-agent/work/m15344`（同 HEAD `dfd36259`，
   无本轮未提交改动）跑 `repro_cmd`，把 got 数值与已提交工件逐位比对；
   在 m15310 工作树重跑两个测试文件。
3. **判据逐条打勾**：C1/C2/C3/R3/R5 用基准行容差，而不是报告自述。
4. **证据链审计**：工件是否入库（`git ls-files`）、工件配置是否等于声明配置、`head_commit` 是否为空、
   repro_cmd 的**真实退出码**。

## ③ 计算结果（ref vs got；零手抄，全部取自工件）

### 3.1 C1 波位置（t=0.14，nx=400，提交工件）

| 量 | ref | got | 相对误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| 稀疏波头 x | 0.334350 | 0.333410 | 0.28093% | <2% | PASS |
| 稀疏波尾 x | 0.490162 | 0.491952 | 0.36524% | <2% | PASS |
| 接触间断 x | 0.629843 | 0.629651 | 0.03049% | <2% | PASS |
| 激波 x | 0.745302 | 0.745543 | 0.03232% | <2% | PASS |

### 3.2 C2 星区（平台均值，提交工件）

| 量 | ref | got | 相对误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| ρ\*_L | 0.42632 | 0.426284 | 0.00850% | <5% | PASS |
| ρ\*_R | 0.26557 | 0.265579 | 0.00363% | <5% | PASS |
| u\* | 0.92745 | 0.927582 | 0.01421% | <5% | PASS |
| p\* | 0.30313 | 0.303092 | 0.01252% | <5% | PASS |

参考值独立性：模块自带 NumPy 精确 Riemann 解（对压力函数二分），与发布值最大差 **0.209%**
（稀疏波尾速度 S_tail，其余 ≤0.0014%）——参考面不是被评代码本身。

### 3.3 C3 全域 L1 收敛阶 —— **FAIL**

| 方案 | nx 100→200 | 200→400 | 400→800 | 判据 | 判定 |
|---|---|---|---|---|---|
| order 1 | 0.6170 | 0.6507 | 0.6600 | ≥1.0 | **FAIL** |
| order 2 | 0.9770 | 0.8943 | 0.8573 | ≥1.0 | **FAIL** |

（单元平均控制实验：order 2 ﬁnest 0.8707，order 1 ﬁnest 0.5892 —— 采样口径不改变结论。）

### 3.4 R3 / R5

* **R3**（解析区内缩窗）：仅第 2 轮工件有，quick 网格 order 2 相邻率 **1.8215**（≥ 2×0.85=1.70 → PASS），
  3 网格指数拟合 2.0316；但取自 50–200 冒烟网格，非声明网格集。
* **R5**（平滑熵波设计阶）：order 1 = 0.9936/0.9968/0.9985（|Δ|≤0.64%，PASS）；
  order 2 = **2.1322 / 2.0958 / 2.0454**。窗口一换结论就翻：最细对 100–800 单边界通过（+2.27%，单侧 3% 内），
  最粗对 50–100 = **1.9382**（−3.09%，超出单侧 3% 带）→ 第 2 轮工件 `smooth_design_order_met.order2=false`，
  第 1 轮工件 `C3_design_order_within_2pct_of_design=true`（单侧 `min(rate) ≥ 2*(1-0.02)`）。**同一锚点、同一格式，两个工件给相反判定**。

### 3.5 守恒/正性

质量漂移 0.0，能量漂移 ≤1.6e−16，`positivity_fixes=0`（全网格）→ 无数值修补掩盖 FAIL。

### 3.6 复算一致性（本次审计亲自跑）

```
PYTHONPATH=src python scripts/repro_fvm_sod1978.py --out /tmp/audit_sod_partial \
    --grids 100,200,400 --orders 1,2 --no-figures        # 干净 HEAD dfd36259
```
* C1/C2 八行 ref/got/误差与提交工件**完全相同**（0.28093/0.36524/0.03049/0.03232；0.00850/0.00363/0.01421/0.01252）；
* `l1_orders_sod` order1 = 0.6170296569436852, 0.6507461456683455；order2 = 0.9769815038332116, 0.8943030492362277（逐位相同）；
* `verdicts` 字典与工件**逐键相等**，含 `all_mission_criteria_met=false`；
* **真实退出码 = 1**（脚本 `return 0 if verdicts['all_mission_criteria_met'] else 1`），即 repro_cmd 可执行但按判据自报失败。

```
PYTHONPATH=src python -m pytest tests/test_repro_fvm_sod1978.py -q   → 6 passed in 40.94s   (m15344)
PYTHONPATH=src python -m pytest tests/test_sod1978_convergence.py -q → 14 passed in 15.88s  (m15310)
```

## ④ 结果图

`docs/audit-m15310-repro-fvm-sod1978/m15310_audit_orders_vs_criterion.png`
（`make_figures.py` 从提交工件 JSON 直读生成，sha256 记于 `figure_summary.json`）：

* 面板 A：Sod 全域 L1 阶 vs C3 底线 1.0 —— 两条曲线全部落在底线之下（0.617–0.977）；
* 面板 B：平滑熵波设计阶 vs 声明带 —— order 2 三点 2.045–2.132 全部在 2.0 上方，"是否在带内"取决于选哪一对网格。

## ⑤ 注意点（分级发现，adversarial-review 口径）

**B1（blocking，基准）** 基准行 C3 实测 FAIL：order 1 = 0.617/0.651/0.660、order 2 = 0.977/0.894/0.857，
判据 `>=1.0 (literal, unchanged)`。本轮任务 `succeeded` 系
`TP_AUTO_APPROVE_DEV_ACCEPTANCE`（admin 自动批准），**行政成功 ≠ 判据满足**；工件自己写着
`all_mission_criteria_met=false`、`c3_literal_met=false`。→ L4 不可认定。

**B2（blocking，证据完整性）** 第 2 轮交付物**全部未提交**：`git status` 显示
`?? docs/repro-fvm-sod1978-convergence/`、`?? scripts/sod1978_convergence_study.py`、
`?? src/tensorfvm/verification/sod1978_convergence.py`、`?? tests/test_sod1978_convergence.py`、
`M src/tensorfvm/compressible/reconstruct.py`（+13/−2，新增 superbee 限制器）；`tensor_dev_missions.head_commit = null`；
`git ls-files` 在 HEAD 上查不到任一 `sod1978_convergence*` 文件。后果：工件只存在于本地工作树，仓库无法复现；
且 `--limiters mc,vanleer,superbee` 依赖未提交的 `superbee`（干净 HEAD 的 `LIMITERS` 只有
`minmod/mc/vanleer`），**按 main 的代码跑第 2 轮 CLI 会直接 ValueError: unknown limiter 'superbee'（实测）**。

**B3（blocking，声明 vs 工件配置）** 第 2 轮唯一工件 `study.json` 的 `config.grids=[50,100,200]`、
`smooth_grids=[50,100]`、`limiter_grids=[100,200]`、`cfl_values=[0.2,0.4]` —— 即 `--quick` 冒烟配置；
CLI 默认声明是 `100..6400`。`tensor_dev_missions.result` 自述"集群 cj1062 至本单结束仍在 queued、未取回"。
因此 `verdicts` 里的 `c3_asymptotic_met`、`sublinear_convergence_observed`（渐近结论）**无细网格数据支撑**。
工件 `verdicts.mechanism` 以陈述句写死 Banks/Aslam/Rider 机制；报告 §⑤2 却写"尚未被本数据确证" —— 工件与报告口径不一致。

**B4（blocking，裁判独立性）** 基准行 id=15 的 `created_at = 2026-11-11 22:20:21`，落在 m15310 窗口
（21:58:13–22:20:40）内、比本任务报告（22:20:20）晚 1 秒；m15310 自己的 result 写"DB：oss_solutions 与
tensor_benchmark_cases 各插入 1 行"。即**裁判行由被审任务在收尾时自建**且遗留 status=pending。
判据文字与历史轮次一致（"literal, unchanged"）故不构成放宽，但"自己设定裁判"必须记账：
本审计的判定只采用该行文字与**实测数值**对齐，不采用被审方任何"通过"自述。

**N1（non-blocking，可复现性隐患）** R5/R3 这类"阶"判据对网格窗口极敏感（见 3.4），
但基准行文字未规定窗口；两轮工件因此给出相反判定。建议基准行补窗口字段（如"R5 用 100→800 相邻对"），
经基准变更流程落地，不允许在核验侧调容差。

**N2（non-blocking）** `docs/repro-fvm-sod1978-convergence/REPORT.md` 的诚实度良好（明写细网格未跑、机制未确证、
C3 字面 FAIL），是本任务唯一没被推翻的自述；问题在工件与提交状态，不在报告粉饰。

**正向证据（支撑 L3 而非更低）**：第 1 轮真实代码 852 行（独立精确 Riemann 参考面 + 波位置估计器 + 收敛研究），
接入主求解器 `compressible.cases.sod → EulerSolver.run`，6+14 个测试通过，主工件逐位复现，C1/C2 大幅通过，
守恒/正性无修补。

## ⑥ 误差判定（是否 <3%）

| 关键量 | 基准/参考 | 实测 | 误差 | <3%？ |
|---|---|---|---|---|
| C1 四波位置 | x0+S·t（发布波速） | nx=400 | 0.031%–0.365% | **是** |
| C2 星区 4 量 | 发布星态（Toro 2009 ch.4） | 平台均值 | ≤0.0142% | **是** |
| C3 全域 L1 阶 | ≥1.0 | 0.617–0.977 | −2.3%…−38.3%（相对 1.0） | **否（FAIL）** |
| R5 设计阶 order 1 | 1.0 | 0.9936–0.9985 | ≤0.64% | **是** |
| R5 设计阶 order 2 | 2.0 | 2.045–2.132 | +2.27%…+6.6% | 边界（最细对是，最粗对否） |

**L4 不成立**：基准 C3 FAIL（唯一裁判的硬判据）。**判 L3**：功能实质=真代码 + 主求解器集成 + 可复现 + 误差分析齐备，
但"基准验证通过"这一级不满足，且本轮新增工件未入库/未跑声明网格。

## 处置（disposition）

* 台账：`tensor_maturity_verdicts` 增一行（repo=TensorFVM, scope=`m15310 REPRO-FVM-SOD1978`, mission_id=15344, **level=3**）。
* 被审方返修项（建议）：① 提交第 2 轮文件（含 reconstruct.superbee 与 4 个新文件）并把 `head_commit` 落值；
  ② 跑完声明网格（≥1600）再出渐近结论，或把工件配置改为与实际网格一致并在文档首页明示；
  ③ 基准行 R5/R3 补窗口定义走基准变更流程，不在核验侧调容差；
  ④ 若 C3 字面判据在格式族内确实不可达（两轮数据显示 order 1≈0.66、order 2≈0.98；文献机制指向接触波次线性），
  应由基准所有者提案改判据并留痕，禁止被审方自行改。