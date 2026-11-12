# 交差核验 · m16077 · CASE-FVM-ETHIER-STEINMAN-3D

- **被核对象**：mission m16077（TensorFVM，状态 succeeded，PR #473 已并入 `e6e254fd`），
  GitHub 分支头 `0ee118f6`（`scripts/ethier-steinman-3d/` + `src/tensorfvm/verification/ethier_steinman.py`
  + `tests/test_ethier_steinman.py`，8 文件 +1386 行）。
- **核验单**：m16176（本单，board-audit，非重做）。
- **裁定级别**：**L3**（系统集成）。**未达 L4**，理由见 §4/F1、F2。
- **报告**：`tensor_reports` id 15224（六段齐全，honest-FAIL 表述正确）。

## 1. 取证：三件套逐项

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 真存在 | ✅（部分） | `scripts/ethier-steinman-3d/artifacts/re1-n4.npz`(5187 B)、`re1-n6.npz`(14839 B)、`study.json`(5714 B) 均随 merge `e6e254fd` 入库；re100/re100_resolved **无工件**（因未收敛，如实缺）。本审计副本 + sha256：`docs/board_audit/m16176/refs/`、`refs/SHA256SUMS.txt` |
| 工件数值自洽 | ✅ | 从 npz 独立重算：`max|u_h-u_ex|/max|u_ex|` = **0.0623793 (n=4) / 0.0388359 (n=6)**，与 RESULTS.md 的 6.238e-02 / 3.884e-02 逐位一致；L2 相对误差 5.7305e-02 / 3.4898e-02 亦一致；体积和 = 8.0（域体积正确） |
| repro_cmd 可执行 | ✅ | `PYTHONPATH=src python3 scripts/ethier-steinman-3d/run_study.py --n-list 4 --cases re1 --time 0.25 --max-iterations 8` 复跑输出 `velocity_relative_linf = 0.0623793291966362`（**与工件逐位相同**，确定性复现） |
| 测试 | ✅ 10 passed / 32.9 s | `tests/test_ethier_steinman.py`（参考自审 4 项 + 求解 4 项 + 网格 + 未收敛必抛错） |
| 基准 ref vs got | ⚠️ **门 FAIL** | 判据（任务 goal）：速度 L∞ 相对误差 **<1%**（Re=1 与 Re=100）。got：Re=1 **6.238% (n=4) → 3.884% (n=6)**；Re=100 两参数化**均未收敛**（Newton `iteration_limit`）。即 **连 3% 线也未达** |
| benchmark 注册表 | ❌ **无此行** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FVM-ETHIER-STEINMAN-3D'` → **0 行**；全表 25 例中亦无 ethier/steinman 关键词（见 F4） |
| 报告六段 | ✅ 内容齐 / ⚠️ 图缺图 | `tensor_reports` id 15224 六段（①案例 ②方法 ③参数表 ④对照表 ⑤注意点 ⑥误差判定含「是否<3%」显式回答）齐；但 `figures[]` 两条只有 `name`+`caption`，**无 png_base64 载荷**，且 `scripts/ethier-steinman-3d/figures/*.png` 未进 merge 提交（F5，本审计已按 npz 重建补图） |

## 2. 独立复算（不 import tensorfvm，直接照报告印刷公式重写）

`docs/board_audit/m16176/independent_reference_check.py` → `independent_reference_check.json`
（随机 12 个内点，固定种子 20261112，中心差分；梯度 h=1e-5，残差旋度 h=1e-3）：

| 工况 | a | d | ν | ‖div u‖∞ | ‖Δu+a²u‖∞ | ‖curl R‖∞ (ν=d²/a²) | ‖curl R‖∞ (错误 ν=1) |
|---|---|---|---|---|---|---|---|
| re1 | 1 | 1 | 1 | 5.6e-11 | 4.4e-08 | 1.10e-06 | 1.10e-06（ν=1 即本工况，作废对照） |
| re100 | 1 | 0.1 | 0.01 | 8.3e-11 | 3.7e-08 | **1.81e-06** | **2.32e+00** |
| re100_resolved | 0.2 | 0.02 | 0.01 | 3.5e-12 | 7.9e-09 | **8.01e-10** | **1.99e-03** |

结论：**参考场确实是精确无外力 NS 解**（散度机器零、Laplacian 本征函数、且在 ν=d²/a² 处动量残差为纯梯度），
且该结论**不是假设**——错误 ν 处残差旋度比正确 ν 处大 1.3e6 倍（re100）与 2.5e6 倍（re100_resolved），
与 m16077 自审记录（8.9e-10 解析旋度、ν=1 处 2.0 的反证）同向。参考这一环**站得住**。

## 3. 级别判定（L0–L5）

| 级 | 判据 | 本单 | 依据 |
|---|---|---|---|
| L1 数学/单元 | 测试断言公式性质 | ✅ | 10 测试，含 `test_reference_is_divergence_free_and_exact`(rtol 1e-10/1e-12)、`test_time_decay_and_frozen_force_consistency`(rtol 1e-12) |
| L2 独立功能 | 真代码 输入→计算→输出 + 工件 + 入口 | ✅ | `src/tensorfvm/verification/ethier_steinman.py:66-529`（atexp 原子场/解析梯度/解析 Laplacian/制造体力/网格/求解/metrics）、入口 `scripts/ethier-steinman-3d/run_study.py:1-366`、工件 npz+study.json |
| L3 系统集成 | 主求解器真调用 | ✅ | `ethier_steinman.py:65` `from ..native_hybrid_ns3d import NativeHybridNavierStokes3D`，`:420-448` 构造并 `solve_newton`；该求解器在仓内被 `native_hybrid_rans3d.py:8,17`、`verification/native_developed_channel.py:36`、`verification/native_hybrid_ns.py:30,73,100`、`verification/native_hybrid_pressure.py:38` 复用（非孤岛脚本） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ | 门 FAIL（6.24%/3.88% vs 1%）；Re=100 两工况**根本没解出来**（无一格单元解）；见 F1/F2 |
| L5 科研工程 | + external_anchor（文献值） | ❌ | 原论文 DOI 10.1002/fld.1650190502 付费闭源，原文表格 WITHHELD；本单**没有**冒充文献对照，故 L5 无从谈起（诚实，不加分也不扣分） |

> 级别=功能实质：附百页报告不改变级别；反过来，诚实 FAIL 也不抹掉已建成的 L3 设施。

## 4. 发现（按严重度）

**F1｜阻断 L4｜门未过**：契约指标 <1%，实测 6.238%(n=4)→3.884%(n=6)，实测阶 1.1688（两分辨率单步），
连 3% 可接受线也未达。任务 goal 与报告都如实承认，不粉饰。

**F2｜阻断 L4｜"完整算例"不成立**：re100（a=1,d=0.1,ν=0.01，单元 Re=50/33）与 re100_resolved（a=0.2，单元 Re=10/6.7）
在 n=4/6 全部 Newton `iteration_limit`；RESULTS.md 该 4 行的 dx/误差列是 `n/a`，属**无测量**而非测量失败。
另注：re100_resolved 在 25 次迭代下 n=4 收敛但误差 137%（NOTES.md §3），说明即便收敛也不具备精度资格。

**F3｜中｜测试阈值比契约松 10 倍**：`tests/test_ethier_steinman.py:129-135`
`assert velocity_relative_linf < 0.10` / `< 0.08`——测试在 6.238% 时绿灯，而契约线是 1%。
即**测试绿 ≠ 算例过**；本套件结构上不可能发现本单的 FAIL。（不是造假，是阈值错配；建议测试断言改成读取预注册容差常量，
或显式标注 "smoke-level bound, contract gate lives in study.json"。）

**F4｜中｜基准注册表缺该锚点**：`tensor_benchmark_cases` 无 `CASE-FVM-ETHIER-STEINMAN-3D` 行，
唯一裁判的 metric/tolerance 只存在于 mission goal 文本里 → 未来轮次无法被机器自动判分。
建议补行：`('CASE-FVM-ETHIER-STEINMAN-3D','三维制造解 NS 验证','速度 L_inf 相对误差 vs 精确解 (Re=1 与 Re=100)','<1%')`。
（本审计不擅自写注册表——它属平台级判据账，需按域注入流程，此处只登记缺口。）

**F5｜轻｜结果图未入库（已部分更正）**：`scripts/ethier-steinman-3d/figures/*.png` 由 `run_study.py:295-362` 生成，
但**未进 merge `e6e254fd`**（该提交 8 文件清单无 figures/；`.gitignore` 亦无 png/figures 规则 ⇒ 是 `git add` 遗漏，非忽略规则）。
**更正（审计自纠）**：原以为 report 15224 的 `figures[]` 无 `png_base64` 也是交付缺陷——本单自己发报告（id 15282，内嵌两张 68 KB/70 KB png）
readback 后同样只剩 `name`+`caption`：**服务端 /tensor/reports 会剥离 png 载荷**，属平台侧存储行为，不判 m16077 的错。
故 F5 收敛为「png 工件未入仓」一条；图已由本审计按入库 npz 重建于 `docs/board_audit/m16176/figures/`。

**F6｜轻｜文档数值不一致**：NOTES.md 写测量阶 **1.14**，RESULTS.md 写 **1.17**；从工件算得 **1.1688**（dx 基），
NOTES 的 1.14 是上一轮（相位修正前）遗留。外推「需 n≈24 到 1%」在 p=1.1688 下应为 n≈19.2（n=24 预测 0.77%），
属**保守**而非夸大，但仍未实跑，只能算诊断。

**F7｜中（平台侧，不属 m16077 交付缺陷）｜maturity_gate 的 L4 判据只查"有没有数字"不查"过没过"**：
`/root/tp-agent/maturity_gate.py:SCHEME[4]` 对 `benchmark` 键只做 `has_numbers()`（正则找数字），对 `repro_cmd` 只查字符串长度。
**实测**：把本单 evidence 以声明级 4 送入 `judge()` → 返回 `(4, [])`（本审计亲测），尽管 benchmark 明确记录 FAIL
（got 6.238%/3.884% vs 1%）。即：**任何审计者只要把失败数字塞进 benchmark 键，就能被机器追认 L4**——这是 gate 的结构性漏洞，
不是 m16077 的问题。建议（零领域语义、通用化）：`tensor_benchmark_cases` 若存在该锚点行，则要求 evidence.benchmark 里
`error <= tolerance` 的机器可判定断言（例如 `benchmark.pass=true` 且由 SQL 复核），否则 bench 键降为 `benchmark_unregistered`。
本审计不利用该漏洞：声明级=3，实测 `judge(3, evidence, 'TensorFVM', 16176) -> (3, [])`。

**正向项（记账不扣分）**：

- **V1** 工件→报告数字逐位可复算（无手抄）。
- **V2** repro_cmd 实跑复现逐位相同 → 确定性可复现（L2/L3 要件成立）。
- **V3** `honest-FAIL` 纪律到位：mission result headline 直书 FAIL；study.json 记录 `converged:false` + `failure_reason`；
  `test_unconverged_case_raises_instead_of_returning_a_field` 断言 stiff 工况**必抛错**而不是返回假场——这类"拒绝造假"的测试应被后续算例仿效。
- **V4** 参考自审先于求解器判定（先审参考再审判求解器），且 ν=d²/a² 由解自身反证，不是搬公式。

## 5. 本审计未做的核验（unknown / not_run，不报 passed）

1. 未访问付费原文（DOI 10.1002/fld.1650190502）→ 论文表值仍为 **WITHHELD**，我无法替它确认文献锚。
2. 未独立验证 `exact_boundary_velocity_means`（order-8 Duffy 三角 Gauss 面积平均）的实现与精度。
3. 未跑完整三工况阶梯（仅复跑 re1 n=4 + 全套 pytest）：re100 的失败是**复述 study.json 记录**，未逐工况重跑；
   故 F2 的证据强度 = "工件记录 + 脚本逻辑一致"，非"本审计亲手重现失败"。
4. 未跑 n=24 外推点（成本 ~1.7e5 未知量直接 LU），F6 的外推仅是诊断。
5. 未核 ecosystem 主仓 `/root/tensor-ecosystem/TensorFVM` 是否已同步该 merge（当前其 HEAD 为 `02e98635`，早于 `e6e254fd`）——
   这是**部署滞后**问题，不属本单交付缺陷，但会影响 L3 "生产侧可见" 的时效。

## 6. 处置（disposition）

**approve with notes → L3 confirmed，L4 退回**：
交付物作为**验证设施（L3）**真实、可复现、诚实；作为**基准算例（L4）**未通过预注册判据，且 2/3 工况无解。
不得以「跑不动但描述详尽」升到 L4/L5。返修承接顺序建议（与 NOTES.md §4 一致，我加一条前置）：
F4 先补注册表行（让下一轮可机器判分）→ F3 测试阈值对齐契约 → Re=100 换瞬态积分或对流稳定化 → Re=1 n=24..32 走集群。

---

### 复现本审计

```bash
cd /root/tp-agent/work/m16176
python3 docs/board_audit/m16176/independent_reference_check.py --json docs/board_audit/m16176/independent_reference_check.json
python3 docs/board_audit/m16176/make_audit_figures.py
python3 -c "import numpy as np; z=np.load('docs/board_audit/m16176/refs/re1-n6.npz'); \
print(np.max(np.abs(z['velocity']-z['reference_velocity']))/np.max(np.abs(z['reference_velocity'])))"
```