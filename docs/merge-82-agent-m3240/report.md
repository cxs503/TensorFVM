# PR #82 (`agent/m3240`) 三分合并治愈报告 —— 一条 Ghia 参考值的两种解释

**任务**：PR #82 对 `main` 变脏（`mergeable=false`），三方合并 `agent/m3240 → main`，
逐文件解冲、语义并集、不静默丢内容，本地测试不低于 `main` 基线，并把结果推回同分支。

**结论（一句话）**：合并已完成且无冲突残留；两侧内容**全部保留**——`main` 的"打印值
不可篡改 + 争议登记 + strict xfail"完整在位，PR 的"该值是排版错误、应改 `-0.38760`"
以机器可读字段 `DISPUTES[...]["competing_transcription"]` 与独立散文记录逐字保留，并由
一条新测试把"已记录但未生效"钉住。因为 PR 的替换值与被评估解自身相差 `1.79e-5`，
采纳它会让 Re=400 的 v 中心线判据变成循环论证，故**不采纳**；该缺陷仍是未结案的科学
问题（打印值 vs Richardson 极限相差 **63.6 %**）。

---

## 1. 案例介绍

| 项 | 值 |
|---|---|
| 仓库 | `TensorFVM`（FVM 不可压求解器库，`src/tensorfvm`） |
| 主线基点 | `main` = `db97955`（= 本工作树起点，`origin/main`） |
| PR #82 头 | `agent/m3240` = `99f8e6e`（"m3240: matured mission deliverables"） |
| 三分合并基点 | `0e9bd51`（`Merge #49 (m2766)`），`main` 领先 19 个提交、PR 领先 1 个提交 |
| 治愈后合并提交 | **`0e2b588`**，父提交 `db97955` + `99f8e6e` |
| 冲突文件 | 2 个内容冲突（`git` 标记）+ 1 个"自动合并成功但语义耦合"的数据文件 |
| 数据资产 | `docs/verification-lid-cavity-ladder/summary.json`（1.61 MB，7 个算例，n=32/64/128，CFL 0.15，tol 1e-7，max_steps 2e5） |

**物理背景**：驱动方腔（Ghia, Ghia & Shin 1982, *J. Comput. Phys.* 48, 387-411）。
`src/tensorfvm/lid_cavity_reference.py` 逐字转录论文 Table I（竖直中心线 x=0.5 的 u）与
Table II（水平中心线 y=0.5 的 v）。Re=400 剖面在 `x=0.9063` 有一个孤立点：打印值
`-0.23827` 与左右邻点 `-0.22847 / -0.44993` 构成一个采样剖面里不该出现的折角，而求解器
在该点二阶收敛到 `-0.38989`。

**两侧各自做了什么（这就是冲突的全部来源）**：

* `main`：把该点登记为**争议**（`DISPUTES[(400.,"v",0.9063)]`，含 n=32/64/128 计算值、
  观测阶、Richardson 极限、指标插值误差界），新增 `smoothness_audit()` 把"剖面折角"机械化，
  并把 PR 关系到的 xfail 测试**保留为 strict xfail**；配套审计脚本
  `scripts/audit_ghia_re400_v_attribution.py`、`scripts/audit_lid_cavity_v_centreline.py`。
  其成文原则：*争议不是修改转录的许可证，打印值保持原样，任何绕过它的判据必须明说*。
* PR #82：把 `V_CENTRELINE[400.0]` 在 `x=0.9063` 的值改为 `-0.38760`（注释
  `# CORRECTED: was -0.23827 (transcription error)`），据此**重新生成了整份
  `summary.json`**，把原 strict xfail 换成 `test_re400_v_centreline_discrepancy_resolved`
  （断言 v 中心线误差 `< 0.03`），并新增 `docs/verification-lid-cavity-correction/`
  （报告 + 3 图）。

两者对**同一条数据点**给出互斥判定，因此这不是排版层面的冲突，而是判据层面的冲突。

---

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**工具链**：`git`（blob-filtered 克隆上的三分合并）、`pytest 9.2.0`（`pythonpath=src`）、
`python3.11` + `matplotlib(Agg)` 生成图；全部脚本落在
`docs/merge-82-agent-m3240/{resolve_pr82.py, make_figs.py}`，可复算。

**流程**：

1. **定位真正的 PR 头**。本地 `refs/heads/agent/m3240`=`5d96650`（2026-11-04 04:58）与
   `refs/remotes/origin/agent/m3240`=`99f8e6e`（2026-11-04 13:01）**不是同一条历史**：前者
   基于 `b61a7fb`，后者基于 `0e9bd51` 且是 `main` 之后的唯一新增提交。以 `origin/agent/m3240`
   为 PR 头（判据：`git rev-list --left-right --count origin/main...origin/agent/m3240` =
   `19 1`，且其提交信息与本单主题一致）。按"禁止机械 overlay"的前科教训，先做此对齐，避免
   把另一条历史静默覆盖进来。
2. **先建基线树**再动手：用临时索引 `GIT_INDEX_FILE=/tmp/idx_base git read-tree HEAD`
   + `checkout-index -a` 把 `main` 导出到 `/tmp/m3860_base`（8023 个文件），作为"不得低于"
   的对照，全程不污染工作树索引。
3. **三分合并**：`git merge --no-commit --no-ff origin/agent/m3240` → 2 个 `CONFLICT (content)`
   + 4 个自动合入（PR 的 3 张图 + 1 份报告）+ 1 个自动合入但语义耦合的数据文件。
4. **判定准则**（写死在 `resolve_pr82.py` 的锚点替换里，每个替换都断言"锚点唯一存在"，
   防止正则批量替换踩过的缩进/重复定义坑）：
   * 语义并集：两侧的**新增**内容都必须落地，不允许"取一边了事"；
   * 数据文件必须以**约定意义上的单一真相源**为准（见 3.3），并把取舍写进提交信息与报告；
   * 任何"删除"都要有替代性载体（登记字段 / 测试 / 文档），否则不删。
5. **验证链**：`AST` 语法检查 → 冲突标记扫描（0 残留）→ 目标测试 → 核心子集基线对照 →
   `--collect-only` 收集数平权 → 提交。
6. **入库**：merge commit `0e2b588`（父 `db97955`+`99f8e6e`），随后提交本报告与图。

---

## 3. 结果

### 3.1 冲突文件清单与逐文件解法

| # | 文件 | 冲突形态 | 解法（语义并集，无静默丢失） | 净效果 |
|---|---|---|---|---|
| 1 | `src/tensorfvm/lid_cavity_reference.py` | 内容冲突（尾部 ~76 行：`main` 新增 `REFERENCE`/`DISPUTES`/`smoothness_audit`，PR 侧该区为空；且 PR 另改了表值、`KeyError` 文案、模块 docstring、文件尾换行） | 保留 `main` 尾部整块；表值**回到打印值** `-0.23827`（注释指向争议）；PR 的替换值与理由**逐字**移入 `DISPUTES[(400.,"v",0.9063)]["competing_transcription"]`（`value / printed_value / source / rationale / rejected_because / kept_at`）；PR 的"Correction history"改写为诚实的"提议 + 未采纳理由"；PR 对 `KeyError` 文案与文件尾换行的改动按 PR 保留 | 两侧新增项 100 % 在位；表值单一化 |
| 2 | `tests/test_lid_cavity_re1000_evidence.py` | 内容冲突（同一段测试体：`main` 的 strict xfail vs PR 的 "resolved" 断言） | 保留 `main` 的 `@pytest.mark.xfail(strict=True)` 测试与更新后的归因理由；PR 的测试**转为** `test_re400_v_centreline_correction_claim_is_recorded_not_applied`：断言打印值仍为 `-0.23827`、争议记录里 `competing_transcription.value == -0.38760`、`source` 溯源到 PR #82、且提交的 `summary.json` 仍按打印值打分、该点 `max_abs_error > 0.03`（即"已记录未生效"）；模块 docstring 两条事实并陈 | 测试数 +1，语义零丢失 |
| 3 | `docs/verification-lid-cavity-ladder/summary.json` | **无 `git` 冲突**（仅 PR 侧修改，自动合入）但语义耦合 | 还原为 `main` 版本（md5 与 `HEAD` 一致）。理由：PR 的整份重生成是**按改写后的参考值**打分的，且把存储浮点截短（`0.00712575` vs `0.007125204387469206`）；保留它会让"已提交证据"与"实际生效的参考值"互相矛盾，并让 `main` 的 strict xfail 变成 XPASS(strict) 而失败。取舍写入提交信息、本报告，PR 自己的分析在 `docs/verification-lid-cavity-correction/report.md` 原样保留 | 证据与参考值同源 |
| 4 | `docs/verification-lid-cavity-correction/{report.md, 3×png}` | 纯新增（自动合入） | 原样保留（PR 的成文记录），另加 `MERGE_NOTE.md` 指回争议登记 | 零丢失 |

冲突标记扫描：合并后 `<<<<<<< / ======= / >>>>>>>` 在三处曾涉及的文件中计数均为 **0**；
`ast.parse` 通过（两个 `.py`）。

### 3.2 测试证据

| 测试面 | 基线（`main`=`db97955`，导出树 `/tmp/m3860_base`） | 治愈后（`0e2b588` 工作树） | 判定 |
|---|---|---|---|
| 收集数 `pytest --collect-only` | 1139 | **1140** | +1（新守卫测试），无用例丢失 |
| 核心验证子集（10 文件 / 115–116 例：Ghia 一致性、ladder 证据、抽取、GCI、解析残差、壁面梯度归因、梯度跳跃、壁函数、Forchheimer、lid_cavity 冒烟） | **113 passed, 1 skipped, 1 xfailed**（102.9 s） | **114 passed, 1 skipped, 1 xfailed**（129.8 s） | 失败 0 = 0；xfail 仍是 xfail（strict 未被"治好"） |
| 冲突涉及的两个文件 | 19 passed, 1 skipped, 1 xfailed | 19 passed, 1 skipped, 1 xfailed | 持平 |

关键点：`xfailed` 在两个树里都是 **1**，说明 Re=400 的缺陷**没有被测试层面粉饰**（strict xfail
一旦意外通过会立即红）。合并后新增的 1 个 passed 即 PR 内容的新载体。

**未完成项（如实）**：全量 1140 例未跑完。本机与 300+ 个同租约任务共享 128 核
（采样时 `loadavg` ≈ 553–578），3D body-fitted 段单个用例 >10 min；前后两次 15 分钟级尝试
分别推进到收集序约 6 % 与 18 %，都在预算内无法收尾。基线树与合并树的全量日志在可比的
前 ~284 例上**肉眼同形但逐字符并不完全一致**（基线先到一段 4 个 `E` 的环境性错误，合并树
此时尚未推进到同一位置），因此**不**据此声称"全量等价"；结论只建立在
"收集数平权（1139→1140）+ 核心子集 115 例基线对照（0 失败、0 退化）"之上。

### 3.3 数值证据（程序化注入，见 `figures_data.json`）

Re=400 v 中心线在 `x=0.9063` 的点：

| n | 计算值 | 对打印值误差 `\|Δ\|` | 对 PR 提议值误差 `\|Δ\|` |
|---|---|---|---|
| 32 | −0.352839 | 0.11457 | 0.03604 |
| 64 | −0.380637 | 0.14237 | 0.00738 |
| 128 | −0.387582 | 0.14931 | 0.00431 |
| Richardson 极限 | −0.38989 | **0.15162**（相对打印值 **63.6 %**） | — |

* 打印值 `-0.23827`；PR 提议值 `-0.38760`；两者相差 `0.14933`。
* **循环性量化**：`|提议值 − n=128 计算值| = 1.79e-5`（相对 0.0046 %）。也就是说，"改参考值"
  与"把解复制进参考值"在数值上不可区分——这正是不能采纳它的机械理由，与是否能拿到论文
  原件无关。
* 邻点对照（n=128，对打印值的偏差）：`x=0.9453 → 0.00431`、`0.8594 → 0.00206`、
  `0.8047 → 0.00086`；争议点偏差 `0.14931` 是三者最大值的 **34.6 倍**——缺陷确实是孤立的，
  不是整体剖面被眯了。
* 受影响域**外**的收敛健康度：u 中心线 `max|Δ|` 在 n=64/128 为 `0.00660 / 0.00135`；
  Re=100 的 v 为 `0.00873 / 0.00915`；Re=1000 (n=64) 为 u `0.02931`、v `0.01973`。

---

## 4. 结果图

图 1–3 由 `docs/merge-82-agent-m3240/make_figs.py` 从**采纳后的** `summary.json` 直接读取
生成（无手抄数字）。

* **fig1_v_centreline_printed_vs_proposed.png** —— Re=400 v 中心线：打印表（空心方块）、
  三条网格计算曲线（n=32/64/128）、PR 提议值（大星）。一眼看到"打印点在光滑剖面上是个折角，
  而提议值正好落在计算曲线上"——冲突的几何形态。
* **fig2_pointwise_error.png** —— 对打印表的逐点误差（半对数）：三条网格曲线在 `x=0.9063`
  都出现单点尖峰并随加密**升高**（0.1146→0.1424→0.1493），其余点稳在 3 % 门附近/以下。
* **fig3_disputed_point_convergence.png** —— 该点值随 `h=1/n` 的二阶收敛：打印值（红虚线）、
  Richardson 极限（灰点线）、PR 提议值（绿点划线）三线对照，并标注提议值与 n=128 解只差
  `1.8e-5`。

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **同名分支两条历史**：本地 `refs/heads/agent/m3240` 与 `origin/agent/m3240` 是不同基点
   （`b61a7fb` vs `0e9bd51`）。若按"本地分支优先"合并，会把 PR 的真实成果整体丢掉而合并
   一条无关历史。**判据**：用 `merge-base` + `rev-list --left-right --count` 确认 PR 头是
   `main` 之后唯一新增的那一侧。
2. **自动合并 ≠ 语义无冲突**：`summary.json` 没有任何 `git` 标记，却是本次最危险的项
   （整份证据被按改写后的参考值重算、浮点被截短）。机械合并会把它当成"PR 的新版"静静留下，
   于是参考值在主模块里被拒、在证据里生效——最坏的半采纳状态。
3. **strict xfail 是资产不是负债**：PR 的动机（让套件变绿）可以理解，但把"参考值改成解的值"
   来清零误差是循环论证；本仓的成文原则（争议不构成篡改转录的许可证）应作为默认判据。
   `xfailed=1` 在两棵树里都保留，正是"没有粉饰"的可执行证据。
4. **未结案的科学问题**：究竟 `-0.23827` 是论文 Table II 的原文（含印刷错误）还是转录错误，
   本工作树没有论文原件/DNS 参照，**无法裁决**。决议径只有两条：拿到印刷页对照，或用独立
   二阶以上 DNS 复算该点。在那之前，正确姿态是"登记 + 明说 + 让判据红着"。
5. **网络受限**：本沙箱对 GitHub 无出网（`git ls-remote` 与 HTTPS 探测均超时，`curl` 返回
   `000`），因此**不能**由本单把分支推回 `origin/agent/m3240`。成果落在分支 `agent/m3860`
   上，由平台自动推送承载；`agent/m3240` 的 old/new SHA 映射记录如下：
   `99f8e6e → 0e2b588`（合并提交；其余分支历史未改写）。
6. **适用范围**：本报告的测试结论限于 Python/pytest 层面的判据与证据一致性，不构成对
   求解器在 `x=0.9063` 附近数值精度（即"壁面邻近单元提取是否失真"）的最终结论——那属于
   `scripts/audit_lid_cavity_v_centreline.py` 的归因结论与后续任务。

---

## 6. 误差判定（逐项，显式回答"是否 < 3 %"）

| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 冲突标记残留（曾冲突的 3 个文件） | 0 | 0 | 0 % | **是** |
| 核心子集失败用例数（合并后 vs `main` 基线） | 0 | 0 | 0 % | **是** |
| 核心子集通过用例数 | 113 | 114 | +0.88 %（方向为增） | **是（无退化）** |
| 收集用例数 | 1139 | 1140 | +0.09 %（方向为增） | **是（无丢失）** |
| Re=400 v 中心线 `x=0.9063`：打印值 vs 独立外推极限（n=32/64/128 二阶 Richardson） | −0.38989 | −0.23827 | **63.6 %** | **否**（故拒绝 PR 的"已修复"结论，保留 strict xfail） |
| Re=400 v 中心线 `x=0.9063`：PR 提议值 vs 计算 n=128 | −0.387582 | −0.38760 | **0.0046 %** | 数值上是；但**基准取自被评估解本身**（循环），不计为验收通过 |
| Re=400 u 中心线 `max\|Δ\|`（n=64 / n=128） | Ghia Table I | 0.00660 / 0.00135 | 0.66 % / 0.135 %（对 lid 速度单位） | **是** |
| Re=100 v 中心线 `max\|Δ\|`（n=64 / n=128） | Ghia Table II | 0.00873 / 0.00915 | 0.87 % / 0.92 % | **是** |
| Re=1000（n=64）u / v `max\|Δ\|` | Ghia Tables I/II | 0.02931 / 0.01973 | 2.93 % / 1.97 % | **是（临界）** |

**一句话裁决**：作为**合并/工程验收**，本单的关键量全部 < 3 %（0 冲突残留、0 测试退化、
收集数无丢失）；作为**科学结论**，"Re=400 v 中心线已修复"**不成立**（打印值与外推极限相差
63.6 %），PR 的 < 3 % 只通过移动基准获得，已如实登记为未采纳的主张，缺陷保持开放。

---

## 附：可复算入口

```bash
# 1) 复现逐文件解法（须在冲突态运行：先 git merge --no-commit --no-ff origin/agent/m3240；
#    脚本用唯一锚点替换，锚点缺失即退出，保证不会在已解冲突的树上二次改写）
python3 docs/merge-82-agent-m3240/resolve_pr82.py
# 2) 复现图与数据
python3 docs/merge-82-agent-m3240/make_figs.py
# 3) 验证
python3 -m pytest tests/test_ghia_reference_consistency.py \
                 tests/test_lid_cavity_re1000_evidence.py -q
```