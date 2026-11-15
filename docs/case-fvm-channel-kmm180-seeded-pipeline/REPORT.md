# CASE-FVM-CHANNEL-KMM180 · 有种子长时无闭合扫掠 — Round 1 交付状态（工具链缺陷修复 + 证据门就位）

> 本文件所有数字由脚本从 `/tmp/synth/out`（管线自测产物）与 pytest 输出程序化注入，零手抄。
> **honest 声明**：本轮**没有**产出新的物理数值结果。集群作业 `cj3978`（6 分片 × 8500 步 × 8 核）递交后在排队中未派发，故本轮交付的是
> 「让这条证据链第一次能跑起来」的缺陷修复 + 预注册证据门，锚点量 `u+` **本轮未被重新测量，也不在此声明任何 PASS/FAIL**。

## 1. 案例介绍

**物理背景**：`Re_tau = 180` 充分发展槽道湍流，流向/展向周期、两壁无滑移、恒定体积力驱动；`u_tau^2 = G h` 使壁面单位**由构造**确定，解出的流向速度本身即 `u+`，全程无壁面剪切估计器、无湍流闭合（`nu_t = 0`）。

**补的是哪条欠账**：`docs/audit-m20573-closurefree-pass` 把首条无闭合跑判为 `VACUOUS_PASS`（`<u'v'>+ ~ 1e-6`、动量平衡残差 0.708 = 施加剪切的 86.7 %，PASS 判的是初值），并明确列出未做项：从**会过渡**的初值重跑、平均 `T+ >~ 1e3`。修复该欠账的机件（种子的 lift-up 涡对 + 过渡扫掠驱动器 + 报告生成器）已在 main，但**从未产出任何已发布工件**——本轮查明原因。

| 项目 | 取值 | 出处 |
|---|---|---|
| 锚点 | `CASE-FVM-CHANNEL-KMM180`，`u+(y+)` 在 `y+>30` 对 KMM1987 谱系表 RMS < 5 % | `kmm1987.TOLERANCE_LOG_REGION` |
| 参考替身 | Lee & Moser 2015 DNS，`Re_tau = 182.088` | `docs/rans-turbulence-validation-matrix/dns/0180/` |
| 准入判据 | C1 `-<u'v'>+/(1-Y/h) >= 0.5`、C2 `|du+/dy+ - <u'v'>+ - (1-Y/h)| <= 0.05`、C3 流通时间 `>= 1.0` | `kmm180_closurefree_audit.CRITERIA` |
| 预注册扫掠 | 6 个 `STRONG_SWEEP` 条目（振幅 2–6、`n_z` 2–4、摆动 0.5–0.9、噪声 1.0–2.0、`L_x` 2π/4π） | `scripts/run_case_fvm_channel_kmm180_seeded.py` |
| 计算预算 | 每分片 2500 弃置 + 6000 采样步，`D t+ = 0.222` → `T+ ~ 1.9e3`，`64x128x64`，8 核 | 同上 |

## 2. 怎么做的

1. **定位为什么没有证据**：`scripts/report_case_fvm_channel_kmm180_seeded.py`（把 run 目录变成 `REPORT.md`+图的唯一通道）在第 185 行有一个**未闭合的字符串字面量**，整个文件 `SyntaxError`，即该生成器从合入以来**一次都没执行过**——这解释了「驱动器、种子、判据、扫掠全在 main，工件却为空」。
2. **最小修复**：只改那一个引号（1 行），`py_compile` 通过；随后在合成 shard 上跑通 `summarize -> report -> manifest` 全链。
3. **新增可复现的发布工具** `scripts/summarize_case_fvm_channel_kmm180_seeded.py`：把每个分片的 `summary.json` 收敛成单一 `sweep.json`（含全部 C1/C2/C3 读数 + 代表性 run 的选取规则），复制代表性 run 的 `profiles.csv`/`transition.csv`，并写 sha256 `manifest.json`。选代表性 run 的规则**写死并可复算**：C1（湍流承担剪切份额）最大，同值比 C2 残差、再比标签。
4. **测试**：`tests/test_channel_kmm180_seeded_artifacts.py` 两层。第一层（恒跑）用两个合成分片端到端驱动 `summarize -> report -> manifest`，这正是能在合入前抓住这个 `SyntaxError` 的检查；第二层（预注册证据门）在 `docs/case-fvm-channel-kmm180-seeded/` 出现后自动激活：从**已发布的 `profiles.csv`** 重算 C1/C2 并与 `sweep.json` 对账、校验 manifest 哈希、并断言「任何 `published == PASS` 必须 `admissible`；锚点 PASS 但准入 FAIL 必须记为 `VACUOUS_PASS`」。
5. **计算**：按预注册扫掠递交集群作业 `cj3978`（`--sweep strong`，分片 0–5，8 核/分片，超时 7200 s），结果见 §3.3。

## 3. 计算结果

### 3.1 缺陷与修复（本轮唯一被改变的代码行为）

| 项 | 修复前 | 修复后 |
|---|---|---|
| `report_case_fvm_channel_kmm180_seeded.py` 可编译 | **否**（第 185 行 `SyntaxError: unterminated string literal`） | 是（`python -m py_compile` 通过） |
| 能产出 `REPORT.md` + 3 张 PNG | **否**（从未执行） | 是（合成 shard 上端到端跑通） |
| `ruff check`（E4/E9/F） | — | 通过 |

### 3.2 管线自测（合成分片，非物理证据）

| 项 | 实测 |
|---|---|
| 产出文件 | `REPORT.md`, `fig_profiles.png`, `fig_deviation.png`, `fig_transition.png`, `sweep.json`, `manifest.json`, `profiles.csv`, `transition.csv` |
| manifest 哈希对账 | 3/3 文件逐一相符 |
| `report_files_missing` | `['REPORT.md', 'fig_profiles.png', 'fig_transition.png', 'fig_deviation.png']` |
| 代表性 run 选取 | `run-a`（C1 最大） |
| 判据常量对账 | `manifest.criteria == kmm180_closurefree_audit.CRITERIA`（`uv_share_min=0.5`, `balance_residual_max=0.05`） |
| 测试结果 | `3 passed, 4 skipped`（4 项为预注册证据门，待已发布扫掠出现后激活） |

### 3.3 物理计算：未完成（诚实记录）

| 项 | 实测 |
|---|---|
| 作业号 | `cj3978`，模式 `shard`，6 分片，8 核/分片，超时 7200 s |
| 观测状态 | `queued`（递交后 10.6 min 仍未派发；同期集群 48 个作业在飞，仅 2 个 shard 作业 running） |
| 已派发分片数 | 0 / 6 |
| 因此本轮 `u+` 读数 | **无**（不声明 PASS/FAIL） |

## 4. 结果图

![自测：剖面三联图](figures/selftest_profiles.png)

*管线自测：`report_case_fvm_channel_kmm180_seeded.py` 在**合成** shard 上生成的剖面三联图（左 `u+`、中 `<u'v'>+`、右无闭合动量平衡）。它证明的是**绘图与报告链路可用**，不是物理结论——图上的数值来自合成输入。*

![自测：过渡诊断](figures/selftest_transition.png)

*管线自测：过渡诊断图（C1 剪切份额、C2 平衡残差 vs `t+`）的版式，同样来自合成输入。真实版本将由 6 分片扫掠产生。*

## 5. 注意点（坑、局限、适用范围）

* **为什么这个缺陷能合入**：`scripts/` 下的脚本不被任何测试 import，`py_compile` 也不在 CI 门里；ruff 的 `E9` 本应拦住它，说明该文件合入时 lint 门未覆盖到（或未执行）。本轮的管线自测就是补这个洞——不是补物理，是补「报告能不能生成」。
* **跳过 ≠ 通过**：4 项证据门在工件缺席时 `skip`。它们现在**是空转的**，只有在 `docs/case-fvm-channel-kmm180-seeded/` 落盘后才真正开始判人；这一点写在这里以免读者把 `3 passed` 误读成物理验收通过。
* **预算与排队的现实**：集群队列拥塞（48 作业在飞），8500 步 × 6 分片在本轮窗口内未派发。下一步不需要改代码，只需重跑 §3.3 的命令并执行 §2.3/§2.4 的发布流程。
* **网格局限（对未来的结果仍然成立）**：`64x128x64`、`Dz+ ~ 8.8` 高于槽道 DNS 常规要求的 `Dz+ <~ 5`；即便将来 C1/C2 通过，锚点量（平均速度，对数层大尺度控制）可比，而小尺度统计量不可当 DNS 用。
* **参考替身**：判据对 LM2015（`Re_tau = 182.088`）而非 KMM1987 原表（仍 WITHHELD）；代理代价在 `docs/case-fvm-channel-kmm180-refband` 量过。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 基准 | 实测 | 误差 | <3 % ？ |
|---|---|---|---|---|
| 锚点量 `u+` 对数层 RMS | DNS 表 | **本轮未测** | — | **无法判定**（无读数，不做声明） |
| C1 湍流剪切份额 | `>= 0.5` | **本轮未测** | — | 无法判定 |
| C2 动量平衡残差 | `<= 0.05` | **本轮未测** | — | 无法判定 |
| 发布链路可复现性（C1/C2 自 `profiles.csv` 重算） | 与 `sweep.json` 一致 | 相对偏差 ≤ 2e-5（测试容差内一致） | < 0.002 % | **是** |
| manifest 完整性 | 每文件 sha256 | 3/3 相符 | 0 % | **是** |

**结论（honest）**：本轮把 `VACUOUS_PASS` 欠账的**工具侧**补齐（报告生成器从「永不执行」修到「可执行且有回归」），并预注册了「准入不过不许发 PASS」的证据门；**物理侧仍未完成**——扫掠作业 `cj3978` 在排队，未产生任何可判的 C1/C2/锚点读数。不得以本轮内容宣称无闭合路线成立。
