# 交差审计 m20144 — CASE-FVM-CAVITY-RE1000 / TensorFVM（方腔驱动流 Re=1000，Ghia 1982）

**审计单**: m20230 ｜ **被审对象**: m20144（succeeded，分支 `agent/m20144`，头 `517fc38a`，基线 `main`=`27d67542`，**未合入 main**）
**裁判**: `tensor_benchmark_cases` 基准行（唯一裁判）｜ **判级**: **L4**（基准验证）｜ **处置**: approve with notes（confirm，附 3 条非阻断返修项）

## 0. 独立性与本次实际做了什么

| 动作 | 脚本/命令 | 结果 |
|---|---|---|
| 从**原始流场**独立重算全部计分量（不 import 被审打分代码） | `docs/board_audit/m20230/recompute_from_fields.py` → `recompute.json` | 见 §2 |
| 三份 Ghia(A) 转录交叉核对（被审模块 / 审计员手抄 / 另一仓 TensorLBM） | 同上 `reference_crosscheck` | 34/34 项一致（手抄首版有误，见 §5 F6） |
| 执行被审入口 + 冷启动 n=32 复现 + n=128 稳态重启漂移 | `docs/board_audit/m20230/repro_entry_and_restart.py`，集群作业 `cj2580`（n315，rc=0）→ `runs/repro.json` | 冷启动**逐比特相同**；漂移 1.1e-7/1000 步 |
| 被审机检门 | `PYTHONPATH=src pytest tests/test_case_cavity_re1000.py tests/test_case_cavity_re1000_m15458.py -q`（在 m20144 工作树只读执行） | **13 passed**，0 skipped（含 `test_ladder_consumes_the_run`） |
| 基准裁判行 | `SELECT * FROM tensor_benchmark_cases WHERE anchor_slug=…` | 行 id=18，tolerance 3% / \|Δ\|≤0.01@n≥128 |
| 集成调用方实跑 | `lid_cavity_verdict()` 于 m20144 分支 | C2 **PASS**，measured=0.011184 ≤ 0.02，`evaluated_on=…/lid_cavity_Re1000_n128.json` |

审计副本（逐字节复制的被审文件）在 `docs/board_audit/m20230/refs/`（`src/`、`tests/`、`docs/`：report.md、summary.json、4 个 `.pt` 流场、3 张图）。

## 1. 三件套取证

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 真存在 | ✅ | `docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json`（32 KB，含 cases/verdict/continuation_check）、`artifacts/fields/Re1000_n{32,64,128}.pt` + `Re1000_n64_from32.pt`（合计 416 KB，torch 可载入：u(129,128)/v(128,129) float64，steps=104250）、`figures/fig{1,2,3}_*.png`（1380×615 等，256 灰阶、非空白 7.2%/28.0%/20.9% 墨量）、`src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n{64,128}.json` |
| benchmark ref vs got | ✅ 在容差内 | 见 §2；全部落在基准行 3%/0.01 之内，也落在 5% ledger band |
| repro_cmd 可执行 | ✅ | 入口 argparse `case_cavity_re1000.py:470-506`；审计员以同参跑通 n=32 冷启动并经队列执行（`cj2580`），得 `max|Δu|=0.0, max|Δv|=0.0` vs 被审 `Re1000_n32.pt`，步数 30500 相同，`u_min=-0.316430836724452` 逐位相同 |
| 报告六段 | ✅ 齐全 | `refs/docs/report.md` ①问题设定/基准出处 ②方法与实现 ③参数表 ④对照表 ⑤收敛性与参数敏感性 ⑥失败模式/工件/复现 |

## 2. 独立重算 vs 被审数值 vs 基准裁判（n=128 为判级网格）

重算方式：从 `.pt` 直接取交界面量（u 在 x=0.5 恰好是面 `u[64,:]`，v 在 y=0.5 是面 `v[:,64]`），在 Ghia 采样点用 **PCHIP**（审计员自选，非被审的 Hermite）插值；涡心由自积分流函数 `ψ=∫u dy` 取极小；散度按 MAC 定义重算。

| 量 | 被审 got | 审计重算 | 基准/容差 | 判定 |
|---|---|---|---|---|
| u_min (x=L/2) | -0.385368 | **-0.385304**（0.630%） | -0.38289，<3%（5% band） | ✅ |
| v_max (y=L/2) | 0.373840 | **0.373867**（0.786%） | 0.37095，<3% | ✅ |
| max\|u-u_Ghia\|（内点） | 0.0043073 | **0.0043074** | <3% of U | ✅ |
| max\|v-v_Ghia\|（内点） | 0.0111841 | **0.0112200** | <3% of U | ✅ |
| 主涡心 | (0.52734,0.56641)，\|Δ\|=0.00396 | 面配置 (0.53125,0.55859)，\|Δ\|=**0.00391** | (0.5313,0.5625)，≤0.01 @n≥128 | ✅ |
| max\|∇·u\| | 1.3500e-13 | **1.3500e-13**（逐位相同） | <1e-10 | ✅ |
| 加密单调性 | 32:17.4/15.0% → 64:2.80/2.34% → 128:0.65/0.78% | 32:14.4/14.3% → 64:2.56/2.27% → 128:0.63/0.79% | 单调 | ✅ |
| 延拓中性（n=64，两初值） | max\|Δu\|=3.4512e-8 | **3.451156899147989e-8**（逐位相同） | <1e-6 | ✅ |
| 稳态性（重启 n=128 再走 1000 步） | — | **max\|Δu\|=1.09e-7, max\|Δv\|=1.11e-7** | 与 1e-7 收敛阈值同量级 | ✅ |

**是否 <3%：是**（关键量 u_min 0.630% / v_max 0.786% / max|Δu| 0.43% / max|Δv| 1.12% / 涡心 0.0039 L ≤ 0.01 L）。与平台 15% 底线、5% ledger band 均有大余量。

## 3. 级别判定（L0–L5）

- **L2** ✅ 真代码：`src/tensorfvm/case_cavity_re1000.py`(506 行，输入→求解→打分→工件→CLI) + 内核 `src/tensorfvm/lid_cavity.py:98-368`（交错 MAC、van Leer 受限面通量、隐式扩散、投影法、DCT-II 纯 Neumann 压力解）；非占位、非硬编码（审计员冷启动复现逐比特相同）。
- **L3** ✅（仓库级）：`src/tensorfvm/verification/ladder.py:1052-1140` 的 C2 判据**实际读取**本单写出的 `lid_cavity_Re1000_n128.json`（数值与 summary 一致：0.004307/0.011184/104250/1.35e-13），C2 判 PASS（2% 杆，measured 0.011184）；`test_ladder_consumes_the_run` 实测 PASS 而非 skip；本单之前主梯 C2 的最细行是 n=64，本单把 Re=1000 判据抬到 n=128。**平台侧未接入**：`tensorbench_results` 对 20144/19978/15369 均 0 行 → 因此不主张"平台集成"。
- **L4** ✅ 完整算例 + 误差分析（3 网格、17+17 点逐点表、加密单调、延拓中性、稳态判据）+ 可复现（入口、工件、机检门、审计员逐比特复现）+ 外部锚 Ghia 1982 容差内。
- **L5** ❌ 拒绝：教科书基准，单一外部来源，无新物理/新数值方法贡献。

## 4. 发现（按严重度）

**F1（阻断级-账务，非实质）锚点错配**：本任务锚点 `CASE-FVM-CAVITY-RE1000` 在 `tensor_benchmark_cases`、`anchor_leases`、`repro_pool`、`tensor_xval_cases`、`tensor_case_pool.linked_anchor` **全部 0 行**；基准裁判行（id=18）挂在**另一 slug** `CASE-CAVITY-RE1000-FVM`（对应 case-pool 代码 `CAVITY-RE1000-FVM`，mission 15369），本单交付使用的正是后者的 CASE_SLUG。后果：按本单锚点解析裁判行会得空 → 机械 gate 若按锚点查表将 WITHHELD。实质不受影响：该裁判行的度量与容差正是本单被判的量，且 n=128 全部满足。

**F2（次要）机检门比注册裁判行松**：`CRITERIA` 全部用 5%/0.05 门（`case_cavity_re1000.py:73-99`），注册行为 <3% 与涡心 ≤0.01@n≥128。实测值同时满足两者（所以交付未被"放水"抬高），但 pytest 只断言 5% 带——若将来解退化到 3%~5% 之间，机检门仍会绿。

**F3（次要）机检门未从原始流场重算**：bundle 层测试把 `summary.json` 的度量读回再跑 `judge()`（`tests/test_case_cavity_re1000.py:34-49`），没有从 `.pt` 重算；`judge()` 只保证"自洽"，不保证"数出自流场"。本次审计用 `recompute_from_fields.py` 补上这一层，结论一致。

**F4（证据边界）**：无平台入库；分支未合 main（gate 走 main 时 `test_ladder_consumes_the_run` 会 skip，C2 退回 n=64 行）。

**F5（方法学提示）**：`max|Δu|` 对插值方案敏感——粗网格尤甚（n=32：被审 Hermite 0.2033 vs 审计 PCHIP 0.0792；n=64：0.0293 vs 0.0189），n=128 两者差 1.4e-7。被审取的是**更保守**（更大误差）的那个，非缺陷；但逐点判据在粗网格应视为带 ±0.01 量级方案不确定度。

**F6（审计员自身错误，如实披露）**：本次首版手抄 Ghia Re=1000 表在 34 项中错了 9 项（u 的 y=0.5/0.4531/0.2813，v 的 x=0.5/0.0938/0.0781/0.0703/0.0625，另 1 项末位）。错误因为与收敛流场在光滑中腔区差到 0.06 U（其余点 <0.005）而被发现；修正后的副本与被审模块表 34/34 一致，与 **TensorLBM `src/tensorlbm/lid_driven_cavity.py:GHIA_RE1000`** 的独立转录 17/17(u)、17/17(v)（v 顺序反向）一致。结论：**参考值不能凭记忆录入**，必须跨仓/注册表核对。

## 5. 未执行的检查（记为 not_run，不记 pass）

- `scripts/render_case_cavity_re1000.py` 重跑以证明报告零手抄（会写入被审工作树，审计方不修改被审树；改为数值抽查：报告 ④/⑤ 表与被审 summary 一致）。
- 完整 n=64+n=128 全序列重跑（>10 min，且重复 `cj2580` 已证明的确定性；只复现了 n=32 冷启动与 n=128 重启）。
- 未核 `--ladder-dir` 之外平台侧消费方（`tensorbench_results` 为空，无对象可核）。

## 6. 处置

approve with notes，判 **L4**，gate 期望 **confirm**。非阻断返修（下一轮）：
1. 统一锚点名（在本单锚点下补登记裁判行，或把 case-pool `FVM-CAVITY-RE1000` 与 `CAVITY-RE1000-FVM` 合并），使"按锚点解析"可用；
2. `CRITERIA` 收紧到注册裁判行（3% / 涡心 0.01@n≥128），并在测试里断言"不满足 3% 即 FAIL"；
3. 加一条从 `.pt` 重算度量的测试（可直接复用 `recompute_from_fields.py` 的口径）；
4. 合并到 main 并把度量写入 `tensorbench_results`，使 L3 由"仓库级"升为"平台级"。