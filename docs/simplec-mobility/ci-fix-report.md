## 1 案例介绍

TensorFVM PR #131（`agent/m4020`）交付 SIMPLEC 压力-速度耦合的验证闭环：二维充分发展槽道
（Poiseuille）层流 + 定义级迁移率守卫 + 变异负控。CI 三个 job 全红
（`test (3.11)` / `test (3.12)` / `lint (advisory)`），本地复现到的唯一红测是
`tests/test_simplec_artifacts.py::test_artifacts_exist`：
`synthetic` 断言 0 >= 3 —— 报告引用的三张 png 图件不在推上去的提交里。

根因不在测试，在**推送门禁**：平台 pusher 的路径过滤器
`\.(db|sqlite3?|png|log)$` 只放行 `docs/` 下的 `*.png`，原提交把三张缩小图放在
`scripts/artifacts/simplec-mobility/evidence/*.png`，于是被静默滤除（原 commit 标题即
「3 个文件被门禁滤除」），GitHub 上的测试看到图件缺失，判红。

| 参数 | 值 |
|---|---|
| 物理算例 | 二维充分发展槽道，解析 `u(y)=6Ub(y/H)(1-y/H)`、`dp/dx=-12 mu Ub/H^2` |
| 网格 | ny = 12（汇总）/ [12, 24]（扫描） |
| Re_H / 收敛判据 | 1 / 残差 1e-06，最多 2000 次外迭代 |
| 解析验收门 | 相对误差 < 3% |
| PR 工件帽 | 204800 B（单文件 < 81920 B） |

## 2 怎么做的

1. 从 `agent/m4020` 取工作树，本地复现红测（`pytest tests/test_simplec_mobility.py
   tests/test_simplec_artifacts.py -x`），定位到图件缺失而非数值失败。
2. 读平台 pusher 的过滤规则，确认 png 只在 `docs/` 前缀下放行 → 结论：图件必须换落点，
   不能靠改测试「宽容缺失」。
3. `scripts/verify_simplec_mobility.py`：新增 `FIGURES = docs/simplec-mobility/evidence`，
   `publish()` 把缩小图写到这里，报告用相对路径 `../../../docs/simplec-mobility/evidence/*.png` 引用；
   `digest()` 保留图件**文件名清单**（不再整体丢弃），使「报告引用的图必须真的在」可机器校验。
4. `tests/test_simplec_artifacts.py`：断言三张被引用的图确实以门禁放行路径入库、
   工件目录不得残留 `*.png/*.db/*.log`（防复发），体积帽改为「摘要 + 图件」合计，
   报告链接与实际渲染前缀一致。
5. 用 `--slim-from` 从原始完整证据（`m4020-ev-2/verification.json`）重派生摘要与报告：
   被测源码（`solver.py` / `simplec_channel.py` / `cases.py`）一字节未动，
   `summary.json.provenance` 的 SHA-256 仍与工作树逐字节一致（该断言由现有守卫强制）。

## 3 计算结果

| 项 | 修复前 | 修复后 |
|---|---:|---:|
| 被门禁滤除的文件数（本单 10 个变更文件模拟 pusher 过滤） | 3 | 0 |
| `tests/test_simplec_artifacts.py` + `tests/test_simplec_mobility.py` | 1 failed, 11 passed | 19 passed |
| 报告引用的图件在仓内 | 0 / 3 | 3 / 3 |

| 仓内产物 | 大小 |
|---|---:|
| `docs/simplec-mobility/evidence/simplec-convergence.png` | 22050 B | convergence.png |
| `docs/simplec-mobility/evidence/simplec-profile.png` | 13497 B | profile.png |
| `docs/simplec-mobility/evidence/simplec-iterations.png` | 9525 B | iterations.png |
| `scripts/artifacts/simplec-mobility/summary.json` | 41075 B |
| `scripts/artifacts/simplec-mobility/report.md` | 8858 B |
| `scripts/artifacts/simplec-mobility/evidence/simplec-audit.md` | 3079 B |
| **合计（工件帽 204800 B）** | **98084 B** |

物理量未变（同一份完整证据重派生）：SIMPLEC(αu=0.8, αp=1) 49 次迭代收敛，
SIMPLE 默认 41 次、调参 43 次。

## 4 结果图

图 1 为修复前后门禁滤除文件数 / 守卫通过数的对照；图 2 是本单守护的物理证据本身：
截面 x = 5H 的速度剖面与解析抛物线重合。

## 5 注意点

1. **门禁是静默的**：pusher 只改 commit 标题里的计数（「N 个文件被门禁滤除」），
   本地测试永远绿、GitHub 上才红；任何二进制图件都必须落在 `docs/` 下。
2. **没有重跑全量审计**：本轮只重派生摘要，`elapsed_s`、pytest 时长等噪声字段保持原值
   （用与提交一致的 `m4020-ev-2` 证据库），被测源码摘要因此不变；要重算请跑
   `python scripts/verify_simplec_mobility.py --output scripts/artifacts/simplec-mobility`。
3. **lint (advisory) 仍红**：整仓 ruff 存量约 1 万条告警，工作流对该 job 设了
   `continue-on-error`；本单改动的两个 .py 文件 `ruff check` 全过（未新增存量债）。
4. **适用范围**：本修复只涉及证据落点与守卫，不改求解器数值路径。

## 6 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测 | 是否 < 3% |
|---|---|---:|---|
| 截面速度 L2（SIMPLEC 0.8/1.0） | 解析 `6Ub(y/H)(1-y/H)` | 0.634% | 是 |
| 下游压力梯度（SIMPLEC） | 解析 `-12 mu Ub/H^2` | 1.370% | 是 |
| 截面速度 L2（SIMPLE 0.7/0.3） | 同上 | 0.634% | 是 |
| SIMPLEC vs SIMPLE 固定点差 | 同一离散解 | ≤ 2.08e-06·Ub | 是 |
| 本单修复目标：被滤除文件数 | 0 | 0（修复前 3） | 是（精确匹配，无容差） |

结论：数值口径误差与修复前逐位一致（速度剖面 0.634%、
压力梯度 1.370%，均 < 3%）；本单唯一改动的是图件落点。
