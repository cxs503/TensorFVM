# FVM-SLIM-GUARD 审计与次优项建议（m24832）

> 状态：审计完成，锚点已交付，提交次优项建议
> 锚点：`FVM-SLIM-GUARD`（TensorFVM 仓库防腐三件套：pre-commit 拦大 blob + CI 体积门 + MIGRATION 骨架）

## 1. 锚点核对结果

| 字段 | 值 |
|---|---|
| slug | `FVM-SLIM-GUARD` |
| title | TensorFVM 仓库防腐三件套（D6：pre-commit 拦大 blob+CI 体积门+MIGRATION 骨架） |
| acceptance | `{"gate":"512KiB 拦截+CI 阈值门","why":"再生 25-35MiB/日,6周长回1GiB"}` |
| status | **in_progress**（但实现已完成） |

### 三件套交付状态

| 组件 | 文件 | 状态 |
|---|---|---|
| pre-commit 拦大 blob | `.githooks/pre-commit` + `scripts/guard/repo_guard.py staged` | ✅ 已交付（m23951/m24008） |
| `.gitattributes` 白名单 | `.gitattributes`（`guard-allow-large` 属性） | ✅ 已交付（m24008） |
| CI 体积门 | `.github/workflows/repo-guard.yml` + `repo_guard.py ci` | ✅ 已交付（m24008） |
| MIGRATION 骨架 | `docs/MIGRATION.md` + `docs/MIGRATION-m23216.md` | ✅ 已交付（m23216/m23832/m24008） |
| 基线 reanchor | `scripts/guard/pack_baseline.json` + `live_baseline.json` | ✅ 已交付（1.3 GiB pack, 188 MiB live） |
| 注入测试 | `scripts/guard/injection_test.py` + `docs/m24008/injection-test.md` | ✅ 已交付 |
| 测试覆盖 | `tests/test_repo_guard.py`（22 测试，全部通过） | ✅ 已交付 |
| 报告图 | `docs/m24008/figures/fig_live_gate.png` | ✅ 已交付 |

### 结论

**FVM-SLIM-GUARD 三件套已全部实现并合并到 main**。feature_ledger 状态仍为 `in_progress` 系状态未更新，非功能缺失。

## 2. 验证证据

- `git show main:.githooks/pre-commit` — 钩子存在，调用 `repo_guard.py staged`
- `git show main:.gitattributes` — 白名单存在（npz/png/svg/pdf/msh/gz/csv）
- `git show main:.github/workflows/repo-guard.yml` — CI workflow 存在，调用 `repo_guard.py ci`
- `scripts/guard/repo_guard.py` — 452 行，覆盖 staged/ci/live/report/reanchor 五子命令
- `pytest tests/test_repo_guard.py` — 22/22 passed
- `python3 scripts/guard/repo_guard.py report` — 输出 JSON census，pack 1.3 GiB, live 188.8 MiB

## 3. 次优项建议

### 建议条目：`FVM-SLIM-GUARD-POST-GUARD-BLEED`

| 字段 | 值 |
|---|---|
| slug | `FVM-SLIM-GUARD-POST-GUARD-BLEED` |
| title | 防腐三件套上线后仓库体积再生监测与自动 reanchor |
| domain | TensorFVM / repo hygiene |
| acceptance | ① 每日 cron 运行 `repo_guard.py report` 并记录 drift；② drift > 50 MiB 时自动触发 reanchor 并告警；③ 生成趋势图（pack + live 随时间变化）；④ 连续 7 天 drift < 10 MiB 则判定再生已止 |
| 验收量化 | ① 报告含 ≥7 天数据点；② drift 计算误差 < 1%；③ 趋势图 png 内嵌 |

### 建议条目：`FVM-SLIM-GUARD-COMMIT-GRAPH-FIX`

| 字段 | 值 |
|---|---|
| slug | `FVM-SLIM-GUARD-COMMIT-GRAPH-FIX` |
| title | D7 commit-graph 修复（fsck rc=16 → 写共享 commit-graph） |
| domain | TensorFVM / repo hygiene |
| acceptance | ① `git commit-graph write --reachable` 执行成功；② `git fsck` rc=0；③ 写入 `.git/objects/info/commit-graph` |

## 4. 给下次自找活的建议线索

1. **FVM-SLIM-GUARD-POST-GUARD-BLEED**（优先级最高）：三件套已上线，但尚未验证"再生是否真的停止"。需要连续监测 7-14 天，生成趋势报告。
2. **FVM-SLIM-GUARD-COMMIT-GRAPH-FIX**（D7 待决）：`git commit-graph` 过期导致 fsck rc=16，需 owner 放行后修复。
3. **历史重写执行**（D1-D5/D7 待决）：filter-repo 方案已就绪（docs/REPO-SLIM-PLAN.md），仅待 owner 裁决。
