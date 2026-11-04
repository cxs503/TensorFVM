# PR #79（agent/m3186）工件瘦身：旧/新大小对照

合并门禁 `artifact_hold` 因工件类 blob（含 `artifacts/`、`*.png`）合计超出 200 KB 扣住 PR。
本次把证据重新以「受限形态」落库，而非删除证据。

## 1 大小对照（字节）

| 文件 | 旧 | 新 | 降幅 |
|---|---:|---:|---:|
| `verification.json` | 167679 | 45101 | -73.1% |
| `evidence/simplec-audit.json` | 157146 | 43424 | -72.4% |
| `evidence/simplec-convergence.png` | 130120 | 35143 | -73.0% |
| `evidence/simplec-profile.png` | 75210 | 20948 | -72.1% |
| `evidence/simplec-iterations.png` | 50978 | 17752 | -65.2% |
| `report.md` | 7457 | 7750 | --3.9% |
| `evidence/simplec-audit.md` | 3079 | 2853 | -7.3% |

合计：591669 → 172971 字节（577.8 KB → 168.9 KB，降 70.8%），
门禁余量 31829 字节（15.5%）。

## 2 语义保全（不是「截断」）

- JSON：仅去缩排（`separators=(",",":")`，浮点仍是 `repr` 全精度，键序不变）。
  逐迭代 `history` 轨迹按 head 3 + 每 10 步 + tail 2 采样，每条轨迹写 `history_sampling`
  （`total_iterations`、`kept`、`stride`、`full_sha256`）——未保留的行可由重生成核对。
- 校验：删去 `history`/`history_sampling` 后，新旧 JSON 的规范化文本**逐字节相同**；
  保留行与原行逐字段相同；`full_sha256` 与原始全轨迹 SHA-256 一致（9 条轨迹 + 1 条失败史全通过）；
  `verification.json["audit"]` 与 `evidence/simplec-audit.json` 规范化文本一致。
- PNG：128 色调色板、像素尺寸不变（未缩放）。
- 可复算：`scripts/slim_simplec_artifacts.py` 从全保真度工件重放得到**逐字节相同**的结果
  （verification.json md5 `d27367820c76aa5e90b7e85e08541539`）。

## 3 测试证据

- `pytest tests/test_simplec_mobility.py tests/test_simplec_artifact_budget.py -q` → **23 passed**（11 守卫 + 12 新增预算/语义守卫），34.7 s。
- 本次改动不触碰 `src/` 与既有测试，故测试面不受影响；新增守卫钉住上限、嵌入审计一致性、
  每 run 门指标（velocity_l2/pressure_gradient 等 10 项）、轨迹记账与图件可读性。
- 复算命令：
  `python scripts/verify_simplec_mobility.py --output scripts/artifacts/simplec-mobility`
  → `python scripts/slim_simplec_artifacts.py --output scripts/artifacts/simplec-mobility`

## 4 提交与状态

- 旧 head `2001f732cd1be659415b5c2304b160b1466be9a5`（PR #79）→ 新提交 `ddb7a2e`
  （父提交即旧 head，线性一条提交，无其它内容改动）。
- 直接 push 被平台安全策略拦截：本单分支 `agent/m3757` 已指向 `ddb7a2e`，
  由平台推送后 governor 下周期自动复验合并（PR #79 head 需更新为 `ddb7a2e`）。
