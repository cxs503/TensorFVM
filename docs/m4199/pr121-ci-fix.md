# PR #121（agent/m3757）CI 红修复：图件被推送门滤除导致 3 个 job 全红

## 1 现象与复现

PR #121 head `61f06c6` 的 CI：`test (3.11)=failure`、`test (3.12)=failure`、
`lint (advisory)=failure`。合并治理配置（`merge_governor.json`）中
`ignore_checks.TensorFVM = ["lint (advisory)"]`，故 lint 只作告警，**阻塞合并的是 test**。

本地在 `61f06c6` 与 `origin/main`(`87d4083`) 的合并树上复现：

```
git checkout -B agent/m4199 origin/m3757-pr && git merge origin/main   # 无冲突
python3 -m pytest -q tests/test_simplec_mobility.py tests/test_simplec_artifact_budget.py
# FAILED tests/test_simplec_artifact_budget.py::test_figures_stay_legible
#   FileNotFoundError: .../scripts/artifacts/simplec-mobility/evidence/simplec-convergence.png
# 1 failed, 22 passed
```

## 2 根因

推送线 `tp_pusher.api_push` 的变更门禁 `_EXCLUDE2 = r'\.(db|sqlite3?|png|log)$'`
只对 `docs/` 下的 `*.png`/`*.svg` 放行（其注释：文档交付图属成果，运行记录图不上仓）。
该 PR 提交信息里的「3 个文件被门禁滤除」正是
`evidence/simplec-{convergence,profile,iterations}.png`，于是远端 head 缺图，
而 `test_figures_stay_legible` 要求这三张图存在且可读——**测试与推送门禁互相矛盾**，
且新增的 `tests/test_simplec_artifact_budget.py` 在 main 的 CI 里必然红。

## 3 修复（只动本 PR 自己的文件）

1. **图件迁到推送门放行的路径**：`scripts/artifacts/simplec-mobility/evidence/*.png`
   → `docs/simplec-mobility/*.png`（沿用门禁对 `docs/` 下文档图的既有例外）。
   图件本身未变（仍是 128 色调色板、像素尺寸不变），内容取自 PR 的原始提交 `b1d041b`。
2. `tests/test_simplec_artifact_budget.py`：新增 `FIGURES = ROOT/"docs"/"simplec-mobility"`，
   `test_figures_stay_legible` 改读该目录；门限不变（宽 ≥1000/900/1000 px、<60 KB、
   palette 模式）。工件目录因此只剩数值证据（4 个文件，104 KB < 200 KB 帽）。
3. `scripts/artifacts/simplec-mobility/report.md`：三张图的相对链接改为
   `../../../docs/simplec-mobility/*.png`。
4. 本 PR 新增的 2 个脚本 + 2 个测试过 ruff：删未用 import（`math` 等）、
   `lambda` 赋值改 `def`（E731）、`import sys` 归位（E402）、`ruff format` 统一格式。

## 4 修复后验证

```
python3 -m ruff check scripts/slim_simplec_artifacts.py scripts/verify_simplec_mobility.py \
    tests/test_simplec_mobility.py tests/test_simplec_artifact_budget.py
# All checks passed!
python3 -m ruff format --check <同 4 文件>
# 4 files already formatted

python3 -m pytest -q tests/test_simplec_mobility.py tests/test_simplec_artifact_budget.py
# 23 passed（此前 1 failed, 22 passed）

python3 -m pytest -q tests/test_audit_native_suboff_run.py tests/test_external_mesh_contract.py \
    tests/test_mutation_coverage.py tests/test_native_viscous_passivity_mesh.py \
    tests/test_wall_law_literature.py
# 35 passed, 9 skipped  ← main 侧会扫仓库/artifacts 目录的 5 个测试面无回归
```

图件实测：`simplec-convergence.png` 1110×720 P/35143 B、`simplec-profile.png` 960×690 P/20948 B、
`simplec-iterations.png` 1050×660 P/17752 B——与预算测试门限一致。

## 5 局限

- 本次只跑了受影响面（PR 新增测试 + 5 个仓库扫描类测试），未跑完整 162 个测试文件
  （单次 shell 调用有 120 s 上限，全量 CI 由 GitHub 侧 3.11/3.12 矩阵跑）。
  合并树在 `src/`、既有 `tests/` 上与 main 逐字节相同（仅新增文件），故回归面受限。
- lint job 仍会因仓库既有 ruff 债（src/tests/scripts 全树）报 advisory failure，
  该 job 在 `merge_governor.json` 的 `ignore_checks` 中已豁免，不阻塞合并。