# TensorFVM 仓库 git 历史瘦身方案（FVM-REPO-SLIM）

状态：**方案稿，交 owner 决策**。按验收契约 `scope: 只出方案禁执行历史改写`，本单未对任何共享仓执行历史改写；全部收益数字来自 `/tmp` 一次性镜像副本上的实测推演（`git-filter-repo a40bce548d2c`），推演副本用后可弃。

- 普查数据：`outputs/repo_slim_census/*.csv`（脚本 `scripts/repo_census.py`，只读）
- 汇总数字：`outputs/repo_slim_census/census_summary.json`
- 图：`fig1_by_extension.png` / `fig2_plan_benefit.png` / `fig3_top_blobs.png`

---

## 1. 大对象普查（实测）

基线（主检出 `/root/tensor-ecosystem/TensorFVM`，2026-11-17）：

| 指标 | 数值 |
|---|---|
| pack 体积 | **1.31 GiB**（count-objects size-pack，与锚点 1.33GiB 一致） |
| blob 逻辑总量 | 2125.23 MiB（19887 个） |
| docs/ 占比 | **1966.42 MiB（92.5%）**，11525 个 blob |
| origin/main 可达 | 1956.17 MiB |
| 仅 agent/PR ref 可达 | 108.56 MiB |
| refs 总数 | 1172（其中本地 agent 分支头 **937**） |

按扩展名（逻辑 MiB）：`.npz` 702.0（881 个）> `.png` 353.9（1984）> `.svg` 324.5（663）> `.pdf` 261.8（863）> `.json` 181.7 > `.msh` 65.8 > `.py` 46.0 > `.pyc` 39.1（**2036 个**）。

Top-10 blob 全部位于 `docs/verification-*/`（npz 场数据、svg 网格图、pdf 报告），单枚 10.5–27.1 MiB，引入提交集中在 2026-10-10~11 的 mission 交付批次；`src/` 内也查到 `__pycache__/*.pyc` 与 `verification/artifacts/**/*.npz` 直接入库。

**出血仍在进行**：近 7 天（2026-10-24 起）main 新增 blob 3607 个 / **156.4 MiB**（≈22 MiB/天）。按此速率，不做防复发门禁的话，任何瘦身收益约 2 个月即被吃回一半。

关键判读：**膨胀主体在 main 历史本身**（1956 MiB），仅清分支不 rewrite 只能回收约 3%（见 §3 方案 C 实测）。

## 2. filter-repo 规则 + 收益估算（/tmp 镜像实测）

推演方法：`git clone --mirror` 主检出 → 副本上跑 `git filter-repo --invert-paths ... --force` → `count-objects`。源代码零损伤已验证：rewrite 前后 `src/ tests/ scripts/ pyproject.toml README.rst` 的 tree 条目 **blob sha 逐字节一致**，差异仅为规则命中的 pyc/npz 删除。

| 方案 | 规则（`--invert-paths --path-glob ...`） | pack 实测 | 收益 | 耗时 |
|---|---|---|---|---|
| 现状 | — | 1341 MiB | — | — |
| **A 保守** | `*.pyc` `*.npz` `*.pt` | **733.1 MiB** | −45.3% | 10.7 s |
| **B 彻底** | A + `*.png` `*.svg` `*.pdf` `*.msh` | **88.9 MiB** | **−93.4%** | 6.7 s |
| C 替代（§3） | 不改写，清 937 分支 + gc | 1310 MiB | −3% | — |

生产命令草案（B，供 owner 批准后执行，须在 §5 备份完成后）：

```bash
git clone --mirror git@github.com:cxs503/TensorFVM.git slim-work.git
cd slim-work.git
git filter-repo --invert-paths \
  --path-glob '*.pyc' --path-glob '*.npz' --path-glob '*.pt' \
  --path-glob '*.png' --path-glob '*.svg' --path-glob '*.pdf' --path-glob '*.msh' \
  --force
# 校验：blob sha 比对（src/tests 零差异）+ pytest 冒烟 + count-objects
git push --force --all origin && git push --force --tags origin
```

代价（B 方案）：HEAD 检出中的历史验证产物一并消失，fresh clone 从 ~1.6 GiB（pack 1.31 G + 检出 ~0.3 G）降到 **~0.26 GiB**（.git 92 MiB + 工作树 171 MiB）；docs/ 下 0 个 npz/png/pdf 残留。若 HEAD 产物仍需随仓分发 → 用 B′：rewrite 后追加一个"产物重灌"提交（仅保留最新一版），或转 Git LFS（见 §3-D）。

## 3. 替代案对比

| 代号 | 方案 | 实测/估算收益 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|---|
| C | 删 937 agent 分支 + reflog expire + `gc --prune=now` | **1.31→1.28 GiB（−3%）** | 无 force-push、零风险 | 治不了 main 历史 | 只值得作为 B 的配套清理，不解决主问题 |
| D | `git lfs migrate import` 转历史大文件 | 与 B 相当的 clone 收益 | 保留产物可下载 | 本机无 git-lfs；LFS 配额/带宽计费；agent 大量小 png/svg 不适合 LFS | 不推荐全量；仅当 D1 选"产物保留"时对 npz/msh 小范围用 |
| E | 全新 orphan 仓（clean-slate）+ 旧仓归档 | 等效 B | 无 force-push | 丢 PR/issue 关联与 sha 谱系，协作面断裂 | 仅当 force-push 协调不可行时的兜底 |
| F | 不改仓：CI/agent 全改 `--filter=blob:none --sparse` 检出 | 服务端 pack 不变 | 零协调成本 | GitHub 服务端与备份仍 1.31 GiB 且继续膨胀 | 作为防复发的辅助手段 |

## 4. force-push 协调面（执行前置条件）

1. **冻结窗口**：选择合并低峰窗口（建议周末），提前 24h 冻结 merge-master 流水线与 agent 递交（本仓 merge 由平台自动推 GitHub，冻结须 owner 在平台侧操作）。
2. **GitHub 侧核对**（本单无 gh 凭证，未核，执行日必查）：branch protection 是否允许 force-push（需临时解除 main 保护）；open PR 数量（rewrite 后全部 PR 的 base sha 失效，需 rebase 或关闭重开）；是否有人 fork（fork 不受影响但会永久保留旧对象）。
3. **本地 937 个 agent 分支**：rewrite 后 sha 全变，建议执行日直接批量删除（`git for-each-ref --format='delete %(refname)' refs/heads/agent/ | git update-ref --stdin`），agent worktree（/root/tp-agent/work/*）全部重新 clone；在飞 mission 的 worktree 须在冻结窗口内确认无未提交成果。
4. **平台 merge 工具**：push-merge 流程引用旧 sha 的部分需在 rewrite 后自检一次（`git fsck` + 试合并一个空 PR）。
5. **通信**：rewrite 提交一个锚点 commit（如 `repo-slim-anchor`），所有协作者以 `git fetch --prune && git reset --hard origin/main` 对齐，禁止旧 clone 上直接 pull --rebase（会复活旧对象）。

## 5. 回滚镜像保全（执行前置条件，顺序在第 0 步）

1. **全量 bundle**：`git bundle create TensorFVM-pre-slim-$(date +%Y%m%d).bundle --all`（≈1.31 GiB），另做 `git clone --mirror` 一份裸镜像，双份异机存放。
2. **登记**：bundle sha256 + 存放路径写入运维台账；保留期建议 ≥6 个月（覆盖旧 PR 引用追溯与审计需求），到期复审。
3. **回滚预案**：若 rewrite 后发现必须回滚——GitHub 侧用 bundle 重建：`git clone TensorFVM-pre-slim.bundle && git push --force --all origin`；回滚同样需要冻结窗口，且 rewrite 后新产生的提交会丢失（故 rewrite 后观察期 48h 内禁止合并）。
4. **GitHub 服务端残留**：force-push 后旧对象进入 GitHub 内部去重池，不可即时回收（可能触发 GitHub support 手动 gc，或等待自然过期）；这不影响"新 clone 变小"的目标，但审计上要知晓旧 sha 在服务端短期内仍可寻址。

## 6. 防复发（无论选哪个方案都必须做，否则收益 2 个月减半）

1. CI 门禁：单 blob > 5 MiB 拒绝入库（白名单机制留给真正的基准数据，走 LFS 或制品服务器）。
2. `.gitignore` 增补：`__pycache__/`、`*.pyc`、`*.npz`、`*.pt`、`docs/verification-*/**.png|svg|pdf`（按 D1 决策裁剪）。
3. 产物外置规范：mission 交付的场数据/大图 → 制品服务器或 release assets，仓库内只留指针（路径+sha256+获取脚本）。

## 7. 决策点（≥3，交 owner）

- **D1 规则选型**：A（−45%，保留全部图表/PDF）vs B（−93%，验证产物全出库）vs B′（B + HEAD 最新一版产物重灌提交）。**建议 B′**：clone 收益接近 B，docs 当前版可视化不丢。
- **D2 产物去向**：HEAD 验证产物是否转 LFS / 制品服务器（影响 D1 的 B′ 细节与 §6.3 规范）。
- **D3 执行窗口与冻结**：何时冻结 merge-master 与 agent 递交；937 个本地 agent 分支是否授权批量删除。
- **D4 回滚镜像保留期与存放**：6 个月是否合适；bundle 放哪台异机。
- **D5 防复发门禁上线时点**：建议先于 rewrite 上线（先止血再瘦身）。

## 8. 本单边界声明

- 未执行：对共享仓/远端的一切历史改写与 force-push（验收契约禁止）。
- 未核：GitHub 侧 open PR 数、branch protection 状态、fork 情况（无 gh 凭证，已列入 §4 执行日清单）。
- 推演副本 `/tmp/fvm-slim-sim/{base,a,b,c}.git` 仅供复核，可随时删除。
