# TensorFVM 仓库 git 历史瘦身方案（FVM-REPO-SLIM）

- 锚点: feature_ledger `FVM-REPO-SLIM`（status=in_progress）
- 范围: **只出方案，禁执行历史改写**（owner 验收口径）。所有数字来自 2026-11-17 对
  `/root/tensor-ecosystem/TensorFVM` 的只读普查 + `/tmp/fvm-exp/mirror-{a,b,c}` 一次性
  克隆上的无损实测（原仓零改动）。
- 结论先行: **推荐 B3（历史大 blob 打桩，HEAD 树 100% 保留）+ 分支/工作树清理 + 回滚镜像**，
  实测 pack 1349.4 MiB → **184.5 MiB（-86.3%，m23910 2026-11-18 回填）**；
  激进版 B2 实测 65.2 MiB（-95.2%）但会删掉当前 HEAD 里
  1023 个验证图/场文件，需 owner 拍板。

## 1. 大对象普查（2026-11-17 实测）

| 指标 | 数值 |
|---|---|
| `.git` 总大小 | 2.1 GiB（pack 1.33 GiB，35 个 pack + 402 loose 4.5 MiB） |
| 可达对象（全部 1118 refs） | 31775 个，raw 2.08 GiB → **repack 后 1.31 GiB**（mirror 实测） |
| HEAD 树 | 12761 文件 / **0.25 GiB**（最大单文件 3.9 MiB，>1 MiB 仅 10 个） |
| main 单分支历史 blob | 13110 个 / 1.91 GiB raw（侧分支仅 +0.09 GiB） |
| 提交数 | main 1152（2026-10-13 ~ 11-17，11-13~17 五天占 491 个） |
| 本地分支 / 远端分支 / worktree | 903 / 574（origin heads）/ 395 |
| remote | `https://github.com/cxs503/TensorFVM.git`，**partial clone（blob:none promisor）** |

可达 blob 质量分布（raw，全部 refs）：

| 类型 | 大小 | 个数 | 说明 |
|---|---|---|---|
| docs/npz | 694 MiB | 835 | 验证算例场量（annular/suboff/overset 等） |
| docs/png | 348 MiB | 1906 | 报告图 |
| docs/svg | 325 MiB | 663 | 报告图 |
| docs/pdf | 262 MiB | 863 | 验证报告 |
| docs/json | 170 MiB | 3330 | 结果台账 |
| docs/msh | 66 MiB | 39 | 网格 |
| **src/pyc** | **37 MiB** | **1794** | **编译产物，本就不该入库** |
| docs/csv / docs/gz | 59 MiB | 422 | 场量/日志 |

Top 单文件: `docs/verification-gpu/annular-1024x256/cpu-fields.npz` 28.4 MiB、
`docs/verification-engineering/annular-1024x256/fields.npz` 28.4 MiB、
`docs/verification-overset-30p30n/background-128/mesh.svg` 22.7 MiB 等（Top40 合计 ~400 MiB）。

**关键判断**
1. 膨胀 100% 在历史，不在工作树（HEAD 0.25 GiB 健康）。
2. `git fsck --unreachable` 报的 18537 个"不可达对象"（raw 1.86 GiB）是 **partial clone
   的 fsck 假阳性**：抽样 200 个中 118 个实际可达（`git log --all --find-object` 证实）；
   mirror 全量克隆 fsck=0 不可达，repack 后 1.31 GiB ≈ 原 pack。→ **单纯 gc 救不了**。
3. 侧分支只多占 0.09 GiB raw → 光删分支收益有限，必须动 main 历史。

## 2. filter-repo 规则 + 收益估算（mirror 克隆实测）

| 方案 | 规则 | pack 结果 | 收益 | HEAD 树代价 |
|---|---|---|---|---|
| A: 仅 gc | `git gc` | 1.31 GiB | -1.5% | 无 |
| B2: 全历史删二进制 | `--invert-paths --path-glob '*.pyc' '*.npz' '*.msh' '*.svg' '*.pdf' '*.png' '*.gz'` | **65.2 MiB（m23910 复测；11-17 为 58.9）** | **-95.2%** | 丢 1023 文件 / 78 MiB（当前 docs 里的图与场） |
| B3: 非 HEAD 大 blob 打桩 | `--blob-callback`：blob∉HEAD 树 且 >100 KiB → 替换为 1 行占位 | **184.5 MiB（m23910 实测，原估 150-250 区间内）** | **-86.3%** | **零**（HEAD 树逐字节不变） |
| B4: 仅删 pyc | `--invert-paths --path-glob '*.pyc'` | ~1.28 GiB（估） | -2.5% | 无 |

B3 实现备注: 本环境 git-filter-repo（a40bce548d2c）`--blob-callback` 在 mirror 上
fast-import "stream ends early"（回调模块可导入仍崩，疑版本 bug）。等价替代实现：
生成「历史大 blob 且当前树无此路径」的路径清单，用 `--invert-paths --path <P>` 逐条传入
（冲突路径=当前树里 >100 KiB 的同名文件，约 50 个，改写后补一个 re-add 提交即可）。
B3 精确数字已回填（m23910，2026-11-18，HEAD 8210a126）：blob 粒度剔除法实测 **184.5 MiB**（剔除 1938 个死 blob / 1706.1 MiB，存活 35884 对象；打桩后死 blob 收敛为 1 个 ~100 B 共享 stub，对 pack 影响 <1 KiB，故剔除法与打桩法等价）。原始数据 `docs/FVM-REPO-SLIM-M23910/measured_pack_B2-B3-R2c-R3.json`。

## 3. 替代案

| 案 | 内容 | 评价 |
|---|---|---|
| A: gc/repack | 只重打包 | 实测 -1.5%，不解决 |
| A2: 分支清理+gc | 删 903 本地/574 远端 agent/* 已合并分支后 gc | 收益 <10%（侧分支仅 +0.09 GiB），但**应作为配套**做（903 分支/395 worktree 本身是运维债） |
| B2: 全删二进制 | 见上 | 收益最大，但当前验证报告图/场从仓库消失，平台报告引用会断 |
| B3: 打桩 | 见上 | 推荐；历史可追溯（占位符注明去向），当前内容不动 |
| C: Git LFS 迁移 | 大文件转 LFS 指针 | 需全历史 rewrite + LFS 服务端 + 所有克隆重拉，成本高于 B3，且 LFS 服务端无现成设施 |
| D: 工件外置 | 大 npz/msh 迁 HPC 集群/对象存储，仓库留 URL 清单 | 与 B3 正交，可作为二期（把 HEAD 里 10 个 >1 MiB 文件也外置，HEAD 再瘦 ~20 MiB） |

## 4. force-push 协调面

- **远端**: origin 574 heads + tags；filter-repo 会重写**所有** ref 的 SHA → 必须
  force-push 全部 574 分支（或先删旧分支只推 main+活跃分支）。
- **本地生态**: 903 本地分支、395 个 worktree（`/root/tensor-ecosystem/TensorFVM/.git/worktrees`）
  全部指向旧 SHA。执行窗口内必须冻结 tp_pusher/merge 流水线（`tp_pusher.py`、
  `ship_merge.py`、`wave_cycle.py` 对 TensorFVM 的推送/合并），否则旧 SHA 回流。
- **跨仓引用**: 12 仓生态若有仓 pin 了 TensorFVM 的 commit SHA（CI/文档/ledger），
  重写后失效。执行前需 grep 各仓对 TensorFVM SHA 的引用（实施单第一步）。
- **partial clone 客户端**: 本机主仓是 blob:none promisor 克隆，重写后必须
  `git remote set-url` 重配 + 重新 `git fetch` 全量，promisor 状态要清
  （`git config --unset remote.origin.partialclone` 或重建克隆）。
- **窗口**: 建议选无在飞 TensorFVM 单的时间窗（查 taskboard 终态），冻结 ≥24h。

## 5. 回滚镜像保全

1. 执行前: `git clone --mirror /root/tensor-ecosystem/TensorFVM /root/tensor-ecosystem/TensorFVM-pre-slim-<date>`
   （bare，含全部 903 分支 + 395 worktree 的 ref），打 tag `pre-slim-<date>`。
2. 该镜像**不入 GitHub**（私有本地保全），保留 ≥90 天；期满由 owner 决定销毁。
3. 重写后新历史打 tag `post-slim-<date>`；两镜像 SHA 清单写入本文件附录（实施单回填）。
4. 回滚路径: 从 pre-slim 镜像 `git push --mirror` 回 origin 即可完整复原（含分支）。

## 6. 交 owner 决策点（≥3）

1. **D1 内容口径**: B2（当前 docs 二进制也删，实测 65.2 MiB / -95.2%）vs B3（当前树零损失，实测 184.5 MiB / -86.3%）？
   若选 B2，平台报告里引用仓库内 png/npz 的链接将失效，需同步改报告引用。
2. **D2 分支策略**: 重写前是否先清 903 本地/574 远端 agent/* 分支（只留 main+活跃）？
   建议清（重写面从 574 缩到个位数，force-push 面大幅缩小）。
3. **D3 回滚镜像位置与保留期**: 本地 bare 镜像保留 90 天是否可接受？是否要额外
   GitHub 私有镜像一份？
4. **D4 执行窗口**: 冻结推送/合并流水线的窗口时长（建议 ≥24h）与执行人（owner 亲批 or
   派 native 单）。
5. **D5 二期**: 是否立项「HEAD 内 >1 MiB 工件外置」（D 案）+ `.gitignore` 补
   `*.pyc`/`*.npz` 防复发？

## 7. m23910 复核（2026-11-18，HEAD 8210a126）

- 工具: `scripts/audit/slim_b3_measure.py`（只读，源仓零写入；存活集喂真实 `git pack-objects --stdout`，
  window=10 depth=50 threads=6，与 m23278 同口径）。
- 基线 keep-all 重打包 **1349.4 MiB**（11-16 为 1344.7，两天 +4.7 MiB ≈ 2.4 MiB/d，
  与 FVM-SLIM-GUARD 后 2.09 MiB/d 的出血率一致 → 守卫有效，漂移可忽略）。
- **B3 估算转实测: 184.5 MiB（-86.3%）**，落在原估 150-250 MiB 区间内，
  推荐案不变；B3p（B3+全仓 pyc）184.1 MiB（pyc 边际仅 0.4 MiB，与 m23278 结论一致）。
- 漂移复核: R2c 385.9 MiB（11-16 381.8，+4.1）、
  R3 66.2 MiB（11-16 63.7，+2.5）→ 各方案排序与降幅稳定。
- 可达对象 31360 → 37822（两天 +6462，多为 <512 KiB 小文件；守卫限幅后体积出血受控，
  但对象数仍在涨，二期建议把「对象数增速」纳入守卫监控）。
- 现场 `.git` 2.3 GiB（11-17 为 2.1 GiB）：瘦身前现场还会继续缓慢变大，执行窗口越晚成本越高（增量小，不改变方案）。

## 附录: 复现命令

```bash
# 普查
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(objectname) %(rest)'
git ls-tree -r -l HEAD | awk '{s+=$4} END{print s}'
# 实测（一次性克隆，勿在原仓跑）
git clone --mirror <repo> m && cd m
git filter-repo --invert-paths --path-glob '*.pyc' --path-glob '*.npz' \
  --path-glob '*.msh' --path-glob '*.svg' --path-glob '*.pdf' \
  --path-glob '*.png' --path-glob '*.gz' --force
git count-objects -vH
```
