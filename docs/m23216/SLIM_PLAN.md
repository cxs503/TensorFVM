# TensorFVM 仓库 git 历史瘦身方案（m23216 ｜ **本单未执行任何改写**）

> 本文件由 `scripts/audit/m23216_git_census.py` 按 `scripts/audit/m23216_slim_plan.md.tpl` 生成：
> prose 在模板里，**所有数字由脚本实测注入，无手抄**。
> 复现：`python3 scripts/audit/m23216_git_census.py --out docs/m23216 --runs runs/m23216`（约 12 秒）。
> 生成时刻 2026-11-16 09:17:10 ｜ HEAD `2548a94e` ｜ 分支 `agent/m23216` ｜ 机器表 `docs/m23216/tables/`、`docs/m23216/census.json`

## 0. 一句话结论

体积不在工作树，在历史：可达 blob 合计 **2044.2 MiB**，其中只有 **183.2 MiB（9.0%）** 仍被 HEAD 引用，
**1861.0 MiB（91.0%）是已删除/已被覆盖的历史工件**（主体于 2026-10-28 由 `b77e7783 Remove reproducible
calculation artifacts` 一次性删除，但对象仍留在 pack 里）。所以删工作树文件不省体积，只有历史改写才省。
主案（`git filter-repo` 路径规则，保图）预估 pack **1367.7 MiB → 475.6 MiB（-65%）**；
另有 **1201.0 MiB** 的 `tmp_pack_*` 垃圾（5 个，2026-11-13 中断 fetch 残留）可零风险回收，与改写完全独立。

## 0.1 验收对照（本锚点 6 项交付 → 本文章节）

| 要求 | 交付位置 |
|---|---|
| ① 大对象普查（top-30 / 后缀族 / 时间窗） | §2.1 top-30（`tables/top30_blobs.csv`）、§2.2 后缀族、§2.3 时间窗；图 1、图 2 |
| ② 主案 filter-repo 路径规则 + 收益预估（含改写后 pack 预估） | §3.1 估算方法、§3.2 场景对照（`tables/scenarios.md`，图 3）、§3.3 规则草案、§3.4 工作树影响 |
| ③ 替代案对照（orphan / 双仓拆分） | §4 表（A/B/B'/C/D 五案收益与代价） |
| ④ 协调面：force-push 影响清单 + 窗口步骤 | §5（共享 gitdir / 323 工作树 / 远端 ref / runner / 文档哈希引用 + 10 步窗口） |
| ⑤ 回滚：镜像保全 + 校验 | §6（bundle、备份 ref、对账指纹、fsck+对象数/提交数判据） |
| ⑥ 决策点 ≥3 条交 owner | §7（D1–D6） |

## 1. 事实基线（实测）

| 指标 | 数值 | 取法 |
|---|---|---|
| `size-pack`（可达 pack 合计） | 1367.7 MiB | `count-objects -vH` / pack 目录实测 |
| pack 文件数 / 最大单包 | 46 / 1191.5 MiB | `objects/pack/*.pack` |
| GitHub 侧仓库体积 | 1303.8 MiB（1335083 KiB） | `GET api.github.com/repos/cxs503/TensorFVM` → `size` |
| 本地 gitdir 占用 | 3151.2 MiB | `du -s`（含垃圾） |
| `tmp_pack_*` 垃圾 | 1201.0 MiB / 5 个 | `count-objects -v` 的 `size-garbage` |
| 本地 ODB 对象数 / in-pack | 38487 / 38350 | `count-objects -vH` |
| 提交数（全引用可达） | 2184 | `rev-list --all --count` |
| main 第一父链提交数 | 820 | `log --first-parent` |
| 历史起点 | 2026-10-13（根 `9b668ded3e00`） | `log --reverse -1` |
| 本地引用 | 1048（{"heads": 833, "remotes": 215}） | `for-each-ref` |
| 远端引用 | {"HEAD": 1, "refs/heads": 555, "refs/pull": 908} | `git ls-remote origin` |
| 开放 PR | 0 | `GET /pulls?state=open` |
| 关联工作树 | 326 个（55 个 prunable） | `git worktree list` |
| 可达 blob / 字节 | 16273 / 2044.2 MiB | `rev-list --objects --all` ⨝ `batch-check` |
| HEAD 唯一 blob / 字节 | 6064 / 183.2 MiB | `ls-tree -r HEAD` 去重 |
| 历史 dead blob / 字节 | 10209 / 1861.0 MiB | 集合差 |
| 压缩标定（pack / 未压缩对象） | 0.640 | 实测 |

## 2. ① 大对象普查

### 2.1 TOP-30 blob（按未压缩体积，全部引用可达、去重）

`n` = 该 blob 在（commit,path）历史中出现次数；首现/末现为提交时间；`dead` = 不在当前 HEAD；最后一列是该路径**当前 HEAD 的状态**
（同名异 blob 表示 HEAD 里还有同名工件但内容已更新 → 按后缀删除会连当前文件一起删）。机器表：`docs/m23216/tables/top30_blobs.csv`。

| # | blob | MiB | 后缀 | 路径族 | n | 首现 | 末现 | dead | 该路径当前状态 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `0e0bd310642f` | 27.1 | `.npz` | `docs/verification-gpu/annular-1024x256/cpu-fields.npz` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 2 | `159e455b8fd5` | 27.1 | `.npz` | `docs/verification-engineering/annular-1024x256/fields.npz` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 3 | `8c45dbcea948` | 26.6 | `.npz` | `docs/verification-gpu/annular-1024x256/cuda-fields.npz` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 4 | `a10d140bd771` | 23.0 | `.npz` | `docs/verification-suboff-single-domain-mesh/extra-fine/native-geometry.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 5 | `9e13dd8ea0de` | 22.9 | `.npz` | `docs/verification-transient-pressure-compatibility/fine-hodge/fields.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 6 | `0b184996dffd` | 21.7 | `.svg` | `docs/verification-overset-30p30n/background-128/mesh.svg` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 7 | `c28d267caeb0` | 19.8 | `.npz` | `docs/verification-benchmarks/lbm-tg-64-ma-0.05/fields.npz` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 8 | `bd416379abdc` | 17.8 | `.svg` | `docs/verification-overset-30p30n/background-128/solution.svg` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 9 | `d747dd21c801` | 17.8 | `.svg` | `docs/verification-overset-30p30n/background-128/error.svg` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 10 | `6638440a1180` | 17.4 | `.npz` | `docs/verification-engineering-export-failure/annular-1024x256/fields.npz` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 11 | `53164341733a` | 14.1 | `.pdf` | `docs/verification-overset-30p30n/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 12 | `f08e85507967` | 13.8 | `.npz` | `docs/verification-suboff-wall-resolved-mesh/attempt-01/native-geometry.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 13 | `896c656e8fd0` | 13.0 | `.msh` | `docs/verification-suboff-single-domain-mesh/extra-fine/suboff-hybrid.msh` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 14 | `419d0a011fdd` | 10.5 | `.svg` | `docs/verification-sa-refinement/figures/sa-256x224-eddy-viscosity.svg` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 15 | `65c162d5a1f6` | 10.5 | `.svg` | `docs/verification-sa-refinement/figures/sa-256x224-speed.svg` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 16 | `b5c3967fa21b` | 10.5 | `.svg` | `docs/verification-sa-refinement/figures/sa-256x224-pressure.svg` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 17 | `73452b3e56a5` | 10.4 | `.msh` | `docs/verification-suboff-wall-resolved-mesh/attempt-01/suboff-hybrid.msh` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 18 | `79f46b6b36ca` | 10.2 | `.pdf` | `docs/verification-gmsh-cylinder-wall/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 19 | `87c630d3ccd1` | 10.2 | `.pdf` | `docs/verification-gmsh-cylinder/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 20 | `ac037c10e922` | 10.0 | `.npz` | `docs/verification-transient-pressure-compatibility/medium-hodge-half-dt/fields.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 21 | `101c8063160d` | 10.0 | `.npz` | `docs/verification-transient-pressure-compatibility/medium-hodge/fields.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 22 | `1503a72b6fb6` | 9.8 | `.pdf` | `docs/verification-gmsh-controls/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 23 | `aeff35496dfa` | 9.5 | `.npz` | `docs/verification-transient-pressure-compatibility/medium-legacy/fields.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 24 | `728e7b58a1fb` | 9.4 | `.npz` | `docs/verification-extended/abc-48/fields.npz` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 25 | `f31461572b4e` | 8.9 | `.pdf` | `docs/verification-gmsh-naca-hybrid/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 26 | `0fcad6af1e4d` | 8.7 | `.pdf` | `docs/verification-gmsh-naca-matched/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 27 | `08ce0e3ab78b` | 8.7 | `.pdf` | `docs/verification-gmsh-naca-matched-initial/report.pdf` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 28 | `3a8207572186` | 8.7 | `.csv` | `docs/verification-30p30n-hlpw4-diagnostic/fields.csv` | 3 | 2026-10-10 | 2026-10-28 | ✓ | 路径已不存在 |
| 29 | `180b81e13b85` | 8.1 | `.npz` | `docs/verification-suboff-single-domain-mesh/fine/native-geometry.npz` | 3 | 2026-10-11 | 2026-10-28 | ✓ | 路径已不存在 |
| 30 | `29295160eb6f` | 7.9 | `.npz` | `docs/verification-native-newton/kovasznay-fine/kovasznay-n96-mu0.025-fields.npz` | 3 | 2026-10-13 | 2026-10-28 | ✓ | 路径已不存在 |

观察：

- TOP-30 全部落在 `docs/verification-*` 证据工件目录（场文件/图/PDF/网格），无一条在 `src/`；
- **30/30 条在普查时已是 dead**：`n=3` 典型 = ① 分支加入 → ② 2026-10-13 合入 main →
  ③ 2026-10-28 被 `b77e7783` 删除；`末现=2026-10-28` 就是那条删除提交；
- 最大单 blob 27.1 MiB，远低于 GitHub 100 MiB 硬限且本仓无 LFS —— 膨胀是「数量 × 反复重生成」型，
  不是单点超大文件；收益来自历史累积，不来自单文件。

### 2.2 后缀族归类（全引用可达、去重）

| 后缀 | 合计 MiB | dead MiB | live MiB | blob 数 | 解释 |
|---|---|---|---|---|---|
| `.npz` | 701.8 | 669.2 | 32.6 | 879 | 场文件(可再生成) |
| `.png` | 351.5 | 283.6 | 68.0 | 1960 | 图(可再生成) |
| `.svg` | 324.5 | 324.3 | 0.2 | 663 | 矢量图(部分被 md 引用) |
| `.pdf` | 261.8 | 261.6 | 0.1 | 863 | 报告(可再生成) |
| `.json` | 174.8 | 135.0 | 39.8 | 3568 | 证据元数据(可检索,保留) |
| `.msh` | 65.8 | 65.8 | 0.0 | 39 | 网格(可再生成) |
| `.py` | 43.3 | 17.6 | 25.7 | 3917 | 源码(保留) |
| `.pyc` | 34.4 | 34.0 | 0.5 | 1794 | 不应入库 |
| `.csv` | 30.0 | 28.5 | 1.5 | 315 | 剖面数据(可再生成) |
| `.gz` | 29.4 | 29.4 | 0.0 | 115 | 压缩工件 |
| `.md` | 16.6 | 9.0 | 7.6 | 1553 | 文档(保留) |
| `.pt` | 2.3 | 0.3 | 2.1 | 18 | 张量存档 |
| `.dat` | 1.7 | 0.5 | 1.2 | 46 | 数据 |
| `.geo` | 1.5 | 0.1 | 1.4 | 16 |  |

`.npz+.png+.svg+.pdf` 四族合计 **1639.6 MiB（80.2%）**，全部可由 `runs/` 重新生成；
`.py+.md+.json` 是可读可 diff、需要保留历史的工程与证据元数据（合计 234.7 MiB）；`.pyc`（34.4 MiB）本不该入库。

### 2.3 时间窗（按日新增可达 blob 字节，main 第一父链）

| 日期 | 当日新增 MiB | 备注 |
|---|---|---|
| 2026-10-13 | 1753.4 | 一次性 import（历史 docs 工件） |
| 2026-10-15 | 1.4 |  |
| 2026-10-20 | 0.0 |  |
| 2026-10-21 | 0.1 |  |
| 2026-10-23 | 0.1 |  |
| 2026-10-28 | 0.1 | `b77e7783` 删除可再生工件 |
| 2026-11-02 | 0.0 |  |
| 2026-11-03 | 1.6 |  |
| 2026-11-04 | 8.4 |  |
| 2026-11-05 | 7.0 |  |
| 2026-11-06 | 5.6 |  |
| 2026-11-07 | 9.8 |  |
| 2026-11-08 | 9.5 |  |
| 2026-11-09 | 6.3 |  |
| 2026-11-10 | 5.7 |  |
| 2026-11-11 | 14.9 |  |
| 2026-11-12 | 19.8 |  |
| 2026-11-13 | 17.4 | agent 交付期，工件重新累积 |
| 2026-11-14 | 26.9 | agent 交付期，工件重新累积 |
| 2026-11-15 | 35.1 | agent 交付期，工件重新累积 |
| 2026-11-16 | 26.5 | agent 交付期，工件重新累积 |

- 2026-10-13 一次性入库 1753.4 MiB（导入历史 docs 工件）→ pack 的主要构成；
- 2026-10-28 `b77e7783` 清理后，该快照的工作树工件仅 18.8 MiB；
- 2026-11-03 起重新累积，最近 3 日合计 88.4 MiB、最近 7 日 146.2 MiB；按此速率**再过约 6 周**
  pack 就会重新长出 1 GiB —— 防腐策略（D6）与改写同等重要。

## 3. ② 主案：`git filter-repo` 路径规则 + 收益预估

### 3.1 估算方法（可复核）

1. `rev-list --objects --all` 得可达 blob 集合（去重 16273 个、2044.2 MiB）；
2. 对每个可达 blob 逐个跑 `zlib.compress(level=6)` 取**实测压缩后大小**（16273 个，非抽样；普查后新增长、缺样本的少数 blob 按同后缀均值兜底），保留集求和；
3. tree/commit 对象按 89.3 / 2.2 MiB × 0.55 / 0.45 折算；
4. **模型自校验 + 标定**：把「一条都别删」代入同一模型得 **1564.9 MiB**，实测 `size-pack` 1367.7 MiB，
   偏差 14.4% —— 该模型不含 delta 压缩、对相似工件偏保守，故统一乘标定系数 **0.874** 后再作为下表数字；
   剩余不确定带按 ±10% 记（同类工件 delta 收益的残差）。

### 3.2 场景对照（改写到「删除后」的 pack 预估）

| 场景 | 删除 MiB | 删除 blob 数 | 保留 MiB | 预估 pack MiB | 相对现状 |
|---|---|---|---|---|---|
| R1 只删 docs 场数据(npz/npy/vtk/vtu/msh/pt/dat/gz/csv) | 822.6 | 1354 | 1221.6 | 697.2 | -49% |
| R2 删场数据+网格+PDF，保留 png/svg 图 | 1084.4 | 2217 | 959.8 | 475.6 | -65% |
| R2b R2 + 全仓 *.pyc | 1118.8 | 4011 | 925.4 | 462.5 | -66% |
| R2c R2 + 删 docs *.svg（几乎全为 dead 图） | 1443.3 | 4674 | 600.9 | 338.3 | -75% |
| R3 删 docs 全部二进制 + 全仓 *.pyc | 1789.8 | 6567 | 254.4 | 56.9 | -96% |
| R5 阈值：删 docs 下 >=1 MiB 的 blob | 1112.2 | 341 | 932.0 | 599.3 | -56% |
| R5b 阈值：删 docs 下 >=512 KiB 的 blob | 1467.0 | 842 | 577.2 | 351.8 | -74% |

**推荐 R2（删场数据/网格/PDF，保留 png/svg 图，被 markdown 引用的图不破）→ 预估 475.6 MiB（-65%）**；
再叠全仓 `*.pyc` 即 R2b = **462.5 MiB**；加上删 docs `*.svg`（历史里 324 MiB、当前仅剩 0.2 MiB 存活）即 R2c = **338.3 MiB**；
若「图也可重新生成」，R3 一步到位 **56.9 MiB（-96%）**；
若不想按后缀、只按体积砍，R5 = 599.3 MiB、R5b = **351.8 MiB**。R1（只删场数据）697.2 MiB 明显不足。

### 3.3 规则草案（**本单不执行**，供窗口执行）

```bash
# 0) 依赖：本机现状 git 2.41.0，filter-repo / git-lfs 均未安装
python3 -m pip install git-filter-repo

# 1) 必须在新鲜克隆上做（filter-repo 拒绝非 fresh clone；本仓是 blob:none 部分克隆，需先补全 blob）
git clone --mirror /path/to/TensorFVM /backup/slim-work.git && cd /backup/slim-work.git
git -c remote.origin.promisor=false -c remote.origin.partialclonefilter= fetch --refetch origin || true

# 2) 干跑核对：导出路径体积清单，人工过一遍再动手
git filter-repo --analyze            # -> .git/filter-repo/analysis/path-all-sizes.txt

# 3) 主案 R2b（保留引用图，删场数据/网格/PDF/pyc）
git filter-repo --invert-paths --force \
  --path-glob 'docs/**/*.npz' --path-glob 'docs/**/*.npy' --path-glob 'docs/**/*.vtk' \
  --path-glob 'docs/**/*.vtu' --path-glob 'docs/**/*.msh' --path-glob 'docs/**/*.pt'  \
  --path-glob 'docs/**/*.dat' --path-glob 'docs/**/*.gz'  --path-glob 'docs/**/*.csv' \
  --path-glob 'docs/**/*.pdf' --path-glob '*.pyc'

# 4) 收尾（仅窗口内、改写后执行）
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git count-objects -vH
```

> filter-repo 默认只改写 `refs/heads`（其余需 `--refs`），且会**移除 `origin`**（防误推）；
> 它输出的 `.git/filter-repo/commit-map` 是 SHA 迁移表，必须随窗口发布。

### 3.4 改写对工作树的影响

- 当前检出中被删的 docs 二进制合计 102.2 MiB；
- docs markdown 引用到的图/数据 45.0 MiB（499/951 个文件有引用），未被引用 57.2 MiB
  → 先出「未引用清单」再删，或直接按 R2b 只删非图工件；
- `.gitattributes` 中的 whitespace 规则只影响内容比对、不改字节，filter-repo 不受影响。

## 4. ③ 替代案对照

| 方案 | 做法 | 预估值 | 收益 | 代价 / 风险 |
|---|---|---|---|---|
| A filter-repo 规则删除（主案 R2/R2b） | 原地改写全部 ref，删 docs 二进制 | **475.6 MiB**（R2b 462.5） | 降 65%，提交与 blame 连续（条目变 short） | 全部 SHA 变更：65 个文件含 95 处 40-hex 哈希；1048 本地 ref / {"HEAD": 1, "refs/heads": 555, "refs/pull": 908} 远端 ref 同步改写 |
| B orphan 重开仓 | 以 HEAD 快照建孤儿根提交，弃旧史 | 快照 183.2 MiB → 估 **131.8 MiB** | 最彻底 | 历史归零：820 个第一父提交的审计链断；不可逆（仅 bundle 可回滚） |
| B' orphan@2026-10-28 清理点 | 以 `b77e7783` 快照重开 | 快照 18.8 MiB → 估 **52.8 MiB** | 更彻底 | 同 B，且丢 10-13 之后全部历史 |
| C 双仓拆分（docs 独立仓） | src/tests/scripts 一仓、docs 一仓 | src 侧 **87.6 MiB**；docs 侧 982.5 MiB | src 侧彻底（91.3 MiB blob） | docs 侧不减（1953.0 MiB），必须叠加过滤；跨仓取证断链；仍要改写 |
| D 只 gc + 防腐策略 | 清 tmp_pack、加 pre-commit 门 | pack 仍 1367.7 MiB，但 gitdir 立刻少 1201.0 MiB（占 88%） | 零风险、约 10 min | pack 本身不变，1.2 GiB 历史工件永留，clone 成本不降 |

建议：立刻做 D，窗口内做 A；B/B' 仅在 owner 判定「历史无审计价值」时采用；C 单独做不解决 docs 体积。

## 5. ④ 协调面：force-push 影响清单与窗口步骤

本仓 gitdir 是**共享**的：`rev-parse --git-common-dir` 指向同一 `.git`，本机 **326 个工作树**
（`/root/tp-agent/work/*` 与本地镜像 `/root/tensor-ecosystem/TensorFVM`）共用同一 `objects/`。
因此受影响面不是「99 个工作树」，而是**全部 326 个同时受影响**。

| 受影响对象 | 实测情况 | 处理 |
|---|---|---|
| 共享 gitdir + 326 工作树 | 同一 `.git`，55 个已 prunable | 窗口内 `worktree prune`；改写后各树 `git reset --hard`（旧 commit 不存在，不能 merge） |
| 本地镜像 `/root/tensor-ecosystem/TensorFVM` | 上述 gitdir 宿主 | 改写前 bundle 备份（§6），改写后重建 |
| 本地 agent 分支 | 833 个 heads（大量一次性 `agent/m*`） | 冻结前确认无未合并交付，否则先落 main |
| 远端引用 | {"HEAD": 1, "refs/heads": 555, "refs/pull": 908}（`refs/pull/*` 908 条） | 改写后旧 PR diff 基线失效；开放 PR = 0，PR 阻塞面为零，但历史 PR「Files changed」不可复现 |
| runner / CI fetch | `.github` 内未发现 clone/fetch 步骤；runner 按 ref 拉取 | 旧 SHA fetch 会 404；`blob:none` 部分克隆缓存全部作废，须清缓存重拉 |
| 文档内哈希引用 | 65 个 docs 文件 / 95 处 40-hex 哈希 | 发布 `commit-map`，加 `docs/MIGRATION-m23216.md` |
| 平台侧（报告/锚点 commit 字段） | 报告与锚点记录 commit | 窗口内冻结平台写入，事后按 map 回填或标注「改写前 SHA」 |

窗口步骤（建议 60–90 min）：

1. T-24h 广播：冻结新交付合入 main，登记在飞分支；
2. T-0 快照：`git bundle create --all` + 记录 `count-objects -vH` / `rev-list --all --count` / `for-each-ref`；
3. 停写：暂停 runner 与平台合入通道；
4. 新建 `--mirror` 克隆做改写（**不在共享 gitdir 上直接 filter-repo**）；
5. 自检：`git fsck --full`；提交数与预估一致；`src`/`tests` 当前树与改写前 `git diff` 应只剩被删 docs 文件；
6. 回灌：`git push --mirror` 到本地 bare，或 `git fetch <slim> '+refs/*:refs/*'`；
7. 正典化：`reflog expire` + `gc --prune=now`；删 `tmp_pack_*`；
8. `git push --force-with-lease --mirror origin`（不推 `refs/pull/*`）；
9. 各工作树 `git reset --hard`；跑一次小集合冒烟（`pytest -q tests/...`）；
10. 解冻 runner（清缓存），发布 `commit-map`。

## 6. ⑤ 回滚：镜像保全与校验

```bash
# R1 全 ref bundle（refs/pull/* 属 GitHub 只读引用，不入 bundle）
git bundle create /backup/TensorFVM-pre-m23216-$(date +%Y%m%dT%H%M).bundle --all
git bundle verify /backup/TensorFVM-pre-m23216-*.bundle

# R2 远端备份 ref（钉住当前 main）
git push origin refs/heads/main:refs/heads/backup/pre-m23216-main

# R3 对账指纹
git count-objects -vH                                > /backup/pre-m23216-count-objects.txt
git rev-list --all --count                           > /backup/pre-m23216-commit-count.txt
git for-each-ref --format='%(objectname) %(refname)' > /backup/pre-m23216-refs.txt
git cat-file --batch-all-objects --batch-check='%(objecttype) %(objectsize)' \
  | awk '{c[$1]++; s[$1]+=$2} END{for(t in c) print t, c[t], s[t]}' > /backup/pre-m23216-objstat.txt
```

回滚流程：

```bash
git clone --mirror /backup/TensorFVM-pre-m23216-*.bundle /backup/restore.git
git -C /backup/restore.git fsck --full                    # 必须无 missing/broken
git -C /backup/restore.git rev-list --all --count         # == pre-m23216-commit-count.txt（2184）
git -C /backup/restore.git count-objects -vH              # in-pack 必须与 pre 一致（38350）
git -C /backup/restore.git push --force --mirror <origin> # 需 owner 二次确认
```

判据（回滚成功）：`fsck --full` 干净；提交数一致（2184）；`in-pack` 与 per-type 对象数同 `pre-m23216-objstat.txt`；
抽查 10 个 TOP-30 blob 的 `cat-file blob | sha256sum` 与改写前一致。

## 7. ⑥ 决策点（交 owner）

1. **D1 docs 工件历史保不保？** (a) R2 保 png/svg 图（475.6 MiB）；(b) R2b 再加全仓 pyc（462.5 MiB）；(c) R2c 再删 svg（338.3 MiB）；
   (d) R3 全删二进制（56.9 MiB）；(e) R1 只删场数据（697.2 MiB）。权衡文档可读性 vs 体积。
2. **D2 改写 / 重开 / 拆分？** filter-repo（存续 820 提交，SHA 全变）/ orphan（131.8 MiB，历史归零）/
   双仓（src 侧 87.6 MiB，docs 侧不减）。
3. **D3 是否接受全 SHA 变更的连带成本？** 65 个文件 95 处 40-hex 哈希、平台报告/锚点 commit 字段、
   {"HEAD": 1, "refs/heads": 555, "refs/pull": 908} 远端 ref、908 条 `refs/pull/*`。
4. **D4 时窗与停写时长**：建议 60–90 min 全冻结（runner + 自动合入 + 平台写入）；是否改为两段（先 gc，再改写）。
5. **D5 tmp_pack 处置授权**：5 个 `tmp_pack_*`（1201.0 MiB，2026-11-13 中断 fetch 残留）本单按令未删，
   需 owner 授权 `rm`（不影响任何对象，回收 1201.0 MiB）。
6. **D6 防腐策略（改写后必须同时上）**：pre-commit 拦 >512 KiB blob；`docs/**` 大工件走对象存储链接；
   CI 体积门（可达 blob 总量超阈值告警）。否则按 §2.3 速率，3 周内长回同等量级。

## 8. 本单边界（honest 声明）

- **零改写、零 force-push、零删除**：`git filter-repo` 未安装（实测 `git: 'filter-repo' is not a git command`），
  本单全部操作为只读（`rev-list` / `cat-file` / `ls-tree` / `log` / `du`）；
- 普查覆盖**本地已有对象**：本仓为 `--filter=blob:none` 部分克隆，理论上远端可能存在本地未取回的 blob；
  交叉校验 GitHub `size = 1335083 KiB` 与本地可达 pack 1367.7 MiB 相差 4.9%，说明本地普查对远端有代表性；
  但「100% 确定」需窗口内 `fetch --refetch` 全量后复测；
- 收益为**估算**（实测压缩模型 + tree/commit 折算），带 ±15% 不确定带，非改写后实测；
- 图 3 的 `now` 柱为 `size-pack` 实测，其余为估算，不可混读；
- 本单未逐条评估「被 md 引用的图若删除后文档站/CI 是否断链」，仅给出引用统计（499/951）。