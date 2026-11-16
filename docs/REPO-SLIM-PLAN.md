# TensorFVM 仓库 git 历史瘦身方案（FVM-REPO-SLIM）

> 状态：**方案（仅规划，禁执行历史改写）** · 锚点 `FVM-REPO-SLIM` · 分支 `agent/m23395`
> 验收口径：只出方案；含「大对象普查 / filter-repo 规则+收益估算 / 替代案 / force-push 协调面 / 回滚镜像保全」五节；决策点 ≥3 交 owner。

---

## 0. 问题契约与现状

- 仓库 `.git/objects/pack/` 合计 **1.33 GiB**（20 个 pack，磁盘 1.43 GB）。
- 主 pack `pack-0bb3171f…` 单包 **1.2 GiB**，占总量 ~90%。
- 全仓对象（`git rev-list --objects --all`）31,639 条；`main` 可达对象 21,083 条。
- **`main` 可达全部对象解压后 1672.6 MiB**，其中：
  - `docs/verification-*`（188 个子目录）**1473.9 MiB = 88%** ← 唯一主因
  - `docs/` 其余 167.3 MiB
  - `src/` 14.9 MiB · `tests/` 4.7 MiB · 其他 11.9 MiB
- **真实源码（src+tests）仅 ~20 MiB**。仓库体积被「验证产物」历史版本完全淹没。
- 工作树已无 >5M 文件（产物已从工作树删除），但**历史仍保留全部旧版本**——这是典型的「提交后删除、历史未清」膨胀。
- 大对象（>1MB）318 个，合计 ~1010 MiB，扩展名分布：`npz`(137) `png`(51) `svg`(50) `pdf`(23) `json`(22) `msh`(16) `gz`(9) `csv`(5)。
- 触及 `docs/verification-*` 的提交 **401 个**。
- 分支：`agent/*` 841 个、总 1076 个、已并入 `main` 149 个。
- 工具现状：**无** `git-filter-repo`、**无** `git-lfs`、**无** `.gitattributes`。
- 先例：`origin/maintenance/artifact-hygiene`、`origin/maintenance/remove-overset-embedded-svg` 两分支为**未并入 main 的旧平行尝试**（与 main 差异 11063 文件），须在协调面中处置，避免重复劳动。

---

## 1. 大对象普查（已执行，证据见上）

| 维度 | 数值 | 说明 |
|---|---|---|
| 全仓 pack | 1.33 GiB / 20 pack | 主包 1.2G |
| main 可达（解压） | 1672.6 MiB | 瘦身后基准 |
| docs/verification-* | **1473.9 MiB (88%)** | 唯一主因 |
| 大对象 >1MB | 318 个 / ~1010 MiB | npz/png/svg/pdf/msh 为主 |
| 源码 src+tests | ~20 MiB | 必须全保留 |
| 触及 verification 的提交 | 401 | 需 filter 覆盖 |

**Top 大对象**（全部在 `docs/verification-*`）：
- `…/annular-1024x256/cpu-fields.npz` 27.1M · `…/cuda-fields.npz` 26.6M · `…/fields.npz` 27.1M
- `…/suboff-single-domain-mesh/extra-fine/native-geometry.npz` 23.0M
- `…/overset-30p30n/background-128/mesh.svg` 21.7M（+solution/error.svg 各 17.8M）
- `…/lbm-tg-64-ma-0.05/fields.npz` 19.8M · `…/report.pdf` 14.1M · `…/suboff-hybrid.msh` 13.0M

**结论**：瘦身目标 = 从历史中移除 `docs/verification-*` 下的二进制验证产物（npz/png/svg/pdf/msh/gz/csv），保留其**文本说明**（.md/.rst）与源码。

---

## 2. filter-repo 规则 + 收益估算

### 2.1 推荐规则（`git filter-repo`）

```bash
# 安装（一次性，需 owner 批准在 CI 镜像装）
pip install git-filter-repo

# 在裸镜像上执行（见 §5 回滚保全），不碰工作树
git clone --mirror <origin> /tmp/tfvm-mirror
cd /tmp/tfvm-mirror
git filter-repo \
  --path-glob 'docs/verification-*/**/*.npz' \
  --path-glob 'docs/verification-*/**/*.png' \
  --path-glob 'docs/verification-*/**/*.svg' \
  --path-glob 'docs/verification-*/**/*.pdf' \
  --path-glob 'docs/verification-*/**/*.msh' \
  --path-glob 'docs/verification-*/**/*.gz' \
  --path-glob 'docs/verification-*/**/*.csv' \
  --invert-paths
```

**关键点**：
- `--invert-paths` = 只删匹配路径，**保留** `docs/verification-*/**/*.md`、`.rst`、`manifest.json`（文本说明与清单）。
- 不删 `src/`、`tests/`、`docs/` 其余、`scripts/`、`pyproject.toml`。
- 若担心误删「小但必要」的 json（如 `manifest.json` 是产物清单），可改为**只删大扩展名**（npz/png/svg/pdf/msh/gz），json/csv 单独评估——见决策点 D1。

### 2.2 收益估算（三档）

| 档位 | 规则 | 保留（解压） | 削减 | 主 pack 预估 |
|---|---|---|---|---|
| **A 激进** | 删全部 `docs/verification-*` 二进制（含 json/csv） | ~198 MiB | **-88%** | 1.2G → ~0.15G |
| **B 推荐** | 只删大扩展名 npz/png/svg/pdf/msh/gz（留 json/csv 清单） | ~250–300 MiB | **-82~85%** | 1.2G → ~0.2G |
| **C 保守** | 只删 >1MB 单文件（按 size 阈值） | ~680 MiB | **-59%** | 1.2G → ~0.5G |

- 三档均**不触碰源码**（src+tests ~20 MiB 全保留）。
- 压缩后 pack 体积约为解压的 60–70%，故 B 档最终 `git gc --aggressive --prune=now` 后预计 **~0.15–0.2 GiB**，较现状 1.33 GiB 降 **~85%**。
- 收益以「解压后 main 可达体积」为硬指标（可复现、可审计），不以 pack 字节数为准（pack 受压缩率/增量影响）。

### 2.3 可复现性

- 普查命令与数字已固化（见 §1 表），执行后可用同一命令复核：`git rev-list --objects main | …` 应回到 ~198–300 MiB。
- 建议在 `docs/REPO-SLIM-PLAN.md` 末尾追加「执行后复核表」，由执行者回填。

---

## 3. 替代案（不执行 filter-repo 的选项）

| 替代案 | 做法 | 收益 | 代价/风险 | 判定 |
|---|---|---|---|---|
| **S1 仅 gc + 删孤儿** | `git gc --prune=now` + 删未并入的 agent 分支 | 只清孤儿对象，**历史大对象仍在** | 收益 <10%，治标不治本 | ❌ 不推荐 |
| **S2 引入 git-lfs** | 大产物改走 LFS 指针 | 新提交瘦身，**历史已提交的仍占** | 需全仓迁移 + 服务端 LFS；历史不回填 | ⚠️ 仅作「未来防再犯」配套 |
| **S3 产物外置**（S3/对象存储） | 验证产物移出仓，仓库只留 manifest + 下载脚本 | 与 filter-repo 同向 | 需改 CI 取数链路；**历史仍须 filter 一次** | ⚠️ 与 filter-repo 叠加，非替代 |
| **S4 重建历史**（squash 到单根） | 重写为单提交 | 体积最小 | 丢失全部提交元数据/审计链，**不可接受** | ❌ 否决 |
| **S5 分仓**（verification 独立仓） | 拆出子仓 | 主仓瘦身 | 跨仓引用/CI 复杂度剧增 | ⚠️ 备选，非首选 |

**结论**：**filter-repo（§2）为唯一能真正清历史的主案**；S2/S3 作为「防再犯」配套在 §4 落地，不替代主案。

---

## 4. force-push 协调面

历史改写必然产生新根/新 SHA，**所有下游引用必须同步**：

1. **分支**：`main` 重写后，841 个 `agent/*` 分支中**已并入 main 的 149 个**其历史被重写（提交仍可达但 SHA 变）；**未并入的 692 个**若其 tip 含被删产物，需逐个评估——**建议只保留仍在飞/近 24h 的 agent 分支**，其余随重写一并清理（决策点 D2）。
2. **PR/合并**：平台自动合并（merge-master）依赖 `main` 的 SHA 锚点；重写后**旧 PR 的 base 失效**，需在重写窗口**冻结合并**（决策点 D3）。
3. **CI/流水线**：`origin/main` 的 `HEAD` 引用、`refs/remotes/origin/HEAD` 需重指；CI 缓存若按 commit SHA 键控会失效。
4. **下游克隆**：所有已 clone 的 worktree（如本单 `/root/tensor-ecosystem/TensorFVM/.git/worktrees/*`）在 force-push 后**必须重新 clone**，旧 worktree 的 `.git` 指向旧对象库，会与重写后对象库不一致——**这是最大协调面**（决策点 D4）。
5. **平台报告/审计**：`tensor/reports` 与 `feature_ledger` 若引用 commit SHA，重写后需回填「新 SHA ↔ 旧 SHA」映射表（建议在重写时输出 `git filter-repo --analyze` 的 mapping 存档）。

**协调面清单**（执行前必须逐项确认）：
- [ ] 冻结合并窗口（owner 批准）
- [ ] 通知所有在飞 agent 单（近 24h）重新 clone
- [ ] 重指 `origin/HEAD`
- [ ] 存档 SHA 映射表
- [ ] 更新 CI 缓存键

---

## 5. 回滚镜像保全

历史改写**不可逆**，必须留完整回滚镜像：

1. **裸镜像**：`git clone --mirror <origin> /srv/tfvm-pre-slim-mirror.git`（含全部 refs、全部 pack），**只读、不参与后续**。
2. **打包归档**：`tar czf /srv/tfvm-pre-slim-<date>.tar.gz` 该镜像，校验 `sha256sum` 并记录到 `docs/REPO-SLIM-PLAN.md` 附录。
3. **保留策略**：镜像保留 **≥90 天**（或直到重写后 3 个完整发布周期），期间任何「重写后数据丢失/误删」均可从镜像恢复。
4. **恢复路径**：若重写后发现问题，`git clone --mirror` 回镜像 → 重新 force-push → 通知下游重新 clone。**恢复成本 = 一次 force-push**，故镜像必须先行。
5. **审计**：镜像的 `git fsck --full` 通过后才允许执行重写（决策点 D5）。

---

## 6. 决策点（≥3，交 owner）

| # | 决策点 | 选项 | 建议 |
|---|---|---|---|
| **D1** | 删除范围：是否连 `json/csv`（含 `manifest.json` 产物清单）一起删？ | A 全删 / B 只删大扩展名 | **B**（保留 json/csv 清单，便于产物可追溯） |
| **D2** | 未并入 main 的 692 个 `agent/*` 分支处置 | 全保留 / 只留近 24h 在飞 / 全清 | **只留近 24h 在飞**（其余为历史死分支，随重写清理） |
| **D3** | 重写窗口是否冻结合并？ | 冻结 / 不冻结 | **冻结**（防 SHA 锚点失效） |
| **D4** | 下游 worktree 重新 clone 的协调 | 全量重 clone / 仅受影响 | **全量重 clone**（旧 worktree 指向旧对象库，不可复用） |
| **D5** | 回滚镜像保留期 | 90 天 / 永久 | **≥90 天**（或 3 个发布周期） |

**owner 批准后**：本方案进入执行阶段（另开执行单，本单仅出方案）。

---

## 7. 防再犯配套（执行阶段一并落地）

- 在 `.gitattributes` 声明 `docs/verification-*/**/*.npz` 等为 `-delta`（防增量膨胀）。
- 引入 `git-lfs` 或「产物外置 S3 + manifest」策略，**新验证产物不再入 git 历史**（见 §3 S2/S3）。
- 在 `docs/VERIFICATION_DEBT_LEDGER.md` 追加「产物入库纪律」条目：验证产物只留文本说明与 manifest，二进制走外置。

---

## 附录 A：执行后复核表（执行者回填）

| 指标 | 执行前 | 执行后 | 判定 |
|---|---|---|---|
| 全仓 pack | 1.33 GiB | ____ | 目标 <0.3 GiB |
| main 可达（解压） | 1672.6 MiB | ____ | 目标 ~250–300 MiB |
| src+tests | ~20 MiB | ____ | 必须不变 |
| `git fsck --full` | — | ____ | 必须通过 |
