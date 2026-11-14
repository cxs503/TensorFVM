# CASE-WOODWARD-COLELLA — 第 2 轮做法的独立复核 + certified-<1% 基准提案（XVAL-FVM-M18737-ROUND2）

**锚点**：`XVAL-FVM-M18737-ROUND2`（「TensorFVM 的 m18737-round2 做法」→ 交叉验证该做法）
**本单性质**：adversarial-review 席 —— 对已合并的 round-2 交叉验证交付物做**独立复核**，
并沿其「给下一手」建议推进 **certified-<1% 基准**（换基准路径）。
**上游**：m18737 = `CASE-WOODWARD-COLELLA` 第 2 轮（已合并，`docs/CASE-WOODWARD-COLELLA-round2-xval.md`，
13 例回归门全绿）。
**验收（总账）**：`【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】`
**仓/改动面**：TensorFVM，只动 `docs/`、`tests/`（**求解器一行未改**）。

---

## ① 独立复核结论（adversarial review，对已合并 round-2 交付物）

对 `docs/CASE-WOODWARD-COLELLA-round2-xval.md` 的**全部关键声明**做独立复算（纯 numpy，
不调用求解器，只读仓内已存工件），逐条核对：

| 声明 | 独立复算值 | 与文档一致？ |
|---|---|---|
| 自收敛（参考解无关）单调，最细档 <2% | [8.078, 5.330, 3.054, 1.765]%，最细 1.765% | ✅ |
| 观测阶 0.60→0.80→0.79 | [0.60, 0.80, 0.79] | ✅ |
| 峰值外推（800/1600/3200，Richardson）阶 2.74，连续极限 6.5148 | 阶 2.736，ρ_max∞=6.5148 | ✅ |
| 连续极限 vs 标定参考峰值（6.3181）= **+3.11%**（参考侧误差） | +3.11% | ✅ |
| **「nx≥800 全 <3%」在 nx=3200 被证伪**（峰值 3.00%、L1 3.08%） | 峰值 2.998%、L1 3.079% | ✅（诚实保留） |
| 总账 <10% 判据 | 全部良态指标 <10% | ✅ |

**裁决**：round-2 交叉验证交付物**无实质缺陷**——声明未被夸大（「全 <3%」的过强声明被显式
证伪并如实记录），误差预算、相位/形状、分辨力、交叉码等度量全部自洽。13 例回归门
（`tests/test_compressible_wc_xval.py`）全绿（4.49s）。

## ② 本单实质增量：certified-<1% 基准（换基准路径）

round-2 的诚实局限是：**「与已发表数值解一致到 3%（参考侧误差主导）」**——参考解（RPJ@3200，
经 PDF 数字化）自身峰值被其离散耗散压低 ~3%，所以任何「<x% vs 参考」都混入参考侧误差。
round-2 明确给出唯一能 certified-<1% 的路径：

> 「真正能 certified <1% 的路径是换基准——用 WENO5@6400 或本仓 nx=12800 的自解作参考」

本单**落实该路径**：用独立格式（WENO5/SSP-RK3/Rusanov，与 TensorFVM 不共享离散）在更细网格
（6400、12800）上作**新高分辨率参考**，再对 TensorFVM MUSCL/HLLC 解重新定级。三道门保证
「certified」一词诚实：

1. **参考自认证**：WENO5@6400 vs WENO5@12800 须 <1%（L1+峰值），否则不宣称 <1%。
2. **跨族地板**：MUSCL/HLLC vs WENO5@6400 残差须**小于** vs 已发表参考残差——即更细独立参考
   确实移除了 round-2 预算归给 RPJ@3200 数字化的参考侧误差。
3. **守恒**：窗质量 ∫ρdx 在 [0.4,1] 上须 <0.1%（格式无关的良态检查）。

**工具链**（本单新增，已落盘）：
- `docs/.../xval/wc_certified_ref.py` —— 三道门 + 定级裁决（纯 numpy，不调求解器）
- `docs/.../xval/wc_certified_figures.py` —— 对比图（MUSCL vs certified 参考 + 参考自收敛）
- `tests/test_compressible_wc_certified_ref.py` —— 5 例回归门

**计算递交**：`run_compute` 递交 `wc_ladder_weno5.py --sizes 6400,12800`（**cj3031**，rc=0，
节点 n330 成功）。**但**：该作业的工件（`ladder_profile_nx{6400,12800}_weno5.npz`）**未回填到
本工作树**——`run_compute fetch` 返回的是提交前 repo 快照，不含作业输出。这是**工件回填缺口**。

## ③ 诚实状态（honest FAIL 不覆盖）

- **已达成**：round-2 交付物独立复核通过（无缺陷）；certified-<1% 的**工具链与三道门**已就位；
  更细 WENO5 计算**已在节点成功**（cj3031 rc=0）。
- **未达成（本单缺口）**：certified-<1% 的**最终数值裁决**——因 WENO5@6400/12800 工件未回填，
  无法在本工作树完成「MUSCL/HLLC nx=3200 vs WENO5@12800 <1%」的逐项判定。
- **用现有数据可先验的部分**：round-2 交叉码已给出 **MUSCL vs WENO5@3200 = 0.972% L1（<1%）**、
  峰值互差 **1.046%（>1%）**——即 **L1 的 <1% 已可用 WENO5@3200 达成**，但**峰值 <1% 需更细参考**
  （这正是 6400/12800 档的用途）。此结论与「参考侧误差 ~3%」的 round-2 归因**自洽**。

## ④ 给下一手的建议（新账条目）

**建议新条目**（slug+量化验收+所属域）：
- `slug`: `XVAL-FVM-M18737-CERTIFIED-REF`
- `title`: 「TensorFVM 的 m18737 换基准：WENO5@6400/12800 独立参考下 certified-<1%」
- `acceptance`: 「MUSCL/HLLC nx=3200 vs 独立 WENO5@12800 参考，L1 与峰值均 <1%（<1% 硬判据）；
  且 WENO5@6400 vs 12800 自收敛 <1%（参考自认证）」——**量化验收**，非 <10% 软判据。
- `所属域`: TensorFVM（compressible / 算例池）
- **前置**：先解决 `run_compute fetch` 的**工件回填缺口**（本单 cj3031 输出未回填），
  或改用 `stage_dir` 直写 + 显式回传，否则该条目无法在本工作树完成数值裁决。

**复现命令**（工件回填后）：
```bash
PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_ref.py
PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_figures.py --outdir artifacts
PYTHONPATH=src python -m pytest tests/test_compressible_wc_certified_ref.py -q
```
