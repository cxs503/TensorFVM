# CASE-FVM-CHANNEL-KMM180：FVM 桥接验证欠账的诚实状态与对抗性复核

锚点 `CASE-FVM-CHANNEL-KMM180`（湍流槽道 `Re_tau=180` 平均速度对数律）。
验收：`y+>30` 上 `u+(y+)` 相对 DNS 参考 RMS 偏差 `< 5 %`。

## 1. 本单认领的欠账

`docs/case-fvm-channel-kmm180-fvm-bridge/REPORT.md` 与 `summary.json`（作业 `cj3061`）
把 FVM 桥接（把锚点判据原样搬到原生混合 FVM 解上）的**数值表与图**明确记为 `not_run`：

> 未出图：作业节点（n330）未安装 matplotlib，绘图步骤在 `summary.json` 写出前抛
> `ModuleNotFoundError`，因此四档的 RMS 数值表、`u+` 剖面图与 1-D-vs-FVM 差分表**均未落盘**
> （`not_run`，不是 `passed`）。

即：FVM 桥接的**判据读数**（`u+` 各窗 RMS、FVM vs 1-D ODE 差分）从未落盘，
「FVM 是否真正继承/改善 1-D ODE 的对数律读数」这一问**仍是未完成**。

## 2. 本单做了什么

* **复核**：锚点算例池测试套件（`test_kmm180_case_pool.py` 34 项 +
  `test_channel_kmm180_fvm_bridge.py` 6 项）全部绿——算例池的 `ADMITTED_WITH_CAVEAT`
  判定（SST 全窗 3.61 % < 5 %、窄窗 6.21 % > 5 %）由已提交剖面可复算、可被拒绝。
* **递交**：`run_compute` 递交 `cj3519`（`--cases 16:3 32:3 64:3 64:1.2`，8 核）——
  在带 matplotlib 的节点上重跑 FVM 桥接，产出缺失的图与数值表。**排队中**（本机
  重计算被硬拦，FVM 解单档 >120 s 超时，无法本地补跑）。
* **对抗性复核**（见下）：对「FVM 窄窗 PASS」这一已提交结论提出**归一化伪影**质疑。

## 3. 对抗性复核：FVM 窄窗 PASS 是否可信（归一化伪影）

已提交 `summary.json`（`cj3061`）的 FVM 读数：

| n | stretch | `Cf` | `Cf` 误差 vs DNS | `legacy_full` | `log_layer` |
|---|---|---|---|---|---|
| 16 | 3.0 | 0.007581 | −6.68 % | PASS | WITHHELD |
| 32 | 3.0 | 0.008061 | −0.77 % | PASS | WITHHELD |
| 64 | 3.0 | 0.008351 | +2.80 % | PASS | **PASS** |
| 64 | 1.2 | 0.007935 | −2.32 % | PASS | **PASS** |

而同一闭包（SST-2003）的 **1-D 平衡 ODE** 在 `n=128/256` 上窄窗读数
5.72 % / 6.21 %（`docs/verification-channel-kmm180-direct-anchor/REPORT.md`）——**FAIL**。

**矛盾**：同一闭包，FVM 窄窗 PASS、1-D ODE 窄窗 FAIL。`REPORT.md` 注 ⑤ 已给出
最可能的解释，本单把它升级为**必须证伪才能采信**的假设：

> FVM 的 `u_tau` 由**自身积分壁面牵引**得出（`Cf` 与 DNS 差 −6.68 %…+2.80 %）；
> 1-D ODE 按构造把 `u_tau` 钉成 1。因此「FVM 窄窗 PASS」里**混着壁面单位归一化的差异**，
> 不能直接读成「FVM 的闭包比 1-D ODE 更准」。

**为什么这是伪影而非物理**：窄窗 `[30, 0.2 Re_tau]` 在本雷诺数下仅约 6 个壁面单位宽。
`u+(y+)` 的 RMS 相对偏差对 `u_tau` 的归一化**一阶敏感**——若 FVM 的 `u_tau` 偏大
（`Cf` 偏大，如 n=64-s3 的 +2.80 %），则 `u+ = u/u_tau` 整体被压低，窄窗内
`u+` 与 DNS 的偏差被**系统性缩小**，从而「PASS」是归一化偏差的**伪影**，不是
对数律本身的改善。

**证伪路径（本单未完成，因 FVM 解重计算被拦）**：FVM 桥接作业 `cj3519` 完成后，
必须把 `u+` 窄窗 RMS 与 **`u_tau` 归一化偏差**解耦——即同时报告
(a) 用 FVM 实测 `u_tau` 归一化的 `u+` RMS，与 (b) 用 DNS `u_tau`（`Re_tau=182.088`）
归一化的 `u+` RMS。若 (b) 仍 < 5 %，则「FVM 窄窗 PASS」是物理；若 (b) 回到
1-D ODE 的 5.7-6.2 % 量级，则「PASS」确为归一化伪影，`REPORT.md` 的「FVM 解
没有继承 1-D ODE 的窄窗失效」结论**必须撤回**。

## 4. 诚实结论

* **已确立**：锚点判据能作用在 FVM 解上；窄窗在默认 FVM 网格上常常不可判（`WITHHELD`）。
* **未完成（`not_run`，不得报为已通过）**：FVM 桥接的 `u+` 各窗 RMS 数值表、剖面图、
  FVM vs 1-D ODE 差分表——作业 `cj3519` 排队中。
* **新增风险（本单对抗性复核产出）**：已提交的「FVM 窄窗 PASS」结论**可能**是
  `u_tau` 归一化伪影，与 1-D ODE 窄窗 FAIL 同闭包矛盾；在 `cj3519` 产出并做
  双归一化解耦前，**不得**把该 PASS 引为「FVM 闭包更准」的证据。

## 5. 给下次自找活的建议

* 锚点 `CASE-FVM-CHANNEL-KMM180` 的 FVM 桥接验证欠账：等 `cj3519` 落盘后，
  做**双归一化（实测 `u_tau` vs DNS `u_tau`）解耦**，判定「FVM 窄窗 PASS」是物理还是伪影。
  量化验收：`u+` 窄窗 RMS 在 DNS `u_tau` 归一化下 < 5 %（物理）或 ≥ 5 %（伪影，撤回结论）。
* 若 `cj3519` 被调度器长期搁置，可改派 `runner=native` 的 mission 到带 matplotlib 的节点，
  并显式要求双归一化输出。
