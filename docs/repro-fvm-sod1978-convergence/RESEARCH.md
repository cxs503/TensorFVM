# RE PRO FVM SOD1978 —— 第 2 轮：调研 · 选型 · 迁移方案

锚点 `REPRO-FVM-SOD1978`（ref: Sod, G. A., *J. Comput. Phys.* **27** (1978) 1-31）
第 1 轮（m15142）复现结论：C1 波位置 4/4 在 2 % 内、C2 星区 4/4 在 5 % 内、
**C3（L1 收敛阶 ≥ 1）字面 FAIL**（二阶 0.977 / 0.894 / 0.857，一阶 0.617 / 0.651 / 0.660）。
同锚点 m15289 honest-FAIL（fail_stage/failure_reason 未留）→ 本单按两轮法先调研再实施，
要求把「为什么达不到」从"现象"变成"机制 + 可迁移方法"，而不是再交一次同样的表。

---

## ① 调研 · 开源横向（oss-cache 实读结果）

**台账查询**：任务书给的
`SELECT * FROM oss_inventory WHERE feature_slug ILIKE '%SOD1978%'`
在本平台 `tensorplatform` 库上**不可执行**——`oss_inventory` 没有 `feature_slug` 列
（实际列：`repo, tier, focus, status, task_key, report_id, lang, license, methods,
multiphase, boundary, gpu, summary, cache_nodes, cache_commit, cache_size_mb`）。
改用 `(methods||summary) ILIKE` 检索：全表 **33 行，全部是 LBM 仓**，
`compressible/Euler/shock` 命中 2 行均为 LBM 术语（palabos 的 Guo-compressible forcing、
pylbm 的 D1Q2… 格子），**没有任何 FVM/有限体积可压缩 Euler 实现**。

**缓存实读**（`/root/oss-cache`，65 MB，3 个仓）：
`pylbm`(474 files) / `hschwane`(116) / `Maarten-vd-Sande_lbm`(40)。
`grep -ril 'hllc|shock tube|euler.*riemann'` 只命中 pylbm 两个 demo。

| 仓 | 同类实现 | 结构可学什么 | 可迁移什么 |
|---|---|---|---|
| `pylbm/demo/1D/riemann_problems/euler.py` (211 行) | D1Q(2,2,2) 动理学格式解 1-D Euler 的 Sod 管，λ=3.0，松弛参数 (1.9,1.5,1.4)，x∈[0,1] | ① 间距 xmid 分段初值 + `exact_solvers.EulerSolver(...)` 注入；② `evaluate(x, t)` 返回 (rho,u,p) 剖面与数值解同网格点对比；③ 六个面板（mass/momentum/energy…）逐一对照 | **验证 oracle 的组织方式**：精确解与求解器完全解耦（同一个 `evaluate` 接口同时服务初始化和误差计算） |
| `pylbm/demo/1D/riemann_problems/exact_solvers/euler_solver.py` (259 行) + `riemann_solvers.py` (190 行) | Toro 型精确 Riemann 解：`_compute_interstate` 用 Newton/bisection (`solve`) 解 p\*，再按 1-波/接触/3-波分段填 rho,u,p | p\*-函数 `_f1/_f2` 返回 (u\*, du\*/dp) 供 Newton 用；波速表 + `values` 回调分段 | **结构对照**：TensorFVM 的 `exact_sample()`（`sod1978.py:264`）是同构实现（同样按 s=ξ 分段、同样用星区量），故 oracle 没有可借的精度增益 |
| 同上 demo 的**判据** | `grep -c error` = **0** | 只有图，没有误差范数、没有收敛阶、没有容差判据 | **无**——开源横向给不出 SOD 的验收协议，只能给 oracle 与画图组织 |

**结论（横向）**：SOD1978 这一类"标准算例复现"在开源侧普遍止步于"画图对得上"，
没有可抄的量化判据；本锚点的判据只能出自文献的收敛性分析（下一节）。
这条结论本身就是负知识：**下次同类锚点不要再指望从 oss-cache 借判据**。

---

## ② 调研 · 方法谱系（文献/教科书标准做法 + 标准算例）

| # | 出处 | 内容 | 对本锚点的作用 |
|---|---|---|---|
| L1 | Sod, *JCP* 27 (1978) 1-31 | 原始算例：(ρ,u,p)_L=(1,0,1)，(ρ,u,p)_R=(0.125,0,0.1)，γ=1.4，膜片 x=0.5；t=0.2 剖面图与波速 | 判据 C1/C2 的 ref 值来源（波位置 x₀+St、星区 ρ\*ₗ/ρ\*ᵣ/u\*/p\*） |
| L2 | Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. (2009) | ch.4 精确 Riemann 解（p\* 迭代 + 五段波结构）；ch.10 HLLC；ch.13–14 TVD 限制器与 MUSCL-Hancock；ch.14.4 收敛性讨论 | 实现侧（精确解、HLLC、MUSCL-Hancock）与"二阶格式在间断处退化为 ≤1 阶"的标准表述 |
| L3 | LeVeque, *Finite Volume Methods for Hyperbolic Problems* (2002), ch.8 | 实测阶 vs 设计阶；L1 范数定义；**在含间断的解上不得用整体误差阶去要求设计阶**，应分别报告光滑区阶与间断处理 | 判据 C3 的正确读法（第 2 轮协议的制度依据） |
| L4 | Banks, Aslam & Rider, *JCP* **227** (2008) 6985-7002, "On sub-linear convergence for linearly degenerate waves in capturing schemes" | 线性退化波（接触间断）在捕捉格式下**次线性收敛**（扰动被线性退化场放大，L1 阶 < 1，典型 ~0.5）；真正非线性波（激波）保持 ≥1 阶 | **本锚点 C3 FAIL 的机制假说**：Sod 的接触波是线性退化 → 整体 L1 阶被它钉在 1 以下 |
| L5 | Engquist & Sjögreen, *SIAM J. Numer. Anal.* **35** (1998) 2464-2482 | 含激波时差分格式的收敛率上界（一般 < 设计阶，L1 常见 0.5–1） | 佐证"整体 L1 阶 < 1 不是实现缺陷" |
| L6 | Jiang & Shu, *JCP* **126** (1996) 202-228 | WENO5：高阶格式在光滑区的阶与 Sod 类算例的 L1 误差表 | 证明"提高设计阶不能消除间断主导项"的对照路径；本轮未采用（改造面过大，见 ③） |
| L7 | Woodward & Colella, *JCP* **54** (1984) 115-173 | 激波管分辨率研究（接触面/激波涂抹宽度随格式与网格的行为） | 解释"涂抹宽度按**单元数**不随 h 缩小 → 误差 ∝ h（一阶）"这一定标 |

**标准算例（本锚点采用，与 L1 一致）**：1-D，x∈[0,1]，膜片 x₀=0.5，γ=1.4，t=0.14（判据时刻；
t=0.2 用于与原文剖面图目视对照）。左 (1,0,1)、右 (0.125,0,0.1)，两端 outflow。
无量纲数：γ=1.4（量热完全气体）、CFL=0.4（主算例）、激波马赫数 Ma_shock = S/c_R = 1.6556（精确解波速，程序算出）、p\*/p_L = 0.30313、ρ\*ₗ/ρ_L = 0.42632、ρ\*ᵣ/ρ_R = 2.12459（星区量由 `exact_star_state()` 计算，非手抄）。

---

## ③ 选型沉淀（方法选型 + 迁移方案）

### 候选与取舍

| 方案 | 能否让"整体 L1 阶 ≥ 1"成立 | 判断 |
|---|---|---|
| A. 换更高阶重构（WENO5 / MP5） | 不能消除 L4 的线性退化波次线性项；只压光滑区误差，改造面 = 重构层 + 限制器 + 测试面全改 | 不采用（成本高、不解决根因） |
| B. 加密网格直到整体阶"看起来" ≥1 | 若指数确为 <1 则永不成立；若为预渐近则可证伪——**必须先测** | 采用为**测量项**，不作判据 |
| C. 判定判据本身不适配，改为文献标准的三段式验证协议 | 是文献（L3/L4）的标准做法：设计阶在光滑解上认证；间断解上报告实测阶 + 机制归因（波位置收敛率指纹：接触 vs 激波） | **采用（主选型）** |
| D. 低耗散限制器（superbee / van Leer）替代 MC | 改变耗散系数，不改指数；可作敏感性对照 | 采用为**对照项** |

### 选型结论（落 `oss_solutions`, feature_slug='REPRO-FVM-SOD1978'）

**第 2 轮协议 = "三种误差范数 + 波位置收敛率指纹 + 光滑解设计阶认证"**：

1. `full`：整体 L1（第 1 轮判据原样保留，**不改判据、不调容差**）——继续如实报告；
2. `cellavg`：以精确**单元平均**（16 点中点法）替代格心点采样，排除采样伪影；
3. `restricted` / `plateaus`：仅在精确解**解析区**（左右未扰动区 + 两个星区平台 + 风扇内部，
   每道波内缩 min(16h, 25 % 窗宽)）计 L1 ——这才是"格式阶"的干净量；
4. `positions`：四道波位置误差对 h 的收敛率。**接触波（线性退化）率 < 激波（真非线性）率**
   ⇒ L4 机制在本次数据上复现；若两者都 ≥1 则说明整体阶 <1 另有原因（需另立项）；
5. 设计阶认证：同求解器跑周期熵波（光滑精确解），实测阶须在声明带内（1.0 / 2.0，±3 %）。

**判据判级（第 2 轮，预先声明、不做事后调整）**：
- `C3_literal` = 整体 L1 最细相邻对（3200→6400）阶 ≥ 1.0 ——判据原义，仍然单独打勾/叉；
- `C3_asymptotic` = 最细 3 个网格上 log-log 最小二乘指数 ≥ 1.0；
- `R3` = 解析区 L1 阶 ≥ 设计阶 ×(1−15 %)；
- `R5` = 光滑熵波实测阶在设计带内；
- `mechanism` = 接触波位置指数 < 激波位置指数（Banks 指纹）。

若 `C3_literal` 与 `C3_asymptotic` 同时为假而 R3/R5 为真、mechanism 为真，
则本锚点的验收结论应写成"**判据不适配（L4 机制）+ 格式经认证为设计阶**"，
并把 C3 的判据改写建议（改为"解析区阶 + 光滑解设计阶"双轨）提交台账——
**不改判据本身，只提交证据与建议**（判据更改权在台账 owner，不在执行方）。

### 迁移方案（落到本仓）

| 层 | 文件 | 内容 |
|---|---|---|
| 方法层 | `src/tensorfvm/verification/sod1978_convergence.py`（新） | `measured`（单网格全量指标）→ `successive_orders` / `fitted_exponent` → `run_study`（全扫描 + 判级 + 图） |
| 格式层 | `src/tensorfvm/compressible/reconstruct.py` | 新增 `superbee` 限制器（Roe 1986，TVD 族最低耗散），纳入 `LIMITERS` 注册表 |
| 入口层 | `scripts/sod1978_convergence_study.py`（新） | `--quick` 冒烟 / 全扫描；输出 `<out>/study.json` + 3 张图 |
| 验证层 | `tests/test_sod1978_convergence.py`（新） | 限制器性质（TVD/对称/单调性）、窗口非空与内缩性质、阶计算器正确性、快扫可复现 |
| 台账层 | `tensor_benchmark_cases`（REPRO-FVM-SOD1978 行）+ `negative_knowledge_ledger`（机制条目） | 补审计（m15142 审计第 5.1 条）指出的"基准台账缺本锚点行"缺口；机制入负知识库 |

**为什么这不是"改判据通关"**：判据字面值、容差、参考量一律未动（`THRESHOLDS` 在扫描前
写死在模块顶部并写进工件 `thresholds_declared_before_run`）；新增的全部是在同一批数字上
**多测了几个量**，并且把新的判级与旧的字面判级**并列输出**——旧的叉依然是叉。