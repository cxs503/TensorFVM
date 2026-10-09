# TensorLBM — 极坐标 O 网格 + 通用曲线坐标 C 网格不可压 NS 求解器（圆柱绕流 / NACA 翼型）

基于 PyTorch 思路重写的**极坐标贴体 O 网格**不可压 Navier–Stokes 求解器，用于 Re=100 圆柱绕流：
拿到**物理正确**的阻力系数，并尝试得到**持续卡门涡街**（von Kármán vortex street）。

> 本目录为可下载代码包（code-only）。所有算例数据（npz/png/log）在 `/workspace/results/`。

## 当前状态（2026-09-26，终版诊断）

| 指标 | 目标 | 实测 | 说明 |
|------|------|------|------|
| 阻力 Cd | ~1.3（含阻塞修正） | ~1.18（无阻塞） | ✅ 量级正确；粘性/压力分解仍有小偏差 |
| 流场 | 无 NaN / 无棋盘 | ✅ | `_lap_face_matrix` 转置 bug 已修（稀疏=稠密 diff=0） |
| 涡街 St | ~0.16–0.18 | **未稳定获得** | 见下：PPM 在现有网格上无法稳定解析非定常尾流 |

### 涡街诊断（重要，已彻底排查）

在 Re=100、现有分辨率（nj≈160–200）下，PPM 格式**无法稳定产生持续卡门涡街**。排查覆盖：

| 尝试 | 结果 |
|------|------|
| PPM，瞬态扰动 amp≤0.02（β=1.6/2.0，nj=160/200） | ✅ 数值稳定，但**衰减回对称定常**（亚临界 Hopf，有效 Re 被数值耗散压在临界 ~47 之下） |
| PPM，amp≥0.025（含 nj=200、Re=250） | ❌ 在 step 3100–4700 **数值发散（overflow→NaN）**：wake 一旦开始非定常脱落，剪切层小尺度被 PPM+Rusanov 放大失稳 |
| PPM，降 Rusanov 耗散 `--weno_alpha 0.3` | ❌ 即使 amp=0.02 也在 step 1038 爆（减耗散即减稳定） |
| central + 四阶超粘性 `--nu_hyp` | ❌ step 222 爆（central 对流本身在此实现未稳健稳定） |
| Rhie–Chow 动量插值 `--rc 1` | ❌ step 22 爆（该实现在当前网格/投影组合下损坏） |
| 抬升物理 Re（--Re 250） | ❌ 活得更久（step 4672）但仍爆 |

**结论**：这是**方案/分辨率硬限制**，不是单个 bug。PPM 数值耗散把有效 Re 压在亚临界之下（稳态可稳），而任何把流场推入非定常的机制（强扰动 / 高 Re）都会触发 PPM 在剪切层小尺度上的绝对不稳定 → 发散。稳态 Re=100 结果已验证可靠；非定常涡街需要下述改进之一。

### 拿到涡街的下一步（按性价比排序，均已实测评估）

1. **更快的 Poisson 求解器（首要瓶颈）**：当前用 SuperLU 直接分解，仅在 N≤40000（即 nj≤200）实用；nj≥300（N≥72000）分解/迭代过慢（实测 26 min 仍在 setup、无一步输出）。换成 **多重网格 / FFT 泊松**后可负担 nj≥300–400，把有效 Re 明确推过亚临界 → 超临界自持涡街（最有希望的路径）。
2. **提高物理雷诺数**：`--Re 200–300` 可让有效 Re 越过临界、自然脱落（已验证 Re=250 能激发非定常，只是仍受 PPM 剪切层不稳定影响）；代价是偏离 Re=100 目标。
3. **换低耗散且稳定的对流格式**：带熵修正的 WENO / 有界中心格式（当前 WENO5 在拉伸 O 网格色散失稳、central 未稳定化、Rhie–Chow `rc=1` 实现损坏）。

## 核心文件

| 文件 | 角色 |
|------|------|
| `polar_ogrid.py` | **核心算子库**：极坐标 O 网格生成、对流（upwind/MUSCL/vanLeer/central/WENO5/PPM）、扩散、Chorin 投影、CBC 谱投影（cb=pair 去 (-1)^j 棋盘）、Poisson 稀疏直接求解（splu） |
| `run_ogrid_primary.py` | **主驱动（投影法）**：圆柱绕流、力/涡量统计、瞬态扰动、结果输出（png + `*_hist.npz`）。新增 `--Re`（物理雷诺数，诊断用）、`--nu_hyp` 四阶超粘性 dt 约束 |
| `run_simple.py` | **SIMPLE 求解器驱动**：与投影法共享底层算子，隐式动量方程 + 压力修正外迭代，用于交叉验证与 3D 铺垫 |
| `polar_ogrid_staggered.py` | 交错 MAC 版（已知对流反耗散不稳定，未用于涡街） |
| `polar_v2.py` | 早期原型 |
| `_forceplot.py` | 读 `*_hist.npz` 画 Cl/Cd 时程 + FFT 求 St |
| `_weno_diag.py` | 区域级失稳定位 |
| `lid_driven_cavity_fvm.py` / `overset_cylinder_fvm.py` | 验证用 FVM 算例（方腔 / 重叠网格） |
| `diag_*.py` / `probe*.py` | 历史诊断与探针（可忽略，保留备查） |

## 运行（稳态 / 出涡街尝试）

```bash
cd /workspace/tensorlbm
# 稳态 Re=100（已验证可靠，Cd≈1.18，无 NaN）
python3 run_ogrid_primary.py \
    --nsteps 16000 --nj 160 --ni 200 --beta 1.6 --aoa 0 \
    --scheme ppm --rc 0 --fck 0 --fp 0 --cb pair \
    --Rf_og 1.4 --Rf_sponge 1.2 \
    --out steady_re100

# 尝试涡街：需更快的 Poisson 求解器以负担 nj≥300（见"下一步"）
# 当前 N≤40000（nj≤200）可跑；nj≥300 因 SuperLU 过慢不可行
python3 run_ogrid_primary.py \
    --nsteps 40000 --nj 200 --ni 200 --beta 1.6 --aoa 0 \
    --scheme ppm --rc 0 --fck 0 --fp 0 --cb pair \
    --Rf_og 1.4 --Rf_sponge 1.2 \
    --pert_amp 0.02 --pert_k0 200 --pert_k1 4000 \
    --out shed_long200        # 实测：瞬态脱落，t>15.6 衰减回定常 (Cl->0, Cd->1.167)
```

参数要点：
- `--scheme ppm`：低耗散对流（TVD/vanLeer 耗散过大，冻结在定常）。
- `--beta 1.6`：近壁拉伸比，越小近壁单元越方、耗散越小（但 <1.4 在强扰动下易爆）。
- `--pert_amp/--pert_k0/--pert_k1`：瞬态扰动（step k0→k1），窄径向包络 + 缓入缓出斜坡，避免注入网格尺度能量。amp 必须 ≤0.02，否则 wake 非定常化后 PPM 发散。
- `--fck 0 --fp 0`：关掉旧默认过度耗散滤波器（否则 Cd 虚高到 2.56）。
- `--cb pair`：CBC 配对投影，消除 (-1)^j 棋盘零空间。
- `--Re`：物理雷诺数（默认 100）；诊断用，抬到 ~250 可验证"耗散压低有效 Re"的判断，但非定常化后仍会发散。

## 后处理

```bash
python3 _forceplot.py steady_re100_hist.npz      # Cl/Cd 时程 + 频谱
```

## 已知问题

1. **非定常涡街不稳定**：PPM 在现有网格上无法稳定解析脱落尾流（见上诊断）。需高分辨率或换格式。
2. **力分解偏差**：`polar_grad` 在 θ=90° 丢法向梯度，压力阻力偏低、粘性阻力偏高（总 Cd 量级仍对）。
3. **St 测量**：仅当流场持续非定常且 t 覆盖 ≥3–4 周期（nsteps≥30000）才能得到可靠 St。

## 第二算法：SIMPLE 版（2026-09-27 新增）

新增 **SIMPLE（Semi-Implicit Method for Pressure-Linked Equations）** 求解器 `run_simple.py`，与投影法共享同一套底层算子（`make_ogrid` / `polar_conv` / `polar_grad` / `polar_div` / `_lap_face_matrix` / `build_poisson` / `cylinder_forces`），所以两版阻力可直接对比，且为后续升 3D 铺路（耦合逻辑/棋盘处理/求解器接口原样照搬）。

### 算法（每时间步）

1. **隐式动量方程** `(I/dt − ν·L) u* = uⁿ/dt − conv(uⁿ) − grad(pⁿ)`，`splu` 求解（`L` = 物理 Laplace，`A` 对称正定）。
2. **压力修正方程** `L p' = div(u*)/dt`，源项是 `u*` 的（Rhie–Chow）质量通量散度；`p ← p + urf_p·p'`。
3. **速度修正** `u = u* − urf_u·dt·grad(p')`；每步重复 6 次**外迭代**（outer iteration）。
4. 末尾 `cb=pair` 谱投影消除 collocated 的 `(-1)^j` 棋盘零空间。

### 与投影法的关键区别

| 维度 | 投影法（run_ogrid_primary） | SIMPLE（run_simple） |
|------|------|------|
| 动量方程 | 显式推进 u* | **隐式** `(I/dt − νL)` 求解 |
| 压力处理 | 每步一次投影 | 每步 **6 次压力修正迭代 + 欠松弛** |
| dt 约束 | `dt_diff=0.25·dx²/ν` 硬约束（cfl=0.2） | 隐式扩散解除 dt 硬约束；**但 cfl=0.5 在非定常下会激发高频数值升力抖动，非定常/涡街探测须 cfl≤0.25–0.3**（见下） |
| 稳态 Cd（细网格） | nj=200 → 1.167 | nj=96 → **1.176**（一致 ✅） |

> SIMPLE 的核心价值是**隐式大 CFL + 压力-速度强耦合鲁棒 + 与投影法交叉验证**，不是解决涡街——Re=100 亚临界是物理/分辨率瓶颈，与耦合算法无关。

### 实现要点（踩过的坑）

- **`rc=True`（Rhie–Chow）会让 SIMPLE 外迭代发散**：Rhie–Chow 把 `pⁿ` 梯度塞进通量源项，而速度修正用 cell-centered `grad(p')`，两者**非自伴** → 修正后散度 ≠ −L·p' → 正反馈爆炸（实测 it1 起 div 9.7→105、p'→10⁶）。**必须用 `rc=False`**（plain 自伴 `polar_div`/`polar_grad`，与投影法同套），此时 `u=u*−dt·grad(p')` 精确消去散度。
- `rc=False` 的 plain Laplace 带 `(-1)^j` 棋盘零空间，用 **`reg_r=200`** 钉死（投影法迭代版也这么干，物理压力不受影响）。
- **`urf_u` 必须 = 1.0**：自伴算子下只有 urf_u=1 才使速度修正精确无散度；urf_u<1 留残散度，下一轮叠加发散。`urf_p=0.7` 压力松弛防过冲。

### 运行

```bash
cd /workspace/tensorlbm
# 稳态 Re=100 交叉验证（SIMPLE，Cd≈1.176，与投影法细网格一致）
python3 run_simple.py \
    --nsteps 6000 --nj 96 --ni 160 --beta 2.0 --aoa 3 \
    --scheme ppm --rc 0 --niters 6 --urf_p 0.7 --urf_u 1.0 --cfl 0.2 \
    --out simple_steady96

# 涡街探测（同投影法物理：Re=100 亚临界，瞬态脱落 + 衰减回定常）
python3 run_simple.py \
    --nsteps 10000 --nj 120 --ni 200 --beta 1.6 --aoa 3 \
    --scheme ppm --rc 0 --niters 6 --urf_p 0.7 --urf_u 1.0 --cfl 0.2 \
    --pert_amp 0.02 --pert_k0 200 --pert_k1 4000 \
    --out simple_shed120
```

### 稳态交叉验证结果

| 求解器 | 网格 | Cd | Cl（稳态） | 符号 |
|--------|------|-----|------|------|
| SIMPLE | nj=48 | 1.090 | ~0.05 | +（正） |
| SIMPLE | nj=96 | **1.176** | ~0.05 | +（正） |
| 投影法 | nj=200 | 1.167 | ~0 | +（正，关掉 fck/fp 后） |

SIMPLE 随分辨率收敛到 ~1.17，与投影法细网格一致 → 算法正确、阻力可靠、无 NaN、无棋盘。

### 涡街

Re=100 在 SIMPLE 下同样**亚临界**（底层 PPM 物理与投影法完全相同）：瞬态扰动能踢出脱落，但有效 Re 被数值耗散压在临界之下，长演化衰减回对称定常。这与投影法的结论一致，进一步印证"亚临界是物理/分辨率瓶颈，非耦合算法问题"。持续涡街仍需更快的 Poisson 求解器（多重网格/FFT）以负担 nj≥300。

#### ⚠️ CFL=0.5 是数值伪像来源（非定常下不可用）

首次涡街探测 `simple_shed120`（nj=120 / ni=200 / aoa=3 / PPM / `pert_amp=0.02` / **`--cfl 0.5`**）出现 Cl 持续 ±0.45 振荡、FFT 主峰 St=1.23——这**不是**物理卡门涡街（物理 St≈0.18）。诊断（详见 `results/shed_compare.png`）：

- Cl 高频分量（2 时间单位滑动平均分离）RMS 在所有窗口恒为 **0.27**（有界、不增长 → cfl=0.5 仍"稳定"但含数值抖动）；低频分量 RMS≈0.005（几乎为零）。
- 粗采样看到的"慢变正弦"是 **混叠假象**：f≈6.16 的真实纹波被每 544 步采样混叠成 ~10 时间单位的假周期。
- **受控对照**（`--cfl 0.2`，其余全同，niters 仍为 6）Cl 收敛到稳态 +0.061、`Cl_amp=0.0051`、`fastRMS=0.0094`——抖动消失，且与投影法同配置（稳态 Cl≈+0.094、Cd≈1.79、无脱落）一致。

| 运行 | Cd | Cl | Cl_amp | 高频 RMS | 判读 |
|------|-----|-----|--------|----------|------|
| 投影法 nj=120 aoa=3（参考） | 1.793 | +0.094 | 0.090（瞬态） | 0.039 | 稳态、亚临界 ✅ |
| SIMPLE **cfl=0.2** | 1.231 | +0.061 | 0.005 | 0.009 | 稳态、亚临界 ✅ |
| SIMPLE **cfl=0.5** | 1.314 | ±0.45 | 0.449 | **0.271** | 高频数值纹波 ❌ |

**结论**：CFL=0.5 的抖动来自对流显式步（隐式扩散只解扩散约束，对流仍显式），被 SIMPLE 外迭代残差放大成持续极限环——**与 `niters`（同用 6）无关，是 CFL 问题**。`cfl=0.2` 下 SIMPLE 干净复现投影法的亚临界稳态。

> 注意：无扰动稳态验证（见上表 nj=48/96）用 cfl=0.5 没问题——抖动需**扰动/非定常**才被激发。但为安全，**涉及扰动或瞬态的统一用 `cfl≤0.25`**；本文档运行示例已改为 `--cfl 0.2`。

#### 阻力量级：哪个更准？（已定论 → SIMPLE 更准）

同网格（nj=120/ni=200/aoa=3/PPM）下，投影法 Cd≈1.79、SIMPLE cfl=0.2 Cd≈1.23。判断依据：**Re=100 圆柱阻力的文献基准 Cd≈1.33–1.42（集中 ~1.35–1.40），且阻力是迎角的偶函数**（3° 下 Cd 应≈零迎角值，仅新增 O(α) 升力 Cl≈Cd·sin3°≈0.07）[Tritton 1.32 / Braza 1.36–1.40 / Meneghini 1.37 / Harichandan 1.35 / Wiesenberger 1.33，St≈0.16–0.17]。

| 求解器 | 实测 Cd | 对基准偏差 | AoA 0→3° 变化 | 判读 |
|--------|---------|-----------|---------------|------|
| SIMPLE cfl=0.2 | 1.23 | 低 ~10%（网格偏保守，合理） | 1.17→1.23（≈持平） | 行为正确 ✅ |
| 投影法 aoa=3 | 1.79 | 高 ~33% | 1.17→1.79（**+53%**） | 违反偶函数对称，物理不可能 ❌ |

**=> SIMPLE 的阻力更准**：1.23 距基准仅 ~10% 且迎角响应正确；投影法 aoa=3 的 1.79 违物理对称，是其**迎角下 clamped 远场 + PPM 触发虚假提前分离/加宽尾流**抬升压力阻的伪像。两版 aoa=0 都 ~1.17（同偏低 ~13%）→ 误差属分辨率/扩散，非算法。当前取 **SIMPLE ~1.23 为最佳估计**；投影法 aoa=3 的 1.79 不可信；要逼近基准 ~1.35 需加密 nj≥200。不影响"亚临界、无涡街"结论。



## 第三套网格：通用曲线坐标 C 网格（圆柱 / NACA 翼型）

在 O 网格基础上新增**通用曲线坐标（generic curvilinear）FVM 算子层** `curve_grid.py`，把 O 网格的极坐标特化推广为任意结构化贴体网格，并用它驱动 **C 型网格**——C 网格沿物面闭合一圈（i 方向），但在下游留下一个**开口尾迹切口**（i 非周期），因此能显式解析尾流，而非像 O 网格那样把尾流折回近场。同一套 `c_div`/`c_grad`/`c_lap`/`c_forces`/`build_poisson` 同时服务 O 与 C 网格，度量一致性由 `_test_curve.py` 回归保证。

### 拓扑（C 网格 vs O 网格）

| 边界 | C 网格含义 | 处理 |
|------|------|------|
| i 方向 | **非周期**，下游留开口尾迹切口 | 切口两列 clamp 自由流 |
| j=0 | 物面（无滑移 u=v=0） | 圆柱：整圈圆；翼型：仅翼型弧段（含上下 TE 折回点），尾迹中心线不算墙 |
| j=nj-1 | 远场 | clamp 自由流 |

远场 + 切口两列 `recv` 每步钳自由流，使 C 网格成为和 O 网格外 fringe 一样的闭合问题，无需背景网格。

### 核心文件

| 文件 | 角色 |
|------|------|
| `curve_grid.py` | **通用曲线坐标算子层**：由节点坐标 X,Y 数值构造面矢量、单元体积 J、逐面距离 h_f；提供 `c_div`/`c_grad`/`c_lap`/`c_forces`/`build_poisson`/`project`。`periodic_i=False` 即 C 网格模式 |
| `cgrid_gen.py` | C 网格代数生成器：圆柱整圆 C 网格、`make_naca_cgrid`（NACA 4 位 0012 对称 / 2412 有弯度），尾迹用几何增长 `_wake_stretch` 避免 TE 交界处大长宽比 |
| `run_cgrid.py` | C 网格 Chorin 投影法驱动（圆柱 / 翼型，扫迎角，`c_forces` 旋转到来流系得 Cd/Cl） |
| `_test_curve.py` | 度量一致性回归：O/C 网格 `div(uniform)=0`、J-加权对称性、Laplacian、对称差 |
| `_diag_poisson2.py` | 泊松矩阵一致化诊断（`c_lap` vs `c_div∘c_grad`） |

### 关键修复：压力泊松矩阵必须与投影"一致"

初版 `build_poisson` 用 `c_lap`（直接面通量 Laplacian = `M⁻¹·K`，`M=diag(J)`）组装矩阵——它只在 **J 加权内积**下对称，在欧氏内积下**非对称且被 J 强烈缩放**；而投影修正用的是 `c_div`/`c_grad`。在翼型网格上 `J` 跨 `6e-5 … 9.8`（**1.6×10⁵** 倍），SuperLU 解该系统既病态又和修正算子**不一致** → 残差散度累积、压力爆炸（`pmax ~1e70`，起于远场/尾迹区）。圆柱因 `J` 跨度仅 ~290，`c_lap ≈ c_div∘c_grad` 才"侥幸"稳定。

修复：新增 `_adj_lap_matrix`，用投影真正使用的算子 **`c_div ∘ c_grad`** 组装泊松矩阵（已替换 `build_poisson`/`build_poisson_iterative`）。投影按构造严格无散：

> `c_div(u_new) = c_div(u*) − dt·(c_div∘c_grad)·p = 0`

任何网格上稳定。纯投影对照：旧矩阵翼型 `pmax→1e70`，新矩阵 `pmax` 衰减到 ~0.2。

### 关于压力场"一块一块"（collocated 网格高频压力，非收敛失败）

初版压力云图在**尾迹 / 远场**出现沿流向的红蓝交替条带（"一块一块"）。诊断（`_diag_pressure_patch.py` 真实格点散射 vs `griddata` 插值）逐项排除：

- **不是渲染伪影**：真实格点散射图（无任何插值）本身就带条带，`griddata(linear)` 只是把它放大成马赛克。
- **速度场完全光滑**（无棋盘）：投影产生的速度严格无散、物理正确；力（Cd/Cl）也正确。
- **已用代码验证：这不是 collocated 压力泊松的"零空间"**。对翼型 C 网格直接算 `c_div(c_grad(φ))`：
  - `φ=(−1)^i` → `|L·φ|max ≈ 1.34e2`（≠0）；`c_grad((−1)^i)` 的 `|gx|max≈16.5`、`|gy|max≈21.6`（梯度**不**失明）。
  - 同理 `(−1)^j`、`(−1)^(i+j)` 也都不是零空间。

> 结论：这些高频模式**携带真实的巨大压力梯度**，是**真实物理内容**，不是"惰性零空间"。它们之所以显形，是因为 **collocated（同位）布局 + 强拉伸网格**在尾迹 / 远场（单元极大、长宽比极端）下，中心差分的 `c_div`/`c_grad` 没有交错网格（MAC）或 Rhie-Chow 动量插值那种抗棋盘机制，于是压力解天然带高频振荡。**它不影响速度无散度、不影响 Cd/Cl**，是压力可视化高频纹路，**不是未收敛**（物理量已稳态、无 NaN）。

**曾经试过并回退的错误修复**：把"棋盘去除"当成零空间、对压力与速度每步做 i 向双向配对投影。代码实测它**发散**（α=4° 跑出 Cd=−5.7、Cl=−143）——因为这些模式并非零空间，移除速度里的真实高频内容会破坏散度-free 场并逐步放大。`c_checkerboard_project` 现已回退为**仅 j 方向的轻度平滑器**（即所有通过验证的运行所用的原始 `cb=pair`），并加注释说明**不可在循环里扩展到 i 方向**。

**真正的工程出路（未来工作，非当前 bug）**：
1. **交错网格 / Rhie-Chow 动量插值** —— 从算子层面消除同位压力棋盘，是标准根治法；
2. **降低远场拉伸**：尾迹 / 远场条纹最重处恰是长宽比极端区，更平缓的径向拉伸可显著削弱视觉条纹（但 C 网格径向拉伸是固有的）；
3. 接受现状：条纹是 cosmetic，力量与速度均正确。

### Rhie–Chow 动量插值实测（按 OpenFOAM 模式实现，2026-09-28）

按"OpenFOAM 模式"实现了 Rhie–Chow 动量插值以消除同位压力棋盘并提升阻力精度。`curve_grid.py` 新增 `_rhie_lap_matrix`（算子 `L = c_div( rAU·c_grad(p) )`，与投影修正 `u ← u − rAU·c_grad(p)` 严格自伴），`build_poisson` / `build_poisson_iterative` 增加 `mode` 参数，`run_cgrid.py` 暴露 `--mode {adjoint,rhie}`，并新增 `_PoissonSolver`（Jacobi 对称缩放 + ILU(0) 预条件 + bicgstab/cg 回退，替代原 SuperLU 以应付变系数矩阵）。`project()` 实现 OpenFOAM 式压力欠松弛 pRelax（对压力场混合而非取消修正）。

**实测结论（修订，见下节 SIMPLE+AMG 升级）：** 早期 Rhie 在拉伸 C 网格上发散的根因是**压力求解器**——旧 `_PoissonSolver`（ILU(0)+BiCGStab）解不动 κ~rAU_ratio·N² ≈ 1e13–1e15 的病态矩阵，返回尖刺压力 → 显式修正崩坏。这不是 Rhie 本身的数学死穴：**把压力方程改用几何多重网格 `_MGSolver`（收敛与条件数无关）后，Rhie–Chow 在拉伸网格上变得可用**，且力与 adjoint 完全一致。

诊断（`_rhie_stab.py`，小网格 ni=121×nj=51，打印 umax/pmax/dmax/corr）与全尺寸 `run_cgrid.py` 两套证据一致：

| 工况（Jratio = Jmax/Jmin） | 算子 | 结果 |
|------|------|------|
| 圆柱 Rf=3（Jratio≈75） | **rhie** rAU=dt/J | ✅ Cd=1.063，稳定，对称 |
| 圆柱 Rf=15（Jratio≈297） | **rhie** | ✅ Cd=0.847，**与 adjoint 完全相同**（近均匀 rAU → Rhie 退化为 adjoint）|
| 圆柱 Rf=15（Jratio≈297） | adjoint | ✅ Cd=0.847，稳定 |
| 翼型 α=4 Rf=15（Jratio≈6.7e4） | adjoint | ✅ Cd=0.034，Cl=0.091 |
| 翼型 α=4 Rf=15（Jratio≈6.7e4） | **rhie**（cb=pair，4000 步） | ✅ Cd=0.036，Cl=0.112（边界稳定，力≈adjoint）|
| 翼型 α=4 Rf=15（Jratio≈6.7e4） | **rhie**（cb=none，600 步） | ✅ 未 NaN（边界稳定，早瞬态 Cd≈0.44）|
| 翼型 α=4 Rf=15（Jratio≈2.2e5，β=4 更拉伸） | **rhie** dtJ / ap | ❌ NaN@10~11（umax→1e8）|
| 翼型 α=4 Rf=40（Jratio≈4.5e7） | **rhie** dtJ / ap（旧 `_PoissonSolver` ILU+BiCGStab） | ❌ umax→310–540（速度 300× 自由流，物理崩坏）|
| 翼型 α=4 Rf=40（Jratio≈4.5e7） | **rhie + AMG**（`_MGSolver`） | ✅ 无 NaN，力与 adjoint 逐位相同（500 步 Cl=0.9874）|
| 所有网格 | adjoint | ✅ 全程稳定（含 Jratio=4.5e7）|

**根因（条件数）**：Rhie 泊松算子 `L = c_div( rAU·c_grad(p) )` 的条件数 ≈ `rAU_ratio · N²`，其中 `rAU_ratio = max(rAU)/min(rAU)`。对 `rAU = dt/J`，`rAU_ratio = Jmax/Jmin = Jratio`（翼型网格达 1e5–1e7）；`N²` 在 N≈1.5e4 时为 ~2.5e8。二者相乘 → 条件数 1e13–1e15，**超过双精度可分辨极限（~1e14）**。任何直接/迭代 Krylov 求解器（SuperLU / ILU+bicgstab / cg）都无法恢复光滑物理压力，返回被放大的尖刺 `p` → 修正 `corr = rAU·|∇p|` 过冲 → 显式 Chorin 步发散。`rAU = 1/A_P`（真正的 OpenFOAM 形式）仅把发散推迟几步（近壁 A_P 大→rAU 小，但总 rAU_ratio 仍 1e4–1e7），同样崩坏。adjoint 算子 `L = c_div∘c_grad`（rAU 恒为 dt，rAU_ratio=1）条件数仅 ~N²≈1e7，可在双精度内求解，且投影按构造无散，故全程稳定。

**为什么 OpenFOAM 能用 Rhie–Chow 而这里（早期）不行**：OpenFOAM 的 Rhie–Chow 是嵌在 **隐式 SIMPLE/PISO 动量耦合**里的——压力修正是隐式外迭代的一部分，且泊松用 **GAMG/AMG** 多重网格求解变系数 Laplace（AMG 的收敛与条件数无关，专门压这种 1e14 量级的变系数系统）。本求解器是**显式 Chorin 投影**（每步一次投影、无隐式动量），但投影循环 `project()` 已经具备 **SIMPLE 结构**（显式动量预估 + 压力欠松弛 pRelax + 多次内迭代 ncorr），缺的只是 AMG 压力求解器。一旦补上几何多重网格（`_MGSolver`），Rhie–Chow 在拉伸网格上就和 OpenFOAM 一样可用了。

### SIMPLE + AMG 升级（2026-09-28，Rhie 现在可在拉伸 C 网格上用）

把压力方程从旧 `_PoissonSolver`（ILU+BiCGStab，解不动 κ~1e14 矩阵）换成手写几何多重网格 `_MGSolver`，使 `--mode rhie` 在强拉伸网格上稳定可用。

- **几何多重网格的合理性**：C 网格是结构化 `ni×nj` 索引，半粗化（只在壁法向 j 方向粗化，i 不变）+ Galerkin `Rᵀ·A·R` 生成粗算子，本质就是 AMG（pyamg 因 numpy 1.x 装不上，故手写）。`project()` 的 SIMPLE 结构（pRelax + ncorr）保持不变，只换压力求解器。
- **两套关键设计**让 MG 在变系数、强拉伸算子上收敛且不溢出：
  1. **用自然（对角占优）算子，不做欧氏对称 Jacobi 缩放**：FV 压力算子 `L=c_div(rAU·c_grad)` 每行行和≈0（非对角同号）→ 对角占优。GS 平滑器在对角占优矩阵上三角回代除以最大对角元、**稳定且收敛**。若做全局 dh 缩放把对角强制成 −1，非对角会留下 ~√Jratio≈2600，破坏对角占优 → GS 三角回代逐层放大 → 溢出。全加权 Galerkin `RᵀAR` 还**继承行和≈0**，所以每个粗算子也对角占优，没有"过小粗对角"问题。
  2. **半粗化（j 方向）+ 对称 Gauss–Seidel 平滑 + BiCGStab 外层（MG 作预条件）**：j 方向拉伸 ~1e7、i 方向仅轻微拉伸，故只对 j 粗化消除主导各向异性；GS 用 `spsolve_triangular` 做对称前后向扫描；最外层用 BiCGStab（自然算子在欧氏意义非对称）以 MG V 循环作预条件，并带热启动（上步压力作初值）。
- **验证**（`_mg_test.py`，翼型 Rf=40，Jratio=7.1e6，N=15330）：
  - 自然 Rhie 算子残差 `‖L·p − b‖/‖b‖ = 7.76e-9`，`|p|max=4.21`（与 SuperLU 直接分解的 4.02 一致），**无解虑溢出**；
  - 全尺寸 `run_cgrid.py --mode rhie --geometry airfoil --Rf 40 --aoa 4` 全程无 NaN；
  - **Rhie+MG 与 adjoint 给出逐位相同的力**（500 步：两者 Cl=+0.9874、Cd=0.1054），证明 MG 解出的是同一物理压力场，Rhie 不再发散。

**对"算准确"的最终结论（修订）**：
- 拉伸 C 网格上现在**两条路都可用**：默认 `mode=adjoint`（unweighted `c_div∘c_grad`，稳定准确，仅遗留 cosmetic 棋盘条纹）与 `mode=rhie`（Rhie–Chow + SIMPLE + AMG，消除同位棋盘）。两者在拉伸网格上给出**相同物理力**。
- `--mode rhie` 现已在**全拉伸范围**（Jratio 达 ~1e7 的翼型 Rf=40）可用；早期"仅近均匀网格可用"的结论已被 SIMPLE+AMG 升级推翻。
- 力的准确性不靠棋盘消除，而靠**网格收敛**：GCI 研究（Rf=40, β=6，t≈10）已显示 Cl≈0.34 网格收敛（GCI~1%，外推 0.335），Cd 仍随分辨率上升未收敛（需 β≥8、nj≥120）。即"算准"靠加密 + 收敛，而非 Rhie 算子本身。

### 运行

```bash
cd /workspace/tensorlbm
# 圆柱 Re=100（C 网格，对照 O 网格）
python3 run_cgrid.py --geometry cylinder --Re 100 --ni 161 --nj 51 \
    --nsteps 3000 --steady --scheme ppm

# NACA0012 扫迎角（近场粗网格 Rf=15，快速）
python3 run_cgrid.py --geometry airfoil --aoa 4 --Re 100 \
    --ni 161 --nj 51 --Rf 15 --beta 3 --nsteps 6000 --steady --scheme ppm

# NACA0012 扫迎角（拉远+近壁加密 Rf=40,β=6，向文献靠拢）
python3 run_cgrid.py --geometry airfoil --aoa 8 --Re 100 \
    --ni 221 --nj 71 --Rf 40 --beta 6 --nsteps 20000 --steady --scheme ppm
```

> 投影算子默认 `--mode adjoint`（unweighted `c_div∘c_grad`，拉伸 C 网格稳定准确，仅遗留 cosmetic 棋盘）；`--mode rhie` 为 OpenFOAM 式 Rhie–Chow + SIMPLE + AMG（现已在拉伸网格可用，见上节）。以上示例未显式写 `--mode` 即走 adjoint 默认。

### 长时程与多核：checkpoint/restart + 并行批处理

**后台时间限制的对策——分段续算**（`run_cgrid.py` 新增）：
```bash
# 第一段：跑 5000 步，每 1000 步落盘一次检查点
python3 run_cgrid.py --geometry airfoil --aoa 4 --Re 100 --Rf 40 --beta 6 \
    --ni 221 --nj 71 --mode rhie --nsteps 5000 --steady \
    --save_every 1000 --out run_seg1
# 被时间限制杀掉后，从上次检查点续跑（自动从 step 衔接，力历史连续）
python3 run_cgrid.py --geometry airfoil --aoa 4 --Re 100 --Rf 40 --beta 6 \
    --ni 221 --nj 71 --mode rhie --nsteps 10000 --steady \
    --restart run_seg1_ckpt.npz --out run_seg2
```
`--save_every N` 每 N 步把 `u/v/p/step/力历史` 写入 `<out>_ckpt.npz`（覆盖式，始终保留最远完成点）；`--restart` 载入后时间循环从 `start_step+1` 续跑，泊松算子由网格参数确定性重建，无需存。

**32 核怎么用——并行独立任务**（`run_batch.py`）：本求解器是显式时间推进，单条长时程**不能跨步并行**；且 ~15000 单元下 MG 的矩阵-向量乘极小，开 BLAS 多线程是**负收益**（线程开销 > 计算量）。所以 32 核的正确用法是**同时跑多条独立任务**（不同 mode / 网格加密 / 攻角扫描），每条单线程：
```bash
python3 run_batch.py        # rhie vs adjoint + 网格加密三连，32 核并行
```
批处理每个 worker 把 `OPENBLAS/MKL_NUM_THREADS` 钉为 1，避免 32 个 worker 超订（oversubscribe）32 个核。

### 验证结果（Chorin 投影，NACA0012 Lref=1 / 圆柱 Lref=2）

| 工况 | 网格 | Re | α(°) | Cd | Cl | 说明 |
|------|------|----|------|-----|-----|------|
| 圆柱 | C ni161 nj51 | 100 | 0 | **1.358** | ≈0 | 对照基线（文献 ~1.0–1.4）|
| NACA0012 | C 近场 Rf=15 | 100 | 0 | 0.436 | ≈0.000 | 对称 ✅ |
| NACA0012 | C 近场 Rf=15 | 100 | 4 | 0.454 | +0.160 | 升力为正（远场过近，偏低）|
| NACA0012 | C 近场 Rf=15 | 100 | 8 | 0.510 | +0.340 | 升力随 α 单调增 |

> ⚠️ **先前"Rf=40 时 Cl=0.384/0.768"的结论已作废**。那组数据以固定 20000 步积分，但 `dt` 随网格缩小，导致各网格到达的**物理时间相差 17 倍**（最细网格只到 t≈1.9，<2 个对流时间，远未稳态）。所谓"高 Cl"是**起始瞬态**，不是收敛值。正确做法见下节。

### 网格收敛研究（GCI）——"力"的误差到底多大

**前提**：三套网格必须**积分到相同物理时间**且各自达稳态，才能做 Richardson 外推。首版"固定步数"违反此前提。

**对齐做法**：固定 Rf=40、β=6、Lw=40，仅加密网格间距（ni 161→221→301，nj 51→71→97，r≈1.367），按 `1/dt` 缩放步数使三套都到 **t≈10**；`run_cgrid.py` 现保存 `t/cd/cl` 时间序列。脚本：`_gci_aligned.py`。

稳态平台均值（公共晚窗 t∈[6,10]）：

| 网格 | ni×nj | dt | Cl | Cd |
|------|-------|-----|-----|-----|
| A 粗 | 161×51 | 1.61e-3 | 0.111 | 0.168 |
| B 中 | 221×71 | 2.37e-4 | **0.338** | 0.395 |
| C 细 | 301×97 | 9.40e-5 | **0.341** | 0.462 |

**结论——升力已收敛、阻力未收敛**：

| 量 | B→C 相对变化 | GCI(细网格, 假设 p=2) | 判读 |
|----|-------------|---------------------|------|
| **Cl** | **0.97%** | **≈1.4%** | ✅ **网格收敛**，外推 Cl≈0.335 |
| **Cd** | 14.4% | ≈21% | ❌ **未收敛**，仍单调上升 |

- **Cl 收敛值 ≈ 0.34**：B、C 从上下两侧收敛到同一平台，差 <1%。与薄翼理论 `2πsin4°=0.438` 差 **−22%**，但**这不是离散误差**（已收敛）——是 Re=100 黏性 + 近壁（β=6）+ 远场（Rf=40）+ collocated 布局的综合模型效应。
- **Cd 未收敛**：0.168→0.395→0.462 单调上升，B→C 仍 +14%，需更细近壁（`β≥8, nj≥120`）与更长积分；Cd 是最后收敛的量。
- **A 粗网格不参与 GCI**：其 Cl=0.11 仅为 B/C 的 1/3（数值耗散抹掉环量），三套不构成渐近序列，故取 B、C 两套做 GCI。
- 图：`_gci_zoom.png`（稳态 Cl/Cd 平台）、`_gci_aligned_timeseries.png`（全程含起始脉冲）。

### 已知限制

1. **远场距离敏感**：Rf=15 时 Cl 偏低（远场闭合过近压低环量）；Rf=40 显著改善。翼型建议 `Rf≥30`、`ni≥220`。
2. **近壁分辨率**：`β=3`（首层 ~0.046c）过粗，`β=6`（~0.009c）更好；要精确定量阻力需 `β≥8`、`nj≥120` 解析层流边界层与分离泡。
3. **Re=100 层流**：与高 Re 实验（Critzos NACA0012 α=0 Cd≈0.006）不可比；本求解器为层流，应和层流 CFD / 低 Re 实验对标（低 Re 层流分离泡使 Cd 偏高）。
4. **稳态统计须按"物理时间"对齐**：`dt` 随网格细化而减小，固定步数会让不同网格停在**不同物理时间**（首版曾差 17 倍，致 Cl 非单调假象）。跨网格比较力时务必固定物理时间（如 t≈10）、各自达稳态后再平均；`run_cgrid.py` 的 `t/cd/cl` 时间序列存于 `*_fields.npz`。
5. **收敛现状**：Cl 已网格收敛（≈0.34，GCI~1%）；Cd 未收敛（B→C 仍 +14%，GCI~21%）。非定常脱落（高 Re）需更长积分与更快泊松求解器（多重网格/FFT）。


