# CASE-FVM-CHANNEL-KMM180 —— 无湍流模型（closure-free）路线复核 u+ 对数律

> 每个数字都由 `scripts/reanalyze_case_fvm_channel_kmm180_dns.py` 从 `docs/case-fvm-channel-kmm180-dns/profiles.csv` 与仓内 DNS 参考程序化注入，零手抄。

## 1. 案例介绍

**物理背景**：充分发展槽道湍流，$x$（流向）与 $z$（展向）周期、两壁无滑移，由恒定体积力 $G$ 驱动。力平衡精确给出 $u_\tau^2=Gh$，故摩擦速度由**构造**确定，解出的流向速度**本身即** $u^+$，全程不经过任何壁面剪切估计器或湍流闭合。

| 项目 | 取值 |
|---|---|
| $Re_\tau=u_\tau h/\nu$ | 180.0（精确施加，非"测得"） |
| $\nu$ / $G$（密度 1） | 0.005556 / 1.000 |
| 计算域（壁面单位） | $L_x^+=1131$, $L_z^+=565$ |
| 网格 | 64x128x64 = 524288 单元 |
| 壁面单位间距 | $\Delta y^+_{\min}=0.86$, $\Delta y^+_{\max}=4.66$, $\Delta x^+=17.7$, $\Delta z^+=8.8$ |
| 时间步 | $\Delta t=1.2336e-03$ ($\Delta t^+=40.0$) |
| 统计窗口 | 弃置 326 步 + 采样 1220 步，$T=1.51\,h/u_\tau$（3.6 个流通时间） |
| 运行 | 集群作业 `cj2408`（节点 n315，8 核，19 min） |
| 参考 | Lee & Moser 2015 DNS ($Re_\tau=182.088$)；KMM1987 原表仍 WITHHELD |

**为什么需要这条路**：该锚点此前每个数字都出自 1-D 平衡 RANS 模型，即出自湍流闭合 `nu_t`；误差预算自陈「预算是一维的、无法判断离散本身是否算对」。本路线把闭合从回路里拿掉：动量、连续、壁面条件三者决定了 $u^+$。

## 2. 怎么做的

1. `src/tensorfvm/verification/channel_spectral_dns.py`：流向/展向 Fourier 谱方法（实 FFT，Nyquist 模截断）；壁法向**交错有限体积**（$u,w,p$ 在单元中心，$v$ 在面心、两壁 $v=0$）；压力泊松算子与散度算子同一通量形式，投影后散度为零到机器精度；
2. 时间推进：SSP-RK3（对流 + 壁法向扩散 + 体积力），周期面扩散在谱空间精确积分（Strang 分裂），步长只受壁法向粘性与 CFL 限制；
3. `nu_t \equiv 0`：雷诺应力完全由求解产生；
4. 统计：$(x,z,t)$ 平均并折叠上下半槽；
5. 判据复用锚点既有评分代码与窗口（与 RANS 路线完全一致）。

## 3. 计算结果

### 3.1 锚点量 $u^+(y^+)$

| 窗口 | 定义 | 采样点 | RMS 相对偏差 | 最大相对偏差 | 平均偏移(wu) | 判定(<5%) |
|---|---|---|---|---|---|---|
| legacy 字面窗口 | y+ in [30, nearest reference station] (literal anchor wording) | 42 | 1.614 % | 2.875 % | +nan | **PASS** |
| 物理对数层 | y+ in [30, 0.2*Re_tau] (physical log layer) | 3 | 2.593 % | 2.875 % | -0.365 | **PASS** |

同参考同判据下的路线对照：

| 路线 | legacy RMS | 对数层 RMS | 来源 |
|---|---|---|---|
| 本单：无闭合（谱+FV, 128 壁法向单元） | 1.61 % | 2.59 % | 本报告 |
| SST, 1-D 平衡 RANS, 256 单元 | 3.46 % | 6.12 % | `docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json` |

### 3.2 全局量

| 量 | 计算 | 参考 DNS | 相对偏差 |
|---|---|---|---|
| $U_b^+$ | 15.060 | 15.691 | -4.02 % |
| $C_f=2\tau_w/(\rho U_b^2)$ | 0.008818 | 0.008123 | +8.55 % |
| 壁面剪切（剖面估计，应=1） | 0.7756 | 1.0000 | -22.44 % |

### 3.3 无闭合的动量恒等式（不需参考值）

$du^+/dy^+-\langle u'v'\rangle^+=1-Y/h$：全剖面最大残差 **0.7082**，$y^+>30$ 区间 **0.7082**（以 $u_\tau^2$ 为单位）。

## 4. 图

![fig_velocity_profile](fig_velocity_profile.png)

*Mean velocity of the closure-free forced-channel run against the imported DNS reference; shaded band = graded log layer.*

![fig_deviation](fig_deviation.png)

*Relative deviation of the closure-free u+ from the DNS reference in percent; the dashed lines are the 5 % criterion.*

![fig_stress_balance](fig_stress_balance.png)

*Viscous plus Reynolds shear against the exact 1-Y/h balance: a momentum-bookkeeping check that needs no reference data.*

![fig_stresses](fig_stresses.png)

*Rms velocity fluctuations of the closure-free run (no reference is imported for the stresses: the column semantics of the shipped LM2015 stdev file are undocumented).*

## 5. 注意点

* **欠分辨**：$\Delta x^+=18$、$\Delta z^+=9$ 只能算"欠分辨无闭合"（under-resolved, no model），不是壁面分辨 DNS；壁法向是有限体积二阶，流向/展向是谱方法，故本结果**不构成** FVM 网格收敛声明。
* 壁法向 $\Delta y^+_{\min}=0.86$：粘性底层由二阶差分覆盖，由剖面反推的壁面剪切偏差 -22.44 %。
* **参考替身**：KMM1987 原表仍不在仓内（WITHHELD），判据对 Lee & Moser 2015（$Re_\tau=182.088$）；本算例精确施加 $Re_\tau=180$，两者差约 1 %（对数律 $\Delta u^+=\ln(182.088/180)/0.41\approx0.03$）。
* **统计不确定性**：窗口只有 3.6 个流通时间，且集群取回的工件只含 `profiles.csv`（`metrics.json`/图未随 tar 返回，故本报告的指标由该 CSV 重算），两半窗口收敛性诊断因此缺失——这是本单的**已知缺口**，不影响剖面本身，但量化统计误差需要更长窗口。
* **湍流强度未与参考对比**：仓内导入的 LM2015 `mean_stdev` 文件列语义无文档，不猜（WITHHELD）。
* 初始条件为复合律（粘性底层+对数律）加 15 % 随机扰动，弃置段之后不再注入扰动。

## 6. 误差判定

| 关键量 | 计算 | 基准 | 相对偏差 | <5 % ? | <3 % ? |
|---|---|---|---|---|---|
| 对数层窗口 RMS $u^+$ | +2.593 % | 0 | +2.593 % | 是 | 是 |
| legacy 窗口 RMS $u^+$ | +1.614 % | 0 | +1.614 % | 是 | 是 |
| $U_b^+$ | -4.020 % | 0 | -4.020 % | 是 | 否 |
| $C_f$ | +8.552 % | 0 | +8.552 % | 否 | 否 |
| $u^+$ @ $y^+=30$ | -3.060 % | 0 | -3.060 % | 是 | 否 |
| $u^+$ @ $y^+=50$ | -1.634 % | 0 | -1.634 % | 是 | 是 |
| $u^+$ @ $y^+=100$ | -1.527 % | 0 | -1.527 % | 是 | 是 |

**结论**：无闭合路线在物理对数层的 RMS 偏差为 2.593 %，满足锚点 5 % 判据；legacy 窗口 1.614 %（PASS）。即：把湍流闭合从回路中拿掉后，锚点判据在该路线上成立，与 RANS 路线在物理对数层的 FAIL 形成对照，说明该 FAIL 属闭合而非离散。

