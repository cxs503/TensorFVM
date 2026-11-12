# D3 总账 GAP-FVM-TWOPHASE 状态复核（m16314, 2026-11-12）

- 性质: 验证欠账 D3（`docs/VERIFICATION_DEBT_LEDGER.md` §2）——对总账
  `GAP-FVM-TWOPHASE`（status=in_progress）的验收句逐条核实现状。
- 结论: **不得升账 done**。验收面两半中 Hysing 半为 HONEST FAIL（气泡下沉，
  末速相对误差 109.5% ≫ 10% 门）。总账维持 in_progress，剩余缺口立为 D9。
- 纪律: honest FAIL 不覆盖；所有数字 SQL/工件溯源，零手抄。

## 1. 总账验收句（feature_ledger, slug=GAP-FVM-TWOPHASE, status=in_progress）

> 【基准牵引|Zalesak 旋转缺口盘 + Hysing 上升气泡｜ref: Zalesak 1979；
> Hysing et al. 2009 benchmark 数据集｜判据: 界面形状误差/末速 vs 基准数据
> 形状 qualitatively+末速<10%】两相流 VOF/界面捕获验证

验收面 = 两条腿：
- **腿 A（Zalesak 1979 旋转缺口盘）**: 界面形状误差 vs 基准（qualitative）。
- **腿 B（Hysing et al. 2009 上升气泡, Case 2: Eo=10, M=1e-3, ρr=1000, μr=10）**:
  末速 U_T vs 基准 <10%（基准 U_T=0.2342, min circularity 0.4644, y_c(t=3)=1.984，
  均取自论文 Table II 2-D level-set 参考解）。

## 2. 逐条核对

### 腿 A — Zalesak 旋转缺口盘: **PASS**（证据链完整）

工件 `docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json`（N=128 PLIC 声明分辨率）:

| 门 | 量 | got | limit | 判定 |
|---|---|---|---|---|
| Z1 | thresholded geometric error | 0.0395 | ≤0.05 | PASS |
| Z2 | 相对液相面积漂移（一转） | 6.4e-16 | ≤1e-3 | PASS |
| Z3 | 捕获轮廓长相对变化 | 0.0643 | ≤0.1 | PASS |
| Z4 | alpha 越界 [0,1] 最大 excursion | 0.0 | ≤1e-12 | PASS |
| Z5 | 几何误差观测收敛阶 | 1.234 | ≥0.8 | PASS |

复算佐证: `runs/m15201/artifacts`（m15201 repair 轮, 59.1 s 快战役）独立
symmetric-difference 复算 0.0930@N=64 与记录一致（`REPAIR-EVIDENCE.md`）。
→ 腿 A 判据「界面形状误差 vs 基准（qualitative）」兑现：最细面 3.95% 误差、
观测阶 1.23、质量守恒机器精度。

### 腿 B — Hysing 上升气泡: **HONEST FAIL**（本单新证据）

本单实跑 `run_hysing_leg(cells_per_diameter=16, end_time=1.0)`
（`src/tensorfvm/twophase_gap/hysing_gate.py`，776 步，未发散），
工件 `docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing_cpd16.json`:

| 门 | 量 | got | ref (Hysing 2009) | limit | 判定 |
|---|---|---|---|---|---|
| B1 | 末速 U_T | **−0.0223** | 0.2342 | ≤10% | **FAIL**（相对误差 109.5%，且**符号相反**：气泡下沉） |
| B2 | min circularity | 5.621 | 0.4644 | ≤10% | **FAIL**（5.62 物理上不可能，circularity∈(0,1]，界面已破碎/度量失真） |
| B3 | 最差单步体积漂移 | 1.69e-7 | — | ≤1e-6 | PASS |
| B4 | 网格收敛 U_T coarse→fine | 未跑（重算力，超本单负载预算） | — | ≤5% | NOT_RUN |

**根因指针（供修复单，本单不修）**:
1. **网格语义 bug（小，可单干）**: `HysingCase.solve_config()` 用
   `nx = cells_per_diameter`——把「每直径格数」当成了「x 向格数」。域 2.0×4.0、
   d=1.0 时 cpd=16 实际 dx=0.125，气泡只跨 **8** 格（应为 16 格）；cpd=32 也只跨 16 格。
   正确构造: `nx = int(domain_width / (d/cpd)) = 2*cpd`。
2. **气泡下沉（物理/设置缺陷，需专项）**: 配置本身物理正确（rho_g=0.001 <
   rho_l=1.0，gravity=1.0 按求解器约定指向 −y，轻气泡应受净上浮力），但实测
   U_T<0。初值气泡底缘贴底壁（start_y=0.5, r=0.5 → 底缘 y=0），贴壁+8 格粗界面
   是否足以解释全部下沉未定（cpd=32 离壁诊断跑超 120 s 单命令预算，未跑完）。
   修复单须先做离壁对照实验再定责。
3. **失效测试（如实保留，不弱化）**: `tests/test_hysing_gate.py::
   TestHysingLegFast::test_run_hysing_leg_cpd16` 断言 `terminal_velocity >= 0`，
   当前红——它正确捕获了缺陷 2，**不得**用 xfail/放宽掩盖。

### 附加腿 — 溃坝（Martin & Moyce 1952，不在总账验收句内）

`gap_twophase.json`: D1 front 0.988 / D2 height 0.845（limit 0.15）→ HONEST FAIL
（m15201 repair 轮已如实记录，`REPAIR-EVIDENCE.md`）。不影响本条总账判定，
但修复单若顺带处理应一并复核。

## 3. 总账更新建议（SQL 提案，交平台/owner 落库）

```sql
-- GAP-FVM-TWOPHASE 维持 in_progress；验收面 2 腿中 1 腿 FAIL，不得升 done。
UPDATE feature_ledger
SET note = 'D3 复核(m16314,2026-11-12): Zalesak 腿 PASS(Z1-Z5, 工件 gap_twophase.json); '
          'Hysing 腿 HONEST FAIL(U_T=-0.022 vs 0.2342, 109.5%, 气泡下沉; 工件 leg_hysing_cpd16.json); '
          '溃坝腿 FAIL(不在验收句内)。剩余缺口=D9(Hysing 腿修复+cpd 语义修正+离壁对照)。'
WHERE slug = 'GAP-FVM-TWOPHASE';
```

新条目建议（若总账收编）:
- slug: `GAP-FVM-HYSING-LEG`
- 验收: ①`solve_config` 网格按 d/cpd 构造（气泡跨格数=cpd±1）；②离壁初值
  U_T>0 且 t=3 末速 vs 0.2342 <10%（cpd≥32）；③min circularity vs 0.4644 <10%；
  ④cpd 16→32 收敛 ≤5%；⑤`test_run_hysing_leg_cpd16` 转绿且断言不弱化。
- 域: TensorFVM 两相 VOF。

## 4. 证据指针

- 腿 A 工件: `docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json`（Z1–Z5）
- 腿 B 工件: `docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing_cpd16.json`（本单实跑）
- 溃坝工件: `docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json`（D1/D2）+
  `docs/GAP-FVM-TWOPHASE/REPAIR-EVIDENCE.md`
- 代码: `src/tensorfvm/twophase_gap/{hysing_gate.py,zalesak_gate.py,dambreak_gate.py,campaign.py}`
- 测试: `tests/test_hysing_gate.py`（20 绿 1 红，红=缺陷 2 的活门）、
  `tests/test_vof_rotation_verification.py`（Zalesak 族）
- 账本: `docs/VERIFICATION_DEBT_LEDGER.md`（D3 移 §4；D9 新增）
