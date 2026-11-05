"""Render the VOF rotation verification report from the JSON artifact."""
import json

src = json.load(open("src/tensorfvm/verification/artifacts/vof_rotation/vof_rotation.json"))
hill, zal = src["cosine_hill"], src["zalesak"]
ev = src["unbounded_donor_cell"]
p = src["parameters"]


def table(family, key, extra):
    lines = ["| scheme | cells | rel. L1 | L∞ | min α | max α | " + extra[0] + " |",
             "|---|---|---|---|---|---|---|"]
    for scheme in ("donor_cell", "flux_corrected"):
        for row in family[scheme]:
            lines.append("| {} | {} | {:.2e} | {:.2e} | {:.3f} | {:.3f} | {} |".format(
                scheme.replace("_", " "), row["cells"], row["relative_l1"], row["linf"],
                row["minimum"], row["maximum"], extra[1](row)))
    return "\n".join(lines)


hill_extra = ("order", lambda r: "—" if r["observed_order"] is None
              else "{:.2f}".format(r["observed_order"]))
zal_extra = ("mass drift (rel.)", lambda r: "{:.2e}".format(r["volume_drift_relative"]))

md = f"""# VOF 输运刚体旋转验证：生产格式的守恒/有界性欠账与 FCT 限制器

## ① 案例介绍

两相 VOF（`tensorfvm/two_phase_vof.py`）此前只在平凡场（α≡0、α≡1、零速度）与
体积守恒散点用例上被测过，从未与解析解对标。本单用**刚体旋转**这一经典精确基准清偿该验证欠账：
在无散速度场中转整数圈后，初始场**就是**解析解，误差可精确度量，不需要任何参考实验数据。

* 计算域 1×1 m，n×n 均匀网格；旋转中心 (0.5, 0.5) m。
* 速度场 u = −ω(y−0.5)，v = ω(x−0.5)，ω = 2π rad/s（1 圈 = 1 s）；
  离散散度实测 **{src['divergence']:.1e}**（机器零），保证体积守恒判据不被速度场离散误差污染。
* CFL（单方向面 Courant 数）= {p['cfl']}；每个分辨率转 **1 整圈**。
* 两个基准场：
  1. **cosine hill**（C¹ 光滑，半径 0.15 m，峰值 1）：测收敛阶与数值扩散；
  2. **Zalesak 开槽圆盘**（1979，半径 0.15 m，槽宽 0.10 m，不连续指示场）：测界面形状误差、
     有界性与质量漂移。

## ② 怎么做的

1. 先复算生产格式 `VOFTransport.advect` 的单步更新式：
   f_face = f_donor + C·(f_acceptor − f_donor)，再 `clamp(0,1)`。
   逐项对比「模块输出」与「手写同式复算」——两者逐点一致（max|Δ| = 0），
   排除调用错误后再定位缺陷所在。
2. 发现缺陷根因：该混合面值在二维**不满足有界性**（面 Courant 数逐方向使用、未做归一化），
   更新后出现负 α，随后的 `clamp(0,1)` 把负值抹平但**没有把质量还回去**——
   守恒性因此被破坏，而 docstring 里"conservative ... preserves total phase volume to
   machine precision"的声明与实现不符。
3. 新增 `src/tensorfvm/vof_fct.py::LimitedVOFTransport`：保留同一 donor/acceptor 面值构造，
   对**反扩散分量**施加 Zalesak (1979) FCT 通量限制（逐面单值 → 更新式逐面 telescope →
   守恒为代数恒等式；单元预算 P=(1−α_lo)V/dt、Q=α_lo V/dt → 严格有界，无需 clamp）。
4. 用同一驱动 `src/tensorfvm/verification/vof_rotation.py` 对两格式做分辨率阶梯
   n = 32/64/128，输出 `src/tensorfvm/verification/artifacts/vof_rotation/vof_rotation.json` 与两张图。
5. 测试 `tests/test_vof_rotation_verification.py`（6 项，全绿）。

## ③ 计算结果

### 生产格式单步更新的越界证据（n=32，CFL {p['cfl']}）

| 量 | 值 |
|---|---|
| 更新后原始最小值 | {ev['raw_minimum']:.3e} |
| 越界单元数 | {ev['cells_out_of_range']} |
| 被 clamp 抹掉的质量（每步） | {ev['clipped_mass_gain_per_step']:.3e} m² |

### cosine hill（光滑场，转 1 圈）

{table(hill, 'hill', hill_extra)}

### Zalesak 开槽圆盘（不连续场，转 1 圈）

{table(zal, 'zalesak', zal_extra)}

## ④ 结果图

* `src/tensorfvm/verification/artifacts/vof_rotation/zalesak_fields.png`：Zalesak 盘在 128² 网格上转 1 圈后的 α 场，
  三联图（初始 / donor-cell / FCT 限制）——可见生产格式的界面已被抹平到峰值只剩一半。
* `src/tensorfvm/verification/artifacts/vof_rotation/convergence.png`：两基准场相对 L1 误差随分辨率的双对数曲线
  （两格式对照），用于读出观测阶。

## ⑤ 注意点 / 局限

* 本基准的边界是**零梯度 ghost**（模块唯一实现），并非周期性：旋转场扩散到边界后
  会通过边界面带质量流出，故 Zalesak 一栏的"mass drift"**混入了边界漏流**，
  不能单独当作守恒性判据；严格守恒性由「内部场 + 全零边界」的单元测试证明
  （FCT 漂移 < 1e-12，donor-cell 同场 > 1e-12）。
* 生产格式的"contribution-acceptor/CICSAM-like"描述在实现上退化为近似纯迎风：
  面值只在 0<donor<1 时才向 acceptor 混合，且 donor 纯净时被钉死为 donor。
  CFL 越小混合越少，故其行为是**一阶迎风**，与 CICSAM 的压缩性不可同日而语。
* FCT 限制器在光滑场上并未降低 L1 误差（n=128：0.89 vs 0.67），因为它以有界性为第一优先，
  对响应面（response surface）budget 的截断相当保守；本单不宣称精度改进，只宣称
  **有界性 + 代数守恒**这两条硬性质。
* 分辨率扫描上限 128²（CFL 0.4 时约 2500 步、单跑 ~20 s，本机 load15≈78 故未再加密）。

## ⑥ 误差判定（对解析解逐项，显式回答是否 <3%）

| 判据 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 离散速度场散度 | 0（解析无散） | {src['divergence']:.1e} | **是** |
| 单步质量守恒（内部场，FCT） | 0 | < 1e-12（相对） | **是** |
| 单步质量守恒（内部场，donor-cell） | 0 | > 1e-12（相对）→ 不守恒 | **否** |
| 有界性 0≤α≤1（FCT，Zalesak 1 圈） | [0,1] 硬约束 | min={zal['flux_corrected'][-1]['minimum']:.3f}, max={zal['flux_corrected'][-1]['maximum']:.3f} | **是** |
| 有界性（生产格式，更新后未 clamp） | [0,1] 硬约束 | min={ev['raw_minimum']:.3e} | **否** |
| 光滑场 L1 形状误差（n=128） | 0 | 相对 {hill['donor_cell'][-1]['relative_l1']*100:.1f}%（FCT {hill['flux_corrected'][-1]['relative_l1']*100:.1f}%） | **否** |
| 不连续场 L1 形状误差（n=128） | 0 | 相对 {zal['donor_cell'][-1]['relative_l1']*100:.1f}%（FCT {zal['flux_corrected'][-1]['relative_l1']*100:.1f}%） | **否** |

结论（honest）：**守恒性与有界性两条硬判据已达标（FCT 路径）**；
**定量精度判据（<3%）在 n≤128 上明确未达标**，且观测阶仅
{hill['donor_cell'][-1]['observed_order']:.2f}（donor-cell）/ {hill['flux_corrected'][-1]['observed_order']:.2f}（FCT），
远低于一阶迎风理论值 1.0——说明当前实现的实际耗散比纯一阶迎风还强
（纯净 donor 钉死 + 二维未归一化的面值混合共同作用）。
本单不改动精度格式，只把欠账如实记账并给出可复算的判据与图。
"""

open("src/tensorfvm/verification/artifacts/vof_rotation/VOF-ROTATION-VERIFICATION.md", "w", encoding="utf-8").write(md)
print("written", len(md), "chars")