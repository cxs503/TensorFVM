# 原生 SST 通道 Reτ=1000.512 验证

## 当前状态

原生 n=64/128 计算已完成并通过数值与原场审计，但两档 Cf 精度均未达 3%；不宣称物理精度或网格收敛通过。
此项用于检查 SST 在更高 Re 壁面湍流中的表现，保留 [Reτ=182.088 的精度失败](../verification-native-channel-sst/README.md)。
通道验证不能代替 SUBOFF 的曲面压力和总阻力验收。

## 独立参考与固定验收标准

Lee & Moser (2015), *Direct numerical simulation of turbulent channel flow up to Reτ=5200*,
[JFM 原始论文](https://doi.org/10.1017/jfm.2015.268)。
采用作者发布的 [Reτ≈1000 均值](https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_1000_mean_prof.dat)
及 [采样标准差](https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_1000_mean_stdev.dat)。
原始文件头给出 Reτ=1000.512、ν=5e-5、h=1、Ub=1、uτ=0.0500256；
由此 Cf=2(uτ/Ub)²=0.00500512131072。
[参考记录](reference/provenance.json)绑定原文件及转录 CSV 的 SHA256、256 个测点、归一化和来源。
均值及标准差网格、参数、y+ 关系已核对；[导入器 5 项测试](reference-import-tests.txt)覆盖错配拒绝和数值保真。
没有凭空给出 Cf 或总体 DNS 不确定度；独立采样标准差只属于已有剖面统计。

计算仍为半通道 h=ρ=uτ=1，无体力、压降驱动、低 Re 壁面条件，不使用壁面函数，不以 DNS 初始化。
生产模块与此前通道相同，先检查 n=64/128，并根据实际加密变化补充下一档。
事先固定沿用原门槛：完整当前 RANS 方程、状态/流入变化 <1e-8，标量绝对收支 <1e-8；
速度体积加权相对 L2 和由出口通量计算的 Cf 对独立 DNS 均 <3%。
每档单独报告数值与精度，不以单档偶然接近 DNS 作为网格认证。

## 复现

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
python scripts/verify_native_developed_channel.py --levels 64 128 --reference docs/verification-native-channel-sst-re1000/reference --output results/channel-re1000-new
python scripts/report_native_developed_channel.py results/channel-re1000-new
```

运行器复制参考文件到各次运行目录；后处理优先使用该冻结副本，并核对哈希。
旧归档保留原参考位置的兼容读取。网格、速度、两种压力、原场与收敛曲线沿用共性报告模块。


## 独立模型参考的实际结果

采用[已独立实现并验证的同条件一维 SST 方程](../verification-native-sst-model-limit/README.md)，
更换为本 DNS 的 Reτ，不修改模型常数或壁面公式。四档初查后继续检查两档更细网格。

| 法向单元 | Cf | 对 DNS 的名义偏差 | 一维相对方程与两次完整更新门槛 | 原始最大绝对 ω 行缺陷 |
|---:|---:|---:|---|---:|
| 64 | 0.0046300306 | 7.4941% | 通过 | 5.821e-11 |
| 128 | 0.0048461096 | 3.1770% | 通过 | 4.657e-10 |
| 512 | 0.0050869864 | 1.6356% | 通过 | 2.98e-08 |
| 2048 | 0.0051592289 | 3.0790% | 通过 | 2.98e-07 |
| 8192 | 0.0051779407 | 3.4529% | 通过 | 0.0002441 |
| 32768 | 0.0051826466 | 3.5469% | **失败** | 0.01562 |

n=512 的小于 3% 不能单独作为达标结论。n=2048→8192 的 Cf 继续增加约 0.3614%，
n=8192 的名义偏差为 3.4529%，模型/壁面组合尚未证明满足此通道的 3% 目标。
n=32768 在 8000 次迭代后局部相对缺陷 2.176e-9，超过预设 1e-10；虽然其 Cf 再变化约 0.0908%，
**该档未收敛，不将其用作认证模型极限或网格通过证据**。

这些一维结果只按所声明的相对方程与连续两次未松弛更新门槛评价，
其极细网格绝对 ω 缺陷显著大于原生三维的 1e-8 门槛，未放宽或替代三维验收。
独立交叉重算存在巨大项相消导致的绝对舍入差异；逐单元归一化重算差异处于浮点量级，原始绝对值也全部保留。
[四档完整证据](independent-reference/summary.json)、[更细诊断含失败状态](independent-fine-diagnostic/summary.json)
及各目录下 `artifact-audit.json`、原场、历史和冻结源码可复查。

![细网格参考明确标出未收敛状态](independent-fine-diagnostic/model-discretization-comparison.png)

```bash
python scripts/verify_sst_channel_reference.py --reference docs/verification-native-channel-sst-re1000/reference --native-runs --levels 64 128 512 2048 --model-source docs/verification-native-sst-model-limit/run/model-source.html --output results/sst-reference-re1000-new
python scripts/verify_sst_channel_reference.py --reference docs/verification-native-channel-sst-re1000/reference --native-runs --levels 8192 32768 --model-source docs/verification-native-sst-model-limit/run/model-source.html --output results/sst-reference-re1000-fine-new
```

第二条预期返回未通过状态；每组可用 `audit_sst_channel_reference.py` 重算，
`plot_sst_channel_reference.py` 对未收敛点单独标记。


## 原生 n=64/128 完整结果

| 法向单元 | 总单元 | 速度相对 L2 | Cf | Cf 对 DNS 偏差 | 当前数值门槛 |
|---:|---:|---:|---:|---:|---|
| 64 | 1024 | 4.33716% | 0.00462875675 | 7.51959% | 通过 |
| 128 | 4096 | 2.40077% | 0.00484577122 | **3.18374%** | 通过 |

两档各 93 次外迭代，约 279.51 / 2062.53 秒。n=128 当前动量 3.715e-12、质量散度 1.238e-12、
k 方程 9.142e-9、ω 方程 4.126e-14；绝对标量收支最大约 3.602e-10。
[冻结原场、参考、源文件及全部历史](native-n64-n128/summary.json)和独立 `artifact-audit.json` 可核查。
前置层流控制同样通过。3.18374% 不能四舍五入后认定小于 3%。

![原生剖面和摩阻对 DNS](native-n64-n128/dns-profiles-friction.png)
![实际网格、速度和两种压力](native-n64-n128/mesh-velocity-pressure.png)

[不重算原场的独立实现对照](implementation-comparison/comparison.json)核查工况、闭合源码、原场与审计哈希：
同档 n=64/128 的 Cf 与独立一维解分别相差 0.027513% / 0.006982%。
相对已通过其相对门槛的 n=8192 一维参考，原生 n=128 的 Cf 仍低 6.41509%，速度 L2 差 3.35062%。
这支持离散误差尚未消除、与模型偏差方向相反的判断；不是另一个物理通过标准。
未收敛的 n=32768 被明确列为排除项，未拿来替代认证参考；一维绝对残差限制继续适用。

当前证据不足以认定此 SST 模型/壁面组合满足本通道 Cf 的 3% 指标，后续不挑选误差抵消的中间网格来宣布达标。
