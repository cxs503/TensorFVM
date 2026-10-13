# 原生 SST 通道 Reτ=1000.512 验证

## 当前状态

已取得并核对原作者均值及统计标准差数据；原生计算尚未完成，不宣称精度或网格收敛通过。
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
