# 三维 ABC 压力诊断：未通过

这是后续问题定位探针，不是已完成的三维 benchmark 报告。2026-10-13，使用本报告同一生产源码（见 `../run64/source-snapshot/`）运行。原始脚本与标准输出保留，可从项目根目录复现：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
/data/TensorFEM/.venv/bin/python docs/verification-native-kovasznay/abc-diagnostics/navier_stokes.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
/data/TensorFEM/.venv/bin/python docs/verification-native-kovasznay/abc-diagnostics/stokes.py
```

独立解析场使用既有 `native_manufactured3d.abc_fields`，U=0.5、波数=0.7、ν=0.1、ρ=1，全部三个速度分量非零且随三个坐标变化。无散 ABC 速度满足 `ΔU=−k²U`。压力为 `0.25 sin(0.7x) cos(0.7y) sin(0.7z)`；解析加速度源来自连续方程，不由离散算子反推。

网格由单位盒的棋盘分组六面体/六分四面体组成，共用三角面，并施加仿射剪切 `(x,y,z)→(x+0.2y+0.1z,y+0.1z,z)`。这比对称挤出案例增加了真实三维偏斜与混合单元。全边界给定解析速度面积均值；压力固定一单元。误差为体积加权 L2，压力比较按既有三维验证规范去除常数规范；没有用固定规范误差替代该指标。

| 单元数 | NS 速度误差 | NS 去均值压力误差 | Stokes 速度误差 | Stokes 去均值压力误差 |
|---:|---:|---:|---:|---:|
| 28 | 5.0208% | 241.9517% | 1.5236% | 65.7534% |
| 224 | 2.9627% | 188.9132% | 0.8705% | 50.9437% |
| 756 | 1.9672% | 139.4630% | 0.5679% | 37.5216% |

所有求解都达到各自残差门槛；**压力精度严重未达标**。Stokes 对照从连续解析体力中去除惯性加速度，与去掉对流后的方程相匹配。Stokes 压力仍有较大误差，因此不能把问题仅归因于 NS 外迭代、对流或湍流模型。

压力参考场的空间变化较小，使该相对指标较敏感；不能因此更换分母、删除压力门槛或宣称达标。下一步应在更简单的非线性压力/剪切制造解上分离黏性应力、体力积分、压力响应、网格偏斜与混合界面影响，再修复共性离散。根因尚未确认。

本探针仅保存脚本和标量输出，没有冻结全部中间场与绘图；待定位后应以完整报告重跑，不能把这些记录升级为完整三维验收证据。新混合后端的 SST、壁面律和 SUBOFF 仍未接入验收。
