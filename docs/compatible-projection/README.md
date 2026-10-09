# 三维相容面通量投影（2026-10-09）

本阶段验收 **离散面质量与压力投影**，`physical_accuracy_qualified=false`。
旧中心 D/G 与七点压力矩阵不相容的失败证据在 `legacy/` 保留；不把旧输出改成通过。

## 实现与原场

压力全部流体单元参与；内部只连接 fluid-fluid 面。固体相邻面和指定法向流量边界
的压力修正梯度为零，出口物理面 p=0 距 owner 半格。矩阵是实际修正的
`-D_face G_face`，对角预条件 PCG 最后重算真实残差。棋盘压力具有非零面梯度，
不通过中心 DG 零模替换压力矩阵。

权威 `face_velocity` 与投影前 `tentative_face_velocity` 存在结果对象和 `face-fields.json`。
x=(z,y,x+1)，y=(z,y+1,x)，z=(z,y,x)，z 是 owner 的上面；周期 lower face
由前一层/前一 rank 提供。JSON 包含 mask、间距、pressure、config 与分区，
多 rank 输出不互相覆盖。`midspan.csv` 中心速度不是权威无散通量。

每一步要求 finite、CFL<1、真实 PCG residual<=target、面连续性与边界质量不平衡
各 <1e-6。原 PCG 1e-8 相对/1e-11 绝对门不变；默认迭代预算增至250。

## 原生实际结果

48×32×12，dt=0.005，50 步，CPU单线程；CUDA为单张RTX3090。

| 案例 | 压力失败步 | 最大面连续性 | 最大边界质量不平衡 | 结论 |
|---|---:|---:|---:|---|
| CPU预算150 | 50 | 1.77929e-7 | 1.02905e-8 | 失败（压力未收敛） |
| CPU预算250 | 0 | 3.28589e-9 | 1.92571e-10 | 投影门通过 |
| CUDA预算250 | 0 | 3.31705e-9 | 1.92571e-10 | 投影门通过 |

250预算实际最多164次，末步160次。CPU末PCG残差2.56850e-8<3.20432e-8，
末面continuity3.43206e-12；中心散度仍0.0831587，单独公开。
`native/study.json` 绑定源代码和每个真实 raw 字段/历史的 SHA256。

复跑：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/run_compatible_projection.py --output /tmp/compatible-projection
```

独立 NumPy 验证与旧源复跑由另一个审查进程完成，证据与脚本在同目录及
`scripts/validate_compatible_projection.py`；独立参考值由 NumPy 重建，并与生产算子输出交叉对照。另用
`scripts/audit_native_projection.py` 从三组原生导出场重建最终面散度、壁面零流、
压力修正及全域通量；全步验收由保存的历史逐步检查，未宣称独立重跑全部 CFD 时间积分。
独立报告在 `independent/report.json` 和 `independent/native-audit.json`，原场与源代码均绑定 SHA256。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/validate_compatible_projection.py
PYTHONPATH=src python scripts/audit_native_projection.py
```

## 限制

中心速度仍用于原型对流/SGS预测，下步重新插值面速度，未资格认证守恒面动量输运。
中心差分对流、显式Euler、可变SGS应力缺项、压力近似力缺黏性力、阶梯固体保持公开。
此次没有认证 LES 统计、升阻力、三维贴体求解、自由液面、移动体或双向FSI。
