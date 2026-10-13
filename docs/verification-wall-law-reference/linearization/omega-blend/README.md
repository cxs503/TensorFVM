# Foundation 12 blended ω/G：原生壁面组件

2026-10-13。新增 `native_sst_wall3d`，实现所选 OpenFOAM Foundation 12 的 blended ω 和生产项 G 公式，以及多壁面的面积加权单元归并。该组件为原生 SST 壁面耦合提供输入，**当前尚未自动接入 RANS，未完成 Spalding 动量与近壁 G 的完整耦合**。

## 固定版本与计算式

以本机 Foundation 12（20250206）`omegaWallFunctionFvPatchScalarField::calculate` 与 `wallCellWallFunctionFvPatchScalarField::patchFieldsToWallCellField` 为依据。官方源文件 SHA256 见 `foundation12-sources.json`；不是对 OpenCFD v2112 或所有 SST 壁面函数的通用等价声明。

给定壁邻单元 k、当前未修改生产项 G、实际壁面距离 y、分子黏度 ν，以及壁面速度法向梯度模 g：

```
Re_y = y sqrt(k) / nu
y+_k = Cmu^(1/4) Re_y
u+ = ln(E y+_k) / kappa
lam = exp(-Re_y/11), turb = 1-lam
uStar = sqrt(lam nu g + turb sqrt(Cmu) k)
omega = lam 6nu/(beta1 y²) + turb uStar/(sqrt(Cmu) kappa y)
G_wall = lam G + turb (uStar g y/u+)²/(nu kappa y+_k)
```

默认 β1=0.075、Cmu=0.09、κ=0.41、E=9.8，均可显式传入。`y_plus_k` 是此 ω 条件的 k 定义坐标，不得当作 Spalding 应力定义的 y+ 输出。k=0 采用解析黏性极限，避免字面计算中的 0/log(0)；不裁剪负 k 或非有限输入。非零梯度下 E·y+_k=1 的对数极点明确报错，不任意加小量掩盖。

本次源码审查确认：同一单元连接多个壁面时，Foundation 12 按**面面积**加权，不能按面数量平均。`area_average_wall_cells` 返回排序的单元索引、面积平均 ω、G 和总壁面面积。将多边形面拆成三角形不应改变结果。当前仅支持普通共形网格的完整壁面覆盖，不实现部分覆盖/非共形单元分数。

## 已执行验证

8 项测试通过，日志见 `tests.log`：

- 四档 k 的 ω/G 与 60 位 Decimal 独立求值对照，包含黏性、混合和湍流区。
- k=0、零梯度、广播、非法输入及对数极点。
- 多壁面的面积平均和面拆分不变性。
- 将计算所得壁邻 ω 通过已有 `omega_constraints` 真正送入原生混合 SST 输运，验证目标值、非零约束反力、局部/全局通量及工作收支。该测试保留约束反力，未删除受约束单元的原方程账本。

最后一项没有安装 G 替换或 Spalding 动量牵引，不能称为完整壁面模型试验。测试脚本和涉及的原生源码快照已保存。

复现：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest tests/test_native_sst_wall3d.py -q
```

## 后续接入要求

1. 壁面速度采样和 Spalding 牵引必须进入实际动量方程及力/工作账本。
2. k 壁面零通量、壁邻 ω 约束及面积平均 G 必须在每次迭代和当前残差审计中使用同一状态。
3. 核对当前 SST-2003 与 Foundation 12 模型内部生产限制的区别，再连接壁面 G；不能直接替换全部 ω 生产项。
4. 通过无体力剪切流及有参考数据的湍流通道后，再运行匹配高 Re、网格与边界的 SUBOFF。

低 Re 原生 SUBOFF 已在[第 47 步](../../../verification-native-suboff-sst-startup/step47-steady-ledgers/README.md)通过当前稳态收支，仍未完成高 Re 实验精度验收。
