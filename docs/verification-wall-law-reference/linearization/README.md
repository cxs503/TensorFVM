# Spalding 牵引线性化与混合求解器接入审查

2026-10-13。此项是壁面本构模块验证，**不是 SUBOFF 壁面函数已接入或物理精度已达标**。

## 本构导数

已有 Spalding 关系为 y+ = f(u+)，其中 u+=U/u_tau、y+=y*u_tau/nu。固定采样距离、分子黏度时，隐式微分得到：

```
du_tau/dU = f'(u+) / [y+ + u+ f'(u+)]
beta = rho*u_tau^2/U
beta_radial = 2*rho*u_tau*du_tau/dU
J_traction = -beta*I - (beta_radial-beta)*e*e^T
```

这里 e 为切向速度方向。零速度极限严格为 J=-mu/y*I，直接实现解析极限，不使用速度截断。新增 `wall_traction_linearization` 返回牵引、摩擦速度、y+、割线系数和解析雅可比。壁面法向投影及速度采样映射必须在边界装配层明确组合。

6 项独立方向差分/极限/旋转与批量广播检查通过，覆盖 y+=0.001、1、30、300、3000；验证牵引耗散、雅可比对称负定、零速度极限及坐标旋转一致性。与既有工程测试合计 12 项通过；随后重新检查批量广播的 6 项与前组重叠，不能相加。代码快照、源码哈希及测试日志随本页保存。工程回归中的旧 CSR CUDA 测试不代表当前混合 SUBOFF 已获 GPU 验证。

## 与本机 OpenFOAM Foundation 12 的实际差异

核查本机 Foundation 12（20250206）官方源代码；文件路径与 SHA256 见 sources.json。对应公开源码：[Spalding](https://github.com/OpenFOAM/OpenFOAM-12/blob/master/src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutUSpaldingWallFunction/nutUSpaldingWallFunctionFvPatchScalarField.C)、[omega 壁面函数](https://github.com/OpenFOAM/OpenFOAM-12/blob/master/src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/omegaWallFunctions/omegaWallFunction/omegaWallFunctionFvPatchScalarField.C)。本页依据本机文件，不能推定所有 OpenFOAM 分支/版本完全相同。

| 接口 | 当前混合求解器 | Foundation 12 的相关行为 |
|---|---|---|
| Spalding 速度采样 | 本构模块接受已投影的切向相对速度；尚未装配为混合壁面模型 | 使用壁邻内部速度与壁面速度之差，结合壁面距离与法向速度梯度 |
| 动量壁面应力 | 精确法向约束与常数 Robin 已有；当前 SUBOFF 用无滑移低 Re 处理 | 通过壁面 nut、梯度与摩擦速度共同确定应力 |
| omega 位置 | 当前低 Re 条件在共享壁面设 60nu/(beta1*y1²) | 壁邻单元约束、边界取内部场，需处理约束反力 |
| blended omega | 混合求解器尚未实现 | 权重 lam=exp(-Re_y/11)，omega=lam*omega_vis+(1-lam)*omega_log；还混合剪切/湍动能定义 uStar，并更新近壁 G |
| 旧 TensorFVM 壁面函数 | 位于另一个 collocated 后端；不能直接标为当前混合后端能力 | 旧项目 sqrt(omega_vis²+omega_log²) 和自选 G 混合不等于本机 Foundation 12 的 blended 公式 |

尤其不能将壁面 omega 的系数从 60 改成 6 后就声称等价：未知量位置、约束方程和近壁生产项也不同。

## 必须完成的接入验证

1. 在当前混合未知量上明确采样速度位置、切向投影和应力装配；报告力与动量方程使用同一个壁面牵引。已有常数 Robin 的通过结果不能代替 Spalding。
2. 将解析雅可比与实际采样映射组合；验证法向零通量、旋转一致性、局部动量/工作收支，以及零速黏性极限。
3. 为 k 零通量、omega 壁邻单元约束及其反力建立完整标量账本；按选定的 Foundation 12 版本处理近壁 G 和混合规则。
4. 先用无体力剪切流与有参考数据的湍流通道检查壁面应力和 y+，然后在实际 SUBOFF 上做匹配网格/工况的 Cp/Cd 对照。仍须完成当前无壁面函数的原生 SST 收敛验证。

本页不改变在运行的 SUBOFF 方程、常数或验收门槛。
