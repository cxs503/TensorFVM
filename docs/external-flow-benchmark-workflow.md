# 圆柱与翼型 benchmark 工作流

## 物理问题与当前范围

| 案例 | 网格 | 工况 | 独立物理指标 |
|---|---|---|---|
| DFG 2D-1 圆柱 | 64×24、128×48、256×96 | 二维稳态层流，Re20，通道2.2×0.41 m，D=0.1 m，入口均速0.2 m/s | Cd、Cl、前后壁面压力差 |
| NACA0012 | 64×26、128×52、256×104 | 二维稳态层流，Re1000，4°，弦长1，20×16计算域 | Cd、Cl；压力/速度场及Cp为实际结果展示 |

圆柱最细24,576控制体、73,728主要未知量；翼型最细26,624控制体、79,872主要未知量。壁面是贴体多边形弦段，翼型 `c-grid` 接口的实际拓扑是周期径向网格。

[圆柱报告](verification-external-cylinder/report.md)、[翼型报告](verification-external-naca/report.md)和原始NPZ是实际计算证据。残差收敛不代表物理达标，三网格存在不代表渐近网格收敛；每项参考误差分别严格小于3%，不能用平均误差掩盖失败。没有匹配的全场公开表时，不把云图展示当成全场误差小于3%的证明。

## 共性模块与求解后端

论文PDF排版/资源去重脚本另需已安装的 `pypdf`；数值求解入口不依赖它。

统一入口 `tensorfvm.verification.external_flow` 复用 Metric/evaluate/save_run、来源哈希与出版绘图。既有默认求解器保持其行为；本工作流使用可选后端。

- `simple`：原SIMPLE方程，CPU SciPy SuperLU；CUDA Torch Krylov。每次真实线性残差必须通过预算。
- `cached`：CPU GMRES求解当前CSR矩阵，缓存LU只作为预条件器，每20次同类线性调用刷新。迭代不满足严格预算时重新直接分解；不使用滞后的矩阵作为物理方程。CUDA路径使用已有Torch后端。
- `anderson`：复用现有8层Anderson混合，配合当前矩阵缓存LU预条件；混合后重新记录实际方程残差，另查<10⁻⁷物理通量固定点门。用于本轮最细翼型。
- `coupled`：只允许已验证的稳态层流CPU圆柱。原SIMPLE推进后执行全网格耦合Picard残差修正，包含实际周期连通、非正交扩散、压力梯度和物理Rhie–Chow。检查矩阵响应与真实线性残差，重新记录所有更新后的实际方程残差，要求最终物理通量缺陷<10⁻⁷。

翼型全耦合修正发生不稳定且被真实残差门拒绝，[失败记录](verification-external-coupled-rejected/README.md)保留；未通过的线搜索原型未发布为可用求解器。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m tensorfvm.verification.external_flow --output results/cylinder \
  --case cylinder --grids 64x24 128x48 256x96 --solver coupled --max-iterations 300

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m tensorfvm.verification.external_flow --output results/naca \
  --case naca --grids 64x26 128x52 256x104 --solver anderson --max-iterations 1500

PYTHONPATH=src python scripts/audit_external_reports.py results/cylinder
PYTHONPATH=src python scripts/audit_external_equations.py results/cylinder
PYTHONPATH=src python scripts/publish_external_manuscript.py results/cylinder
PYTHONPATH=src python scripts/publish_external_frontmatter.py results/cylinder
```

本轮翼型报告复用已实际收敛的两个CPU直接解网格，再合并缓存LU预条件与既有Anderson混合加速的实际最细网格；每个运行绑定其生产代码版本和源快照，见 `scripts/finish_external_benchmarks.py`。复现上述统一缓存后端命令时，迭代数及最后舍入位允许随线性后端预算变化，独立方程和物理门必须仍满足。

## 参考值与审计

[FeatFlow](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)提供圆柱高精度标量参考。翼型取[Di Ilio 2020](https://arxiv.org/html/2006.10487)图10/11的 present-study 矢量标记，[提取事实](naca-reference-extraction.json)和脚本保存页面/图形哈希、坐标、轴映射及±0.0001图形读取区间。验收采用区间端点最坏误差；区间不涵盖作者数值误差。旧0.205/0.120参考及历史通过结论已撤回。

独立NumPy审计从原场重新构建最小二乘梯度、完整层流动量、面质量守恒、无伪时间的物理Rhie–Chow通量、壁面无滑移梯度和压力；仅复用几何生成与给定边界数据。独立牵引积分分别重算压力/黏性受力及气动轴旋转，圆柱压力差另查周期插值。最后稳态记录必须对应所有修正后的实际场。

[CPU/CUDA五步一致性](verification-external-device-check/report.md)只证明辅助后端场一致，不证明稳态、物理误差或性能加速比。单次 `elapsed_s` 包含实际迭代和线性组装/求解，排除网格构造、导出和作图。

## 下一阶段：由易到难

1. 先定位圆柱小升力的压力/黏性分力误差、一次迎风输运和壁面压力重建误差。预定义贴壁加密和守恒二阶对流方案，再做至少三网格验证；不能根据参考值挑选有利的牵引公式。
2. 对NACA做同样的输运/壁面误差分解，并检查远场尺寸及尾流分辨率。当前两级加密明显改变升阻力，旧粗网格接近参考不能用作合格证据。
3. 通过稳态层流门后，再开发DFG Re100非稳态涡脱落，检查时间步、升阻力振幅与Strouhal数。
4. 再推进NACA更高攻角非稳态、公开湍流翼型工况与壁面函数。需要匹配Re、攻角、Mach数/不可压缩假设及参考资料，不能用高Re实验认证当前层流模型。
5. 匹配TensorLBM的实际几何、边界与物理参数后比较精度与耗时，再讨论贴体网格优势；当前没有匹配运行，不作跨软件优劣排名。

## Mandatory body-fitted mesh gate

Cylinder and NACA benchmarks require physical body-fitted quadrilateral meshes;
Cartesian obstacle masks or staircase boundaries cannot substitute for them.
The resolution denotes circumferential and radial cell counts. Both current
meshes have a periodic radial O topology. The historical NACA API name `c-grid`
does not describe a true open-wake C topology.

Run `python scripts/audit_external_mesh.py` before accepting external-flow
results. This independent NumPy audit checks the saved wall vertices against the
circle or independently generated NACA0012 profile polygon, positive signed
cell areas, saved cell volumes, periodic seam closure, physical wall face
midpoints, wall face counts and the presence of oblique faces. A failed check
raises an error requiring mesh generation repair. The checks describe polygonal
body conformity, not exact curved-face representation or sufficient mesh quality
for aerodynamic accuracy. The six current saved meshes pass these checks;
physical errors remain above the 3% acceptance gate.

Results and raw field SHA256 identifiers: [mesh audit](verification-external-mesh/audit.json).
