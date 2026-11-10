# README 一致性巡检（自动生成）

抽查项 20：PASS 20，FAIL 0，SKIP 0。

| # | 面 | 声明 | 状态 | 实测证据 |
|---|---|---|---|---|
| A1 | 示例命令/选项 | README 每个示例命令的选项都被目标程序接受 | PASS | 15 条命令、36 个选项 token；全部可解析 |
| A2 | 示例命令/选项 | README 中的 `python -m tensorfvm.*` 模块可执行 | PASS | 8 个模块；全部存在且可执行 |
| A3 | 控制台入口 | pyproject [project.scripts] 的条目都指向真实模块函数 | PASS | 9 个入口；全部可解析 |
| A4 | 示例命令/选项 | README 引用的 scripts/*.py 存在 | PASS | 6 个脚本指针；全部存在 |
| B1 | 公开 API | README 列出的导出名可由 `import tensorfvm` 取得 | PASS | 6 个名称；全部可导入 |
| B2 | 配置默认值 | `SolverConfig` 默认 mesh_type=cartesian（README 兼容性说明） | PASS | 实测 mesh_type='cartesian' |
| B3 | 配置默认值 | README 默认算例（4×2 域、中心 (1,1)、R=0.2、U=1、rho=1、Re=20、mu=rho U 2R/Re） | PASS | mu=0.02 = rho*U*(2R)/Re，与 8 个默认字段一致 |
| B4 | 网格约束 | README 的贴体/C 型网格约束（body-fitted nx≥8 且 4 的倍数、ny≥4；c-grid nx≥16 且 4 的倍数） | PASS | 5 个边界用例逐一符合 |
| B5 | 输出文件 | README 列的运行输出（fields.csv 含 x,y,u,v,p,fluid 等 10 个文件）由导出处写出 | PASS | 10 个文件名全部出现在 cylinder.py；fields.csv 表头 x,y,u,v,p,fluid |
| C1 | 基准常数 | DFG Re=20：Cd=5.579535、最细网格 256×96、误差门 3% | PASS | REFERENCE_DRAG=5.579535, GRIDS=((64, 24), (128, 48), (256, 96)), ERROR_LIMIT=0.03 |
| C2 | 基准常数 | NACA 0012 Re=1000 α=4°：Cl≈0.2009636、Cd≈0.1250065、读图 ±0.0001、3% 门 | PASS | lift=0.2009636, drag=0.1250065, band=0.0001 |
| C3 | 基准常数 | 平板：Cf=0.074 Re^-1/5、Re_L=1e5、平均 Cf 误差门 20%、首层 y+<1 | PASS | ERROR_LIMIT=0.2, Re=100000, Cf(1e5)=0.0074 |
| C4 | 基准常数 | Re=3900 外流圆柱：参考 Cd=1.12、St=0.20 | PASS | mean_drag=1.12, strouhal=0.2 |
| C5 | 基准常数 | 30P30N：原始表 Cl=2.167089、Cd=0.033243 | PASS | lift=2.167089, drag=0.033243 |
| D1 | 文档指针 | README 引用的 docs/* 路径在检出中存在 | PASS | 38 个指针；全部存在 |
| D2 | 数字可溯源 | README 引用的实测数字都能在 docs/ 证据中找到同一数值 | PASS | 41 个三位以上有效数字；全部可溯源到 docs/ |
| D3 | 测试入口 | README 记的测试运行方式与本仓库测试语料一致 | PASS | tests/ 共 233 个文件：176 个 import pytest、26 个定义 unittest.TestCase、1399 个模块级 test 函数；README 当前写的是 pytest |
| D4 | 前置依赖 | README 文档化的 benchmark/verification 命令所需可选依赖有安装说明 | PASS | verification/report.py 需要 ['matplotlib', 'numpy']；pyproject extras=['amg', 'benchmark', 'dev', 'mesh']；安装节已说明 extras 安装方式 |
| E1 | 证据树只读 | 文档化的示例命令默认输出不落在已发布的 docs/ 证据树内 | PASS | benchmark_suite_channel 默认 --output='results/suite-channel'（results/ 下，与其它 benchmark 一致） |
| E2 | 示例可运行 | README 的贴体圆柱示例按文档参数小规模重跑，产出全部声明文件 | PASS | exit=2（0=收敛 2=达迭代上限）；产出 ['cells.csv', 'fields.csv', 'history.json', 'nodes.csv', 'pressure.svg', 'summary.json', 'velocity.svg']；表头 'x,y,u,v,p,fluid' |
