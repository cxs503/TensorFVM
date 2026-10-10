# 共性 FVM benchmark 与论文报告流程

## 首批案例

- 平行板 Poiseuille：固壁稳态、发展段后的速度、压力梯度/曲线、壁面剪切和力平衡。
- 周期 Taylor–Green：非定常面动量、连续解析速度/压力、空间与时间加密、每步质量/动量/能量。
- TensorLBM D2Q9 BGK 对照：共同周期问题、参数和终止时间，网格32/64及Mach减半；调用外部生产共性模块。
- 两个常规失败对照，以及另行保存的128网格非线性发散。成功不能覆盖失败证据。

正式FVM运行全部物理误差严格小于3%，还需通过适用的残差、质量和能量门。不认证圆柱/翼型、自由液面、移动体或破冰。

## 安装和执行

~~~bash
python -m pip install -e '.[benchmark,dev]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tensorfvm-benchmark   --output docs/verification-benchmarks   --lbm-repo /absolute/path/to/TensorLBM
python -m tensorfvm.verification.audit --directory docs/verification-benchmarks
python -m pytest -q tests/test_verification_workflow.py
~~~

也可用 PYTHONPATH=src python -m tensorfvm.verification。输出目录必须为空；每次新运行使用新目录保留旧结果。--max-channel-iterations 1 用于非收敛检查，返回非零。未指定 --lbm-repo 时只运行FVM，不伪造LBM。

重放128网格失败：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src   python scripts/reproduce_periodic_mac_failure.py --output /tmp/fvm-n128-failure
~~~

## 模块职责

| 模块 | 共性职责 |
|---|---|
| verification/core.py | 固定3%门、相对误差、JSON/NPZ/CSV导出、哈希、来源、观察阶 |
| verification/cases.py | 案例适配：配置实际求解器，提供解析参考和原场 |
| verification/lbm.py | 加载指定外部TensorLBM，验证实际来源、单位映射和共同参数 |
| verification/report.py | 压力/速度云图、曲线、收敛图、对照图，统一报告 |
| verification/audit.py | NumPy重建参考及完整MAC/BGK步骤；检查门和来源 |
| verification/__main__.py | 空间/时间/失败对照矩阵、统一执行与验收 |

新案例提供 config/metrics/history/fields/reference，通过统一 save_run 和验收。不得在报告脚本复制碰撞、压力或动量求解器。区域、单位、范数、规范和时钟须明确。参考为零时采用物理尺度绝对门，不能除以零或筛选少数非零点。

## 输出与论文材料

- report.md：中文完整报告，问题、控制方程、方法、使用、结果、讨论、参考依据和LBM对照。
- report.pdf：英文论文结构稿与图版，包含对照和局限。
- report.tex：可编辑LaTeX稿；从输出目录运行 pdflatex report.tex 编译，需本地TeX环境。
- report.html：本地可打印报告，无远程资源依赖。
- PNG（300 dpi）、PDF、SVG：带SI坐标、色标和说明的独立图。
- CSV：压力和速度剖面数值。
- fields.npz、history、result、summary、audit、manifest：复算证据。

中心展示速度不代替权威面速度。Poiseuille只验证发展段后的指定区域；Taylor–Green速度用步末、压力用midpoint。图由原始计算场生成，解析值只用于比较。

## 对照与优势

报告展示FVM实际速度精度、不可压压力约束、相容面质量、逐步能量账本及主状态存储结构。LBM压力更准时如实记录。单次CPU时间、不同z维度和步数不能证明FVM更快；主数组字节不是峰值内存。保留LBM局部更新和GPU融合优势，后续以同精度、同维度的总成本对比。

## 新增案例入口

- 三维 ABC/Beltrami 与带平均流剪切波：
  `python -m tensorfvm.verification.extended --output results/extended --lbm-repo ../TensorLBM`
- DFG Re20 的完整证据与严格 Cd/Cl/压力差门：
  `python -m tensorfvm.verification.dfg --output results/dfg --grids 64x24 128x48 256x96`
- 从已验算原场生成新增案例的详细表格与同网格 LBM 曲线：
  `python scripts/summarize_extended_benchmarks.py --directory results/extended`

以上扩展共用 `verification/core.py` 的指标门、原场导出、来源和哈希，
以及 `verification/report.py` 的论文绘图输出；实际方程仍调用现有生产求解器。
`extended_audit.py` 独立重建每个 MAC/BGK 接受步；圆柱侧审计限定为保存的
壁面压力/梯度牵引积分和权威面质量通量，没有声称独立复算全部 SIMPLE。
零参考压力采用事先声明的动态尺度绝对误差。三维 ABC 不与二维 D2Q9 比较。

当前已交付范围与失败案例见 [benchmark 总览](benchmarks-index.md)。
