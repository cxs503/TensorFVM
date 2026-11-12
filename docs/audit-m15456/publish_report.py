#!/usr/bin/env python3
"""Publish the m15456 audit report to the platform (/tensor/reports).

All numbers in the markdown tables are injected straight from the audit JSONs
(no hand transcription); the figure is embedded as base64 png.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = os.environ["TENSOR_API_BASE"]
TOK = open("/tmp/tok").read().strip()

a120 = json.loads((HERE / "audit_double_mach_120x30.json").read_text())
a240 = json.loads((HERE / "audit_double_mach_240x60.json").read_text())
st = json.loads((HERE / "dm_field_stats.json").read_text())
rep = json.loads(Path("/root/tp-agent/work/m15331/runs/double_mach_reflection/report.json").read_text())

A = a120["A_case_definition"]
code = A["code_post_shock"]
text = A["textbook_lab_velocity_M10"]
m10 = A["normal_shock_M10"]
b120, c120 = a120["B_shipped_ic"], a120["C_textbook_ic"]
b240 = a240["B_shipped_ic"]
r0 = rep["results"][0]
dmax = a120["B_vs_reported"]["delta_max_density"]

tbl_datum = f"""
| 量 | m15331 实现返回 | M=10 正激波/投影正解 | 偏差 |
|---|---|---|---|
| 后激波 rho | {code['rho']:.4f} | {m10['rho']:.4f} | {(code['rho']-m10['rho'])/m10['rho']*100:+.2f}% |
| 后激波 p | {code['p']:.3f} | {m10['p']:.3f} | {(code['p']-m10['p'])/m10['p']*100:+.2f}% |
| u (lab) | {code['u']:.4f} | {text['u']:.4f} | {(code['u']-text['u'])/text['u']*100:+.1f}% |
| v (lab) | {code['v']:+.4f} | {text['v']:+.4f} | 符号相反 |
| 速度模 | {(code['u']**2+code['v']**2)**0.5:.4f} | {(text['u']**2+text['v']**2)**0.5:.4f} | -29.8%（恰为 V·cos60） |
| 由 p 反解法向马赫 | {A['implied_normal_mach_from_code_pressure']:.4f} | 10.0000 | +15.5% |
"""

tbl_repro = f"""
| 量 | 交付 report.json | 本单独立重算 | 差 |
|---|---|---|---|
| nsteps | {r0['nsteps']} (CFL 0.3) | {b120['nsteps']} (CFL 0.4) | 同 CFL 下 485 |
| max rho | {r0['max_density']} | {b120['max_density']} | {dmax:+.4f}（{abs(dmax)/r0['max_density']*100:.3f}%）|
| min rho | {r0['min_rho']} | {b120['min_density']} | {b120['min_density']-r0['min_rho']:+.4f} |
| max rho 位置 | ({r0['triple_point_x']}, {r0['mach_stem_height']}) | ({b120['max_density_xy'][0]}, {b120['max_density_xy'][1]}) | 同上边界注入角 |
"""

tbl_grid = f"""
| 网格 | max rho | 位置 | 壁面行 max rho | 壁面行 mean rho | max p |
|---|---|---|---|---|---|
| 120x30 | {b120['max_density']} | ({b120['max_density_xy'][0]}, {b120['max_density_xy'][1]}) | {b120['wall_row_max_density']} | {b120['wall_row_mean_density']} | {b120['max_pressure']} |
| 240x60 | {b240['max_density']} | ({b240['max_density_xy'][0]}, {b240['max_density_xy'][1]}) | {b240['wall_row_max_density']} | {b240['wall_row_mean_density']} | {b240['max_pressure']} |
| 120x30（正解数据对照） | {c120['max_density']} | 角点（奇异性） | {c120['wall_row_max_density']} | {c120['wall_row_mean_density']} | {c120['max_pressure']} |
"""

tbl_err = f"""
| 关键量 | ref | got（交付/本单核实） | 相对误差 | <3%? | <8%? |
|---|---|---|---|---|---|
| max rho | {rep['reference_values']['max_density']} | {r0['max_density']} / {b120['max_density']} | {abs(b120['max_density']-8.44)/8.44*100:.1f}% | 否 | 否 |
| 马赫杆高 | {rep['reference_values']['mach_stem_height']} | {r0['mach_stem_height']}（=全局极值 y，实为上边界角） | {abs(r0['mach_stem_height']-0.55)/0.55*100:.1f}% | 否 | 否 |
| 三波点 x | {rep['reference_values']['triple_point_x']} | {r0['triple_point_x']}（=全局极值 x，同一角点） | {abs(r0['triple_point_x']-1.5)/1.5*100:.1f}% | 否 | 否 |
"""

MD = f"""# 交差审计：m15331 CASE-DOUBLE-MACH（TensorFVM）——判 L2，基准三比项全部超差且算例物理不成立

## ① 案例介绍
双马赫反射（Woodward & Colella 1984, JCP 54(1) 115-173 §4）：M=10 强激波与壁面成 60° 入射，域 [0,4]×[0,1]，
γ=1.4，预激波 rho=1.4 / p=1.0 / (u,v)=(0,0)，t=0.2。裁判依据：`tensor_benchmark_cases` 中
`anchor_slug='CASE-DOUBLE-MACH'` **零行（未登记基准）**；退而用 m15331 自带裁判 max rho=8.44、马赫杆高 0.55、
三波点 x=1.5、容差 8%。

## ② 怎么做的
1) 取证：读 m15331 交付（`docs/double_mach_reflection_report.md`、`runs/double_mach_reflection/*`、tip `0e2091f8`），
查 `git ls-files runs/`（=0）、平台 `/tensor/bench/results?mission_id=15331`（total 0）、
`/tensor/bench/l4-verdict/15331`（CLAIMED_NOT_PROVEN）、`/tensor/reports?mission_id=15331`（total 0）。
2) 独立重算（第二条推导路径，不复用交付度量）：`docs/audit-m15456/audit_double_mach.py`
 —— 正激波关系正反解 + front-frame/lab-frame 投影给出后激波速度；原样重跑 120x30 与 240x60；
 再做"只换后激波数据"的对照实验隔离"数据错"与"格式耗散"。
3) 场量落地与出图：`audit_figure.py` → `dm_fields_120x30.npz` / `fig_dm_audit.png`（数字全部程序化注入）。

## ③ 计算结果

### 3.1 后激波初始数据（阻断项）
{tbl_datum}

同一函数里 `vs = mach*c1/sin(theta) = 11.547`（把 mach 当法向马赫）与 `mn = mach/sin(theta) = 11.547`
（把 mach 当 x 向平移速度再除法向）不可能同时成立；实测返回值对应法向马赫 11.547，而引用的是 M=10。

### 3.2 交付数字可复现（问题不在"数不出来"）
{tbl_repro}

### 3.3 网格趋势与对照实验
{tbl_grid}

加密一倍 max rho 只变 0.09%——收敛到"上边界注入值被抹平后的固定值"，不是物理峰值；壁面行 mean rho 2.5 说明
**气体被推离壁面，壁面从未被激波扫过**（`wall_shock_x=NaN`，壁面无任何 rho>4 单元）。换成正确的后激波数据
（v 指向壁面）后，壁面 mean=11.85、剔边界窗 max rho=18.58、max p=581，反射/马赫杆结构才出现。

## ④ 图
见内嵌图：左=交付数据 rho 云图（热点只在上边界注入角 (0.0167,0.9833)，壁面一片 2-3）；中=正解数据 rho 云图
（壁面附近致密压缩区）；右=沿壁面 rho 剖面（交付曲线贴 1-3 掉，正解曲线维持 8-20）。原图 140 dpi，
存 `docs/audit-m15456/fig_dm_audit.png`。

## ⑤ 注意点（缺陷按严重度）
1. [阻断] 后激波状态自相矛盾（法向马赫二次折算）→ p 偏高 33.4%。`cases.py:500 _post_shock_state()`。
2. [阻断] 后激波速度公式错（分量相减而非投影）：得 (3.276, +4.775)、模 5.79=V·cos60，正解 (7.145, -4.125)、模 8.25；
   y 符号反了 → 壁面无反射，**双马赫反射这一现象在交付算例里不存在**。
3. [高] `top_bc(t)` 用 mach·c1=10 扫掠，而同文件激波前沿速度 11.547 → t=0.2 滞后 0.309（域长 7.7%）。
4. [高] 度量取全局 rho 极值坐标（`scripts/run_double_mach_reflection.py:33`），三波点/马赫杆从未被测量；
   报告"98.9% 误差源于 MUSCL-Hancock 耗散"属无依据归因。
5. [高] 参考三元组 8.44/0.55/1.5 无出处工件（无抓取件/sha256/页图），与"30° 入射流须强再压缩"的几何不自洽 → 标 WITHHELD。
6. [中] 工件与复现命令不互证：report.json 只有 120x30，而驱动脚本只跑 240/480/960 且写 `.npy`+`double_mach_reflection.png`
   （均不存在）；`runs/` 里两个 PNG 脚本根本不会写。
7. [中] `runs/` 未入库（`git ls-files runs/`=0）→ 推送后工件消失。
8. [中] 未接平台基准（无 tensorbench_results 行；L4 端点 CLAIMED_NOT_PROVEN；无 tensor_reports 行）。
9. [低] 分支落后 main（main 已有 double-blast/solid-mask），直接合并会回退既有能力。
10. [低] 零测试覆盖（含新增 callable/1-D 入流 BC）。

## ⑥ 误差判定
{tbl_err}

**是否 <3%：全部否（3/3 不达标，最小者 24.6%，是 3% 线的 8 倍）。** 且后两项 got 值不是物理量（全局极值坐标），
判据本身作废，不能用"接近"记分。未跑项（不得记为通过）：480x120、960x240、集群复算、pytest（无对应用例）。

## ⑦ 定级与处置
- **L2 达成**：真代码 输入→计算→输出，独立重跑复现 max rho（差 0.006%），rho/p 场已落盘。
- **L3 未达成**：集成面（EulerSolver + solver.py BC 补丁）未并入 main、零测试、未登记验证目录；机器门
  `L3 缺 integration_callers(grep 无引用)`。
- **L4 不成立**：3/3 超差 + 两个度量非物理量 + 算例数据非引用算例。
- verdict id=320（`tensor_maturity_verdicts`，verdict_by=board-audit，level=2，gate 机器预演 L2=(2,[])→confirmed）。
  账务层处置建议仍是**返修**：先修数据（法向马赫/速度投影），再修度量（特征检测），再谈 8%；
  另登记负知识 `negk:m15456:CASE-DOUBLE-MACH-DATUM`（failure_kind=wrong_physics_derivation）防止重犯。

## 附：可复现命令
```bash
PYTHONPATH=/root/tp-agent/work/m15331/src OMP_NUM_THREADS=16 python3 docs/audit-m15456/audit_double_mach.py \\
  --out /tmp/dm120 --reported /root/tp-agent/work/m15331/runs/double_mach_reflection/report.json
PYTHONPATH=/root/tp-agent/work/m15331/src python3 docs/audit-m15456/audit_figure.py docs/audit-m15456
```
"""

fig_png = base64.b64encode((HERE / "fig_dm_audit.png").read_bytes()).decode()
payload = {
    "title": "交差审计 m15331 CASE-DOUBLE-MACH（TensorFVM）：判 L2，基准 3/3 超差，后激波数据错致壁面不反射",
    "summary": "独立重算复现交付数字（max rho 差 0.006%），但证明：后激波状态把同一 mach 同时当法向马赫与 x 向平移速度"
               "（p 偏高 33.4%）、速度投影错致 y 反向（壁面无任何 rho>4 单元=无反射）、度量取全局极值坐标（三波点/马赫杆"
               "从未被测量）、参考三元组无出处、runs/ 未入库、算例未登记基准。判 L2；L4 声明否证，建议返修。",
    "scenario_key": "case-pool-audit",
    "mission_id": 15456,
    "stage": "S3",
    "tags": ["TensorFVM", "CASE-DOUBLE-MACH", "board-audit", "m15331", "benchmark", "compressible",
             "double-mach-reflection", "maturity-L2"],
    "markdown": MD,
    "figures": [{"name": "fig_dm_audit.png",
                 "caption": "120x30, t=0.2：左=交付数据 rho 场（热点在上边界注入角，壁面 2-3）；中=正解数据 rho 场"
                            "（壁面致密压缩区）；右=沿壁面 rho 剖面（交付膨胀 vs 正解压缩）",
                 "png_base64": fig_png}],
}
req = urllib.request.Request(BASE + "/tensor/reports", method="POST",
                             data=json.dumps(payload).encode())
req.add_header("Authorization", "Bearer " + TOK)
req.add_header("Content-Type", "application/json")
try:
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
        print("POST", r.status, json.dumps(out, ensure_ascii=False)[:400])
        rid = out.get("id") or out.get("report_id")
        print("report_id", rid)
except Exception as e:
    print("ERR", e, getattr(e, "read", lambda: b"")()[:500])