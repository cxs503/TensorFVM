import base64
import json
import os
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
u, pw = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=180) as r:
        return json.loads(r.read() or b"{}")


tok = api("POST", "/auth/login", {"username": u, "password": pw})["access_token"]

D = "docs/board_audit/m16176"
figs = []
for name, cap in [
    ("convergence.png",
     "Re=1 网格阶梯（从已入库 npz 重建）：速度 L∞/L2 相对误差 vs dx，含 1% 契约门线与 3% 可接受线；实测阶 1.1688"),
    ("re1-slice-n6.png",
     "Re=1, n=6, x≈0 细胞面切片：精确 u_x / 计算 u_x / 误差场（按入库工件重绘；原报告 figures[] 只有标题无载荷）"),
]:
    b = base64.b64encode(open(os.path.join(D, "figures", name), "rb").read()).decode()
    figs.append({"name": name, "caption": cap, "png_base64": b})

md = """## 1 案例介绍
被核对象：mission **m16077**（TensorFVM，succeeded，PR #473 → merge `e6e254fd`，分支头 `0ee118f6`），锚点
**CASE-FVM-ETHIER-STEINMAN-3D**。物理背景：三维制造解（manufactured solution）Navier–Stokes 验证——用一个精确无外力解
`u_i = -a[e^{a x_i} sin(a(x_j+x_k)) + e^{a x_k} cos(a(x_i+x_j))] e^{-d^2 t}`（(i,j,k) 循环，ρ=1，无外力当且仅当 ν=d²/a²），
在 TensorFVM 原生混合 3D 求解器 `NativeHybridNavierStokes3D`（稳态，故按冻结时刻 t=0.25 s 把 du/dt 移入体力）上求解并与精确场比对。
域 [-1,1]³，六面全 Dirichlet（精确面积平均速度），convection order 2，Newton 容差 1e-10。
参数：re1（a=1,d=1,ν=1,Re=1）、re100（a=1,d=0.1,ν=0.01,Re=100）、re100_resolved（a=0.2,d=0.02,ν=0.01,Re=100）；网格 n³，n=4/6。
契约判据（唯一裁判）：速度 L∞ 相对误差 **<1%**（Re=1 与 Re=100）。
**本单是交差核验（非重做）**：核三件套真伪、级别判定、落 `tensor_maturity_verdicts`（verdict id 432）。

## 2 怎么做的
1. 取证：读 m16077 的 `scripts/ethier-steinman-3d/{RESULTS.md,NOTES.md,study.json,artifacts/*.npz}`、`src/tensorfvm/verification/ethier_steinman.py`、`tests/test_ethier_steinman.py`；查基准注册表、`tensor_reports` id 15224、`tensor_dev_missions` goal。
2. 工件重算（审计侧独立脚本，不 import tensorfvm）：从 npz 的 `velocity/reference_velocity/errors/cell_volumes` 重算 L∞/L2 相对误差与体积和。
3. 参考独立复核：照报告印刷公式重写场，随机 12 内点中心差分核 div u、`Δu+a²u`、动量残差旋度（正确 ν=d²/a² vs 错误 ν=1 作负对照）。
4. 复现核验：跑 `run_study.py --n-list 4 --cases re1` 比对是否逐位相同；跑全套 `pytest tests/test_ethier_steinman.py`。
5. 级别判定按 L0–L5 的**功能实质**（不按报告篇幅），并用平台 `maturity_gate.judge()` 预演机器复核。
6. 落表 tensor_maturity_verdicts（repo=TensorFVM, scope='m16077 CASE-FVM-ETHIER-STEINMAN-3D', mission_id=16176, level=3, verdict_by=board-audit）→ verdict id 432，gate_state pending；预演 judge 返回 (3, [])。
7. 补图：原报告 `figures[]` 只有 caption 无 png 载荷、`figures/*.png` 未入 merge，故从已入库 npz 重建两张图（本报告内嵌）。

## 3 计算结果（关键数字，全部程序化取自工件/本审计复算）
| 核验项 | 结果 | 出处 |
|---|---|---|
| 工件真存在 | re1-n4.npz 5187 B、re1-n6.npz 14839 B、study.json 5714 B 随 merge 入库 | `refs/SHA256SUMS.txt` |
| 工件数值 vs 报告 | n=4 **0.0623793291966362**、n=6 **0.03883591044093914**（重算逐位相同） | `refs/re1-n*.npz` |
| repro_cmd | 复跑 re1 n=4 → 0.0623793291966362（**逐位相同**，确定性） | 实跑输出 |
| 测试 | `pytest tests/test_ethier_steinman.py -q` → **10 passed / 32.9 s** | 实跑 |
| 参考独立复核（re100） | ‖div u‖∞=8.3e-11；‖Δu+a²u‖∞=3.7e-8；‖curl R‖∞(ν=d²/a²)=**1.81e-6** vs (ν=1)=**2.32**（比 1.3e6） | `independent_reference_check.json` |
| 基准表注册 | 该锚点在基准注册表 **0 行**（全表 25 行无 ethier/steinman） | SQL |
| 报告六段 | 齐（含⑥显式回答「是否<3%」）；但 `figures[]` 无 png 载荷 | `tensor_reports` 15224 |
| verdict | id 432，level **3**，gate_state pending（judge 预演 (3,[])） | `tensor_maturity_verdicts` |

## 4 结果图
图 A：网格阶梯误差 vs dx（含 1% 契约线与 3% 可接受线，实测阶 1.1688）；图 B：n=6 x≈0 切片精确/计算/误差 u_x。
两图均由**已入库 npz** 重建——既补上原报告缺图，也证明工件可读、可重算。

## 5 注意点（踩过的坑 / 局限 / 适用范围）
- **F1 门未过**：Re=1 速度 L∞ 相对误差 6.238%(n=4)→3.884%(n=6)，实测阶 1.1688（单步）。连 3% 线也未达。
- **F2 "完整算例"不成立**：Re=100 两参数化在 n=4/6 全部 Newton `iteration_limit`（无一解），RESULTS 表对应行 dx/误差为 `n/a`——**无测量**不是测量失败。
- **F3 测试阈值比契约松 10 倍**：`tests/test_ethier_steinman.py:129-135` 断言 `<0.10`/`<0.08`，在 6.2% 时仍绿灯 ⇒ 测试绿 ≠ 算例过。
- **F4 基准注册表缺该锚点**：metric/tolerance 只活在 mission goal 文本，未来轮次无法机器判分。
- **F5 报告图"有名无图"**：report 15224 的 `figures[]` 只有 name+caption；`figures/*.png` 未进 merge `e6e254fd`（已由本审计重建）。
- **F6 文档数值不一致**：NOTES.md 写阶 1.14，RESULTS.md 写 1.17，工件算得 **1.1688**（NOTES 陈旧）；「需 n≈24 到 1%」在 p=1.1688 下偏保守（应为 n≈19.2），但仍未实跑。
- **F7（平台侧）maturity_gate L4 判据只查有无数字不查过没过**：`judge(4, 本单 evidence, …)` 实返 `(4, [])`，尽管 benchmark 记录 FAIL ⇒ 失败数字塞进 benchmark 键即可被机器追认 L4。本审计**不利用**该漏洞（声明级 3）。
- 局限：未访问付费原文（DOI 10.1002/fld.1650190502，仍 WITHHELD）；未独立验证 order-8 Duffy 面积平均；未亲手重跑 re100 失败工况（引用 study.json 记录）；未跑 n=24 外推点。

## 6 误差判定（逐项，显式回答）
| 关键量 | 基准/契约 | got | 是否 <3% | 是否 <1% |
|---|---|---|---|---|
| 速度 L∞ 相对误差 @Re=1, n=4 | <1% | 6.238e-02 | ❌ | ❌ |
| 速度 L∞ 相对误差 @Re=1, n=6 | <1% | 3.884e-02 | ❌ | ❌ |
| 速度 L∞ 相对误差 @Re=100 | <1% | 未收敛（无解） | ❌（无法测量） | ❌ |
| 离散散度 max|div u| | <1e-12（次要门） | 0.0 | ✅ | ✅ |
| 参考自审：div u / ‖Δu+a²u‖ / 残差旋度 | 机器零 | 1.1e-16 / 0（解析） | —（参考自洽） | — |
| 本审计独立复核：‖div u‖∞ / ‖Δu+a²u‖∞ | — | ≤8.3e-11 / ≤4.4e-8 | ✅ | ✅ |

**结论：不满足 <3%，更不满足 <1% ⇒ 门 FAIL（honest-FAIL）。** 但作为**验证设施**交付真实：真代码 + 真调用主求解器 + 真工件 + 确定性复现 + 参考自身可证 ⇒ 判 **L3（系统集成）**，L4（基准验证）**拒绝**，L5 无从谈起（无文献锚）。
处置：**approve with notes**——m16077 按 L3 消账，不追认 L4；返修顺序 F4（补注册表行）→ F3（测试阈值对齐契约）→ Re=100 换瞬态/稳定化 → Re=1 n=24..32 上集群。
"""

payload = {
    "title": "交差核验 m16077 CASE-FVM-ETHIER-STEINMAN-3D：L3 判定（基准门 honest-FAIL 未过）",
    "summary": ("核验 m16077 三件套（工件/ref-vs-got/repro）真伪并判定成熟度：工件数值逐位可复算、10 测试通过、"
                "参考经本审计独立复核为精确无外力 NS 解；但契约 <1% 未达（6.238%/3.884%），Re=100 两工况无解 ⇒ 判 L3，L4 拒绝。"
                "另报 7 项发现（含 maturity_gate L4 漏洞）。verdict id 432。"),
    "scenario_key": "BOARD-AUDIT-MATURITY",
    "mission_id": 16176,
    "tags": ["TensorFVM", "board-audit", "maturity-verdict", "Ethier-Steinman",
             "manufactured-solution", "honest-FAIL"],
    "markdown": md,
    "figures": figs,
}
r = api("POST", "/tensor/reports", payload, tok)
print({k: r.get(k) for k in ("id", "title", "mission_id")})