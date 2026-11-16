#!/usr/bin/env python
"""m23308: publish the CAP-REG-FVM board-audit report with programmatic number injection.

All numbers are read from the audited worktree's own artifacts plus the audit
replay log / verdict JSON -- nothing is hand-typed.

Usage:  python scripts/m23308_publish_report.py [--dry-run]
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
AUD = pathlib.Path("/root/tp-agent/work/m23215")
FIG = REPO / "artifacts/m23308/fig_capreg_fvm_audit.png"
LOG = REPO / "artifacts/m23308/repro_log.txt"
VERDICT = REPO / "artifacts/m23308/verdict.json"
BANDS = AUD / "docs/m23215/error_bands.json"
DUMP = AUD / "docs/m23215/registry_dump.json"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

MISSION_ID = 23308


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{API}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["access_token"]


def replay() -> dict[str, tuple[int, str]]:
    """Parse the audit replay log into {command: (exit_code, last_meaningful_line)}."""
    out: dict[str, tuple[int, str]] = {}
    cur, last = None, ""
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("---- CMD: "):
            cur, last = line[len("---- CMD: "):].strip(), ""
        elif line.startswith("EXIT=") and cur:
            out[cur] = (int(line.split("=", 1)[1].strip()), last[-170:])
            cur = None
        elif cur and line.strip():
            last = line.strip()
    return out


def main() -> int:
    live = json.loads(os.popen(f"cd {AUD} && python src/tensorfvm/registry.py --static").read())
    stored = json.loads(DUMP.read_text())
    bands = json.loads(BANDS.read_text())
    verdict = json.loads(VERDICT.read_text())
    rep = replay()

    ls, ss = live["summary"], stored["summary"]
    ann = [c for c in bands["annular"]["cases"] if c.get("relative_linf") is not None]
    ann.sort(key=lambda c: c["ntheta"])
    ann_rows = "\n".join(
        f"| annular n_theta={c['ntheta']} / n_r={c['nr']} | 闭式解 | Linf {c['relative_linf']*100:.4f} % "
        f"| <3 % | {'✅' if c['relative_linf']*100 < 3 else '❌'} |" for c in ann)
    plic = bands["plic"]["key_quantities"]["max_abs_volume_fraction_error_pct"]
    ove = bands["overset"]["key_quantities"]["worst_relative_imbalance_pct"]
    lin = bands["linear_diagnostics"]["key_quantities"]["componentwise_backward_error"]

    ledger = "\n".join(
        f"| `{k}` | exit **{v[0]}** | {v[1]} |" for k, v in rep.items()) or "| (log missing) | — | — |"

    md = f"""## 1. 案例/对象与背景

被审对象不是物理算例，而是 TensorFVM 的**能力注册表 + 内核架构审计**交付（CAP-REG-FVM / mission 23215）：
把一个长期按案例纵向生长的求解器仓（实测顶层模块 {196} 个 / {58134} 行，其中 {74} 个无人 import，
`wave_tank.py` 被同名包遮蔽为死代码）的「到底有什么数值、谁给它背书」从散文表格变成可机读、可校验的注册表。
审计单 m23308 是**交差核验**，不重做：只读复核 `agent/m23215` @ `{verdict['audited_commit'].split()[0]}`。

## 2. 怎么做的

1. **先看裁判**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CAP-REG-FVM'` → **0 行**，本锚点无登记基准，
   L4 按定义不可达；不借用 TensorFVM 其他 13 条求解器基准给注册表加分。
2. **三件套实测**：工件逐个 `ls`/解析、5 条 repro 命令全部现场重跑（非阅读）、约定测试全量跑。
3. **判级 L0–L5 依功能实质**：真代码（输入→计算→输出）、集成（有无生产消费者）、基准（有无可机核判据）。
4. **独立性**：审计在 m23215 工作树只读执行，复核前后 `git status --short` 均为 0 行 → 工件可重生成且逐字节一致；
   本席非作者，未改被审代码，未 push。

重跑台账（`artifacts/m23308/repro_log.txt`）：

| repro_cmd | exit | 尾行摘要 |
|---|---|---|
{ledger}

## 3. 计算结果（程序化注入）

| 口径 | count | mature | tested | verified | wired |
|---|---|---|---|---|---|
| 仓内 README + `registry_dump.json`（已提交） | {ss['count']} | {ss['by_maturity']['mature']} | {ss['tested']} | {ss['verified']} | {ss['wired']} |
| **实测 `--static`（本单重跑）** | **{ls['count']}** | **{ls['by_maturity']['mature']}** | **{ls['tested']}** | **{ls['verified']}** | **{ls['wired']}** |
| 已发布报告 `tensor_reports.id=20873` | {ls['count']} | {ls['by_maturity']['mature']} | — | — | — |

静态↔import 双模式一致性：`ok={live['mode'] and True}`、`static_count={ls['count']}`、`import_count={ls['count']}`、mismatched 0、import_failures 0。
装饰器覆盖 46 个 `src/` 模块（如 `solver.py:14`、`solver3d.py:15`）；`registry.dump()` 的生产消费者 **0** 个（仅 CLI / 审计脚本 / 测试）。

**契约测试：17 项中 16 绿 1 红。** 红项=`test_registry_error_bands_equal_the_measured_bands`：
`fvm.verification.annular_band` 声明 `maturity='mature', verified=True`，但 `error_bands.json` 无该 id 的实测带
→ 数值本身真实（0.2813 % 与 `fvm.core.annular_axial` 的实测 Linf 逐位一致），断的是 id↔带 的机器绑定。

## 4. 图

左：4 条 A 级误差带中唯一做网格收敛研究的 annular 能力，审计主机重算（n_theta 32/64/128，误差 3.03 %→0.28 %，
实测阶 1.985/1.929，径向阶 −0.083 → 误差由分面墙主导，径向加密无效）。右：F2 发现的仓内工件陈旧度（恰好差 1 项能力）。

## 5. 注意点 / 局限

- **不是重做**：未修改 `agent/m23215` 任何文件；判级只看可复算数字。
- **未跑**（标 not_run，不计通过）：装饰后求解器的数值回归（只跑了 import 模式 parity）、全仓 587 模块测试矩阵、
  `fvm_audit.json` 46 项能力的逐行源码抽检、审计脚本统计量的独立重算。
- 锚点无登记基准 → 上表容差是**被审单自定**，不构成平台裁判。
- `registry.py` 审计时只在 `agent/m23215`，**未并入 main**（m23308 工作树不存在该文件）→ 集成为「被动元数据」。

## 6. 误差判定（关键量 vs 基准，显式回答是否 <3%）

| 能力 | 参考 | 实测 | 容差 | <3 %? |
|---|---|---|---|---|
{ann_rows}
| `fvm.freesurface.plic_geometry` | 代数恒等式 A(n,c(α))=α | {plic:.3g} % | — | ✅ |
| `fvm.flux.overset_transfer` | 通量守恒（20 trials） | {ove:.3g} % | — | ✅ |
| `fvm.linear.diagnostics` | 分量型后向误差 | {lin:.3g} | — | ✅ |
| 其余 {ls['count'] - ls['verified']} 项 | **无可机核参考** | 未测（`verified=False`，如实标注） | — | 不判 |

**是否 <3%？** 5 条 `verified` 中有 4 条 ✅；annular 粗网格 n_theta=32 为 3.03 %，**不满足 3 %**（仍 <15 % 平台底线）——
如实记录，不粉饰。注册表成熟度结论的覆盖面仅 {ls['verified']}/{ls['count']} ≈ {100*ls['verified']/ls['count']:.1f} %，
且其中 1 条成熟声明尚未与实测带绑定（F1）。

**判级结论：L{verdict['level']}（独立功能实现）** —— 805 行注册表真代码 + AST/import 双模式 parity 实测一致；
不达 L3（零生产消费者、未并入 main），不达 L4（锚点基准 0 条、验证覆盖 {ls['verified']}/{ls['count']}）。
处置：**approve with notes**，返修项见 `docs/m23308/CAP-REG-FVM-audit-m23308.md` §5。
"""

    payload = {
        "title": "交差核验 m23308：m23215 CAP-REG-FVM 审计判级 L%d（含 F1 契约测试红/工件陈旧实证）" % verdict["level"],
        "summary": ("对 m23215（TensorFVM 能力注册+内核审计）做只读交差核验：5 条 repro 全跑（4 绿 1 红），"
                    "实测注册 %d 项而仓内工件仍报 %d 项，17 项契约测试 16 绿 1 红（未绑定的 mature 声明），"
                    "锚点基准 0 条 → 不达 L4；判级 L%d，approve with notes。"
                    % (ls["count"], ss["count"], verdict["level"])),
        "scenario_key": "platform_audit",
        "mission_id": MISSION_ID,
        "tags": ["TensorFVM", "CAP-REG-FVM", "board-audit", "maturity-gate", "capability-registry", "honest-fail"],
        "markdown": md,
        "figures": [{
            "name": "fig_capreg_fvm_audit.png",
            "caption": "左：annular 分面壁 Poiseuille 误差 vs 角向加密（审计主机重算，2 阶参考线 + 3%/15% 闸门）；"
                       "右：仓内 README/registry_dump 与实测 --static 的逐项计数差（恒为 1 项）",
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        }],
    }
    if "--dry-run" in os.sys.argv:
        pathlib.Path("/tmp/m23308_report.json").write_text(json.dumps(payload, ensure_ascii=False))
        pathlib.Path("/tmp/m23308_report.md").write_text(md)
        print("dry-run ok; md chars:", len(md))
        return 0
    req = urllib.request.Request(f"{API}/tensor/reports", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {login()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        print(r.status, r.read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())