#!/usr/bin/env python3
"""Register the audit's negative-knowledge entry (distillation duty).

The double-Mach datum defect found in m15331 is not a one-off typo: any future
mission that writes an oblique-shock initial datum from a "mach" scalar will
hit the same two pitfalls (double use of the mach number; component instead of
front-frame projection).  Register it so the next mission greps it first.
"""
import subprocess

E = {
    "entry_id": "negk:m15456:CASE-DOUBLE-MACH-DATUM",
    "mission_id": 15456,
    "title": "双马赫反射算例：法向马赫数二次折算 + 速度分量式投影，导致壁面不反射（算例≠引用算例）",
    "hypothesis": "把 WC84 双马赫反射的后激波状态按 vs=mach*c1/sin(theta) 与 mn=mach/sin(theta) 给出，"
                  "并令 u_lab=vs-un*sin(theta)、v_lab=+un*cos(theta)，即可复现该基准算例",
    "falsification_evidence": "独立重算（docs/audit-m15456/audit_double_mach.py）：返回 p=155.389 对应法向马赫 11.547（正激波反解 11.5469），而 WC84 标称 M=10 → p 应 116.500，偏高 33.4%；速度模 5.79 恰等于 V*cos60=5.773（上游切向速度），正解应为 (7.145,-4.125)、|v|=8.25，且 y 分量符号反了。实测 120x30/240x60 壁面行 rho 只有 3.48/3.49（mean 2.52），壁面无任何 rho>4 单元（wall_shock_x=NaN），即从未发生壁面反射；换成正解数据后壁面 mean=11.85、剔边界窗 max rho=18.58、max p=581，反射结构才出现",
    "failure_reason": "同一标量 mach 被同时当作法向马赫数（算 vs）与 x 向平移速度（算 mn），二者不可兼得；"
                      "后激波速度用原始分量相减而非激波坐标系/实验室坐标系投影",
    "failure_kind": "wrong_physics_derivation",
    "applicable_conditions": '["任意斜激波初值(oblique/shock-wall reflection)", '
                             '"把文献 Mach 数搬进初值时"]',
    "keywords": '["TensorFVM", "CASE-DOUBLE-MACH", "oblique-shock", "Rankine-Hugoniot", "initial-datum"]',
    "domain_tags": '["CFD", "compressible", "benchmark-datum"]',
    "differentiation_required": '["先反解：由 p2 反算法向马赫数并与标称 Mach 对齐（差>1% 即判数据错）", '
                               '"后激波速度必须用 front-frame 投影，模长应等于 Rankine-Hugoniot 的 |u2|，'
                               '方向应由壁面反射是否出现来反证"]',
    "source_artifacts": '["docs/audit-m15456/REPORT.md", "docs/audit-m15456/audit_double_mach.py", "docs/audit-m15456/audit_double_mach_120x30.json", "docs/audit-m15456/audit_double_mach_240x60.json", "docs/audit-m15456/dm_field_stats.json", "docs/audit-m15456/fig_dm_audit.png", "tensor_maturity_verdicts id=320"]',
    "confidence": 0.9,
    "registered_by": "m15456 board-audit",
}

cols = ["entry_id", "mission_id", "title", "hypothesis", "falsification_evidence", "failure_reason",
        "failure_kind", "applicable_conditions", "keywords", "domain_tags",
        "differentiation_required", "source_artifacts", "confidence", "registered_by", "status"]


def esc(v):
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


vals = list(E.values())
rows = list(zip(cols, vals))
rows = [(c, "'active'") if c == "status" else (c, esc(v)) for c, v in rows]
sql = ("INSERT INTO negative_knowledge_ledger (" + ",".join(c for c, _ in rows) + ") VALUES ("
       + ",".join(v for _, v in rows) + ") ON CONFLICT (entry_id) DO NOTHING")
r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-c", sql],
                   capture_output=True, text=True)
print(r.stdout[-200:], r.stderr[-400:])
r2 = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|",
                     "-c", "SELECT entry_id, mission_id, failure_kind, confidence FROM "
                           "negative_knowledge_ledger WHERE entry_id='negk:m15456:CASE-DOUBLE-MACH-DATUM'"],
                    capture_output=True, text=True)
print(r2.stdout, r2.stderr[-200:])