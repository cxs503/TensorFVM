"""Post the P167W2B-N03 paper-grade report to the platform.

Reads ``docs/p167-n03/report.md`` (markdown), the gate artefacts under
``runs/p167`` and the figures under ``docs/p167-n03/figures``, and publishes the
report through ``POST /tensor/reports``.  All numbers in the markdown come from
the artefact files, so nothing is transcribed by hand.

Usage::

    python3 scripts/publish_p167_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request

DOC = "docs/p167-n03/report.md"
FIG_DIR = "docs/p167-n03/figures"
MISSION_ID = "12422"
SCENARIO = "tensorfvm-two-phase-vof-verification"

FIGURES = [
    ("fig1_zalesak_fields.png",
     "Zalesak 旋转圆盘绕行一圈后（64²，CFL 0.5）：精确解 / PLIC / CICSAM / FCT 的 "
     "α 云图与 0.5 等值线；FCT 已把狭缝抹平，CICSAM 与 PLIC 仍分辨狭缝。"),
    ("fig2_zalesak_errors.png",
     "左：界面 L1 误差随圈数（CICSAM/PLIC/FCT，64² 实线、更细网格虚线，对数纵轴）；"
     "右：质量台账 |V(T)−V(0)|/V(0) 随圈数，全部落在 1e-15（比 0.1% 判据低 12 个量级）。"),
    ("fig3_interface_thickness.png",
     "界面厚度（1/max|Δα|，单位：网格）：各方案各圈数的中位数与 90 分位，黑点线为 2 格判据。"),
    ("fig4_dam_break_front.png",
     "溃坝波前轨迹：Martin & Moyce (1952) 数字化曲线、Ritter 浅水极限、"
     "CICSAM/FCT 模拟曲线；系统性滞后是本节点唯一未通过的判据。"),
    ("fig5_hydrostatic.png",
     "静水压力分布误差：well-balanced 表面模型三档网格（机器零）vs 非 well-balanced "
     "负对照（1.56%），虚线为 1% 判据。"),
]


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _token(base: str) -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    doc = json.load(urllib.request.urlopen(req, timeout=30))
    return doc.get("access_token") or doc["token"]


def build_payload() -> dict:
    with open(DOC, encoding="utf-8") as fh:
        markdown = fh.read()
    figures = []
    for name, caption in FIGURES:
        path = os.path.join(FIG_DIR, name)
        if os.path.exists(path):
            figures.append({"name": name, "caption": caption,
                            "png_base64": _b64(path)})
    summary_path = os.path.join(FIG_DIR, "..", "artifacts", "summary.json")
    criteria = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            criteria = json.load(fh).get("criteria", {})
    g1 = criteria.get("G1_zalesak_mass", {}).get("value")
    g2 = criteria.get("G2_cicsam_sharpness", {}).get("value")
    g4 = criteria.get("G4_hydrostatic_pressure", {}).get("value")
    g5 = criteria.get("G5_dam_front", {}).get("value")
    summary = (
        "TensorFVM 新增 CICSAM（Ubbink & Issa 1999）压缩型 VOF 输运与两相流验证门："
        f"Zalesak 1/2/4/8 圈体积累计漂移 {g1:.2e}（判据<0.1%，通量形式+局部质量精确界修复）；"
        f"界面厚度 90 分位 {g2:.2f} 格（判据<2 格；64² 同场对比 PLIC 0.048、CICSAM 0.087、FCT 0.581 L1）；"
        f"静水 ρgh 压力振荡 {g4:.1e}（well-balanced，机器零；非 well-balanced 负对照 1.56%）；"
        f"溃坝波前 vs Martin & Moyce 最差 {100 * (g5 or 0):.0f}% —— honest FAIL（判据<10%，已如实记录，"
        "加密到 200×60 无收敛趋势，属前缘度量与 2D 建模局限）。16 例 pytest 覆盖方案性质与回归。"
    )
    return {
        "title": "P167W2B-N03 TensorFVM 两相流 VOF 验证面：CICSAM 界面压缩 + Zalesak 多圈质量守恒 + 静水压力 + 溃坝波前实验对标",
        "summary": summary,
        "scenario_key": SCENARIO,
        "mission_id": MISSION_ID,
        # the reports API caps the tag list at 10
        "tags": ["tensorfvm", "two-phase", "vof", "cicsam", "zalesak",
                 "martin-moyce", "dam-break", "hydrostatic",
                 "verification-gate", "p167w2b-n03"],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    payload = build_payload()
    with open("/tmp/report_p167.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print("payload:", len(payload["markdown"]), "chars,",
          len(payload["figures"]), "figures")
    if args.dry_run:
        return 0
    base = os.environ["TENSOR_API_BASE"]
    token = _token(base)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/tensor/reports", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token})
    out = json.load(urllib.request.urlopen(req, timeout=120))
    print(json.dumps({k: out[k] for k in out if k != "markdown"},
                     ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())