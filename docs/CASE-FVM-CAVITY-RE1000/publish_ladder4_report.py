"""Publish the fourth-level cavity audit to the platform report API."""
from __future__ import annotations

import base64
import json
import os
import pathlib

import requests

DOCS = pathlib.Path(__file__).resolve().parent
DOC = DOCS / "GRID-LADDER-4LEVEL.md"
PNG = DOCS / "figures" / "fig_ladder4_two_references.png"
d = json.loads((DOCS / "runs" / "grid_ladder_4level.json").read_text())
lv, v, b = d["levels"], d["verdict"], d["bands"]
pct = lambda x: f"{100 * x:.3f}%"  # noqa: E731

body = DOC.read_text()
md = (
    "# CASE-FVM-CAVITY-RE1000：第四级网格（n=192）落库与「双参考」误差归因\n\n"
    + body.split("\n", 1)[1]
)
rep = {
    "title": "CASE-FVM-CAVITY-RE1000：第四级网格 n=192 + 双参考误差归因（表值吻合度≠精度）",
    "summary": (
        f"取回 cj3849 的 n=192 场，四级原始 MAC 面量全部独立重算（对账 1e-12）：登记门在 n=192 仍 PASS"
        f"（余量 2.7×→{0.03 / lv[3]['err_u_min_vs_ghia']:.2f}×），但对 Ghia 的吻合度在第四级变差 "
        f"{v['u_min']['cancellation_finest']:.2f}×（u_min）/ {v['v_max']['cancellation_finest']:.2f}×（v_max），"
        f"对独立谱极限仍按 p={d['orders']['u_min_node']['pair_128_192']['order']:.2f} 单调下降，"
        f"GCI 带 {pct(b['n128_u_min']['gci_fine'])}→{pct(b['n192_u_min']['gci_fine'])} ⇒ 残差判决 "
        f"{v['u_min']['verdict']}（非 unresolved）。"
    ),
    "scenario_key": "CASE-FVM-CAVITY-RE1000",
    "mission_id": 22630,
    "tags": ["TensorFVM", "cavity", "Re1000", "grid-convergence", "GCI",
             "verification-debt", "ghia1982", "spectral-reference"],
    "markdown": md,
    "figures": [{
        "name": "fig_ladder4_two_references.png",
        "caption": "左：x=L/2 反流井，四级 FVM 剖面 vs Ghia 表点与谱极限；"
                   "右：对 Ghia 与对谱极限的相对误差 vs h（双对数）",
        "png_base64": base64.b64encode(PNG.read_bytes()).decode(),
    }],
}

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8002/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "ci")
pwd = os.environ.get("TENSOR_AGENT_PASS", "ci")
sess = requests.Session()
tok = ""
for path in ("auth/login", "auth/token", "login"):
    try:
        r = sess.post(f"{base}/{path}", json={"username": user, "password": pwd}, timeout=20)
        if r.ok:
            j = r.json()
            tok = j.get("access_token") or j.get("token") or j.get("data", {}).get("access_token", "")
            if tok:
                break
    except Exception as exc:  # noqa: BLE001
        print("login attempt failed", path, exc)
print("token_len", len(tok))
r = sess.post(f"{base}/tensor/reports", json=rep,
              headers={"Authorization": f"Bearer {tok}"} if tok else {}, timeout=120)
print(r.status_code, r.text[:400])