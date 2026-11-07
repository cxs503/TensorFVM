"""Publish the bubble-rise report to the platform report API.

Reads the generated artefacts (``summary.json``, ``controls.json``,
``report.md`` and the PNG figures), compresses each figure below the API's
payload budget, and POSTs a paper-grade report for ``mission_id`` 5037.

    PYTHONPATH=src python scripts/submit_bubble_rise_report.py --dir docs/bubble-rise-2d
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import urllib.request
from pathlib import Path

MISSION_ID = 5037
MAX_PNG_BYTES = 60_000


def _login(base: str) -> str:
    payload = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read())["access_token"]


def _shrink(png: Path) -> str:
    """Base64 of the PNG, re-encoded smaller if it exceeds the payload budget."""
    raw = png.read_bytes()
    if len(raw) > MAX_PNG_BYTES:
        from PIL import Image
        image = Image.open(io.BytesIO(raw))
        scale = 0.8
        while True:
            width = max(200, int(image.width * scale))
            height = max(150, int(image.height * scale))
            buffer = io.BytesIO()
            image.resize((width, height)).convert("P").save(buffer, format="PNG", optimize=True)
            raw = buffer.getvalue()
            if len(raw) <= MAX_PNG_BYTES or scale < 0.25:
                break
            scale *= 0.8
    return base64.b64encode(raw).decode()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("docs/bubble-rise-2d"))
    args = parser.parse_args(argv)
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    token = _login(base)
    summary = json.loads((args.dir / "summary.json").read_text())
    markdown = (args.dir / "report.md").read_text()
    figures = []
    captions = {
        "fig1-interface-snapshots.png": "气泡分数场快照（最细网格，4 个时刻）",
        "fig2-history.png": "形心高度 / 上升速度 vs 锚与 ±15% 带 / 宽高比历史",
        "fig3-convergence-conservation.png": "终端速度网格收敛与全程体积守恒",
        "fig4-clift-map.png": "实测 (Re, Eo) 在 Clift-Grace-Weber 图上的位置",
        "fig5-controls.png": "控制实验：基线 / 密度比 100 / 侧向 8d 与寄生电流地板",
    }
    for name, caption in captions.items():
        path = args.dir / name
        if path.exists():
            figures.append({"name": name, "caption": caption, "png_base64": _shrink(path)})
    crit = summary["criteria"]
    verdict = "PASS" if summary["all_pass"] else "FAIL"
    payload = {
        "title": "TensorFVM 两相第二刀：二维单气泡上升 benchmark（PLIC VOF）",
        "summary": (
            f"2-D 单气泡上升（Eo=1.23, M=2.63e-11）三网格基准：终端速度 "
            f"{crit['C4_terminal_velocity']['U_terminal']:.3f} vs Fan-Tsuchiya 锚 "
            f"{crit['C4_terminal_velocity']['U_anchor']:.3f}"
            f"（{100 * crit['C4_terminal_velocity']['signed_error']:+.1f}%）；体积守恒最坏单步 "
            f"{crit['C2_mass_conservation']['worst_step_relative_drift']:.1e}；"
            f"形状带与几何验证通过；总判定 {verdict}（终端速度未进 15% 容差，已定位主因）。"),
        "scenario_key": "tensorfvm-bubble-rise-2d",
        "mission_id": MISSION_ID,
        "tags": ["TensorFVM", "two-phase", "VOF", "PLIC", "bubble-rise", "benchmark", "audit"],
        "markdown": markdown,
        "figures": figures,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/tensor/reports", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read())
        print("posted:", json.dumps(result)[:400])
    except urllib.error.HTTPError as exc:
        print("HTTPError", exc.code, exc.read()[:500])
        return 1
    check = urllib.request.Request(
        base + f"/tensor/reports?mission_id={MISSION_ID}",
        headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(check, timeout=60) as response:
        listing = json.loads(response.read())
    items = listing if isinstance(listing, list) else listing.get("items", [])
    print("reports for mission:", [(i.get("id"), i.get("title", "")[:40]) for i in items][:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
