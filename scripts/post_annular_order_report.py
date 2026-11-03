"""Publish the annular design-order verification report to the platform API.

Reads the generated ``scripts/verification-annular-order/report.md`` and the three
figures (down-scaled to keep the JSON payload well under the API limit), then
POSTs to ``/tensor/reports`` with a bearer token from the agent environment.
"""
import base64
import io
import json
import os
from pathlib import Path
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
ROOT = Path(__file__).resolve().parent / "verification-annular-order"
FIGURES = (
    ("profile.png", "Fig.1 velocity profiles at three angular positions vs the exact annular "
                    "Poiseuille solution, and the pointwise absolute error (finest level "
                    "nr=32, ntheta=128).", 760),
    ("convergence.png", "Fig.2 log-log relative error vs radial spacing h for all six reported "
                        "quantities with the slope-2 guide, and the O(h^2) geometric area defect "
                        "of the straight-edge annulus.", 760),
    ("field.png", "Fig.3 axial-velocity cloud map on the production mesh (left) and its absolute "
                  "error field (right); white disc = no-slip inner cylinder.", 620),
)


def post(path, payload, token=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                    headers={"Content-Type": "application/json"})
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def main():
    from PIL import Image
    login = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                 "password": os.environ["TENSOR_AGENT_PASS"]})
    token = login.get("access_token") or login.get("token")
    if not token:
        raise RuntimeError(f"login failed: {login}")
    figures = []
    for name, caption, width in FIGURES:
        image = Image.open(ROOT / name).convert("RGB")
        image = image.resize((width, int(image.height * width / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        raw = buffer.getvalue()
        print(name, len(raw))
        figures.append({"name": name.replace(".png", ".jpg"), "caption": caption,
                        "png_base64": base64.b64encode(raw).decode()})
    payload = {
        "title": "TensorFVM 环形层流后端设计阶验证：解析基准自检 + 二阶收敛判定 (PASS)",
        "summary": (
            "生产后端 tensorfvm.annular（贴体曲面网格+fitted 扩散核+无滑移壁面，Stokes 极限）设计阶验证。"
            "先机器自检解析基准：动量方程残差 3.4e-10、闭式流量 vs scipy quad 1.9e-15、壁面剪应力全局平衡恒等式 1.2e-16。"
            "再在 nr=8/16/32、ntheta=4nr 三级网格测量：速度 L2 观测阶 1.98→2.00，nr=32 时速度 L2 1.22e-3、流量 1.06e-3、"
            "内壁剪应力 6.2e-4、外壁 1.0e-4、峰值速度 2.3e-4，全部 <3%（ERROR_LIMIT）。几何量先行：直边多边形环面积缺陷 O(h^2)。裁决 PASS。"),
        "scenario_key": "tensorfvm-annular-design-order",
        "mission_id": 2675,
        "tags": ["TensorFVM", "verification", "order-of-accuracy", "annular-poiseuille",
                 "analytic-benchmark", "selfstart"],
        "markdown": (ROOT / "report.md").read_text(),
        "figures": figures,
    }
    result = post("/tensor/reports", payload, token)
    print("REPORT", json.dumps(result)[:800])


if __name__ == "__main__":
    main()