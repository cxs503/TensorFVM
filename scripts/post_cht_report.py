"""Post the P295W1R2-N01 report to the TensorPlatform API (report discipline)."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "cht-stefan-ice-steel"


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]

caps = {
    "fig1_stefan_neumann.png": "图1 C2：Stefan 相变前沿 X(t) vs Neumann 相似解（左）与 t=1200 s 温度剖面（右）。前沿误差 1.10/0.53/0.21 % @300/600/1200 s。",
    "fig2_ice_steel_field.png": "图2 C3：冰层(33.4 mm)-接触热阻-钢板(10 mm) 二维温度场云图（顶面温度沿横向 -1.8→-11.8 ℃）与剖面 vs 一维级数解析解。",
    "fig3_interface_mutations.png": "图3 C1：界面审计（对数坐标）——干净工况与三种注入变异；判据线 1e-6（热流）/1e-2（温度连续），变异全部变红。",
    "fig4_w4n08_crosscheck.png": "图4 C4：同一冰况五项表面热平衡 vs W4-N08 工件（左）；冻结份额随结构背面温度的 conduction-aware 扫掠 vs W4-N08 绝热基底假设（右）。",
}
figures = [{"name": n, "caption": c,
            "png_base64": base64.b64encode((DOCS / "figures" / n).read_bytes()).decode()}
           for n, c in caps.items()]
body = (DOCS / "REPORT.md").read_text()
payload = {
    "title": "TensorFVM CHT 验证面：Stefan 相变 + 冰-钢温度场 + 与 W4-N08 积冰互裁（P295W1R2-N01）",
    "summary": ("新增 enthalpy-porosity Stefan 求解器（前沿 vs Neumann 1.10/0.53/0.21 %）、"
                "两域耦合 CHT 界面审计（温度连续 3.5e-14、热流残差 1.5e-12、三种变异全红）、"
                "冰/钢稳态场（vs 级数解析 ~1e-14）、与 W4-N08 积冰模块同冰况互裁"
                "（五项通量 ≤1.7e-7、温度场 L2 8.99e-15）；并发现在冷舱工况下 W4-N08 绝热基底假设"
                "使结冰率偏小（5.57→10.26 mm/h）。15 个 pytest 用例全绿。"),
    "scenario_key": "tensorfvm_cht_stefan_ice_steel",
    "mission_id": 10735,
    "tags": ["TensorFVM", "CHT", "Stefan", "phase-change", "ice-steel", "contact-resistance",
             "W4-N08", "mutual-validation", "P295W1R2-N01"],
    "markdown": body,
    "figures": figures,
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
res = post("/tensor/reports", payload, token)
print(json.dumps(res)[:600])