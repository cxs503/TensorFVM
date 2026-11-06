"""Build /tmp/report.json for mission 4694 and POST it to the platform.

Figures are embedded as base64 so the payload stays far below the 250 KB
report-endpoint ceiling (the PNG is ~61 KB -> ~82 KB encoded).
"""
import base64
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent
markdown = (HERE / "report.md").read_text()
png = base64.b64encode((HERE / "lint-and-suite-before-after.png").read_bytes()).decode()

payload = {
    "title": "TensorFVM main CI 修复: lint 门禁 55 -> 0 (test 作业本地未复现)",
    "summary": (
        "ruff 0.15.0 在 bd06c3fb 上于 tests/ 报 55 条 finding (48 F401 / 5 F841 / 2 E401), "
        "来自 #187 被推送门禁滤除的 34 个文件。修复后 ruff 全绿、compileall 通过、"
        "34 个被改测试文件 239 passed、全量套件 1426 passed / 0 failed。"
        "test(3.11)/(3.12) 红在本机 (py3.11/torch2.7.1) 未复现: 37 个 skip 中 gmsh 门控用例"
        "与 3.12 腿为残余风险, 如实声明未验证。"
    ),
    "scenario_key": "tensorfvm-ci-repair",
    "mission_id": 4694,
    "tags": ["TensorFVM", "CI", "lint", "ruff", "F401", "F841", "pytest", "honest-FAIL"],
    "markdown": markdown,
    "figures": [{
        "name": "lint-and-suite-before-after.png",
        "caption": (
            "左: 修复前后各规则 finding 数 (55 -> 0); "
            "右: 本分支全量套件结果计数 (1426 passed / 37 skipped / 2 xfailed / 0 failed)"
        ),
        "png_base64": png,
    }],
}

body = json.dumps(payload, ensure_ascii=False).encode()
pathlib.Path("/tmp/report.json").write_bytes(body)
print("payload bytes", len(body))

token = os.environ.get("M4694_TOKEN") or pathlib.Path("/tmp/tok.txt").read_text().strip()
url = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1") + "/tensor/reports?mission_id=4694"
req = urllib.request.Request(url, data=body, method="POST",
                            headers={"Authorization": "Bearer " + token,
                                     "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        print("HTTP", resp.status, resp.read()[:500].decode(errors="replace"))
except Exception as exc:  # noqa: BLE001 - report the failure, do not hide it
    print("POST failed:", type(exc).__name__, exc)
    if hasattr(exc, "read"):
        print(exc.read()[:800].decode(errors="replace"))