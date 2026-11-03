"""Publish the VOF rotation verification report to the platform report API."""
import base64
import io
import json
import os
import urllib.request

ENV = "/root/tp-agent/agent.env"
ART = "src/tensorfvm/verification/artifacts/vof_rotation"


def env():
    values = {}
    for line in open(ENV, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def request(url, payload=None, token=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode())


def png_b64(path, width=900):
    from PIL import Image
    image = Image.open(path).convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    config = env()
    base = config["TENSOR_API_BASE"]
    login = request(base + "/auth/login",
                    {"username": config["TENSOR_AGENT_USER"],
                     "password": config["TENSOR_AGENT_PASS"]})
    token = login.get("access_token") or login.get("token")
    markdown = open(os.path.join(ART, "VOF-ROTATION-VERIFICATION.md"), encoding="utf-8").read()
    figures = [
        dict(name="zalesak_fields.png", path=os.path.join(ART, "zalesak_fields.png"),
             caption="Zalesak 开槽圆盘：128² 网格转 1 圈后的 α 场（初始 / donor-cell / FCT 限制）。"
                     "生产格式峰值仅剩 0.52，界面被数值扩散抹平。"),
        dict(name="convergence.png", path=os.path.join(ART, "convergence.png"),
             caption="cosine hill 与 Zalesak 盘相对 L1 误差随分辨率（32/64/128）的双对数曲线，"
                     "donor-cell 与 FCT 两格式对照。"),
    ]
    payload = dict(
        title="TensorFVM VOF 刚体旋转验证：生产格式既不守恒也不有界，FCT 限制器清偿",
        summary=("用刚体旋转精确基准（cosine hill + Zalesak 盘，1 圈，32/64/128）复核两相 VOF 输运。"
                 "生产 VOFTransport 的混合面值在二维越界（原始最小 -5.05e-3，4 个单元），"
                 "随后的 clamp 破坏守恒（每步 +8.82e-6）。新增 Zalesak FCT 限制器 LimitedVOFTransport："
                 "面通量单值 → 守恒为代数恒等式（内部场漂移<1e-12），逐单元预算 → 严格有界。"
                 "度量精度：观测阶 0.55（donor）/0.45（FCT），Zalesak 相对 L1 1.19（128²），"
                 "honest 未达 <3% 精度门。6 项验证测试全绿，27 项 VOF 相关测试通过。"),
        scenario_key="tensorfvm-vof-rotation-verification",
        mission_id=2562,
        tags=["TensorFVM", "VOF", "verification", "Zalesak", "conservation", "FCT"],
        markdown=markdown,
        figures=[dict(name=f["name"], caption=f["caption"], png_base64=png_b64(f["path"]))
                 for f in figures],
    )
    size = len(json.dumps(payload).encode())
    result = request(base + "/tensor/reports", payload, token)
    print("payload_bytes", size)
    print("report_id", result.get("id") or result.get("report_id"))
    print(json.dumps({k: v for k, v in result.items() if k != "markdown"}, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()