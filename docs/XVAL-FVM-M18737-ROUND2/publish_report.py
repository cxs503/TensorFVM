import base64, json, os, requests

DOC = "docs/XVAL-FVM-M18737-ROUND2/W1-NORM-CERTIFICATION.md"
PNG = "docs/XVAL-FVM-M18737-ROUND2/figures/fig_w1_norm.png"
b = json.load(open("docs/XVAL-FVM-M18737-ROUND2/figures/w1_norm_budget.json"))
rows = {r["nx"]: r for r in b["graded_family"]["rows"]}
fin = max(rows)
eps = b["reference_family"]["eps_ref_w1"]
bound = b["verdict"]["P4"]["honest_bound"]

md = open(DOC).read()
md = md.replace("![W1 认证四联](figures/fig_w1_norm.png)",
                "（图见文末 figures 字段：fig_w1_norm.png）")
md += (
    "\n\n---\n\n**产物与证据指针**：`docs/XVAL-FVM-M18737-ROUND2/` 下 "
    "`PREDECLARATION-w1-norm.md`（判据冻结）、`W1-NORM-CERTIFICATION.md`（本报告全文）、"
    "`wc_w1_norm.py`（度量 + 判据 + 探针）、`wc_w1_norm_figures.py`、`render_w1_doc.py`、"
    "`figures/w1_norm_budget.json`、`figures/fig_w1_norm.png`；"
    "回归门 `tests/test_compressible_wc_w1_norm.py`（9 测全绿，5 s）；"
    "commit `67997743`（分支 `agent/m21711`）。\n"
)

rep = {
    "title": "XVAL-FVM-M18737-ROUND2 R5：输运范数（W1）认证——certified<1% 可证明，"
             "但「多得一阶」被证伪",
    "summary": (
        "Woodward-Colella 双爆轰：同一批存档剖面改在输运范数（累积质量 / W1 距离）下定级，"
        f"诚实上界 {bound:.5f} % < 1 %（P4 PASS，余量 {1/bound:.0f}x）；逐点 L1 仍是 "
        f"{rows[fin]['d_l1']:.4f} %（R4 的 nx_ref≈1.33e5 结论未变）。"
        "度量经注入误差闭式校验（刚体平移 / 定点搬质量 / 1% 幅值错 比值 1.0000/1.0000/1.0001）"
        "与负控制（数字化已发表曲线落在上界 4.53x 之上）。预声明 9 项判据 7 PASS / 2 honest FAIL："
        "P1「积分后多得一阶」被证伪（p_W1 0.987 vs p_L1 0.833；残差是对撞区 O(h) 净质量通量缺陷，"
        "W1 的优势是 ~100x 常数而非阶）、P7 方向鲁棒性 1/6 对越界（nx=1600 达 0.46）。"
    ),
    "scenario_key": "xval-fvm-m18737-round2-w1-norm",
    "mission_id": 21711,
    "tags": ["TensorFVM", "verification", "Woodward-Colella", "negative-norm", "W1",
             "certified", "honest-FAIL"],
    "markdown": md,
    "figures": [{
        "name": "fig_w1_norm.png",
        "caption": ("(a) 同一批剖面在逐点 L1 与输运 W1 两范数下的误差阶梯（d_W1 恒低约 100x，"
                    "斜率仍是一阶）；(b) 累积差 D(x)：残差集中在 x~0.765 对撞区，终值仅峰值 ~0.2 %；"
                    "(c) 认证阶梯对 1 % 判据线，误差棒为参考侧 eps_ref，虚线为诚实上界；"
                    "(d) 注入误差探针实测 vs 闭式（y=x 线），橙线为负控制（数字化已发表曲线）。"),
        "png_base64": base64.b64encode(open(PNG, "rb").read()).decode(),
    }],
}

tok = open("/tmp/tok").read().strip()
r = requests.post(os.environ["TENSOR_API_BASE"] + "/tensor/reports",
                  headers={"Authorization": f"Bearer {tok}"}, json=rep, timeout=180)
print(r.status_code, r.text[:400])