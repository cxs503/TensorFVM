import base64, io, json, os, urllib.request
from PIL import Image

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
ROOT = "/root/tp-agent/work/m4105"
DOCS = os.path.join(ROOT, "docs", "m4105-ci-remediation")


def req(path, data=None, token=None):
    body = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(BASE + path, data=body, method="POST" if body else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=240) as resp:
        return json.loads(resp.read().decode() or "{}")


def login():
    err = None
    for p in ("/auth/login", "/tensor/auth/login", "/login"):
        try:
            out = req(p, {"username": USER, "password": PASS})
            tok = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
            if tok:
                print("login ok", p)
                return tok
        except Exception as e:
            err = e
    raise SystemExit("login failed: %s" % err)


def png(path):
    img = Image.open(path).convert("RGB")
    if img.width > 760:
        img = img.resize((760, round(img.height * 760 / img.width)), Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


table = open(os.path.join(DOCS, "metrics_table.md")).read().strip()
audit = json.load(open(os.path.join(DOCS, "reference_edit_audit.json")))
d = audit["dispute"]
nb = audit["neighbour_gaps_n128"]
gap = abs(d["computed"]["128"]["computed"] - d["published"])
rel = gap / abs(d["published"])
corr = open(os.path.join(ROOT, "docs/verification-lid-cavity-correction/report.md")).read()
rem = open(os.path.join(DOCS, "report.md")).read()
report = open(os.path.join(ROOT, "docs/m4105-ci-remediation/pytest-full-partial.log")).read()
tail = "\n".join(report.strip().splitlines()[-6:])

figs = []
for name, cap in (
    ("reference_edit_audit.png", "左：已提交 ladder 的 Re=400 v-中心线(n=64) 与按原文发表的 Table II 点，红叉=被否决的替换值 -0.38760；右：v(x=0.9063) 对 1/n^2 二阶收敛到登记 Richardson 极限 -0.38989，两条候选参考值均画出。"),
    ("error_vs_grid.png", "u/v 中心线 max|err| 随 n（对数轴，红线=3% 验收带）：恢复发表参考值后 u 收敛良好，Re=400 v 行随加密恶化 0.1146→0.1424→0.1493，欠账如实保留。"),
):
    raw = png(os.path.join(DOCS, name))
    figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(raw).decode()})
    print(name, len(raw) // 1024, "KB")

md = f"""## 0. 结果表（程序化生成，`scripts/m4105_ci_remediation_report.py` 输出）

{table}

争议点量化（n=128）：发表值 {d['published']:.5f}，求解器 {d['computed']['128']['computed']:.5f}，
二阶 Richardson 极限 {d['richardson_limit']:.5f}；gap = {gap:.5f}（{rel*100:.1f}%）= 度量路径插值误差界的
{d['gap_over_bound_n128']:.0f} 倍、最坏邻点 gap（{nb['0.9453']:.4f}）的 {gap/nb['0.9453']:.0f} 倍。
被否决值 -0.38760 与登记极限仅差 0.59%，且与求解器 n=128 值仅差 2e-5 —— 用它当参考即
“拿求解器自己的外推给自己打分”，属循环验证。

## 1. 物理背景 / 几何 / 参数（案例介绍）

Ghia, Ghia & Shin (1982, JCP 48, 387–411) 顶盖驱动方腔：单位方腔、三面静止壁、
y=L 顶盖 U=1；参考数据为其 Table I（u at x=0.5）与 Table II（v at y=0.5）的逐字转录。
Re = U·L/ν ∈ {{100, 400, 1000}}，n ∈ {{32, 64, 128}}，CFL=0.15，残差门限 1e-7，步数上限 2e5。
本单是验证数据完整性事故的修复：PR #126（agent/m3884, head 48fd44b）在解 #82 冲突时
把 Re=400,y=0.9063 的发表值 -0.23827 改成 -0.38760，并据此重算证据包与测试，CI 变红。

## 2. 怎么做（方法 / 工具链）

1. 取 PR head 到工作树，`python3 -m pytest -q` 本地复现：`tests/test_ghia_reference_consistency.py`
   三条不变量同时失败（audit["ok"]=False；gap 1.8e-5 不再 >5×邻点；gap/bound=0.04 不再 >100）。
2. 正负对照审计 `scripts/audit_m4105_reference_edit_rejection.py`：发表表下
   `flagged=['400/v/0.9063','400/v/0.9453']` 且 `ok=True`；把替换值仅在内存注入后
   `flagged=[]`、`registered_flagged=[]`，登记指纹消失 → 门禁失效（这就是 CI 红的根因）。
3. 判据来自仓内两条既有规则：`lid_cavity_reference.py` 明文“争议不等于可以改转录”；
   登记记录裁决条件“需要纸质 Table II 或独立 DNS，本检出无离线副本”。
4. 修复：参考表、ladder 证据包、测试三件产物回滚到 main 内容；保留严格 xfail 让欠账
   继续可见；文档改写为“否决 + 复现路径”。
5. 验证：定向回归 `pytest -q tests/test_ghia_reference_consistency.py
   tests/test_lid_cavity_re1000_evidence.py tests/test_lid_cavity.py` → 27 passed, 1 skipped,
   1 xfailed；全量套件跑到 43%（0 失败）后为控预算主动停止，尾部日志：

```
{tail}
```

## 3. 注意点（坑 / 局限 / 适用范围）

- 全量 pytest 未跑完（43%，0 失败）；CI 的 test (3.11/3.12) 结论需读 GitHub run。
- lint (advisory) 的 9822 条 ruff 错误是 main 既有存量（job 本身 continue-on-error），
  不在本单范围；本单触及文件 `ruff check` 干净、新脚本 `ruff format` 干净。
- 未重跑 ladder（证据包回滚到已提交版本，与早期记录可比）。
- 争议未关闭：拿到纸质 Table II / 独立 DNS 之前不得宣称 <3%。

## 4. 误差判定（逐项，显式回答是否 <3%）

| 量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| Re=100 u / v | Ghia Table I / II（发表值） | 0.0045–0.0069 / 0.0078–0.0091 | 是 |
| Re=400 u | Ghia Table I | 0.0013 (n=128) | 是 |
| Re=400 v | Ghia Table II | 0.1146 / 0.1424 / 0.1493 (n=32/64/128) | **否** |
| Re=1000 u / v (n=64) | Ghia Table I / II | 0.0293 / 0.0197 | 是（u 临界） |
| 全部行散度 | 精确 div u = 0 | ≤6.8e-13 | 是 |

**结论：Re=400 v-中心线不满足 3%，本单如实记为未结欠账（honest-FAIL）。**
本单修复的是“不得改写发表参考数据”，不是“把数字调好看”。

---

# 附 A：修复说明（docs/m4105-ci-remediation/report.md）

{rem}

---

# 附 B：参考数据裁决（docs/verification-lid-cavity-correction/report.md）

{corr}
"""

payload = {
    "title": "PR #126 修复：撤销 Ghia Re=400 参考数据改写，恢复登记争议（TensorFVM m4105）",
    "summary": ("PR #126 把 Ghia(1982) Table II 在 Re=400,y=0.9063 的发表值 -0.23827 换成本解器"
                "Richardson 极限 -0.38760 并重算证据包，test_ghia_reference_consistency 三条不变量因此失败。"
                "本单回滚参考表/证据包/测试到 main、恢复严格 xfail，并用正负对照审计证明：发表表下审计只标出"
                "登记争议点(gap 0.1493=149×插值界)，替换值注入后登记指纹消失。定向回归 27 passed/1 skipped/"
                "1 xfailed；Re=400 v 中心线误差 0.1493 如实保留为未结欠账（honest-FAIL）。"),
    "scenario_key": "tensorfvm-verification-integrity",
    "mission_id": 4105,
    "tags": ["TensorFVM", "CI-repair", "Ghia-1982", "verification-integrity", "honest-FAIL", "PR-126"],
    "markdown": md,
    "figures": figs,
}
size = len(json.dumps(payload))
print("payload", size // 1024, "KB")
assert size < 240 * 1024, "payload too large"
json.dump(payload, open("/tmp/report.json", "w"))
print("post:", json.dumps(req("/tensor/reports", payload, login()))[:300])