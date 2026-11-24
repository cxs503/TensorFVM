#!/usr/bin/env python3
"""Post the round-26 verification report to the platform (figures inlined)."""
import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
CASE = "docs/case-periodic-hills"


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


login = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                             "password": os.environ["TENSOR_AGENT_PASS"]})
token = login.get("access_token") or login.get("token")
assert token, login

figs = []
for name, cap in [
    ("fig_round26_archive_health.png",
     "14 个归档的 max|u|（对数轴）：绿=可用(8)，红=全零 dump(6)，虚线=零场门 1e-9"),
    ("fig_round26_bubble_objects.png",
     "四件代表算例：归一化壁剪(蓝)+归一化贴壁速度(橙)+认证壁剪气泡(红带)+场侧贴壁区间(绿线)+DNS 5.41 参考线(点线)"),
    ("fig_round26_definition_band.png",
     "每件的四种合法口径取值(点)、取值范围(红竖线)、带宽百分比与 ±10% 验收带(绿)"),
    ("fig_round26_cells_laminar.png",
     "层流 Re=2800 与 SST r18 的 u/U_b 云图 + u=0 等值线 + 连通再循环单元跨度：层流解只有 1 个单元"),
]:
    with open(f"{CASE}/{name}", "rb") as f:
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(f.read()).decode()})

markdown = """## ① 案例介绍

**CASE-FVM-PERIODIC-HILLS（ERCOFTAC UFR 3-30 / 周期山丘流）**：槽道在流向周期（L=9H），下壁为六个三次多项式
描述的山丘（丘高 H=1，丘顶 x/H=0，平底段 1.929–7.071，关于 x/H=4.5 镜像），槽顶 3.036H；上壁平直、周期边界。
驱动力按 ERCOFTAC 体平均定义 `u_b = 1/2.035 ∫ u dy` 每步调整，使 `Re = U_b H/ν` 成为与网格无关的输入。
验收量：再附长度 `x_r/H` 与 `y/H=0.5` 处平均速度剖面，参考值 `x_r/H = 5.41 @ Re=2800`（Breuer et al. 2009,
*Comput. Fluids* 38:433, DOI 10.1016/j.compfluid.2008.05.002；数值取自 ERCOFTAC KB `UFR 3-30 Evaluation`），容差 ±10%。

本回合（第 26 回合）回答两条**验证欠账**：round 23「给下次」第 1 条（SST 加密网格看 `x_r/H` 收敛）与
round 25 遗留（闭式无关 Re=2800 网格阶梯），并新增一个此前无人问过的问题——**被壁剪配对成「气泡」的那段区间，
是否对应速度场里一个闭合再循环单元**。全部只读归档，零求解器运行。

## ② 怎么做的

新增 `src/tensorfvm/verification/periodic_hill_archive_audit.py`：

| 组件 | 作用 |
|---|---|
| `artifact_health` | 归档完整性门：`usable` / `zero_field` / `non_finite` / `missing_keys`（判据 `max|u| ≤ 1e-9`） |
| `recirculation_cells` | 场侧拓扑口径：`u<0` 的二维连通块，x 方向周期连通；给面积、x/行跨度、贴壁区间 |
| `wall_spans` | 认证符号穿越算子作用在**壁面相邻单元 u** 上（输入独立于壁面法向导数） |
| `bubble_object_audit` | 壁剪气泡区间是否被闭合单元包住；报告抬离带与该带内 `max|τ|`/剖面 `max|τ|` |
| `definition_routes`/`definition_band` | 四种合法口径并列 + 带宽 + `definition_sensitive` 判据（带宽 > 2×容差×参考值） |
| `ladder_summary` | 阶梯只用可用档；可用档 <3 一律 `insufficient_rungs`，被排除档单独列出 |

驱动 `scripts/round26_archive_audit.py` 读 `docs/case-periodic-hills/*.npz` 全部 14 件，产出 JSON、本报告与 4 张图
（正文/表格数字全部由脚本注入，零手抄）。测试 `tests/test_periodic_hill_archive_audit.py` 19 条全绿
（合成夹具：两个独立单元 / 抬离单元 / 跨接缝单元 / 非有限场 / 缺键；真实归档回归 1.254、8.264、7.682、6 退化、两阶梯 insufficient）。
同批回归：`test_periodic_hill*` + `wall_shear_bubbles/bubble_object/field_metrics/round23` 共 95 passed / 2 skipped，ruff 全绿。

## ③ 计算结果

### 归档完整性：8/14 可用，6 件全零 dump

| artefact | 状态 |
|---|---|
| round11_sst_r18_120x60 / 144x72 | zero_field |
| round20_laminar_Re1400_96x48 | zero_field |
| round20_laminar_Re2800_144x72 / 192x96 / 240x120 | zero_field |

速度、时均速度、壁剪三者全零。零剪切的 `mean_tau_b` 让认证算子返回 `status="no_recirculation"`、`x_r/H=nan`
——与「流动完全附着」**不可区分**。两条网格阶梯的细档全部落在退化档内 ⇒ 判决 `insufficient_rungs`
（SST r18 可用 2/4，唯一可用对 72→96 的 `x_r/H` 变化 **1.07 %**；层流可用 1/4）。

### 每件的四口径与口径带宽

| artefact | nx | 单元数 | 壁剪 first | 壁剪 largest | 场侧单元末贴壁点 | 带宽 | 对象判决 |
|---|---|---|---|---|---|---|---|
| round20_laminar_Re2800_96x48 | 96 | 1 | **1.254** | **8.264** | 8.258 | **129.6 %** | object_incomplete |
| round20_laminar_Re2800_96x48dt01 | 96 | 1 | 1.287 | 8.230 | 8.223 | 128.3 % | object_incomplete |
| round20_laminar_Re700_96x48 | 96 | 1 | 1.255 | 8.078 | 8.077 | 126.1 % | object_incomplete |
| round11_sst_r18_96x48 | 96 | 1 | 7.682 | 7.682 | 7.676 | 0.1 % | object_supported |
| round11_sst_r18_72x36 | 72 | 1 | 7.600 | 7.600 | 7.591 | 0.2 % | object_supported |
| round9_96x48 (SA) | 96 | 1 | 7.651 | 7.651 | 7.644 | 0.1 % | object_supported |

层流锚点：速度场里只有 **1 个连通再循环单元**，但壁剪配对出 **2 个气泡**；该单元在 `x/H≈1.25` 只是**抬离壁面**，
抬离带 `1.25–3.9` 内 `max|τ|` 仅为剖面峰值的 **2.15 %**，随后在 `x/H≈8.26` 重新贴壁 ⇒ `object_incomplete`。
阈值扫描 `u<0 / -0.002 / -0.005 / -0.01 / -0.02 / -0.05` 全部给出 **1 个单元**，结论不依赖阈值。

## ④ 结果图

4 张（内嵌）：归档健康图、壁剪气泡 vs 场侧贴壁区间对照图、口径带宽图、再循环单元云图。

## ⑤ 注意点 / 局限

1. 单元口径 = `u<0` 连通块（单元中心、4-连通、x 周期）；贴壁行是第一层单元中心 `y≈0.033H`，不是壁面本身。
2. 完整性门只看 `mean_u`；归档无 `samples`/收敛记录，「发散 / 作业被杀 / 写盘失败」在本单只能统一记作 `zero_field`。
3. 两条口径共享同一套穿越插值实现，只独立于**输入**，不独立于插值。
4. **未重跑任何求解器**：round 23/25 的遗留（加密档是收敛还是系统性过预测）**仍未回答**，本单只把「答不了」的原因钉成数据事实。
5. 层流「−76.8 %」是 `wall_shear:first` 口径的产物；同一解在 largest / 场侧口径下是 **+52.8 %**，此后引用必须带口径。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 口径 | x_r/H | 对 5.41 误差 | <3 % ? | <10 % ? |
|---|---|---|---|---|
| 层流 Re=2800 壁剪 first（round 23 发布值） | 1.254 | −76.8 % | 否 | 否 |
| 层流 Re=2800 壁剪 largest | 8.264 | +52.8 % | 否 | 否 |
| 层流 Re=2800 场侧单元末贴壁点 | 8.258 | +52.6 % | 否 | 否 |
| SST round 18 (96×48) | 7.682 | +42.0 % | 否 | 否 |
| SA round 9 (96×48) | 7.651 | +41.4 % | 否 | 否 |

**没有任何口径达到 <3 %，也没有任何口径达到 <10 %** —— 锚点保持 **honest FAIL**。
口径不变的一点：三种口径下 SST 都最接近参考，所以「最优闭包 = SST」不是口径伪影；随口径移动的只是误差量级（层流带宽 129.6 %，验收带只有 20 %）。

## ⑦ 给下次的线索

1. 要让 round 23/25 真正可答，须按 round-23 探针钉住的稳定 `dt`（144×72→0.00665、192×96→0.0025）重跑并将**非零**场入档，先过 `artifact_health` 门。
2. 验收条目应写死口径（`wall_shear:first` 或 `field_cell_last_wall_span`）——现有总账只写「再附长度」，在本求解器族里不唯一。
3. `lift_off_band` + `lift_off_shear_ratio` 可作为闭包对比的附加指标（层流 0.0215，SST 无抬离带）。

---

产物：commit `a636d030`（分支 agent/m22685）；`src/tensorfvm/verification/periodic_hill_archive_audit.py`；
`scripts/round26_archive_audit.py`；`tests/test_periodic_hill_archive_audit.py`（19 passed）；
`docs/case-periodic-hills/{round26_archive_audit.json, ROUND26.md, fig_round26_*.png}`；欠账账本 D28。
"""

payload = {
    "title": "CASE-FVM-PERIODIC-HILLS 第26回合：归档完整性门与再循环单元对象审计（x_r/H 口径带宽 129.6%）",
    "summary": ("周期山丘算例 14 个归档中 6 个是全零 dump（两条网格阶梯细档均不可用→insufficient_rungs）；"
                "层流解只有 1 个连通再循环单元但壁剪配对出 2 个气泡，x_r/H 在四种口径下于 1.254–8.264 摆动"
                "（带宽 129.6%，判 definition_sensitive，抬离带剪切仅峰值 2.15%）；SST/SA 为单单元 object_supported；"
                "无任何口径 <10%，锚点保持 honest FAIL。零求解器运行。"),
    "scenario_key": "CASE-FVM-PERIODIC-HILLS",
    "mission_id": 22685,
    "tags": ["TensorFVM", "verification", "periodic-hill", "ERCOFTAC-UFR-3-30",
             "reattachment", "artifact-integrity", "definition-sensitivity"],
    "markdown": markdown,
    "figures": figs,
}
res = post("/tensor/reports", payload, token)
print(json.dumps(res)[:600])