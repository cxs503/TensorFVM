"""Render the human-readable views of the TensorFVM OSS inventory registry.

``docs/oss-inventory-fvm/registry.json`` is the single source of truth (built by
``build_registry.py``); this script derives ``registry.md`` (tiered table) and
``gap-proposals.md`` (the GAP-FVM-* anchors).  Nothing is typed by hand, so the
markdown can never drift from the JSON.

Usage::

    python scripts/oss_inventory_fvm/render_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "oss-inventory-fvm" / "registry.json"


def _meta_cells(repo: dict) -> tuple[str, str, str]:
    """(stars, licence, last push) with an explicit marker when unmeasured."""
    meta = repo["metadata"]
    source = meta.get("source")
    if source == "github_api":
        return (str(meta.get("stars")), meta.get("license_spdx") or "n/a",
                (meta.get("pushed_at") or "")[:10])
    if source == "upstream_not_github":
        return ("n/m", "n/m", "n/m")
    return ("closed", "closed", "n/m")


def render_registry(reg: dict) -> str:
    repos = reg["repos"]
    lines = [
        "# TensorFVM 开源 FVM 技术盘点：仓级注册表与分级（registry + tier）",
        "",
        "> 程序化生成，请勿手改：`python scripts/oss_inventory_fvm/build_registry.py`"
        " → `python scripts/oss_inventory_fvm/render_report.py`。",
        f"> 生成时间 {reg['generated_at']}；目标仓 {reg['target_repo']}；"
        f"mission {reg['mission']}。",
        "",
        "## 0. 证据口径（读表前必读）",
        "",
        "| 通道 | 含义 | 可信度 |",
        "|---|---|---|",
        "| `metadata.source = github_api` | 本轮实测：stars / SPDX licence / 最近 push / 语言，"
        "逐仓记录 slug 与抓取时间 | **实测** |",
        "| `metadata.source = upstream_not_github` | 项目主仓不在 GitHub（自建 GitLab / darcs / "
        "SourceForge / NASA 发布），本轮**无法**测量元数据 | 未测（字段留空，不得引用为现状） |",
        "| `metadata.source = commercial` | 闭源商业软件，仅作验收基线 | 不可引用 |",
        "| `curation.*` | 专家标注：技术族/离散方式/能力标签/分级/缺口映射 | **标注**（本轮未逐仓读源码复核） |",
        "",
        f"计数：{reg['counts']['repos']} 个项目，T1={reg['counts']['by_tier']['T1']} / "
        f"T2={reg['counts']['by_tier']['T2']} / T3={reg['counts']['by_tier']['T3']}，"
        f"元数据实测 {reg['counts']['measured_metadata']} 个，缺口锚点 {reg['counts']['gaps']} 个。",
        "",
        "## 1. 分级定义",
        "",
        "| Tier | 含义 |",
        "|---|---|",
    ]
    for tier, text in reg["tiers"].items():
        lines.append(f"| {tier} | {text} |")
    lines += ["", "## 2. 全量注册表", "",
              "| 项目 | id | Tier | 技术族 | 离散方式 | stars | licence | 最近 push | 缺口映射 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for repo in sorted(repos, key=lambda r: (r["tier"], r["id"])):
        stars, lic, pushed = _meta_cells(repo)
        gaps = ", ".join(repo["gap_targets"]) or "—"
        lines.append(f"| {repo['name']} | `{repo['id']}` | {repo['tier']} | {repo['family']} | "
                     f"{repo['discretisation']} | {stars} | {lic} | {pushed} | {gaps} |")
    lines += ["", "## 3. 分级理由与可吸收内容（Tier 1 全列）", ""]
    for repo in sorted(repos, key=lambda r: (r["tier"], r["id"])):
        if repo["tier"] != "T1":
            continue
        meta = repo["metadata"]
        slug = meta.get("slug") or repo.get("homepage") or "n/a"
        lines += [
            f"### {repo['id']} — {repo['name']}",
            "",
            f"- 来源：{slug}（metadata `{meta.get('source')}`）",
            f"- 分级理由：{repo['tier_rationale']}",
            f"- 可吸收：{repo['absorb']}",
            f"- 许可风险：{repo['license_risk']}",
            f"- 缺口映射：{', '.join(repo['gap_targets']) or '—'}",
            "",
        ]
    lines += ["## 4. 未实测元数据的项目（诚实清单）", "",
              "| 项目 | 主仓 | 说明 |", "|---|---|---|"]
    for repo in repos:
        meta = repo["metadata"]
        if meta.get("source") in ("upstream_not_github", "commercial"):
            lines.append(f"| {repo['name']} | {repo.get('homepage') or '—'} | "
                         f"{meta.get('note', '')} |")
    lines += ["", "> 上述项目的能力标注同样只是标注；若纳入采纳路线，第一步是拉取源码做一次"
                  "源读审计（本波未做）。", ""]
    return "\n".join(lines)


def render_gaps(reg: dict) -> str:
    repos = {r["id"]: r for r in reg["repos"]}
    gaps = reg["gaps"]
    lines = [
        f"# TensorFVM 缺口锚点提议（GAP-FVM-*，本波 {len(gaps)} 条）",
        "",
        f"> 程序化生成，勿手改。源：`docs/oss-inventory-fvm/registry.json` 的 `gaps` 段"
        f"（生成时间 {reg['generated_at']}）。",
        "> 每条的「现状证据」都必须是本仓可 grep 到的路径；「验收判据」必须是机器可裁的数值门；"
        "「外部锚」必须落在文献/解析解，不得用本仓自己的实现自证。",
        "",
        "## 排序总表",
        "",
        "| # | 锚点 | 优先级 | 参考实现（Tier） | 工作量 | 判据要点 |",
        "|---|---|---|---|---|---|",
    ]
    for i, gap in enumerate(gaps, 1):
        refs = ", ".join(f"{repos[x]['name']}({repos[x]['tier']})"
                         for x in gap["reference_repos"][:3])
        lines.append(f"| {i} | `{gap['id']}` | {gap['priority']} | {refs} | {gap['effort']} | "
                     f"{gap['acceptance'].split(';')[0]} |")
    lines += ["", "## 逐条详述", ""]
    for gap in gaps:
        refs = []
        for rid in gap["reference_repos"]:
            repo = repos[rid]
            stars = (f", {repo['metadata'].get('stars')}★"
                     if repo["metadata"].get("source") == "github_api" else "")
            refs.append(f"{repo['name']}[{repo['tier']}{stars} → 吸: {repo['absorb']}]")
        risks = sorted({repos[x]["license_risk"] for x in gap["reference_repos"]})
        lines += [
            f"### `{gap['id']}` — {gap['title']}",
            "",
            f"- **优先级**：{gap['priority']}；**工作量**：{gap['effort']}",
            f"- **现状（TensorFVM）**：{gap['current_state']}",
            f"- **现状证据**：{gap['evidence_note']}",
            "- **证据文件**：" + ", ".join(f"`{p}`" for p in gap["evidence_in_tensorfvm"]),
            "- **参考实现**：" + ", ".join(refs),
            f"- **验收判据（机器可裁）**：{gap['acceptance']}",
            f"- **外部锚**：{gap['external_anchor']}",
            "-- 许可提醒：" + "；".join(risks),
            "",
        ]
    lines += [
        "## 两条 P0 的落地顺序建议", "",
        "1. **先 `GAP-FVM-LOWMACH-PRECOND` 后 `GAP-FVM-IMPLICIT-DUAL-TIME`**：低马赫预条件会改变"
        "方程组的谱，隐式线性化必须在它之后标定，否则 LU-SGS 的 CFL 上界要重做一次。",
        "2. `GAP-FVM-AMG-KRYLOV` 与上面两条 P0 共享验收载体（同一个低马赫算例的线性求解次数），"
        "可以并行立项但共用一套诊断。",
        "3. `GAP-FVM-AMR-QUADTREE` 的验收刻意复用本波的 Sod 外部锚"
        "（`artifacts/sod_literature_benchmark.json`），AMR 版与均匀 400 格解比对，"
        "避免再引入一个未验证的基准。", "",
        "## 与既有账本的关系", "",
        "- `docs/verification-gap-audit-m7555.md` 已把「燃烧/组分输运」「辐射换热」列为验证缺口，"
        "本波把二者升级为**能力缺口**并补上外部锚与参考实现。",
        "- 平台 campaign_nodes 已有多条 FVM 两相/VOF/热对流节点（本波检索命中），故 VOF 两相"
        "**不**列为缺口：TensorFVM 侧已有 PLIC/CICSAM 与三条验证面。",
        "- ALE/6DOF 与 overset 在本仓已有实现（`moving_mesh.py`/`six_dof.py`/`overset*.py`），"
        "本波不重复立项。", "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=str(DEFAULT_JSON))
    args = ap.parse_args(argv)
    path = Path(args.registry)
    reg = json.loads(path.read_text(encoding="utf-8"))
    outdir = path.parent
    (outdir / "registry.md").write_text(render_registry(reg) + "\n", encoding="utf-8")
    (outdir / "gap-proposals.md").write_text(render_gaps(reg) + "\n", encoding="utf-8")
    print(json.dumps({"registry_md": str(outdir / "registry.md"),
                      "gap_proposals_md": str(outdir / "gap-proposals.md")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())