#!/usr/bin/env python3
"""FVM-REPO-SLIM B3 场景**实测**（只读审计工具，源仓库零写入）。

背景
----
docs/repo-slim-plan.md（m23856）推荐案 B3 = 「历史大 blob 打桩：blob ∉ HEAD 树 且
>100 KiB → 替换为 1 行占位」，但当时 git-filter-repo 的 ``--blob-callback`` 在
mirror 上崩（fast-import "stream ends early"），B3 数字只能给区间估算
（~150-250 MiB），方案里明确「B3 精确数字在实施单里用该路径法复测后回填」。

本脚本把 B3 从估算变成实测，方法与 ``measure_rewrite_pack.py`` 完全同构：
1. 存活对象集 = 全部可达对象 − 死 blob 集；
2. 死 blob 集（B3，**blob 粒度**，比路径粒度更准）：
   ``size > 100 KiB 且 oid 不在 HEAD 树`` 的 blob；
3. 存活集喂给真实 ``git pack-objects --stdout``，量字节数（含 git 真实 delta
   复用，无标定系数）。打桩后所有死 blob 收敛为 1 个 ~100 B 的共享 stub，
   对 pack 体积影响可忽略（<1 KiB），故直接剔除死 blob 即等价。

同时复测 B2（m23856 口径：全仓路径 glob 删二进制，含 HEAD）与 R2c/R3
（m23832 runbook 口径），用于当前 HEAD 的漂移复核。

用法
----
    python3 scripts/audit/slim_b3_measure.py \
        --repo /root/tensor-ecosystem/TensorFVM \
        --out docs/FVM-REPO-SLIM-M23910
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_rewrite_pack import (  # noqa: E402
    MiB, SCENARIOS, occurrence_map, reachable_objects, object_sizes,
    measure_pack, git,
)

B3_THRESHOLD = 100 * 1024  # 与方案 B3 定义一致

# m23856 方案 B2 口径：全仓（不限 docs/）按扩展名删二进制
B2_EXTS = (".pyc", ".npz", ".msh", ".svg", ".pdf", ".png", ".gz")


def _b2(path: str) -> bool:
    return path.lower().endswith(B2_EXTS)


def head_tree_blobs(repo: str) -> dict[str, int]:
    """HEAD 树 path -> blob oid（含大小需另查；这里只要 oid 集合）。"""
    cp = git(repo, "ls-tree", "-r", "-z", "HEAD")
    if cp.returncode != 0:
        raise SystemExit(f"ls-tree failed: {cp.stderr.decode()[:400]}")
    out: dict[str, str] = {}
    for entry in cp.stdout.decode("utf-8", "replace").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        fields = meta.split()
        # <mode> <type> <oid>\t<path>
        out[path] = fields[2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/tensor-ecosystem/TensorFVM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()
    repo = args.repo

    occ = occurrence_map(repo)
    oids = reachable_objects(repo)
    info = object_sizes(repo, oids)
    blobs = {o: s for o, (t, s) in info.items() if t == "blob"}
    head = head_tree_blobs(repo)
    head_blob_set = set(head.values())
    print(f"[i] reachable={len(oids)} blobs={len(blobs)} "
          f"head_files={len(head)} head_blobs={len(head_blob_set)}", flush=True)

    # HEAD 树里 >100 KiB 的文件 = B3 路径法回退的「冲突路径」（删路径会误伤 HEAD）
    head_big = {p: b for p, b in head.items() if blobs.get(b, 0) > B3_THRESHOLD}

    # 基线：keep-all 重打包
    baseline = measure_pack(repo, oids, args.window, args.depth, args.threads)
    print(f"[i] baseline keep-all pack = {baseline / MiB:.1f} MiB "
          f"({time.time() - t0:.0f}s)", flush=True)

    results = []

    def run_tag(tag: str, desc: str, dead: set[str]) -> None:
        dropped = sum(blobs[o] for o in dead)
        survivors = [o for o in oids if o not in dead]
        meas = measure_pack(repo, survivors, args.window, args.depth, args.threads)
        results.append({
            "tag": tag,
            "desc": desc,
            "dropped_blobs": len(dead),
            "dropped_MiB": round(dropped / MiB, 2),
            "measured_pack_MiB": round(meas / MiB, 2),
            "reduction_pct": round(100.0 * (1 - meas / baseline), 2),
            "n_surviving_objects": len(survivors),
        })
        print(f"[i] {tag}: dropped {len(dead)} blobs / {dropped / MiB:.1f} MiB "
              f"-> measured pack {meas / MiB:.1f} MiB ({time.time() - t0:.0f}s)",
              flush=True)

    # B3：blob 粒度打桩（推荐案）
    b3_dead = {o for o, s in blobs.items() if s > B3_THRESHOLD and o not in head_blob_set}
    run_tag("B3", f"历史大 blob 打桩（>100 KiB 且 ∉ HEAD 树；HEAD 树零改动）", b3_dead)

    # B3+pyc：B3 + 全仓 *.pyc（pyc 本就不该入库，HEAD 里若有一并剔除后 re-add）
    b3p_dead = set(b3_dead)
    for o, s in blobs.items():
        paths = occ.get(o, ())
        if paths and all(p.lower().endswith(".pyc") for p in paths):
            b3p_dead.add(o)
    run_tag("B3p", "B3 + 全仓 *.pyc", b3p_dead)

    # B2：m23856 口径（全仓路径 glob 删二进制，含 HEAD 树）
    b2_dead = set()
    for o, s in blobs.items():
        paths = occ.get(o, ())
        if paths and all(_b2(p) for p in paths):
            b2_dead.add(o)
    run_tag("B2", "全历史删二进制（*.pyc/npz/msh/svg/pdf/png/gz，含 HEAD）", b2_dead)

    # 漂移复核：R2c / R3（m23832 runbook 口径）
    for tag in ("R2c", "R3"):
        desc, rule = SCENARIOS[tag]
        dead = set()
        for o, s in blobs.items():
            paths = occ.get(o, ())
            if paths and all(rule(p) for p in paths):
                dead.add(o)
        run_tag(tag, desc, dead)

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": os.path.abspath(repo),
        "git_head": git(repo, "rev-parse", "HEAD").stdout.decode().strip(),
        "git_head_short": git(repo, "rev-parse", "--short", "HEAD").stdout.decode().strip(),
        "pack_params": {"window": args.window, "depth": args.depth,
                        "threads": args.threads},
        "b3_threshold_bytes": B3_THRESHOLD,
        "reachable": {
            "n_objects": len(oids),
            "n_blobs": len(blobs),
            "blob_MiB": round(sum(blobs.values()) / MiB, 2),
        },
        "head_tree": {
            "n_files": len(head),
            "n_blobs": len(head_blob_set),
            "files_over_100KiB": len(head_big),
            "files_over_1MiB": sum(1 for b in head_blob_set
                                 if blobs.get(b, 0) >= MiB),
        },
        "baseline_keepall_pack_MiB": round(baseline / MiB, 2),
        "scenarios": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "measured_pack_B2-B3-R2c-R3.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"[i] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
