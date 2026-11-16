"""scripts/audit/measure_rewrite_pack.py 的单元测试（小合成仓，秒级）。

守的是三个**不变量**，不是「能跑」：
1. 删除集判定用「全部历史路径」：同一 blob 同时出现在 `docs/`（会被规则删）与
   `src/`（保留）时，该 blob **不得**被判死；
2. 判死的 blob 必须真的命中规则，且判定与 `git rev-list` 的单路径近似区分开；
3. 实测重打包：删掉对象后 pack 字节数必须严格变小且非零（量的是真 pack，不是估计）。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "measure_rewrite_pack", REPO_ROOT / "scripts" / "audit" / "measure_rewrite_pack.py"
)
mrp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mrp)


def _git(repo: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(repo), *args],
                        capture_output=True, text=True, check=False)
    assert cp.returncode == 0, cp.stderr
    return cp.stdout


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    r = tmp_path_factory.mktemp("slimrepo")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")

    # 可压缩性差的随机字节，保证体积可观测
    field = os.urandom(200_000)
    shared = os.urandom(50_000)
    _write(r / "docs" / "caseA" / "field.npz", field)
    _write(r / "docs" / "caseA" / "report.pdf", os.urandom(80_000))
    _write(r / "src" / "code.py", b"print('hi')\n")
    # 同一内容、两条路径：一条会被 R1 删，一条保留 -> 该 blob 必须存活
    _write(r / "docs" / "caseA" / "shared.npz", shared)
    _write(r / "src" / "shared.npz", shared)
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "c1")

    # 后续再改一次 docs 工件，制造历史版本（delta 机会）
    _write(r / "docs" / "caseA" / "field.npz", field + b"v2")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "c2")
    # 并在之后删除，模拟「已删除但对象仍在」
    _git(r, "rm", "-q", "-r", "docs")
    _git(r, "commit", "-q", "-m", "c3 remove artifacts")
    return r


def test_occurrence_map_keeps_all_paths(repo: Path) -> None:
    occ = mrp.occurrence_map(str(repo))
    oids = mrp.reachable_objects(str(repo))
    info = mrp.object_sizes(str(repo), oids)
    targets = [o for o, (t, _) in info.items()
               if t == "blob" and occ.get(o) == {"docs/caseA/shared.npz", "src/shared.npz"}]
    assert targets, "共享内容的 blob 应同时记录 docs/ 与 src/ 两条路径"
    # 该 blob 在 src/ 侧存活 -> R1 不得判死
    rule = mrp.SCENARIOS["R1"][1]
    for oid in targets:
        assert not all(rule(p) for p in occ[oid])


def test_dead_set_matches_rule_and_spares_shared_blob(repo: Path) -> None:
    occ = mrp.occurrence_map(str(repo))
    oids = mrp.reachable_objects(str(repo))
    info = mrp.object_sizes(str(repo), oids)
    blobs = {o: s for o, (t, s) in info.items() if t == "blob"}
    rule = mrp.SCENARIOS["R1"][1]
    dead, alive_paths = set(), set()
    for oid in blobs:
        paths = occ.get(oid, set())
        assert paths, "合成仓无 merge，所有 blob 应有路径"
        (dead.add(oid) if all(rule(p) for p in paths) else alive_paths.add(oid))
    dead_paths = {p for oid in dead for p in occ[oid]}
    assert "docs/caseA/field.npz" in dead_paths
    assert "src/shared.npz" not in dead_paths
    assert any("src/code.py" in occ[o] for o in alive_paths)


def test_measure_pack_shrinks_and_is_positive(repo: Path) -> None:
    oids = mrp.reachable_objects(str(repo))
    info = mrp.object_sizes(str(repo), oids)
    occ = mrp.occurrence_map(str(repo))
    rule = mrp.SCENARIOS["R1"][1]
    dead = {o for o, (t, _) in info.items()
            if t == "blob" and occ.get(o) and all(rule(p) for p in occ[o])}
    assert dead
    all_bytes = mrp.measure_pack(str(repo), oids, 10, 50, 2)
    kept_bytes = mrp.measure_pack(str(repo), [o for o in oids if o not in dead], 10, 50, 2)
    assert kept_bytes > 0
    assert kept_bytes < all_bytes, "删掉工件后重打包必须变小"


def test_threshold_rules_take_size_argument() -> None:
    """R5/R5b 是体积阈值规则：签名不同，必须被 SIZE_RULES 标记到。"""
    assert {"R5", "R5b"} <= set(mrp.SCENARIOS)
    assert mrp.SCENARIOS["R5"][1]("docs/x.bin", 2 * 1024 * 1024) is True
    assert mrp.SCENARIOS["R5"][1]("docs/x.bin", 10) is False
    assert mrp.SCENARIOS["R5b"][1]("docs/x.bin", 600 * 1024) is True
    assert mrp.SCENARIOS["R5b"][1]("src/x.bin", 600 * 1024) is False


def test_extension_families_are_disjoint_from_text() -> None:
    """R3 的文本/二进制边界：源码与文档不得被当二进制删掉。"""
    for path in ("src/tensorfvm/solver.py", "docs/README.md", "docs/report.json"):
        assert mrp.SCENARIOS["R3"][1](path) is False, path
    for path in ("docs/fig/a.png", "docs/fig/a.npz", "src/pkg/mod.pyc"):
        assert mrp.SCENARIOS["R3"][1](path) is True, path