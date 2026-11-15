"""Chunked local driver for the long-time study (parts merged by ``--merge``)."""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tensorfvm.shallow_water import long_time as lt  # noqa: E402
from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET  # noqa: E402
from tensorfvm.shallow_water.exact import t_star_to_time  # noqa: E402

CASES = {"ritter-dry": RITTER_DRY, "stoker-wet": STOKER_WET}
OUT = pathlib.Path("/tmp/lt-parts")
OUT.mkdir(exist_ok=True)

PARTS = {
    "p1": [1.0, 2.0, 5.0],
    "p2": [10.0],

}


def run_part(name: str) -> None:
    t_stars = PARTS[name]
    t0 = time.perf_counter()
    for key, case in CASES.items():
        payload = lt.run_long_time_study(cases=(case,), t_stars=tuple(t_stars),
                                         controls=False)
        json.dump(payload, open(OUT / f"{name}_{key}.json", "w"))
        v = payload["cases"][key]["verdict"]
        print(f"{name} {key} t*={t_stars} pass={v['pass']} "
              f"wall={payload['wall_time_s']:.1f}s", flush=True)
    print(f"part {name} total {time.perf_counter()-t0:.1f}s", flush=True)


def run_controls() -> None:
    """Cheap falsification controls: scheme order and one CFL halving at t*=5."""
    t_star = 5.0
    rows = {}
    t0 = time.perf_counter()
    for label, kwargs in (("cfl0.4", {"cfl": 0.4, "order": 2}),
                          ("cfl0.2", {"cfl": 0.2, "order": 2}),
                          ("order1", {"cfl": 0.4, "order": 1})):
        rows[label] = {}
        for key, case in CASES.items():
            row = lt.run_point(case, t_star, dx=lt.DX, **kwargs)
            row.pop("_similarity", None)
            rows[label][key] = {k: row[k] for k in
                                ("front_error", "x_over_t_error", "speed",
                                 "depth_l1", "local_extrema", "steps")}
        print(f"controls {label} done {time.perf_counter()-t0:.1f}s", flush=True)
    json.dump({"t_star": t_star, "rows": rows}, open(OUT / "controls.json", "w"))


def merge() -> None:
    payload = {"criteria": {"tolerance": lt.TOLERANCE,
                            "speed_tolerance": lt.SPEED_TOLERANCE,
                            "collapse_tolerance": lt.COLLAPSE_TOLERANCE,
                            "step_drift_tolerance": lt.STEP_DRIFT_TOLERANCE,
                            "dx": lt.DX, "cfl": lt.CFL, "order": lt.ORDER,
                            "t_stars": list(lt.T_STARS), "t_ref": lt.T_REF,
                            "pad_c0": lt.PAD_C0},
               "cases": {}}
    for key, case in CASES.items():
        rows, profiles, exact_profile, xi = [], [], None, None
        for part in PARTS:
            data = json.loads((OUT / f"{part}_{key}.json").read_text())
            entry = data["cases"][key]
            rows.extend(entry["rows"])
            profiles.extend(entry["similarity"]["profiles"])
            exact_profile = entry["similarity"]["exact_profile"]
            xi = entry["similarity"]["xi"]
        rows.sort(key=lambda r: r["t_star"])
        profiles.sort(key=lambda p: p["t_star"])
        xi_arr = np.asarray(xi, dtype=float)
        span = xi_arr[-1] - xi_arr[0]
        ref = next(p for p in profiles if abs(p["t_star"] - lt.T_REF) < 1e-12)
        h_ref = np.asarray(ref["h"], dtype=float)
        collapse = []
        for p in profiles:
            d = np.abs(h_ref - np.asarray(p["h"], dtype=float))
            collapse.append({"t_reference": ref["t"], "t": p["t"],
                             "t_star": p["t_star"],
                             "linf_over_h0": float(d.max()),
                             "l1_over_h0": float(d.sum() / (xi_arr.size * span)),
                             "xi_span": [float(xi_arr[0]), float(xi_arr[-1])]})
        speed_rows = [dict(r["speed"], t_star=r["t_star"]) for r in rows]
        worst = max((r for r in speed_rows if r.get("defined")),
                    key=lambda r: r["speed_error"])
        speed = {"per_t_star": speed_rows, "defined": True, "worst": worst}
        payload["cases"][key] = {
            "reference": case.reference,
            "rows": rows,
            "similarity": {"xi": xi, "exact_profile": exact_profile,
                           "profiles": profiles,
                           "collapse": collapse,
                           "exact_collapse": [
                               {"t_reference": ref["t"], "t": p["t"],
                                "t_star": p["t_star"], "linf_over_h0": 0.0,
                                "l1_over_h0": 0.0} for p in profiles]},
            "speed": speed,
            "verdict": lt._verdict(key, rows, collapse, worst),
        }
    ctrl = json.loads((OUT / "controls.json").read_text())
    rows = ctrl["rows"]
    payload["controls"] = {
        "t_star": ctrl["t_star"],
        "cfl_refinement": {"0.4": rows["cfl0.4"], "0.2": rows["cfl0.2"]},
        "order_reference": {"2": rows["cfl0.4"], "1": rows["order1"]},
    }
    payload["wall_time_s"] = 0.0
    dest = pathlib.Path("src/tensorfvm/verification/artifacts/"
                        "shallow-water-long-time")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "study.json").write_text(json.dumps(payload, indent=1))
    # render the summary + figures from the merged payload
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "runner", pathlib.Path(__file__).with_name("run_shallow_water_long_time.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    (dest / "SUMMARY.md").write_text(runner.render_summary(payload))
    figdir = pathlib.Path("docs/case-fvm-dambreak-stoker-long-time")
    figdir.mkdir(parents=True, exist_ok=True)
    runner.figure_errors(payload, figdir / "long_time_errors.png")
    runner.figure_collapse(payload, figdir / "long_time_collapse.png")
    print("merged ->", dest / "study.json")
    for key, case in payload["cases"].items():
        print(key, "PASS" if case["verdict"]["pass"] else "FAIL",
              [c["name"] for c in case["verdict"]["criteria"] if not c["pass"]])


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "merge":
        merge()
    elif what == "controls":
        run_controls()
    else:
        run_part(what)