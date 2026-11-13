"""Digitise Figure 21 of Dumbser et al. (2019) as a published reference profile.

The figure (arXiv:1902.07773, page 21, "Figure 21: Woodward-Colella. Comparison
between the MOOD and non-MOOD schemes on B3 at T = 0.038") plots four density
curves for the two interacting blast waves of Woodward & Colella (1984) at
``T = 0.038``: three solutions of the B3/MOOD scheme family and one curve the
legend labels "Reference", which the text of Sec. 5.1.4 states is *"the
expected solution ... computed via the pure RPJ on a mesh of 3200 cells"*.

The curves are stored in the PDF as vector paths, so they are recovered at path
precision (no raster reading).  Two independent digitisation runs are made —
panel (a), the 400-cell plot, and panel (b), the 800-cell plot — of the same
"Reference" curve; their mutual deviation is the digitisation error budget.

Outputs (written next to this file unless ``--out`` is given):
  * ``curves.json`` — every recovered path in data coordinates + calibration,
  * ``reference_density.csv`` — the two digitised "Reference" curves on a
    common x grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

PANEL_WINDOWS = {"a": (58.0, 292.0), "b": (294.0, 525.0)}
Y_BAND = (352.0, 470.0)
MIN_ITEMS = 50
REFERENCE_COLOR = (0.0, 0.0, 0.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def affine(pairs):
    """Least-squares ``pixel = a*value + b`` over (pixel, value) pairs."""
    pairs = sorted(set(pairs))
    if len(pairs) < 2:
        raise SystemExit(f"not enough ticks: {pairs}")
    v = np.array([p[1] for p in pairs], float)
    p = np.array([p[0] for p in pairs], float)
    A = np.stack([v, np.ones_like(v)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, p, rcond=None)
    resid = p - (a * v + b)
    return {"a": float(a), "b": float(b), "n_ticks": int(len(pairs)),
            "max_abs_residual_pt": float(np.abs(resid).max()),
            "ticks": [[float(pp), float(vv)] for pp, vv in pairs]}


def calibrate(page, window):
    """x calibration from bottom tick labels, y calibration from left ones."""
    words = page.get_text("words")
    xlab, ylab = [], []
    for x0, y0, x1, y1, txt, *_ in words:
        try:
            val = float(txt)
        except ValueError:
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if not (Y_BAND[0] <= cy <= Y_BAND[1] + 25):
            continue
        if not (window[0] <= cx <= window[1]):
            continue
        # x tick labels sit just below the axis, y tick labels just left of it
        if 464.0 <= cy <= 474.0 and 0.3 <= val <= 1.05:
            xlab.append((cx, val))
        elif abs(val - round(val)) < 1e-9 and 0 <= val <= 20 and cy < 464.0:
            ylab.append((cy, val))
    return affine(xlab), affine(ylab), xlab, ylab


def paths(page, window):
    out = {}
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if len(d["items"]) < MIN_ITEMS:
            continue
        if not (Y_BAND[0] <= r.y0 and r.y1 <= Y_BAND[1]):
            continue
        if not (window[0] <= r.x0 and r.x1 <= window[1]):
            continue
        col = tuple(round(c, 2) for c in (d.get("color") or ()))
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts.append((it[1].x, it[1].y))
                pts.append((it[2].x, it[2].y))
            elif it[0] == "c":
                for p in it[1:]:
                    pts.append((p.x, p.y))
        if len(pts) < 20:
            continue
        out.setdefault(col, []).append({"width": d.get("width"), "points": np.array(pts)})
    return out


def to_data(groups, fx, fy):
    res = {}
    for col, plist in groups.items():
        xs, ys = [], []
        for p in plist:
            xs.append((p["points"][:, 0] - fx["b"]) / fx["a"])
            ys.append((p["points"][:, 1] - fy["b"]) / fy["a"])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        order = np.argsort(x, kind="stable")
        res[col] = {"x": x[order], "y": y[order],
                    "n_paths": len(plist),
                    "width": plist[0]["width"],
                    "x_range": [float(x.min()), float(x.max())],
                    "y_range": [float(y.min()), float(y.max())]}
    return res


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--pdf", type=Path, default=here / "pdf" / "1902.07773.pdf")
    ap.add_argument("--page", type=int, default=21)
    ap.add_argument("--out", type=Path, default=here)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page]
    report = {"source": "Dumbser et al., J. Comput. Phys. 2019 (arXiv:1902.07773)",
              "pdf": args.pdf.name, "pdf_sha256": sha256(args.pdf),
              "page": args.page, "caption": "Figure 21: Woodward-Colella. Comparison "
              "between the MOOD and non-MOOD schemes on B3 at T = 0.038.",
              "panels": []}
    refs = {}
    for tag, window in PANEL_WINDOWS.items():
        fx, fy, xlab, ylab = calibrate(page, window)
        groups = paths(page, window)
        data = to_data(groups, fx, fy)
        entry = {"panel": tag, "window_pt": list(window),
                 "x_calibration": fx, "y_calibration": fy,
                 "curves": {}}
        for col, d in sorted(data.items(), key=lambda kv: -len(kv[1]["x"])):
            entry["curves"][str(col)] = {
                "n_paths": d["n_paths"], "width": d["width"],
                "n_points": int(d["x"].size),
                "x_range": d["x_range"], "y_range": d["y_range"]}
            if col == REFERENCE_COLOR:
                refs[tag] = d
        report["panels"].append(entry)

    if set(refs) != {"a", "b"}:
        raise SystemExit(f"reference curve missing: found {list(refs)}")

    lo = max(refs["a"]["x"].min(), refs["b"]["x"].min())
    hi = min(refs["a"]["x"].max(), refs["b"]["x"].max())
    grid = np.linspace(lo, hi, 1201)
    ya = np.interp(grid, refs["a"]["x"], refs["a"]["y"])
    yb = np.interp(grid, refs["b"]["x"], refs["b"]["y"])
    dev = np.abs(ya - yb)
    report["reference"] = {
        "colour_rgb": list(REFERENCE_COLOR),
        "overlap_x": [float(lo), float(hi)],
        "n_grid": int(grid.size),
        "rho_max": float(max(ya.max(), yb.max())),
        "rho_min": float(min(ya.min(), yb.min())),
        "digitisation_selfconsistency": {
            "max_abs_dev": float(dev.max()),
            "median_abs_dev": float(np.median(dev)),
            "rms_abs_dev": float(np.sqrt((dev ** 2).mean())),
            "max_abs_dev_percent_of_local": float((dev / np.maximum(1e-9, np.abs(ya))).max() * 100),
            "x_of_max_abs_dev": float(grid[int(dev.argmax())]),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "curves.json").write_text(json.dumps(report, indent=2))
    lines = ["x,rho_reference_panel_a,rho_reference_panel_b"]
    lines += [f"{x:.6f},{a:.6f},{b:.6f}" for x, a, b in zip(grid, ya, yb)]
    (args.out / "reference_density.csv").write_text("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in report["reference"].items()}, indent=2))
    for entry in report["panels"]:
        print(f"panel {entry['panel']}: x cal {entry['x_calibration']['a']:.3f} pt/unit "
              f"(resid {entry['x_calibration']['max_abs_residual_pt']:.3f} pt), "
              f"y cal {entry['y_calibration']['a']:.3f} pt/unit "
              f"(resid {entry['y_calibration']['max_abs_residual_pt']:.3f} pt)")
        for col, info in entry["curves"].items():
            print(f"   {col} n_pts={info['n_points']} x={info['x_range']} y={info['y_range']}")


if __name__ == "__main__":
    main()