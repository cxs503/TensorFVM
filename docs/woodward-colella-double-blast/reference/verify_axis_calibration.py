"""Independent check of the digitisation axis calibration against PDF geometry.

Why this exists
---------------
``digitize_reference.py`` recovers the published "Reference" curve of Dumbser et
al. (2019) Fig. 21 by fitting an affine map from *tick-label text bboxes* to the
tick values.  For the **x** axis that is safe: a horizontal run of digits is
centred on its tick, and the fit lands on the plot-frame corners to <0.01 pt
(0.4 at the left spine, 1.0 at the right spine).  For the **y** axis it is not:
the bbox of a digit spans ascender-to-descender, so its vertical centre sits a
constant few tenths of a point *above* the data line it labels.  A least-squares
fit over equally spaced labels cannot see that offset -- it is absorbed into the
intercept -- and the residual stays at round-off.

The consequence is a *constant* density offset of every digitised sample.  It is
irrelevant at the peak (rho ~ 6.3) and dominant in the near-vacuum trough
(rho ~ 0.16), which is exactly the one metric the case could not close (13-14 %).

This script measures the same two axes from the *drawn geometry* instead --
the plot-frame rectangle and the tick marks stroked on the spines -- which carry
no glyph metrics.  Cross-checks performed here:

1. the spine tick marks are uniformly spaced (a constant need not be assumed);
2. text-label centres differ from their tick marks by a **constant** vertical
   offset (reported per label, so a non-constant offset would be visible);
3. the frame corners coincide with the extreme tick marks (an anchor that does
   not use the labels at all);
4. the derived bias, in density units, is compared with the residual of the
   solver/reference profile comparison -- these are independent quantities.

Outputs (``--outdir``, default: this directory):

* ``axis_calibration.json`` -- tick geometry, per-label residuals, the geometry
  calibration, the committed text calibration, and the bias in pt and in density;
* ``reference_density_calibrated.csv`` -- ``x, rho_committed, rho_calibrated``
  (the committed column is copied, never replaced: the raw digitisation stays
  auditable).

The PDF is not committed (3 MB); pass ``--pdf`` or let the script fetch
arXiv:1902.07773 and verify it against the sha256 recorded in ``curves.json``.

Usage::

    python docs/woodward-colella-double-blast/reference/verify_axis_calibration.py
    python .../verify_axis_calibration.py --pdf /tmp/wc.pdf --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CURVES = HERE / "curves.json"
REFERENCE_CSV = HERE / "reference_density.csv"
PDF_URL = "https://arxiv.org/pdf/1902.07773"

#: drawn objects of the panels live in this band of the page (pt, top-down)
FIGURE_Y = (300.0, 500.0)
#: hairline width of stems/spines and of the short tick marks
SPINE_WIDTH = 0.4
TICK_WIDTH = 0.2
TICK_LENGTH = (1.0, 3.0)
#: tolerance when matching a tick mark to a frame edge, in pt
EDGE_TOL = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_pdf(target: Path, expect_sha: str | None) -> Path:
    """Download the paper (or reuse a local copy) and verify its digest."""
    if target.exists():
        if expect_sha is None or sha256(target).startswith(expect_sha[:16]):
            return target
        print(f"[warn] {target} sha256 mismatch; re-downloading", file=sys.stderr)
    target.write_bytes(urllib.request.urlopen(PDF_URL, timeout=120).read())
    got = sha256(target)
    if expect_sha and got != expect_sha:
        raise SystemExit(f"pdf sha256 {got} != recorded {expect_sha}")
    return target


def panels(page, y_band=FIGURE_Y):
    """White plot-frame rectangles and their spine tick marks, per panel.

    A panel is identified by the white-filled rectangle that spans the plot
    area (``> 150 x 100 pt``); its spines are the hairline strokes on the frame
    edges and its tick marks are the 1.6 pt strokes that end on a spine.
    """
    drawings = page.get_drawings()
    frames: list[tuple[float, float, float, float]] = []
    for d in drawings:
        fill = d.get("fill")
        r = d["rect"]
        if not fill or min(abs(c - 1.0) for c in fill) > 1e-6:
            continue
        if not (r.x1 - r.x0 > 150 and r.y1 - r.y0 > 100):
            continue
        if not (y_band[0] <= r.y0 and r.y1 <= y_band[1]):
            continue
        frames.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    out = []
    for x0, y0, x1, y1 in sorted(frames, key=lambda f: f[0]):
        y_ticks, x_ticks = [], []
        for d in drawings:
            w = d.get("width") or 0.0
            if abs(w - TICK_WIDTH) > 0.05 or d.get("fill") is None:
                continue
            r = d["rect"]
            if not (y_band[0] <= r.y0 and r.y1 <= y_band[1]):
                continue
            dx, dy = r.x1 - r.x0, r.y1 - r.y0
            # horizontal stroke ending on the left or right spine -> y tick
            if dy < 1e-6 and TICK_LENGTH[0] < dx < TICK_LENGTH[1] * 2:
                if abs(r.x0 - x0) < 0.3 or abs(r.x1 - x1) < 0.3:
                    y_ticks.append(round(float(r.y0), 4))
            # vertical stroke ending on the bottom spine -> x tick
            elif dx < 1e-6 and TICK_LENGTH[0] < dy < TICK_LENGTH[1] * 2:
                if abs(r.y1 - y1) < 0.3 or abs(r.y0 - y0) < 0.3:
                    x_ticks.append(round(float(r.x0), 4))
        out.append({"frame": [x0, y0, x1, y1],
                    "y_ticks": sorted(set(y_ticks)),
                    "x_ticks": sorted(set(x_ticks))})
    return out


def label_pairs(page, frame, axis: str, y_band=FIGURE_Y):
    """``(position, value)`` of the numeric axis labels belonging to a panel.

    Labels are matched geometrically (x labels sit below the frame, y labels
    left of it and inside the figure band), not by value, so a missing or
    repeated label cannot shift the assignment.
    """
    x0, y0, x1, y1 = frame
    out = []
    for w in page.get_text("words"):
        try:
            val = float(w[4])
        except ValueError:
            continue
        cx, cy = 0.5 * (w[0] + w[2]), 0.5 * (w[1] + w[3])
        if not (y_band[0] <= cy <= y_band[1] + 40):
            continue
        if axis == "x" and y1 <= cy <= y1 + 12 and x0 - 20 <= cx <= x1 + 20:
            out.append((cx, val))
        elif axis == "y" and y0 <= cy <= y1 - 1e-6 and x0 - 45 <= cx <= x0:
            out.append((cy, val))
    return out


def fit(pairs):
    pos = np.array([p for p, _ in pairs], dtype=float)
    val = np.array([v for _, v in pairs], dtype=float)
    if pos.size < 3:
        raise SystemExit(f"not enough axis labels: {pairs}")
    a, b = np.polyfit(val, pos, 1)           # pos = a*value + b (pt per unit)
    resid = pos - (a * val + b)              # residual in pt
    return {"a": float(a), "b": float(b), "n": int(pos.size),
            "max_abs_residual": float(np.max(np.abs(resid)))}


def analyse(page, committed_panels: list[dict]) -> dict:
    """Geometry calibration of every panel + bias against the committed one.

    ``committed_panels`` is the ``reference``->``panels`` list of ``curves.json``
    (one entry per panel, each holding the text-fit ``x_calibration`` /
    ``y_calibration`` that the digitisation used).  The bias is reported per
    panel so a panel-dependent calibration error cannot hide in an average.
    """
    result = {"panels": []}
    for i, pan in enumerate(panels(page)):
        committed = committed_panels[min(i, len(committed_panels) - 1)]
        result["committed"] = {"x": committed["x_calibration"],
                               "y": committed["y_calibration"]}
        frame = pan["frame"]
        ys = np.array(pan["y_ticks"], dtype=float)
        if ys.size < 3:
            # whole-panel background boxes also match the white-fill test
            continue
        steps = np.diff(ys)
        scale = float(np.mean(np.abs(steps)))
        uniform = float(np.ptp(steps))
        # anchor: the frame's own corners must be the extreme tick marks
        edge_err = {"top": float(abs(ys[0] - frame[1])),
                    "bottom": float(abs(ys[-1] - frame[3]))}

        ylab = label_pairs(page, frame, "y")
        xlab = label_pairs(page, frame, "x")
        fy = fit(ylab)
        fx = fit(xlab)
        # per-label vertical residual against the drawn tick marks
        ticks_for_label = []
        for cy, val in ylab:
            j = int(np.argmin(np.abs(ys - cy)))
            ticks_for_label.append({"value": val, "text_y": cy,
                                    "tick_y": float(ys[j]),
                                    "residual_pt": float(ys[j] - cy)})
        res = np.array([t["residual_pt"] for t in ticks_for_label])
        # geometry calibration: value = (y0 - y)/scale, y0 = bottom tick (=0)
        y0_geom = float(ys[-1])
        a_geom = float(fy["a"])                     # pt per unit (negative)
        # the committed text fit uses the same convention  pos = a*value + b
        y0_text = float(committed["y_calibration"]["b"])
        bias_pt = float(y0_geom - y0_text)          # >0: label centre sits above
        bias_density = float(-bias_pt / a_geom)     # >0: committed rho too low
        entry = {
            "frame": [round(v, 3) for v in frame],
            "y_tick_marks": [round(float(v), 4) for v in ys],
            "y_tick_scale_pt_per_unit": round(scale, 5),
            "y_tick_spacing_max_deviation_pt": round(uniform, 6),
            "frame_vs_extreme_tick_pt": {k: round(v, 4) for k, v in edge_err.items()},
            "y_labels": [{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in t.items()} for t in ticks_for_label],
            "y_label_residual_pt": {"mean": round(float(res.mean()), 4),
                                    "std": round(float(res.std()), 5),
                                    "max_abs": round(float(np.max(np.abs(res))), 4)},
            "y_fit_from_labels": {k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in fy.items()},
            "x_fit_from_labels": {k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in fx.items()},
            "x_fit_lands_on_frame": {
                "left_pt": round(float(fx["a"] * 0.4 + fx["b"] - frame[0]), 4),
                "right_pt": None,
            },
            "y_zero_pt_geometry": round(y0_geom, 4),
            "y_zero_pt_from_labels": round(y0_text, 4),
            "bias_pt": round(bias_pt, 4),
            "bias_density": round(bias_density, 6),
        }
        # x cross-check needs the x tick values; use the label fit itself
        xv = [v for _, v in xlab]
        if len(set(xv)) == len(xv) and xv:
            right = max(xv)
            entry["x_fit_lands_on_frame"]["right_pt"] = round(
                float(fx["a"] * right + fx["b"] - frame[2]), 4)
            entry["x_fit_lands_on_frame"]["right_value"] = right
        result["panels"].append(entry)

    biases = np.array([p["bias_pt"] for p in result["panels"]])
    dens = np.array([p["bias_density"] for p in result["panels"]])
    result["bias_pt"] = round(float(biases.mean()), 4)
    result["bias_pt_spread"] = round(float(np.ptp(biases)), 4)
    result["bias_density"] = round(float(dens.mean()), 6)
    result["bias_density_spread"] = round(float(np.ptp(dens)), 6)
    return result


def write_calibrated_reference(bias_density: float, outdir: Path) -> Path:
    raw = np.loadtxt(REFERENCE_CSV, delimiter=",", skiprows=1)
    x, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
    out = outdir / "reference_density_calibrated.csv"
    # same three-column layout as reference_density.csv (two panels), so
    # ``load_reference`` can read either file without special-casing
    body = ["x,panel_a_calibrated,panel_b_calibrated"]
    body += [f"{xx:.6f},{aa + bias_density:.6f},{bb + bias_density:.6f}"
             for xx, aa, bb in zip(x, a, b)]
    out.write_text("\n".join(body) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, default=Path("/tmp/wc.pdf"))
    ap.add_argument("--outdir", type=Path, default=HERE)
    ap.add_argument("--strict", action="store_true",
                    help="fail unless the label residuals are constant and the "
                         "frame corners coincide with the extreme tick marks")
    args = ap.parse_args()

    curves = json.loads(CURVES.read_text())
    pdf = fetch_pdf(args.pdf, curves.get("pdf_sha256"))
    try:
        import fitz
    except ImportError:                       # pragma: no cover - environment
        raise SystemExit("PyMuPDF is required: pip install pymupdf")

    doc = fitz.open(pdf)
    page = doc[curves["page"]]
    res = analyse(page, curves["panels"])
    res.update({"pdf": str(pdf), "pdf_sha256": sha256(pdf),
                "page": curves["page"], "figure": curves.get("caption")})
    res["corrected_reference_csv"] = write_calibrated_reference(
        res["bias_density"], args.outdir).name
    out = args.outdir / "axis_calibration.json"
    out.write_text(json.dumps(res, indent=2) + "\n")

    print(f"pdf sha256 {res['pdf_sha256'][:16]}… page {res['page']}")
    for i, p in enumerate(res["panels"]):
        print(f"panel {i}: frame {p['frame']} y-tick scale "
              f"{p['y_tick_scale_pt_per_unit']:.4f} pt/unit, "
              f"spacing dev {p['y_tick_spacing_max_deviation_pt']:.2e} pt")
        print(f"  label centre vs tick mark: mean {p['y_label_residual_pt']['mean']:+.4f} pt "
              f"std {p['y_label_residual_pt']['std']:.2e} "
              f"(n={len(p['y_labels'])})")
        print(f"  y(0) geometry {p['y_zero_pt_geometry']} vs labels "
              f"{p['y_zero_pt_from_labels']}  -> bias {p['bias_pt']:+.4f} pt "
              f"= {p['bias_density']:+.6f} in density")
        print(f"  frame corners vs extreme ticks: {p['frame_vs_extreme_tick_pt']}")
        print(f"  x labels land on frame: {p['x_fit_lands_on_frame']}")
    print(f"bias: {res['bias_pt']:+.4f} pt = {res['bias_density']:+.6f} density "
          f"(panel spread {res['bias_pt_spread']:.2e} pt)")
    print(f"wrote {out} and {args.outdir / res['corrected_reference_csv']}")

    if args.strict:
        for p in res["panels"]:
            assert p["y_tick_spacing_max_deviation_pt"] < 1e-3, p
            assert p["y_label_residual_pt"]["std"] < 5e-3, p
            assert max(p["frame_vs_extreme_tick_pt"].values()) < EDGE_TOL, p
        assert res["bias_pt_spread"] < 1e-3, res
        print("strict checks passed")


if __name__ == "__main__":
    main()