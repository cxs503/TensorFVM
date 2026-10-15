#!/usr/bin/env python3
"""NACA 0012 (TMR sharp trailing edge) O-type far-field mesh with a Gmsh quad boundary layer.

usage: make_naca_mesh.py OUTDIR [layer_thickness=0.005] [near=0.004] [ratio=1.18]
                         [wake_size=0.03] [wake_distance=0.5] [max_turn_deg=1.5]
                         [first_layer=2e-6] [alpha_split_deg=10] [far_size=25] [radius=500]
                         [split_x] [thick_layer=0.04] [grading_distance=25]
grading_distance: distance at which the background size reaches far_size (linear growth from
wake_size at wake_distance).  The default 25 gives cells of ~1x the distance to the airfoil, which
leaves total-pressure errors of +-2-3 % in the outer flow at 1-5 c; grading_distance=400 gives
h ~ 0.03 + 0.06 d (an O-mesh-like ~6 % radial grading) for only a few thousand extra cells.
Writes OUTDIR/mesh.geo, OUTDIR/mesh.msh (v2.2 ASCII) and OUTDIR/metadata.json.
Physical groups: main (airfoil), inlet, outlet, far-field, fluid  -- same far-field
arc assignment as src/tensorfvm/hlpw30p30n.write_mesh_geo (circle centred at (0.5,0),
corner points rotated by alpha_split so the outlet half faces downstream).
Needs: export LD_LIBRARY_PATH=/home/user/.stublibs  (headless gmsh wheel).
"""
import sys, json, math, numpy as np
from pathlib import Path

def naca0012(x):
    # TMR sharp-TE definition (y(1) = 0)
    return 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x**2
                          + 0.291984971 * x**3 - 0.105174606 * x**4)

def dense_profile(n=40000):
    # cosine-like clustering in x towards both ends, upper surface TE->LE then lower LE->TE
    t = np.linspace(0, 1, n)
    x = 0.5 * (1 - np.cos(np.pi * t))          # 0 .. 1
    x = x**1.0
    y = naca0012(x)
    upper = np.c_[x[::-1], y[::-1]]            # TE -> LE
    lower = np.c_[x[1:], -y[1:]]               # LE -> TE (skip duplicate LE)
    P = np.vstack([upper, lower])
    # clockwise orientation (body on the right when walking): TE -> lower -> LE -> upper -> TE
    return P[::-1]

def resample(P, max_turn_deg, near, te_size=1e-3, te_zone=0.97):
    """Greedy facet selection: next kept point is the farthest dense point such that the
    tangent direction has not turned more than max_turn and the facet is shorter than the
    local size limit (te_size for x > te_zone, else near)."""
    seg = np.diff(P, axis=0); ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    keep = [0]; i = 0; n = len(P)
    max_turn = math.radians(max_turn_deg)
    while i < n - 1:
        j = i + 1
        best = j
        while j < n - 1:
            local = te_size if max(P[i, 0], P[j, 0]) > te_zone else near
            if np.linalg.norm(P[j] - P[i]) > local: break
            if np.abs(ang[i:j] - ang[i]).max() > max_turn: break
            best = j; j += 1
        # also stop at the last point
        if j >= n - 1 and np.linalg.norm(P[n - 1] - P[i]) <= (te_size if P[i, 0] > te_zone else near) \
                and (np.abs(ang[i:n - 1] - ang[i]).max() <= max_turn):
            best = n - 1
        keep.append(best); i = best
    Q = P[keep]
    if np.linalg.norm(Q[-1] - Q[0]) < 1e-12: Q = Q[:-1]
    return Q

def write_geo(path, Q, *, first_layer, layer_thickness, ratio, near, wake_size, wake_distance,
              far_size, radius, alpha_split, fan_elements=12, grading_distance=25.0,
              split_x=None, thick_layer=None):
    L = []
    L += [f"Mesh.CharacteristicLengthMin = {first_layer:.12g};",
          f"Mesh.CharacteristicLengthMax = {far_size:.12g};",
          "Mesh.MeshSizeFromPoints = 0;", "Mesh.MeshSizeFromCurvature = 0;",
          "Mesh.MeshSizeExtendFromBoundary = 0;", "Mesh.Algorithm = 6;",
          "Mesh.ElementOrder = 1;", "Mesh.SaveAll = 0;"]
    n = len(Q)
    for k, (x, y) in enumerate(Q, start=1):
        L.append(f"Point({k}) = {{{x:.16g}, {y:.16g}, 0, {near:.12g}}};")
    curves = []
    for k in range(1, n + 1):
        a, b = k, (k % n) + 1
        L.append(f"Line({k}) = {{{a}, {b}}};"); curves.append(k)
    L.append(f"Curve Loop(1) = {{{', '.join(map(str, curves))}}};")
    th = math.radians(alpha_split); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    corners = np.array([(0, -radius), (radius, 0), (0, radius), (-radius, 0)]) @ R.T + np.array([0.5, 0.0])
    p0 = n + 1; centre = n + 5
    for k, (x, y) in enumerate(corners):
        L.append(f"Point({p0 + k}) = {{{x:.12g}, {y:.12g}, 0, {far_size:.12g}}};")
    L.append(f"Point({centre}) = {{0.5, 0, 0, {far_size:.12g}}};")
    arcs = []
    for j in range(4):
        tag = n + 1 + j
        L.append(f"Circle({tag}) = {{{p0 + j}, {centre}, {p0 + (j + 1) % 4}}};"); arcs.append(tag)
    L.append(f"Curve Loop(2) = {{{', '.join(map(str, arcs))}}};")
    L.append("Plane Surface(1) = {2, 1};")
    L += [f"Physical Curve(\"main\") = {{{', '.join(map(str, curves))}}};",
          f"Physical Curve(\"outlet\") = {{{arcs[0]}, {arcs[1]}}};",
          f"Physical Curve(\"far-field\") = {{{arcs[2]}}};",
          f"Physical Curve(\"inlet\") = {{{arcs[3]}}};",
          "Physical Surface(\"fluid\") = {1};"]
    cl = ', '.join(map(str, curves))
    L += ["Field[1] = Distance;", f"Field[1].CurvesList = {{{cl}}};", "Field[1].Sampling = 100;",
          "Field[4] = Threshold;", "Field[4].InField = 1;",
          f"Field[4].SizeMin = {near:.12g};", f"Field[4].SizeMax = {wake_size:.12g};",
          f"Field[4].DistMin = {layer_thickness:.12g};", f"Field[4].DistMax = {wake_distance:.12g};",
          "Field[5] = Threshold;", "Field[5].InField = 1;",
          f"Field[5].SizeMin = {wake_size:.12g};", f"Field[5].SizeMax = {far_size:.12g};",
          f"Field[5].DistMin = {wake_distance:.12g};", f"Field[5].DistMax = {grading_distance:.12g};",
          "Field[6] = MathEval;", f"Field[6].F = \"F4+F5-{wake_size:.12g}\";",
          "Background Field = 6;",
          ]
    # Optional second boundary-layer field: facets whose both end points lie at x > split_x get a
    # thicker quad layer (Gmsh 2D accepts several BoundaryLayer fields; tests whether the junction
    # between different layer thicknesses is meshed robustly).
    thin = [k for k in curves if not (split_x is not None and Q[k - 1][0] > split_x and Q[k % n][0] > split_x)]
    thick = [k for k in curves if k not in thin]
    L += ["Field[3] = BoundaryLayer;", f"Field[3].CurvesList = {{{', '.join(map(str, thin))}}};",
          f"Field[3].hwall_n = {first_layer:.12g};", f"Field[3].thickness = {layer_thickness:.12g};",
          f"Field[3].ratio = {ratio:.12g};", "Field[3].Quads = 1;",
          "Field[3].FanPointsList = {1};", f"Field[3].FanPointsSizesList = {{{fan_elements}}};",
          "BoundaryLayer Field = 3;"]
    if thick:
        L += ["Field[7] = BoundaryLayer;", f"Field[7].CurvesList = {{{', '.join(map(str, thick))}}};",
              f"Field[7].hwall_n = {first_layer:.12g};", f"Field[7].thickness = {thick_layer:.12g};",
              f"Field[7].ratio = {ratio:.12g};", "Field[7].Quads = 1;", "BoundaryLayer Field = 7;"]
    Path(path).write_text("\n".join(L) + "\n")

def main():
    a = sys.argv[1:]
    out = Path(a[0]); out.mkdir(parents=True, exist_ok=True)
    g = lambda i, d: float(a[i]) if len(a) > i else d
    layer_thickness, near, ratio = g(1, 0.005), g(2, 0.004), g(3, 1.18)
    wake_size, wake_distance, max_turn = g(4, 0.03), g(5, 0.5), g(6, 1.5)
    first_layer, alpha_split, far_size, radius = g(7, 2e-6), g(8, 10.0), g(9, 25.0), g(10, 500.0)
    split_x = float(a[11]) if len(a) > 11 and a[11] not in ("-", "none") else None; thick_layer = g(12, 0.04)
    grading_distance = g(13, 25.0)
    Q = resample(dense_profile(), max_turn, near)
    seg = np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1)
    geo = out / "mesh.geo"
    write_geo(geo, Q, first_layer=first_layer, layer_thickness=layer_thickness, ratio=ratio, near=near,
              wake_size=wake_size, wake_distance=wake_distance, far_size=far_size, radius=radius,
              alpha_split=alpha_split, split_x=split_x, thick_layer=thick_layer,
              grading_distance=grading_distance)
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(geo)); gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2); gmsh.write(str(out / "mesh.msh"))
    types, tags, _ = gmsh.model.mesh.getElements(2)
    counts = {int(t): len(x) for t, x in zip(types, tags)}
    gmsh.finalize()
    meta = dict(wall_points=int(len(Q)), min_facet=float(seg.min()), max_facet=float(seg.max()),
                cells=int(sum(counts.values())), element_counts=counts,
                layer_thickness=layer_thickness, near=near, ratio=ratio, wake_size=wake_size,
                wake_distance=wake_distance, max_turn=max_turn, first_layer=first_layer,
                alpha_split=alpha_split, far_size=far_size, radius=radius,
                split_x=split_x, thick_layer=thick_layer, grading_distance=grading_distance)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))

if __name__ == "__main__":
    main()
