"""Post-processing kernels shared by the case drivers and the acceptance gate.

All functions take raw stored fields (cell centres, volumes, velocities and
per-face wall data) plus the frozen TMR reference files, and return plain
numbers.  The gate recomputes them from the artefacts, so every error in the
report is reproducible from what is committed.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

REFERENCE = "TMR (tmbwg.github.io/turbmodels mirror of turbmodels.larc.nasa.gov)"


def read_tmr(path):
    """Read a TMR ASCII data file into a float array, skipping header lines."""
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "variables", "zone")):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue
    return np.array(rows)


def read_tmr_zones(path):
    """Read a TMR file containing Tecplot-style zones; return {title: array}."""
    zones, title, rows = {}, None, []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.lower().startswith("zone"):
                if title is not None and rows:
                    zones[title] = np.array(rows)
                rows = []
                if 't="' in stripped:
                    title = stripped.split('t="')[1].split('"')[0]
                else:
                    title = f"zone{len(zones)}"
                continue
            if not stripped or stripped.startswith(("#", "variables")):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue
    if title is not None and rows:
        zones[title] = np.array(rows)
    return zones


def columns(centers, tolerance_scale=1e-3):
    """Group cells into vertical columns (constant streamwise coordinate).

    TMR grids are very slightly skewed in x within a column, so clustering uses
    a fraction of the domain length rather than exact rounding.
    """
    x = centers[:, 0]
    span = float(x.max() - x.min())
    resolution = max(tolerance_scale * span, 1e-12)
    rounded = np.round(x / resolution)
    order = np.argsort(rounded)
    grouped = {}
    for index in order:
        grouped.setdefault(rounded[index], []).append(index)
    return [(float(x[indices[0]]), np.array(indices)) for _, indices in
            sorted(grouped.items())]


def station_profile(centers, field, station, halfwidth=None):
    """Return ``(y, value)`` for the cell column closest to ``station``."""
    unique = np.unique(np.round(centers[:, 0], 9))
    if halfwidth is None:
        index = int(np.argmin(np.abs(unique - station)))
        x_column = unique[index]
        gaps = [unique[index + 1] - unique[index] if index + 1 < unique.size else np.inf,
                unique[index] - unique[index - 1] if index > 0 else np.inf]
        halfwidth = 0.55 * float(np.min(gaps))
    selected = np.where(np.abs(centers[:, 0] - station) <= halfwidth)[0]
    if selected.size < 3:
        return None
    y = centers[selected, 1]
    order = np.argsort(y)
    selected, y = selected[order], y[order]
    # Keep one cell per distinct station (multi-zone grids may contribute two
    # cells within the sampling band).
    keep = [0]
    for index in range(1, y.size):
        if y[index] - y[keep[-1]] > 1e-12:
            keep.append(index)
    keep = np.array(keep)
    return y[keep], field[selected[keep]]


def flat_plate_metrics(fields, wall, reference_dir, viscosity, density=1.0,
                       inlet_velocity=1.0, u_tau_station_retheta=10000.0):
    """Skin friction versus Re_theta and u+ versus y+ at Re_theta = 1e4."""
    centers = fields["centers"]
    velocity = fields["velocity"]
    volumes = fields["volumes"]
    cf_curve = read_tmr(str(reference_dir / "FlatPlate_validation_cf_K-S.dat"))
    coles = read_tmr(str(reference_dir / "FlatPlate_validation_u+y+.dat"))
    coles = coles[np.argsort(coles[:, 0])]
    karman = PchipInterpolator(cf_curve[:, 0], cf_curve[:, 1], extrapolate=False)
    wall_x = wall["x"]
    # ``shear`` stores the tangential x-component of the surface traction
    # sigma.n; multiplying by sign(n_y) turns it into the streamwise wall shear
    # stress tau_w (positive when the near-wall flow runs in +x).
    wall_shear = wall["shear"] * np.sign(wall["normal_y"])
    # Only the plate (lower) wall carries skin friction for this case.
    plate = wall["y"] < 1e-12
    rows = []
    for x_column, indices in columns(centers):
        if x_column <= 0.0:
            continue
        y = centers[indices, 1]
        u = velocity[indices, 0]
        order = np.argsort(y)
        y, u = y[order], u[order]
        if y.size < 8:
            continue
        edge = np.max(u)
        if edge <= 0:
            continue
        # Momentum thickness over the boundary layer only: the integral stops at
        # the 99.5 % thickness, otherwise the outer-region numerical velocity
        # deficit (a few 1e-3 of U) integrates over the whole channel height and
        # inflates theta far beyond any physical boundary layer.
        reached = np.nonzero(u >= 0.995 * edge)[0]
        if reached.size == 0:
            continue
        thickness = float(y[reached[0]])
        inside = y <= thickness
        y_integral = np.concatenate(([0.0], y[inside], [thickness]))
        u_integral = np.concatenate(([0.0], u[inside], [edge]))
        theta = np.trapezoid(u_integral / edge * (1 - u_integral / edge), y_integral)
        # The plate face under a column shares that column's x coordinate; the
        # nearest face is used so clustered leading-edge spacing cannot reject
        # a legitimate column.
        plate_x = wall_x[plate]
        nearest = int(np.argmin(np.abs(plate_x - x_column)))
        tau = float(wall_shear[plate][nearest])
        cf = tau / (0.5 * density * inlet_velocity ** 2)
        rows.append(dict(x=float(x_column), theta=float(theta),
                         re_theta=float(edge * theta / viscosity),
                         cf=float(cf), u_tau=float(np.sqrt(max(tau, 1e-30) / density)),
                         u_edge=float(edge)))
    table = [row for row in rows if 4000.0 <= row["re_theta"] <= 13000.0]
    for row in table:
        reference = float(karman(row["re_theta"]))
        row["cf_KS"] = reference
        row["cf_relative_error"] = abs(row["cf"] - reference) / abs(reference)
    # u+ versus y+ at the station whose Re_theta is closest to 1e4.
    station = min(rows, key=lambda row: abs(row["re_theta"] - u_tau_station_retheta))
    profile_x = station["x"]
    column = [ind for x, ind in columns(centers) if abs(x - profile_x) < 1e-9][0]
    y = centers[column, 1]
    u = velocity[column, 0]
    order = np.argsort(y)
    y, u = y[order], u[order]
    u_tau = station["u_tau"]
    y = np.concatenate(([0.0], y))
    u = np.concatenate(([0.0], u))
    y_plus = y * u_tau / viscosity
    u_plus = u / u_tau
    u_plus_law = PchipInterpolator(coles[:, 0], coles[:, 1], extrapolate=False)(y_plus)
    valid = np.isfinite(u_plus_law) & (y_plus > 0)
    log_layer = valid & (y_plus > 30) & (y_plus < 300)
    metrics = dict(
        cf_table=table,
        cf_relative_error_rms=float(np.sqrt(np.mean([row["cf_relative_error"] ** 2
                                                     for row in table]))),
        cf_relative_error_max=float(np.max([row["cf_relative_error"] for row in table])),
        re_theta_range=[float(min(row["re_theta"] for row in table)),
                        float(max(row["re_theta"] for row in table))],
        profile_station=dict(x=float(profile_x), re_theta=float(station["re_theta"]),
                             u_tau=float(u_tau)),
        u_plus_y_plus=dict(y_plus=y_plus.tolist(), u_plus=u_plus.tolist(),
                           u_plus_law=u_plus_law.tolist()),
        u_plus_error_rms=float(np.sqrt(np.mean((u_plus[valid] - u_plus_law[valid]) ** 2))),
        u_plus_error_rms_log_layer=float(np.sqrt(np.mean((u_plus[log_layer]
                                                          - u_plus_law[log_layer]) ** 2))),
        u_plus_error_max=float(np.max(np.abs(u_plus[valid] - u_plus_law[valid]))),
        plate_area=float(np.sum(wall["area"][plate])),
        first_cell_y_plus=float(y[1] * u_tau / viscosity))
    metrics["profile"] = dict(x=[float(v) for v in y], u_plus=u_plus.tolist(),
                              y_plus=y_plus.tolist())
    return metrics


WINDOW = 20.0  # x/H comparison window: beyond it the TMR grid zone 4 has
# dx ~ 2.6 H, which cannot support a grid-independent near-wall solution.


def bottom_faces(wall, tolerance=1e-12):
    """Bottom-wall faces of the backward-facing-step case, ordered by x."""
    mask = wall["y"] < tolerance
    order = np.argsort(wall["x"][mask])
    return mask, order


def near_wall_owner(centers, wall, mask, order):
    """Owner cells of the bottom-wall faces (first cell above the wall)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    points = np.column_stack((wall["x"][mask][order], wall["y"][mask][order]))
    _, owner = tree.query(points)
    return owner


def backstep_metrics(fields, wall, reference_dir, viscosity, density=1.0,
                     reference_station=-4.0, profile_stations=(1.0, 4.0, 6.0, 10.0)):
    """Reattachment length, Cf, Cp and velocity profiles against D&S data."""
    centers = fields["centers"]
    velocity = fields["velocity"]
    pressure = fields["pressure"]
    near = np.where(np.abs(centers[:, 0] - reference_station) < 1.0)[0]
    middle = near[np.argmin(np.abs(centers[near, 1] - 5.0))]
    u_ref = float(velocity[middle, 0])
    q_ref = 0.5 * density * u_ref ** 2
    mask, order = bottom_faces(wall)
    x = wall["x"][mask][order]
    shear = (wall["shear"] * np.sign(wall["normal_y"]))[mask][order]
    pressure_wall = wall["pressure"][mask][order]
    owner = near_wall_owner(centers, wall, mask, order)
    near_wall_velocity = velocity[owner, 0]
    cf = shear / q_ref
    cp = (pressure_wall - pressure[middle]) / q_ref
    # Reattachment from the first-cell velocity (robust at the singular corner).
    crossings = []
    for index in range(1, x.size):
        if near_wall_velocity[index - 1] < 0.0 <= near_wall_velocity[index]:
            weight = (-near_wall_velocity[index - 1]
                      / (near_wall_velocity[index] - near_wall_velocity[index - 1]))
            crossings.append(float(x[index - 1] + weight * (x[index] - x[index - 1])))
    reattachment = crossings[-1] if crossings else float("nan")
    cf_crossings = []
    for index in range(1, x.size):
        if cf[index - 1] < 0.0 <= cf[index]:
            weight = -cf[index - 1] / (cf[index] - cf[index - 1])
            cf_crossings.append(float(x[index - 1] + weight * (x[index] - x[index - 1])))
    window = (x > 0.0) & (x <= WINDOW)
    exp_cf = read_tmr(str(reference_dir / "Backstep_validation_cf.exp.dat"))
    exp_cp_zones = read_tmr_zones(str(reference_dir / "Backstep_validation_cp.expnew.dat"))
    exp_profile_zones = read_tmr_zones(str(reference_dir / "Backstep_validation_profiles.exp.dat"))
    exp_cp = None
    for title, data in exp_cp_zones.items():
        if "bottom" in title:
            exp_cp = data
    if exp_cp is None:
        exp_cp = next(iter(exp_cp_zones.values()))
    exp_cf_interp = PchipInterpolator(exp_cf[:, 0], exp_cf[:, 1], extrapolate=False)
    cf_reference = exp_cf_interp(x)
    valid = np.isfinite(cf_reference) & window
    cf_scale = float(np.max(np.abs(cf_reference[valid])))
    exp_crossings = []
    for index in range(1, exp_cf.shape[0]):
        if exp_cf[index - 1, 1] < 0.0 <= exp_cf[index, 1]:
            weight = -exp_cf[index - 1, 1] / (exp_cf[index, 1] - exp_cf[index - 1, 1])
            exp_crossings.append(float(exp_cf[index - 1, 0]
                                       + weight * (exp_cf[index, 0] - exp_cf[index - 1, 0])))
    exp_reattachment = exp_crossings[0] if exp_crossings else float("nan")
    # cp_orig is referenced to the conditions near x/H = -4, i.e. the same
    # normalisation used here, so no arbitrary shift is applied.
    exp_cp_orig = exp_cp[:, 2] if exp_cp.shape[1] > 2 else exp_cp[:, 1]
    cp_interp = PchipInterpolator(exp_cp[:, 0], exp_cp_orig, extrapolate=False)
    cp_exp = cp_interp(x)
    cp_valid = np.isfinite(cp_exp) & window
    cp_scale = float(np.max(np.abs(cp_exp[cp_valid])))
    profile_rows = []
    for station in profile_stations:
        sample = interpolated_station(centers, velocity[:, 0], station)
        if sample is None:
            continue
        y_cells, u_cells = sample
        key = [title for title in exp_profile_zones
               if abs(float(title.split("=")[-1]) - station) < 1e-6]
        if not key:
            continue
        data = exp_profile_zones[key[0]]
        ref_interp = PchipInterpolator(data[:, 1], data[:, 2], extrapolate=False)
        reference = ref_interp(y_cells)
        inside = np.isfinite(reference)
        denominator = float(np.max(np.abs(data[:, 2])))
        profile_rows.append(dict(
            station=station,
            y=[float(v) for v in y_cells],
            u_over_uref=[float(v / u_ref) for v in u_cells],
            u_exp=[float(v) for v in reference[inside]],
            y_exp=[float(v) for v in y_cells[inside]],
            error_rms=float(np.sqrt(np.mean((u_cells[inside] / u_ref
                                             - reference[inside]) ** 2)) / denominator)))
    profile_error = (float(np.mean([row["error_rms"] for row in profile_rows]))
                     if profile_rows else float("nan"))
    return dict(
        u_reference=u_ref,
        reference_cell=dict(x=float(centers[middle, 0]), y=float(centers[middle, 1])),
        reattachment=reattachment,
        reattachment_experiment=exp_reattachment,
        reattachment_relative_error=float(abs(reattachment / exp_reattachment - 1)),
        reattachment_from_wall_shear=cf_crossings[-1] if cf_crossings else float("nan"),
        cf=dict(x=x.tolist(), cf=cf.tolist()),
        cf_window=[0.0, WINDOW],
        cf_error_rms_relative=float(np.sqrt(np.mean(
            (cf[valid] - cf_reference[valid]) ** 2)) / cf_scale),
        cf_error_max_relative=float(np.max(np.abs(cf[valid] - cf_reference[valid]))
                                    / cf_scale),
        cp=dict(x=x.tolist(), cp=cp.tolist(),
                cp_exp=[float(v) for v in cp_exp[cp_valid]],
                x_exp=[float(v) for v in x[cp_valid]]),
        cp_error_rms_relative=float(np.sqrt(np.mean(
            (cp[cp_valid] - cp_exp[cp_valid]) ** 2)) / cp_scale),
        cp_error_max_relative=float(np.max(np.abs(cp[cp_valid] - cp_exp[cp_valid]))
                                    / cp_scale),
        near_wall_velocity=dict(x=x.tolist(), u=near_wall_velocity.tolist()),
        profiles=profile_rows,
        profile_error_rms_relative=profile_error,
        wall_extreme_bottom_faces=int(mask.sum()))


def interpolated_station(centers, field, station, tolerance=1e-3):
    """Interpolate a vertical profile to ``station`` between adjacent columns."""
    grouped = columns(centers, tolerance)
    coordinates = np.array([x for x, _ in grouped])
    if coordinates.size < 2:
        return None
    index = int(np.searchsorted(coordinates, station))
    index = min(max(index, 1), coordinates.size - 1)
    left, right = grouped[index - 1], grouped[index]
    x_left, x_right = coordinates[index - 1], coordinates[index]

    def profile(group):
        y = centers[group, 1]
        values = field[group]
        order = np.argsort(y)
        return y[order], values[order]

    y_left, u_left = profile(left[1])
    y_right, u_right = profile(right[1])
    if y_left.size != y_right.size or not np.allclose(y_left, y_right, atol=1e-9):
        # Different column layouts (zone boundary): fall back to the nearer one.
        group = left if abs(station - x_left) <= abs(station - x_right) else right
        return profile(group[1])
    weight = 0.0 if x_right == x_left else (station - x_left) / (x_right - x_left)
    return y_left, u_left + weight * (u_right - u_left)
