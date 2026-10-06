"""Run a finite-volume case and export portable, cell-centred CSV data."""

import argparse
import csv
from html import escape
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .solver import SimpleSolver, SolverConfig


def _write_fitted_contour(result, values, title, destination):
    """Write cell-constant values on the physical quadrilateral mesh to SVG."""
    mesh = result.mesh
    vertices = mesh.vertices.detach().cpu()
    values = values.detach().cpu()
    low, high = float(values.min()), float(values.max())
    bounds = vertices.reshape(-1, 2)
    xmin, ymin = (float(bounds[:, i].min()) for i in range(2))
    xmax, ymax = (float(bounds[:, i].max()) for i in range(2))
    width = 900
    height = width * (ymax - ymin) / (xmax - xmin)
    margin = 50

    def position(point):
        return (margin + (float(point[0]) - xmin) * width / (xmax - xmin),
                margin + (ymax - float(point[1])) * width / (xmax - xmin))

    def color(value):
        fraction = (value - low) / (high - low) if high > low else 0.5
        fraction = max(0.0, min(1.0, fraction))
        return f"rgb({round(255 * fraction)},70,{round(255 * (1 - fraction))})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" '
        f'height="{height + 150:g}" viewBox="0 0 1000 {height + 150:g}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="25" font-family="sans-serif" font-size="18">'
        f'{escape(title)}</text>',
    ]
    for j in range(result.config.ny):
        for i in range(result.config.nx):
            polygon = (vertices[j, i], vertices[j, i + 1],
                       vertices[j + 1, i + 1], vertices[j + 1, i])
            points = " ".join(f"{x:g},{y:g}" for x, y in map(position, polygon))
            parts.append(
                f'<polygon points="{points}" fill="{color(float(values[j, i]))}"/>'
            )
    parts.extend([
        f'<rect x="{margin}" y="{margin}" width="{width:g}" height="{height:g}" '
        'fill="none" stroke="black"/>',
        f'<text x="{margin}" y="{height + 2 * margin + 15:g}" '
        'font-family="sans-serif">x</text>',
        '<text x="5" y="55" font-family="sans-serif">y</text>',
    ])
    for i in range(100):
        parts.append(
            f'<rect x="{margin + 9 * i}" y="{height + 2 * margin + 35:g}" '
            f'width="9" height="15" fill="{color(low + (high - low) * i / 99)}"/>'
        )
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<text x="{margin + width * fraction:g}" y="{height + 2 * margin + 70:g}" '
            f'font-family="sans-serif">{low + (high - low) * fraction:.4g}</text>'
        )
    parts.append("</svg>")
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_result(result, directory: Path) -> None:
    """Export fields, residual history and configuration without pickle files."""
    directory.mkdir(parents=True, exist_ok=True)
    config = result.config
    u, v = (field.detach().cpu() for field in result.cell_center_velocity())
    p = result.p.detach().cpu()
    fluid = result.fluid.detach().cpu()
    x, y = result.x.detach().cpu(), result.y.detach().cpu()
    if x.ndim == 1:
        y, x = torch.meshgrid(y, x, indexing="ij")
    with (directory / "fields.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "u", "v", "p", "fluid"))
        for j in range(config.ny):
            for i in range(config.nx):
                writer.writerow(
                    (
                        float(x[j, i]),
                        float(y[j, i]),
                        float(u[j, i]),
                        float(v[j, i]),
                        float(p[j, i]),
                        int(fluid[j, i]),
                    )
                )
    if config.mesh_type in ("body-fitted", "c-grid", "flat-plate"):
        vertices = result.mesh.vertices.detach().cpu()
        with (directory / "nodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("node", "x", "y"))
            for index, point in enumerate(vertices.reshape(-1, 2)):
                writer.writerow((index, float(point[0]), float(point[1])))
        with (directory / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("cell", "node0", "node1", "node2", "node3"))
            stride = config.nx + 1
            for j in range(config.ny):
                for i in range(config.nx):
                    node = j * stride + i
                    writer.writerow((j * config.nx + i, node, node + stride,
                                     node + stride + 1, node + 1))
    else:
        for name in ("nodes.csv", "cells.csv"):
            (directory / name).unlink(missing_ok=True)
    if config.mesh_type == "c-grid":
        mask = result.mesh.masks["airfoil"].detach().cpu()
        owners = result.mesh.owner.detach().cpu()[mask]
        centers = result.mesh.face_centers.detach().cpu()[mask]
        pressure = result.p.detach().cpu().reshape(-1)[owners]
        scale = 0.5 * config.density * config.inlet_velocity ** 2
        with (directory / "airfoil.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x_over_chord", "y_over_chord", "x", "y", "p", "cp"))
            for point, value in zip(centers, pressure):
                writer.writerow((
                    float((point[0] - config.airfoil_x) / config.airfoil_chord),
                    float((point[1] - config.airfoil_y) / config.airfoil_chord),
                    float(point[0]), float(point[1]), float(value), float(value / scale),
                ))
    if config.mesh_type != "c-grid":
        (directory / "airfoil.csv").unlink(missing_ok=True)
    if getattr(result, "nu_tilde", None) is not None:
        nu_tilde = result.nu_tilde.detach().cpu()
        nu_t = result.turbulent_kinematic_viscosity.detach().cpu()
        with (directory / "turbulence.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y", "nu_tilde", "nu_t"))
            for j in range(config.ny):
                for i in range(config.nx):
                    writer.writerow((float(x[j, i]), float(y[j, i]),
                                     float(nu_tilde[j, i]), float(nu_t[j, i])))
    else:
        (directory / "turbulence.csv").unlink(missing_ok=True)
    if config.mesh_type in ("body-fitted", "c-grid", "flat-plate"):
        speed = torch.sqrt(u.square() + v.square()) / config.inlet_velocity
        pressure = p / (config.density * config.inlet_velocity ** 2)
        _write_fitted_contour(
            result, speed, "Computed speed |V|/U", directory / "velocity.svg"
        )
        _write_fitted_contour(
            result, pressure, "Computed gauge pressure p/(rho U^2)",
            directory / "pressure.svg",
        )
    else:
        for name in ("velocity.svg", "pressure.svg"):
            (directory / name).unlink(missing_ok=True)
    with (directory / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(result.history, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "converged": result.converged,
                "iterations": len(result.history),
                "config": asdict(config),
                "final_residuals": result.history[-1] if result.history else {},
                **({"aerodynamic_coefficients": result.aerodynamic_coefficients}
                   if getattr(result, "aerodynamic_coefficients", None) is not None else {}),
                **({"surface_forces": result.surface_forces}
                   if getattr(result, "surface_forces", None) is not None else {}),
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def main(argv=None) -> int:
    """Command-line entry point; return 2 when the iteration limit is reached."""
    defaults = SolverConfig()
    parser = argparse.ArgumentParser(
        description="PyTorch 有限体积 SIMPLE：二维不可压缩流动"
    )
    parser.add_argument("--mesh-type", choices=("body-fitted", "cartesian", "c-grid"),
                        default="body-fitted",
                        help="body-fitted 为圆柱 O 网格，c-grid 为 NACA 翼型网格")
    parser.add_argument("--nx", type=int, default=defaults.nx,
                        help="贴体模式为周向网格数（c-grid >=16，其余 >=8，均为4的倍数）")
    parser.add_argument("--ny", type=int, default=defaults.ny,
                        help="贴体模式为径向网格数；笛卡尔模式为y方向")
    parser.add_argument(
        "--reynolds", type=float, default=defaults.reynolds,
        help="基于圆柱直径（c-grid 时为翼型弦长）的 Re"
    )
    parser.add_argument("--airfoil-code", default=defaults.airfoil_code,
                        help="c-grid 使用的四位 NACA 翼型编号")
    parser.add_argument("--airfoil-chord", type=float, default=defaults.airfoil_chord)
    parser.add_argument("--airfoil-x", type=float, default=defaults.airfoil_x,
                        help="翼型前缘 x 坐标")
    parser.add_argument("--airfoil-y", type=float, default=defaults.airfoil_y,
                        help="翼型弦线 y 坐标")
    parser.add_argument("--angle-of-attack", type=float, default=defaults.angle_of_attack,
                        help="来流相对翼型弦线的攻角（度）")
    parser.add_argument("--inlet-profile", choices=("uniform", "parabolic"),
                        default=defaults.inlet_profile,
                        help="入口速度剖面；parabolic 仅适用于圆柱 O 网格")
    parser.add_argument("--domain-length", type=float, default=defaults.length)
    parser.add_argument("--domain-height", type=float, default=defaults.height)
    parser.add_argument(
        "--max-iterations", type=int, default=defaults.max_iterations
    )
    parser.add_argument("--tolerance", type=float, default=defaults.tolerance)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument("--threads", type=int, default=1, help="CPU 计算线程数")
    parser.add_argument(
        "--output", type=Path, default=Path("results/cylinder"), help="结果输出目录"
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads 必须大于零")
    torch.set_num_threads(args.threads)
    try:
        config = SolverConfig(
            nx=args.nx,
            ny=args.ny,
            reynolds=args.reynolds,
            length=args.domain_length,
            height=args.domain_height,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            device=args.device,
            mesh_type=args.mesh_type,
            airfoil_code=args.airfoil_code,
            airfoil_chord=args.airfoil_chord,
            airfoil_x=args.airfoil_x,
            airfoil_y=args.airfoil_y,
            angle_of_attack=args.angle_of_attack,
            inlet_profile=args.inlet_profile,
        )
        result = SimpleSolver(config).solve()
        export_result(result, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"计算失败：{error}\n")
    status = "已收敛" if result.converged else "未收敛（达到迭代上限）"
    print(f"{status}，迭代次数：{len(result.history)}，输出：{args.output}")
    print(json.dumps(result.history[-1] if result.history else {}, ensure_ascii=False))
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
