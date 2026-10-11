#!/usr/bin/env python3
"""Read-only reconstruction audit of a frozen native 3D ABC manufactured case.

Quadratic reconstruction uses sampled analytic pressure solely to measure
consistency. It neither changes the production solver nor computes a new NS
pressure solution. The default imports the case's archived production source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / 'docs/verification-native-suboff-mms/coarse'
NATIVE_MODULES = ('native_mesh3d', 'native_collocated3d', 'native_manufactured3d',
                  'native_operators3d', 'native_viscous3d')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def triangle_integrals(mesh, field, order=5):
    """Independent Duffy Gauss averages and integrated advective traction."""
    t, w = leggauss(order)
    t, w = (t + 1) / 2, w / 2
    bary = np.array([[1-r, r*(1-z), r*z] for r in t for z in t])
    weights = np.array([2*r*wr*wz for r, wr in zip(t, w) for z, wz in zip(t, w)])
    F = len(mesh.owner)
    q, pressure = np.zeros(F), np.zeros(F)
    convection = np.zeros((F, 3))
    for start in range(0, F, 512):
        ids = np.arange(start, min(start+512, F))
        tri = mesh.vertices[np.array([mesh.face_vertices[f] for f in ids])]
        points = np.einsum('qa,fak->fqk', bary, tri)
        exact = field(points.reshape(-1, 3))
        velocity = exact['velocity'].reshape(len(ids), len(weights), 3)
        normal_flux = np.einsum('fqk,fk->fq', velocity, mesh.area_vectors[ids])
        q[ids] = normal_flux @ weights
        pressure[ids] = exact['pressure'].reshape(len(ids), len(weights)) @ weights
        convection[ids] = np.einsum('q,fqk,fq->fk', weights, velocity, normal_flux)
    return q, pressure, convection


def volume_integrals(mesh, field):
    """Positive centroid-to-facet tetra fans, independent degree-2 quadrature.

    Four barycentric nodes integrate every polynomial of degree <=2 exactly.
    Trigonometric fields are approximate; the Gauss pressure surface integral
    versus the independent volume gradient quantifies integration consistency.
    """
    cells, triangles = [], []
    for face, owner in enumerate(mesh.owner):
        tri = mesh.vertices[list(mesh.face_vertices[face])]
        cells.append(int(owner)); triangles.append(tri)
        if mesh.neighbor[face] >= 0:
            cells.append(int(mesh.neighbor[face])); triangles.append(tri[::-1])
    cells, triangles = np.array(cells), np.array(triangles)
    tetra = np.concatenate([mesh.cell_centers[cells, None], triangles], axis=1)
    volume = abs(np.linalg.det(tetra[:, 1:] - tetra[:, :1])) / 6
    fan_volume = np.bincount(cells, weights=volume, minlength=len(mesh.cells))
    closure = float(np.max(abs(fan_volume-mesh.cell_volumes)/mesh.cell_volumes))
    if closure > 1e-10:
        raise ValueError('Tetrahedral fan volume does not match native cell volume')
    bary = np.full((4, 4), .1381966011250105)
    np.fill_diagonal(bary, .5854101966249685)
    result = {name: np.zeros((len(mesh.cells), 3)) for name in
              ('acceleration', 'pressure_gradient', 'velocity', 'convection')}
    for start in range(0, len(cells), 1000):
        ids = slice(start, start+1000)
        points = np.einsum('qa,tak->tqk', bary, tetra[ids])
        exact = field(points.reshape(-1, 3))
        for name in ('acceleration', 'pressure_gradient', 'velocity'):
            np.add.at(result[name], cells[ids],
                      exact[name].reshape(-1, 4, 3).mean(1)*volume[ids, None])
        conv = np.einsum('cij,cj->ci', exact['velocity_gradient'], exact['velocity'])
        np.add.at(result['convection'], cells[ids],
                  conv.reshape(-1, 4, 3).mean(1)*volume[ids, None])
    return result, closure


def quadratic_basis(d):
    x, y, z = d.T
    return np.column_stack((x, y, z, x*x, y*y, z*z, x*y, x*z, y*z))


def quadratic_probe(mesh, exact, reference_face_pressure, integrated_force):
    """Polynomial-exact diagnostic on actual adjacency, with no fitted MMS parameters."""
    N = len(mesh.cells); internal = mesh.neighbor >= 0
    adjacency = [set() for _ in range(N)]
    for owner, neighbor in zip(mesh.owner[internal], mesh.neighbor[internal]):
        adjacency[owner].add(int(neighbor)); adjacency[neighbor].add(int(owner))
    pressure = exact['pressure']
    coefficients, scales = np.zeros((N, 9)), np.zeros((N, 3))
    depths, conditions, polynomial_errors = [], [], []
    for cell in range(N):
        visited, frontier = {cell}, {cell}
        for depth in range(1, 6):
            frontier = set().union(*(adjacency[k] for k in frontier))-visited
            visited |= frontier
            if len(visited) < 19:
                continue
            neighbors = np.array(sorted(visited-{cell}))
            d = mesh.cell_centers[neighbors]-mesh.cell_centers[cell]
            scale = np.sqrt(np.mean(d*d, axis=0))
            if np.any(scale <= 0):
                continue
            basis = quadratic_basis(d/scale)
            weight = 1/np.sqrt(np.sum(d*d, axis=1))
            matrix = basis*weight[:, None]
            column_scale = np.linalg.norm(matrix, axis=0)
            if np.any(column_scale <= 0):
                continue
            matrix /= column_scale
            # Fourth-degree/trigonometric solution is not used to select the stencil.
            inverse, _, rank, singular = np.linalg.lstsq(matrix, np.eye(len(neighbors)), rcond=1e-12)
            if rank == 9:
                response = inverse*weight[None, :]/column_scale[:, None]
                coefficients[cell] = response @ (pressure[neighbors]-pressure[cell])
                scales[cell] = scale
                conditions.append(float(singular[0]/singular[-1])); depths.append(depth)
                polynomial_errors.append(float(np.max(abs(response@basis-np.eye(9)))))
                break
        else:
            raise ValueError(f'Rank-deficient real-neighbor quadratic stencil at cell {cell}')
    gradient = coefficients[:, :3]/scales
    volume = mesh.cell_volumes
    gradient_error = np.sqrt(np.sum(volume[:, None]*(gradient-exact['pressure_gradient'])**2)
                             /np.sum(volume[:, None]*exact['pressure_gradient']**2))
    vertices = mesh.vertices[np.array(mesh.face_vertices)]
    offsets = vertices-mesh.face_centers[:, None]
    # Uniform triangle covariance: sum_a (vertex_a-centroid)(...)^T / 12.
    covariance = np.einsum('fai,faj->fij', offsets, offsets)/12
    predictions = []
    for cells in (mesh.owner, np.where(internal, mesh.neighbor, mesh.owner)):
        displacement = (mesh.face_centers-mesh.cell_centers[cells])/scales[cells]
        face_covariance = covariance/scales[cells, :, None]/scales[cells, None, :]
        basis = quadratic_basis(displacement)
        for col, left, right in ((3,0,0),(4,1,1),(5,2,2),(6,0,1),(7,0,2),(8,1,2)):
            basis[:, col] += face_covariance[:, left, right]
        predictions.append(pressure[cells]+np.einsum('fj,fj->f', coefficients[cells], basis))
    d = mesh.cell_centers[np.where(internal, mesh.neighbor, mesh.owner)]-mesh.cell_centers[mesh.owner]
    weight = np.zeros(len(internal))
    weight[internal] = np.einsum('fi,fi->f', (mesh.face_centers-mesh.cell_centers[mesh.owner])[internal],
                               mesh.area_vectors[internal])/np.einsum('fi,fi->f', d[internal], mesh.area_vectors[internal])
    shared_face = predictions[0]*(1-weight)+predictions[1]*weight
    force_error = np.zeros((N, 3))
    delta = (shared_face-reference_face_pressure)[:, None]*mesh.area_vectors
    np.add.at(force_error, mesh.owner, delta)
    np.add.at(force_error, mesh.neighbor[internal], -delta[internal])
    force_relative = np.sqrt(np.sum(force_error**2/volume[:, None])
                             /np.sum(integrated_force**2/volume[:, None]))
    area = np.linalg.norm(mesh.area_vectors, axis=1)
    rms = lambda mask: float(np.sqrt(np.sum(area[mask]*(shared_face[mask]-reference_face_pressure[mask])**2)/area[mask].sum()))
    return dict(diagnostic_only=True, production_solver_modified=False,
        new_NS_pressure_solution_computed=False,
        pressure_gradient_relative_L2=float(gradient_error),
        pressure_force_consistency_relative_L2=float(force_relative),
        boundary_face_pressure_absolute_RMS=rms(~internal),
        internal_face_pressure_absolute_RMS=rms(internal),
        neighbor_BFS_depth_counts={str(k): depths.count(k) for k in sorted(set(depths))},
        normalized_design_condition_max=max(conditions),
        degree_2_local_polynomial_coefficient_reproduction_max=max(polynomial_errors))


def audit(mesh, fields, field, solver, weighted_error, nu):
    N = len(mesh.cells); volume = mesh.cell_volumes
    exact = field(mesh.cell_centers)
    for key in ('velocity', 'pressure', 'flux'):
        if key not in fields or not np.isfinite(fields[key]).all():
            raise ValueError(f'Missing/nonfinite frozen {key}')
    if fields['velocity'].shape != (N,3) or fields['pressure'].shape != (N,) or fields['flux'].shape != (len(mesh.owner),):
        raise ValueError('Frozen fields do not match the native mesh')
    for key in ('velocity','pressure'):
        stored = 'reference_'+key
        if stored in fields and not np.allclose(fields[stored], exact[key], rtol=1e-13, atol=1e-14):
            raise ValueError('Frozen reference differs from specified analytic parameters')
    internal = mesh.neighbor >= 0; boundary = ~internal
    types = np.array(mesh.cell_types)
    boundary_cells, interface_cells = np.zeros(N,bool), np.zeros(N,bool)
    boundary_cells[mesh.owner[boundary]] = True
    interface_faces = internal.copy()
    interface_faces[internal] = types[mesh.owner[internal]] != types[mesh.neighbor[internal]]
    interface_cells[mesh.owner[interface_faces]] = True; interface_cells[mesh.neighbor[interface_faces]] = True
    groups = {t: types==t for t in np.unique(types)}
    groups.update(boundary_cells=boundary_cells, interface_cells=interface_cells,
                  bulk_cells=~boundary_cells & ~interface_cells)
    pe = fields['pressure']-exact['pressure']; pe -= np.sum(pe*volume)/volume.sum()
    pref = exact['pressure']-np.sum(exact['pressure']*volume)/volume.sum()
    error_energy = np.sum(volume*pe*pe)
    partitions = {}
    for name, mask in groups.items():
        partitions[name] = dict(cells=int(mask.sum()), volume_fraction=float(volume[mask].sum()/volume.sum()),
            pressure_error_energy_fraction=float(np.sum(volume[mask]*pe[mask]**2)/error_energy),
            pressure_absolute_RMS=float(np.sqrt(np.sum(volume[mask]*pe[mask]**2)/volume[mask].sum())))
    volume_exact, fan_closure = volume_integrals(mesh, field)
    q, pf, convection_face = triangle_integrals(mesh, field)
    native_pf = solver.pressure.face_matrix@exact['pressure']+solver.p_face_source
    area = np.linalg.norm(mesh.area_vectors,axis=1)
    face_errors = {}
    for name, mask in [('boundary',boundary), ('internal',internal), ('type_interface',interface_faces)]:
        face_errors[name] = dict(pressure_absolute_RMS=float(np.sqrt(np.sum(area[mask]*(native_pf[mask]-pf[mask])**2)/area[mask].sum())),
                                pressure_max_absolute=float(np.max(abs(native_pf[mask]-pf[mask]))))
    D = solver.viscous.vector_divergence_matrix
    pressure_force = solver.pressure.pressure_force_matrix@exact['pressure']+solver.p_force_source
    pressure_force_quad = D@(pf[:,None]*mesh.area_vectors).ravel()
    solver.u = exact['velocity'].copy()
    ac, source = solver._convection(q)
    convection = ac@exact['velocity'].ravel()+source
    viscous = -solver.viscous.momentum_diffusion_matrix@exact['velocity'].ravel()-solver.diffusion_source
    viscous_exact = nu*field.parameters['wavenumber']**2*volume_exact['velocity'].ravel()
    force_quad = volume_exact['acceleration'].ravel()*solver.settings.density
    point_force = solver.force_integrated
    def norm(z):
        return float(np.sqrt(np.sum(z.reshape(N,3)**2/volume[:,None])))
    denominator = norm(force_quad)
    residuals = {
        'point_vs_volume_forcing':point_force-force_quad,
        'pressure_face_reconstruction':pressure_force-pressure_force_quad,
        'pressure_quadrature_vs_volume_gradient':pressure_force_quad-volume_exact['pressure_gradient'].ravel(),
        'viscous_discretization':viscous-viscous_exact,
        'convection_discretization':convection-volume_exact['convection'].ravel(),
        'total_exact_state_residual':viscous+convection+pressure_force-point_force}
    consistency = {name:dict(volume_acceleration_relative_L2=norm(z)/denominator,
                              integrated_residual_L2=float(np.linalg.norm(z))) for name,z in residuals.items()}
    regions = {}
    for name, z in residuals.items():
        energy = np.sum(z.reshape(N,3)**2,axis=1)/volume
        regions[name] = {region:float(energy[mask].sum()/energy.sum()) for region,mask in groups.items()}
    _, _, _, correction, flux_source = solver._system(q)
    reconstructed = solver.velocity_flux_matrix@exact['velocity'].ravel()+flux_source
    continuity = {}
    for name, flux in [('analytic_surface_quadrature',q), ('velocity_reconstruction_no_RC',reconstructed),
                       ('velocity_reconstruction_with_RC',reconstructed+correction@exact['pressure'])]:
        div = solver.viscous.op.integrate(flux)/volume
        continuity[name] = dict(max_absolute_divergence=float(np.max(abs(div))),
                               volume_weighted_divergence_RMS=float(np.sqrt(np.sum(volume*div*div)/volume.sum())))
    ls_gradient = np.column_stack([g@exact['pressure'] for g in solver.pressure.G])
    gradient_error = float(np.sqrt(np.sum(volume[:,None]*(ls_gradient-exact['pressure_gradient'])**2)/np.sum(volume[:,None]*exact['pressure_gradient']**2)))
    return dict(global_errors=dict(velocity_relative_L2=weighted_error(fields['velocity'],exact['velocity'],volume),
        pressure_relative_L2=weighted_error(fields['pressure'],exact['pressure'],volume,remove_mean=True)),
        cells=N, faces=len(mesh.owner), volume=float(volume.sum()), partitions=partitions,
        partition_note='Cell types form a partition; boundary/interface categories can overlap.',
        face_pressure_errors=face_errors, momentum_consistency=consistency,
        residual_energy_regions=regions, continuity_consistency=continuity,
        production_pressure_LS_gradient_relative_L2=gradient_error,
        pressure_error_projection_on_reference=float(np.sum(volume*pe*pref)/np.sum(volume*pref*pref)),
        tetra_fan_volume_relative_max=fan_closure,
        quadratic_probe=quadratic_probe(mesh,exact,pf,force_quad.reshape(N,3)))


def report_text(result, mesh_path, fields_path):
    e = result['global_errors']; q = result['quadratic_probe']; c = result['momentum_consistency']
    table = '\n'.join(f"| {name} | {row['volume_acceleration_relative_L2']:.6%} |" for name,row in c.items())
    return f'''# Native SUBOFF MMS reconstruction audit

This audits the frozen converged coarse field, not a new NS solve. Mesh: `{mesh_path}`; fields: `{fields_path}`. Source and input SHA256 bindings are in `audit.json`.

## Production field errors

Velocity volume-weighted relative L2: **{e['velocity_relative_L2']:.6%}**. Pressure after volume-mean gauge removal: **{e['pressure_relative_L2']:.6%}**. These are centroid point-value errors, not cell-average norms. MMS is mathematical verification; prescribed hull velocity is not physical no-slip SUBOFF.

## Independent continuum consistency

At each cell, e is an integrated momentum residual. The reported relative norm is sqrt(sum(|e_c|²/V_c))/sqrt(sum(|F_c|²/V_c)), where F is independent volume-quadrature analytic acceleration integrated over the cell. Contributions are not an error budget of the coupled pressure solution.

| Contribution | Relative volume-weighted acceleration L2 |
|---|---:|
{table}

Faces use order-5 Duffy Gauss integration. Each convex native cell is decomposed into centroid-to-triangle tetrahedra, with four-point degree-2 volume quadrature. Trigonometric integrals are approximate; pressure surface-versus-volume-gradient agreement is reported independently. Native tetra fan volume closure is {result['tetra_fan_volume_relative_max']:.3g}. No production residual defines the analytic force.

The continuum ABC definition is u=U(sin(kz)+cos(ky), sin(kx)+cos(kz), sin(ky)+cos(kx)), p=rho U² sin(kx)cos(ky)sin(kz). It obeys div u=0 and laplacian u=-k²u. Hence f=(u·grad)u+grad(p)/rho+nu k²u. This is a self-defined analytic verification, with no experimental reference.

## Read-only quadratic reconstruction probe

The probe fits the nine scaled polynomial terms x,y,z,x²,y²,z²,xy,xz,yz through actual mesh adjacency. Geometry/rank determine 2–4-ring stencils; analytic pressure samples do not select a stencil or tune parameters. A shared face value averages owner/neighbor predictions with native interpolation weights. Triangle quadratic means include covariance sum(vertex-centroid outer products)/12, so the same face force is added and subtracted for its two cells.

| Quantity | Production reconstruction | Quadratic consistency probe |
|---|---:|---:|
| Pressure gradient relative L2 | {result['production_pressure_LS_gradient_relative_L2']:.6%} | {q['pressure_gradient_relative_L2']:.6%} |
| Pressure force consistency | {c['pressure_face_reconstruction']['volume_acceleration_relative_L2']:.6%} | {q['pressure_force_consistency_relative_L2']:.6%} |
| Boundary face pressure absolute RMS | {result['face_pressure_errors']['boundary']['pressure_absolute_RMS']:.8g} | {q['boundary_face_pressure_absolute_RMS']:.8g} |
| Internal face pressure absolute RMS | {result['face_pressure_errors']['internal']['pressure_absolute_RMS']:.8g} | {q['internal_face_pressure_absolute_RMS']:.8g} |

The maximum normalized design condition is {q['normalized_design_condition_max']:.6g}; the independently checked degree-2 coefficient reproduction defect is {q['degree_2_local_polynomial_coefficient_reproduction_max']:.3g}. **The quadratic probe is not in the production solver and computes no new NS pressure solution. It does not establish pressure below 3%.**

## Interpretation and next checks

The cell-type and geometric-region energy distributions are in the JSON. Most domain volume lies in external tetrahedra; refining only the hull layer cannot establish global MMS convergence. Pressure reconstruction has a measurable consistency improvement under polynomial reconstruction. Production integration would require compatible skew/Hessian terms in Rhie–Chow compact and interpolated gradient fluxes, retained shared-face conservation, and physical velocity-wall free-pressure treatment. Do not remove RC or impose exact manufactured pressure as a production wall repair.

Independent source-volume integration removes a smaller measured residual, but full coupled ablations are needed to determine its pressure-solution impact. Viscous stress and advective reconstruction also require convergence checks. Keep fixed analytic parameters, solve on genuinely refined external meshes, and report native continuity and momentum residuals. This audit neither establishes grid convergence nor experimental drag/Cp acceptance.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,default=DEFAULT_CASE/'input.msh')
    parser.add_argument('--fields',type=Path,default=DEFAULT_CASE/'fields.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/verification-native-suboff-mms/reconstruction-audit')
    parser.add_argument('--source-root',type=Path,default=None)
    parser.add_argument('--amplitude',type=float,default=.5)
    parser.add_argument('--wavenumber',type=float,default=.7)
    parser.add_argument('--nu',type=float,default=.1)
    args = parser.parse_args()
    if args.nu <= 0 or args.amplitude <= 0 or args.wavenumber <= 0:
        parser.error('positive fixed analytic parameters required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty to preserve audit evidence')
    archived = args.fields.parent/'source-snapshot/src'
    source_root = (args.source_root or (archived if archived.exists() else ROOT/'src')).resolve()
    sys.path.insert(0,str(source_root))
    from tensorfvm.native_mesh3d import NativeMesh3D
    from tensorfvm.native_collocated3d import NativeSteadySettings, NativeSteadySolver3D
    from tensorfvm.native_manufactured3d import abc_fields, exact_boundary_flux, volume_weighted_error
    field = lambda points: abc_fields(points,amplitude=args.amplitude,wavenumber=args.wavenumber,nu=args.nu)
    field.parameters = dict(amplitude=args.amplitude,wavenumber=args.wavenumber,nu=args.nu,rho=1.)
    production_config = args.fields.parent/'config.json'
    production_result = args.fields.parent/'result.json'
    config = json.loads(production_config.read_text()) if production_config.exists() else None
    producer = json.loads(production_result.read_text()) if production_result.exists() else None
    if producer is not None and not producer.get('converged',False):
        raise ValueError('Producer field is not marked converged')
    used_sources = {Path(__file__).resolve(): 'scripts/'+Path(__file__).name}
    for name in NATIVE_MODULES:
        module = sys.modules['tensorfvm.'+name]
        path = Path(module.__file__).resolve(); key = 'src/tensorfvm/'+name+'.py'
        used_sources[path] = key
        if config and config['source_sha256'].get(key) != digest(path):
            raise ValueError('Producer source hash mismatch: '+key)
    inputs = dict(mesh=dict(path=str(args.mesh.resolve()),sha256=digest(args.mesh)),
                  fields=dict(path=str(args.fields.resolve()),sha256=digest(args.fields)))
    if config and config.get('mesh_sha256') != inputs['mesh']['sha256']:
        raise ValueError('Producer mesh hash mismatch')
    for name,path in [('producer_config',production_config),('producer_result',production_result)]:
        if path.exists(): inputs[name]=dict(path=str(path.resolve()),sha256=digest(path))
    mesh = NativeMesh3D.from_gmsh(args.mesh,triangulate_faces=True)
    exact = field(mesh.cell_centers); boundary = mesh.neighbor<0
    solver = NativeSteadySolver3D(mesh,boundary,field(mesh.face_centers)['velocity'],
        settings=NativeSteadySettings(viscosity=args.nu),body_force=exact['acceleration'],
        boundary_flux=exact_boundary_flux(mesh,field),gauge_value=exact['pressure'][0])
    print(f'AUDIT_OPERATOR_READY cells={len(mesh.cells)}',flush=True)
    with np.load(args.fields) as fields:
        result = audit(mesh,fields,field,solver,volume_weighted_error,args.nu)
    result.update(schema='tensorfvm.native-mms-reconstruction-audit/1',
        analytic_parameters=field.parameters, inputs=inputs,
        production_source_root=str(source_root), producer_converged=None if producer is None else producer['converged'],
        production_pressure_solution_unchanged=True, grid_convergence_qualified=False,
        experimental_accuracy_qualified=False, source_sha256={key:digest(path) for path,key in used_sources.items()})
    for label,item in inputs.items():
        if digest(Path(item['path'])) != item['sha256']:
            raise RuntimeError('Input changed during audit: '+label)
    args.output.mkdir(parents=True,exist_ok=True)
    for path,key in used_sources.items():
        target=args.output/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if digest(target)!=result['source_sha256'][key]:raise RuntimeError('Source changed during audit: '+key)
    write(args.output/'audit.json',result)
    (args.output/'report.md').write_text(report_text(result,args.mesh,args.fields))
    print(json.dumps(dict(errors=result['global_errors'],quadratic=result['quadratic_probe']),indent=2))


if __name__ == '__main__':
    main()
