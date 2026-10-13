import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra
from tests.test_native_rans3d import box


def thin_mixed_mesh(height=.01):
    vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                         [0,0,height],[1,0,height],[1,1,height],[0,1,height],
                         [.5,.5,1.],[.5,-.5,.5]], float)
    return NativeMesh3D.from_cells(vertices, [list(range(8)), [4,5,6,7,8], [4,8,5,9]],
                                   ['hexahedron','pyramid','tetra'], triangulate_faces=True)


@pytest.mark.parametrize('mesh_factory', [two_tetrahedra, thin_mixed_mesh, lambda: box(2)])
@pytest.mark.parametrize('components', [1,3])
def test_hybrid_homogeneous_dirichlet_energy_and_shared_flux(mesh_factory, components):
    mesh = mesh_factory()
    op = NativeHybridViscous3D(mesh, components=components)
    matrix = op.matrix.toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.linalg.eigvalsh(matrix)[0] > 0
    u = np.random.default_rng(19).normal(size=(op.N, components))
    result = op.equilibrate_faces(u)
    assert result['kinetic_energy_rate'] < 0
    assert result['energy_form'] > 0
    assert result['interface_flux_defect_max'] < 1e-10
    assert result['energy_balance_error'] < 1e-9
    # Every cell must balance its incident shared numerical face fluxes.
    integrated = np.zeros((op.N, components))
    np.add.at(integrated, mesh.owner, result['face_flux'])
    np.add.at(integrated, mesh.neighbor[~op.boundary], -result['face_flux'][~op.boundary])
    np.testing.assert_allclose(result['cell_reaction'], -integrated, atol=1e-10)


@pytest.mark.parametrize('mesh_factory', [two_tetrahedra, thin_mixed_mesh])
@pytest.mark.parametrize('components', [1,3])
def test_hybrid_affine_solution_and_analytic_traction(mesh_factory, components):
    mesh = mesh_factory()
    gradient = np.array([[.7,.2,-.1],[-.3,.4,.6],[.15,-.25,.1]])[:components]
    base = np.array([1.2,.8,-.4])[:components]
    exact = base+mesh.cell_centers @ gradient.T
    boundary = base+mesh.face_centers @ gradient.T
    op = NativeHybridViscous3D(mesh, 2.3, components=components)
    result = op.solve(boundary_values=boundary)
    np.testing.assert_allclose(result['cell_values'], exact, atol=2e-11)
    stress = gradient if components == 1 else gradient+gradient.T-2/3*np.trace(gradient)*np.eye(3)
    expected = 2.3*mesh.area_vectors @ stress.T
    np.testing.assert_allclose(result['face_flux'], expected, atol=2e-10)
    assert result['linear_relative_residual'] < 1e-12
    assert result['energy_balance_error'] < 1e-9


def test_hybrid_solid_rotation_has_no_stress_or_stabilization_force():
    mesh = thin_mixed_mesh()
    gradient = np.array([[0,-2,1],[2,0,-.3],[-1,.3,0]])
    op = NativeHybridViscous3D(mesh)
    result = op.equilibrate_faces(mesh.cell_centers @ gradient.T, mesh.face_centers @ gradient.T)
    np.testing.assert_allclose(result['face_flux'], 0., atol=1e-10)
    assert abs(result['energy_form']) < 1e-10


def test_hybrid_scalar_sine_poisson_refinement():
    errors = []
    for n in (2,4,8):
        mesh = box(n)
        x = mesh.cell_centers
        exact = np.prod(np.sin(np.pi*x), axis=1)
        source = (3*np.pi**2*exact)[:, None]
        result = NativeHybridViscous3D(mesh, components=1).solve(source)
        errors.append(np.sqrt(np.sum(mesh.cell_volumes*(result['cell_values'][:,0]-exact)**2)))
        assert result['interface_flux_defect_max'] < 1e-11
    assert np.log2(errors[-2]/errors[-1]) > 1.8, errors


def test_hybrid_cg_matches_direct_face_equilibrium():
    op = NativeHybridViscous3D(thin_mixed_mesh())
    values = np.random.default_rng(20).normal(size=(op.N, 3))
    direct = op.equilibrate_faces(values)
    iterative = op.equilibrate_faces(values, linear_method='cg')
    np.testing.assert_allclose(iterative['face_values'], direct['face_values'], atol=1e-10)
    assert iterative['face_linear_relative_residual'] < 1e-9


def test_hybrid_symmetric_deviatoric_vector_diffusion_refinement():
    errors = []
    # With the Korn jump term, 8->16 is still pre-asymptotic (1.79961).
    # Keep the 1.8 gate and verify the next refinement pair.
    for n in (4,8,16,32):
        mesh = box(n)
        sin = np.sin(np.pi*mesh.cell_centers); cos = np.cos(np.pi*mesh.cell_centers)
        phi = np.prod(sin, axis=1)
        source = np.column_stack((10/3*phi, -cos[:,0]*cos[:,1]*sin[:,2]/3,
                                  -cos[:,0]*sin[:,1]*cos[:,2]/3))*np.pi**2
        exact = np.column_stack((phi, np.zeros((len(phi), 2))))
        result = NativeHybridViscous3D(mesh).solve(source, linear_method='cg')
        errors.append(np.sqrt(np.sum(mesh.cell_volumes[:,None]*(result['cell_values']-exact)**2)))
    assert np.log2(errors[-2]/errors[-1]) > 1.8, errors
