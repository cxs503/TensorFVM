from dataclasses import replace
import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.overset3d import CartesianGrid3D
from tensorfvm.native_face_geometry3d import (
    polygon_triangles, polygon_face_covariance, merge_coplanar_faces,
)
from tensorfvm.native_quadratic3d import NativeQuadratic3D
from tensorfvm.native_quadratic_maps3d import face_mean_matrix, face_gradient_matrices
from tensorfvm.native_manufactured3d import (
    abc_fields, exact_boundary_flux, exact_boundary_velocity_means,
)


def mesh(mixed=False):
    base = CartesianGrid3D.from_bounds([0, 0, 0], [1, 1, 1], [3, 3, 3])
    points = base.vertices@np.array([[1, .18, .12], [0, 1, .2], [0, 0, 1]]).T
    original = NativeMesh3D.from_cells(points, base.cells, base.cell_types,
                                      triangulate_faces=True)
    if not mixed:
        return original
    center = len(points)
    points = np.vstack((points, original.cell_centers[0]))
    faces = np.flatnonzero((original.owner == 0)|(original.neighbor == 0))
    cells = list(base.cells[1:])+[list(original.face_vertices[f])+[center] for f in faces]
    return NativeMesh3D.from_cells(points, cells, ['hexahedron']*26+['tetra']*len(faces),
                                  triangulate_faces=True)


def test_exact_polygon_covariance_and_quadratic_integral():
    rectangle = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0.]])
    np.testing.assert_allclose(polygon_face_covariance(rectangle),
                               np.diag([4/12, 9/12, 0]), atol=1e-15)
    shifted = rectangle+[8, -2, 5]
    np.testing.assert_allclose(polygon_face_covariance(shifted),
                               polygon_face_covariance(rectangle), atol=1e-15)
    # A parallelogram happens to share this formula; a general quad does not.
    trapezoid = np.array([[0, 0, 0], [2, 0, 0], [1, 3, 0], [0, 3, 0.]])
    fan = polygon_triangles(trapezoid)
    areas = np.linalg.norm(np.cross(fan[:, 1]-fan[:, 0], fan[:, 2]-fan[:, 0]), axis=1)
    centroid = (areas[:, None]*fan.mean(axis=1)).sum(axis=0)/areas.sum()
    centered = trapezoid-centroid
    assert not np.allclose(centered.T@centered/12, polygon_face_covariance(trapezoid))
    triangle = rectangle[:3]
    offsets = triangle-triangle.mean(axis=0)
    np.testing.assert_allclose(polygon_face_covariance(triangle), offsets.T@offsets/12,
                               atol=2e-16)
    # A convex pentagon exercises a genuinely polygonal, non-quadrilateral fan.
    theta = np.linspace(0, 2*np.pi, 6)[:-1]
    pentagon = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(5)))
    moment = polygon_face_covariance(pentagon)
    assert np.linalg.eigvalsh(moment).min() >= -1e-15
    assert len(polygon_triangles(pentagon)) == 3


@pytest.mark.parametrize('points', [
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, .1]],
    [[0, 0, 0], [1, 0, 0], [.2, .2, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]],
])
def test_reject_nonplanar_concave_and_folded_faces(points):
    with pytest.raises(ValueError):
        polygon_face_covariance(points)


@pytest.mark.parametrize('mixed', [False, True])
def test_merge_preserves_cells_owner_neighbor_patches_closure_and_exact_moments(mixed):
    original = mesh(mixed)
    merged, metadata = merge_coplanar_faces(original)
    assert metadata['merged_pairs'] > 0
    assert any(len(ids) == 4 for ids in merged.face_vertices)
    assert merged.cells is original.cells
    assert merged.cell_centers is original.cell_centers
    assert merged.cell_volumes is original.cell_volumes
    assert metadata['audit']['maximum_relative_closure'] < 1e-14
    assert metadata['audit']['minimum_face_projection'] > 0
    for face, group in enumerate(metadata['original_face_groups']):
        np.testing.assert_array_equal(metadata['original_to_merged'][list(group)], face)
        assert len(set(original.owner[list(group)])) == 1
        assert len(set(original.neighbor[list(group)])) == 1
        assert len(set(original.patch_labels[list(group)])) == 1
        np.testing.assert_array_equal(merged.area_vectors[face],
                                      original.area_vectors[list(group)].sum(axis=0))
        weights = np.linalg.norm(original.area_vectors[list(group)], axis=1)
        np.testing.assert_allclose(merged.face_centers[face],
                                   np.sum(weights[:, None]*original.face_centers[list(group)], axis=0)/weights.sum(), atol=1e-16)
        covariance = polygon_face_covariance(original.vertices[list(merged.face_vertices[face])],
                                              merged.face_centers[face])
        expected = np.zeros((3, 3))
        for index, weight in zip(group, weights/weights.sum()):
            center = original.face_centers[index]-merged.face_centers[face]
            expected += weight*(polygon_face_covariance(original.vertices[list(original.face_vertices[index])],
                                                         original.face_centers[index])+np.outer(center, center))
        np.testing.assert_allclose(covariance, expected, atol=2e-16)
    if mixed:
        # Shared hex-tet facets each have a distinct tet neighbor and therefore
        # must never merge into a quad spanning different control volumes.
        for face, neighbor in enumerate(merged.neighbor):
            if neighbor >= 0 and merged.cell_types[merged.owner[face]] != merged.cell_types[neighbor]:
                assert len(metadata['original_face_groups'][face]) == 1
    again, second = merge_coplanar_faces(merged)
    assert second['merged_pairs'] == 0
    assert again.face_vertices == merged.face_vertices


def test_perpendicular_and_nonplanar_facets_and_distinct_patches_remain_split():
    cube = CartesianGrid3D.from_bounds([0, 0, 0], [1, 1, 1], [1, 1, 1])
    points = cube.vertices.copy()
    points[np.argmax(points.sum(axis=1)), 2] += .1
    warped = NativeMesh3D.from_cells(points, cube.cells, cube.cell_types, triangulate_faces=True)
    merged, metadata = merge_coplanar_faces(warped)
    assert metadata['merged_pairs'] < 6
    for group in metadata['original_face_groups']:
        if len(group) == 2:
            normals = warped.area_vectors[list(group)]
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            np.testing.assert_allclose(normals[0], normals[1], atol=1e-12)
    original = mesh()
    _, metadata = merge_coplanar_faces(original)
    pair = next(group for group in metadata['original_face_groups']
                if len(group) == 2 and original.neighbor[group[0]] < 0)
    labels = original.patch_labels.astype('U30').copy()
    labels[pair[0]] = 'different-physical-patch'
    _, separate = merge_coplanar_faces(replace(original, patch_labels=labels))
    assert separate['original_to_merged'][pair[0]] != separate['original_to_merged'][pair[1]]


@pytest.mark.parametrize('mixed', [False, True])
def test_polygon_boundary_quadrature_split_merged_consistency_and_quadratic_means(mixed):
    original = mesh(mixed)
    merged, metadata = merge_coplanar_faces(original)
    field = lambda x: {'velocity': np.column_stack((1+x[:, 0]**2,
                        2+x[:, 1]**2, x[:, 0]*x[:, 1]+x[:, 2]))}
    constant = lambda x: {'velocity': np.tile([.7, -.2, 1.3], (len(x), 1))}
    for producer in (field, abc_fields, constant):
        old_flux = exact_boundary_flux(original, producer)
        new_flux = exact_boundary_flux(merged, producer)
        old_means = exact_boundary_velocity_means(original, producer)
        new_means = exact_boundary_velocity_means(merged, producer)
        for face, group in enumerate(metadata['original_face_groups']):
            np.testing.assert_allclose(new_flux[face], old_flux[list(group)].sum(),
                                       rtol=2e-14, atol=1e-15)
        if producer is not field:
            assert abs(new_flux.sum()) < 1e-13
            weights = np.linalg.norm(original.area_vectors[list(group)], axis=1)
            np.testing.assert_allclose(new_means[face],
                                       (weights[:, None]*old_means[list(group)]).sum(axis=0)/weights.sum(),
                                       rtol=2e-14, atol=1e-15)
    op = NativeQuadratic3D(merged, merged.neighbor < 0)
    expected = field(merged.face_centers)['velocity']
    expected[:, 0] += op.face_covariance[:, 0, 0]
    expected[:, 1] += op.face_covariance[:, 1, 1]
    expected[:, 2] += op.face_covariance[:, 0, 1]
    means = exact_boundary_velocity_means(merged, field)
    np.testing.assert_allclose(means[op.boundary], expected[op.boundary], atol=2e-15)
    values = field(merged.cell_centers)['velocity']
    boundary = field(merged.face_centers)['velocity']
    np.testing.assert_allclose(op.face_mean(values, boundary), expected, atol=2e-12)
    # Analytic divergence is affine, so volume-centroid integration is exact.
    exact_volume_source = np.dot(2*merged.cell_centers[:, 0]
                                 + 2*merged.cell_centers[:, 1]+1,
                                 merged.cell_volumes)
    assert exact_boundary_flux(merged, field).sum() == pytest.approx(exact_volume_source, abs=3e-14)


@pytest.mark.parametrize('boundary', ['dirichlet', 'mixed', 'free'])
def test_polygon_sparse_quadratic_maps_replay_original_action(boundary):
    merged, _ = merge_coplanar_faces(mesh(True))
    mask = merged.neighbor < 0
    if boundary == 'mixed':
        mask[np.flatnonzero(mask)[::2]] = False
    elif boundary == 'free':
        mask[:] = False
    op = NativeQuadratic3D(merged, mask)
    values = np.random.default_rng(91).normal(size=(op.N, 3))
    for prescribed in (False, True):
        np.testing.assert_allclose(face_mean_matrix(op, prescribed)@values,
                                   op.face_mean(values, 0., boundary_face_means=0. if prescribed else None),
                                   rtol=3e-13, atol=5e-13)
    for compact in (False, True):
        maps = face_gradient_matrices(op, compact)
        np.testing.assert_allclose(np.stack([g@values for g in maps], axis=-1),
                                   op.face_gradient(values, 0., compact=compact),
                                   rtol=3e-13, atol=3e-12)
