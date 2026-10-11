"""Plot reconstruction must use producer face indices, including planar merging."""
import importlib.util
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_face_geometry3d import merge_coplanar_faces

script=Path(__file__).parents[1]/'scripts/plot_native_suboff_fields.py'
spec=importlib.util.spec_from_file_location('native_suboff_face_policy',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def cube():
    vertices=[(0,0,-1),(1,0,-1),(1,1,-1),(0,1,-1),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    m=NativeMesh3D.from_cells(vertices,[tuple(range(8))],['hexahedron'],triangulate_faces=True)
    return replace(m,patch_labels=np.full(len(m.owner),'wall'))


def setup(tmp_path,monkeypatch,option):
    mesh=cube();monkeypatch.setattr(NativeMesh3D,'from_gmsh',classmethod(lambda cls,*a,**k:mesh))
    mesh_path=tmp_path/'native.msh';mesh_path.write_text('mock loader; genuine cube topology')
    if option is not None:(tmp_path/'config.json').write_text(json.dumps({'merge_coplanar_faces':option}))
    return mesh,mesh_path,tmp_path/'fields.npz'


@pytest.mark.parametrize('option,expected_faces',[(None,12),(False,12),(True,6)])
def test_config_face_policy_preserves_legacy_or_reconstructs_polygons(tmp_path,monkeypatch,option,expected_faces):
    _,path,fields=setup(tmp_path,monkeypatch,option)
    mesh,policy,_=module.load_plot_mesh(path,fields)
    assert len(mesh.owner)==expected_faces
    assert policy['merge_coplanar_faces'] is (option is True)
    assert policy['face_policy']==('merged_coplanar_polygon_faces' if option else 'triangulated_native_faces')
    polygons,ids=module.cell_section_polygons(mesh,value=0.)
    assert ids.tolist()==[0]
    assert len(polygons)==1
    polygon=polygons[0]
    area=.5*abs(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))
    assert area==pytest.approx(1)
    if option:assert len(polygon)==4
    polygons,ids=module.cell_section_polygons(mesh,value=1.)
    assert ids.tolist()==[0]
    assert len(polygons[0])==4


def test_merged_plot_matches_flux_faces_and_preserves_face_pressure_ids(tmp_path,monkeypatch):
    import csv
    original,path,fields=setup(tmp_path,monkeypatch,True)
    merged,_=merge_coplanar_faces(original);face_pressure=50+np.arange(len(merged.owner))
    np.savez(fields,velocity=np.ones((1,3)),pressure=np.array([7.]),flux=np.zeros(len(merged.owner)),pressure_faces=face_pressure)
    summary=module.plot_fields(path,fields,tmp_path/'plots')
    assert summary['merged_pairs']==6
    assert summary['reconstructed_faces']==6
    assert summary['config_sha256']
    assert (tmp_path/'plots/run-config.json').read_bytes()==(tmp_path/'config.json').read_bytes()
    mapping=np.load(tmp_path/'plots/face-topology-map.npz')
    assert len(mapping['original_to_merged'])==12
    with (tmp_path/'plots/wall-pressure.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==6
    for row in rows:assert float(row['plotted_wall_pressure'])==face_pressure[int(row['native_face_index'])]


@pytest.mark.parametrize('wrong_kind',['flux','pressure_faces'])
def test_merged_face_arrays_with_old_triangle_length_are_rejected(tmp_path,monkeypatch,wrong_kind):
    original,path,fields=setup(tmp_path,monkeypatch,True)
    values=dict(velocity=np.ones((1,3)),pressure=np.array([7.]),flux=np.zeros(6),pressure_faces=np.ones(6))
    values[wrong_kind]=np.ones(len(original.owner));np.savez(fields,**values)
    with pytest.raises(ValueError,match='face|flux'):module.plot_fields(path,fields,tmp_path/'plots')
    assert not (tmp_path/'plots/fields.png').exists()


def test_non_boolean_merge_configuration_is_rejected(tmp_path,monkeypatch):
    _,path,fields=setup(tmp_path,monkeypatch,'false')
    with pytest.raises(ValueError,match='JSON boolean'):module.load_plot_mesh(path,fields)


def test_same_shape_mesh_with_wrong_producer_hash_is_rejected(tmp_path,monkeypatch):
    _,path,fields=setup(tmp_path,monkeypatch,True)
    (tmp_path/'config.json').write_text(json.dumps({'merge_coplanar_faces':True,'mesh_sha256':'0'*64}))
    with pytest.raises(ValueError,match='SHA256'):module.load_plot_mesh(path,fields)
