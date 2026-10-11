import importlib.util
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D

script=Path(__file__).parents[1]/'scripts/plot_native_suboff_fields.py'
spec=importlib.util.spec_from_file_location('native_suboff_plots',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def mixed_mesh():
    cube=[(0,0,-1),(1,0,-1),(1,1,-1),(0,1,-1),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    wedge=[(2,0,-1),(3,0,-1),(2,1,-1),(2,0,1),(3,0,1),(2,1,1)]
    tetra=[(4,0,-1),(5,0,-1),(4,1,-1),(4,0,1)]
    return NativeMesh3D.from_cells(cube+wedge+tetra,[tuple(range(8)),tuple(range(8,14)),tuple(range(14,18))],['hexahedron','wedge','tetra'],triangulate_faces=True)


def polygon_area(p):
    return .5*abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))


def test_mixed_native_cell_sections_have_exact_areas_and_field_indices():
    polygons,ids=module.cell_section_polygons(mixed_mesh())
    assert ids.tolist()==[0,1,2]
    assert np.allclose([polygon_area(p) for p in polygons],[1,.5,.125])
    actual_cell_pressure=np.array([7.,-3.,11.])
    assert actual_cell_pressure[ids].tolist()==[7.,-3.,11.]
    # The tetra slice is halfway between its tip and base. Projecting its
    # off-plane vertices would incorrectly extend this section to x=5.
    assert polygons[2][:,0].max()==4.5


def test_coplanar_facets_cancel_diagonals_and_skip_tip_tangency():
    polygons,ids=module.cell_section_polygons(mixed_mesh(),value=1.)
    assert ids.tolist()==[0,1]
    assert np.allclose([polygon_area(p) for p in polygons],[1,.5])
    assert len(polygons[0])==4


def test_plane_outside_mesh_has_no_fake_projected_polygons():
    polygons,ids=module.cell_section_polygons(mixed_mesh(),value=3.)
    assert polygons==[] and len(ids)==0


def test_plot_uses_supplied_wall_face_pressure_and_marks_unqualified(tmp_path,monkeypatch):
    import csv,json
    from dataclasses import replace
    mesh=mixed_mesh()
    mesh=replace(mesh,patch_labels=np.where(mesh.neighbor<0,'wall','internal'))
    monkeypatch.setattr(NativeMesh3D,'from_gmsh',classmethod(lambda cls,*a,**k:mesh))
    mesh_path=tmp_path/'native.msh';mesh_path.write_text('mock native loader; topology supplied above')
    face_pressure=100.+np.arange(len(mesh.owner))
    fields=tmp_path/'fields.npz'
    np.savez(fields,velocity=np.ones((len(mesh.cells),3)),pressure=np.array([1.,2.,3.]),flux=np.zeros(len(mesh.owner)),pressure_faces=face_pressure)
    summary=module.plot_fields(mesh_path,fields,tmp_path/'plots',case_kind='manufactured')
    assert summary['status_text']=='Diagnostic / unqualified'
    assert 'Reconstructed native wall-face pressure' in summary['wall_pressure_definition']
    assert summary['qualification_inferred_from_figures'] is False
    with (tmp_path/'plots/wall-pressure.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert rows
    for row in rows:
        index=int(row['native_face_index'])
        assert float(row['plotted_wall_pressure'])==face_pressure[index]
        assert float(row['adjacent_cell_pressure'])==[1.,2.,3.][mesh.owner[index]]
