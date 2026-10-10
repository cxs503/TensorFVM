"""Check archived and current producer sources, artifacts and per-run provenance."""
import json
from pathlib import Path
from tensorfvm.verification.core import sha

root=Path(__file__).resolve().parents[1]
count=0
for path in sorted((root/'docs').glob('*/manifest.json')):
    m=json.loads(path.read_text());directory=path.parent;count+=1
    for file,value in m.get('source_sha256',{}).items():
        source=directory/m['source_snapshot_root']/file if m.get('source_snapshot_root') else root/file
        if not source.is_file() or sha(source)!=value:raise ValueError('Source mismatch: '+str(source))
    for file,value in m.get('artifacts_sha256',{}).items():
        if not (directory/file).is_file() or sha(directory/file)!=value:raise ValueError('Artifact mismatch: '+str(directory/file))
for name in ('cylinder','naca'):
    directory=root/'docs'/('verification-external-'+name);s=json.loads((directory/'summary.json').read_text())
    if len(s['runs'])!=3:raise ValueError('Incomplete three-grid study')
    for row in s['runs']:
        if row['passed']!=all(q['passed'] for q in row['metrics']):raise ValueError('False case badge')
        for file,value in row.get('producer_provenance',{}).get('source_sha256',{}).items():
            if sha(directory/row['source_snapshot_root']/file)!=value:raise ValueError('Per-run producer mismatch')
        if any(q['limit']!=.03 for q in row['metrics'] if q['name'].endswith('relative_error')):raise ValueError('Relaxed physical gate')
print('Evidence verified:',count,'manifests; six actual external-flow grids; all recorded producer snapshots and artifact hashes')

# New physical Gmsh reports include immutable-input and continuation evidence.
import numpy as np
external_count=0
for path in sorted((root/'docs').glob('verification-gmsh-*/summary.json')):
    directory=path.parent;summary=json.loads(path.read_text());manifest=json.loads((directory/'manifest.json').read_text())
    for file,value in manifest.get('input_sha256',{}).items():
        if sha(root/file)!=value:raise ValueError('Input field mismatch: '+file)
    def check_producer(producer,snapshot_root):
        for file,value in producer.get('source_sha256',{}).items():
            source=directory/snapshot_root/file if snapshot_root else root/file
            if sha(source)!=value:raise ValueError('Numerical producer mismatch: '+str(source))
    check_producer(summary['provenance'],manifest.get('source_snapshot_root'))
    if 'original_numerical_producer_provenance' in summary:
        check_producer(summary['original_numerical_producer_provenance'],summary['numerical_source_snapshot_root'])
    equations=json.loads((directory/'equations-audit.json').read_text())
    mesh_audits=json.loads((directory/'mesh-audit.json').read_text())
    for row in summary['runs']:
        external_count+=1
        if row.get('producer_provenance'):check_producer(row['producer_provenance'],row['source_snapshot_root'])
        if row.get('continuation'):
            c=row['continuation'];check_producer(c['initial_numerical_producer_provenance'],c['initial_source_snapshot_root'])
            if sha(root/c['initial_fields_path'])!=c['initial_fields_sha256']:raise ValueError('Continuation input mismatch')
            history=json.loads((directory/row['directory']/'history.json').read_text())
            initial_path=(root/c['initial_fields_path']).parent/'history.json'
            initial_history=json.loads(initial_path.read_text())
            if history[:c['initial_iterations']]!=initial_history:raise ValueError('Initial iteration history changed')
            if row['iterations']!=c['initial_iterations']+c['additional_iterations']:raise ValueError('False continuation count')
        if row['passed']!=all(q['passed'] for q in row['metrics']):raise ValueError('False Gmsh badge')
        for metric in row['metrics']:
            if metric['name'].endswith('relative_error') and (metric['limit']!=.03 or metric['passed']!=(metric['value']<.03)):raise ValueError('Relaxed physical gate')
        equation=next(q for q in equations['runs'] if q['resolution']==row['resolution'])
        mesh_audit=next(q for q in mesh_audits['runs'] if q['resolution']==row['resolution'])
        if row['passed'] and not (row['converged'] and equation['physical_fixedpoint_passed'] and mesh_audit['passed']):raise ValueError('Unverified accepted run')
        if row['case']=='naca' and row['passed'] and not row.get('reference_domain_matched'):raise ValueError('Unmatched NACA reference accepted')
        if row.get('wall_postprocessed_without_resolving'):
            if sha(root/row['upstream_fields_path'])!=row['upstream_fields_sha256']:raise ValueError('Wall postprocessing input changed')
            with np.load(root/row['upstream_fields_path']) as before,np.load(directory/row['directory']/'fields.npz') as after:
                for key in ('cell_velocity_m_s','cell_pressure_pa','face_mass_flux','cell_volumes_m2','vertices_m','cell_polygons_m','cell_centers_m','face_owner','face_neighbor','face_area_vectors_m','surface_pressure_force','surface_wall_pressure_pa'):
                    np.testing.assert_array_equal(before[key],after[key])
print('Gmsh evidence verified:',external_count,'reported results; source/input/continuation hashes, immutable solved fields, strict physical gates and independent equation/mesh audits')
