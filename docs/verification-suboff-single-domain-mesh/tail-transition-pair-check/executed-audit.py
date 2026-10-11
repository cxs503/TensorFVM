"""Check fixed structured geometry across the exterior-only meshing experiment."""
import json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
pairs=[('docs/verification-suboff-single-domain-mesh/coarse','docs/verification-suboff-single-domain-mesh/coarse-tail-transition'),('docs/verification-suboff-hybrid-mesh/medium','docs/verification-suboff-single-domain-mesh/medium-tail-transition')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];bindings={}
def sorted_rows(a):return a[np.lexsort(a.T)]
for old,new in pairs:
 files=[root/name/'native-geometry.npz' for name in (old,new)]
 for f in files:bindings[str(f.relative_to(root))]=sha(f)
 with np.load(files[0]) as a,np.load(files[1]) as b:
  def data(z):
   structured=z['cell_types']!='tetra';wall=z['patch_labels']=='wall'
   return (sorted_rows(np.column_stack((z['cell_centers'][structured],z['cell_volumes'][structured]))),
           sorted_rows(np.column_stack((z['face_centers'][wall],z['area_vectors'][wall]))))
  x,y=data(a),data(b)
  checks=dict(structured_cell_centers_and_volumes_bitwise_equal=np.array_equal(x[0],y[0]),wall_face_centers_and_area_vectors_bitwise_equal=np.array_equal(x[1],y[1]))
  assert all(checks.values()),checks
  rows.append(dict(baseline=old,candidate=new,structured_cells=len(x[0]),wall_triangles=len(x[1]),checks=checks))
(out/'result.json').write_text(json.dumps(dict(passed=True,pairs=rows,scope='fixed structured volume and wall facet geometry; does not imply identical exterior stencils or improved flow accuracy'),indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps(dict(input_sha256=bindings,artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2)+'\n')
print(json.dumps(rows))
