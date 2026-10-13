import numpy as np
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_velocity_means,volume_weighted_error
for n in (2,4,6):
 base=channel_mesh(n);cells=[];kinds=[]
 pieces=((0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6))
 for i,cell in enumerate(base.cells):
  if (i%n+i//n%n+i//n**2)%2:
   cells.append(cell);kinds.append('hexahedron')
  else:
   cells.extend([[cell[j] for j in t] for t in pieces]);kinds.extend(['tetra']*6)
 points=base.vertices@np.array([[1,0,0],[.2,1,0],[.1,.1,1]])
 m=NativeMesh3D.from_cells(points,cells,kinds,triangulate_faces=True)
 e=abc_fields(m.cell_centers);bd=exact_boundary_velocity_means(m,abc_fields)
 s=NativeHybridStokes3D(m,m.neighbor<0,bd,viscosity=.1,body_force=e['acceleration']-np.einsum('cij,cj->ci',e['velocity_gradient'],e['velocity']),gauge_value=e['pressure'][0],coupled_linear_method='reused_lu')
 r=s.solve(algorithm='coupled',tolerance=1e-8)
 print('RESULT',n,len(m.cells),r['numerically_converged'],volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes),volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True),r['history'][-1],flush=True)
