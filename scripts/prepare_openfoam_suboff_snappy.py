#!/usr/bin/env python3
"""Prepare a body-fitted AFF-1 snappyHexMesh pilot using the mother profile.

Literature-informed independent setup, not an exact AFF-8/v2112 reproduction.
"""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from tensorfvm.openfoam_case import header,write_rans_case
from tensorfvm.suboff_geometry import radius,SOURCE_SHA256

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--surface-level',type=int,default=6)
p.add_argument('--tail-level',type=int,choices=[6,7,8],help='Optional local volume refinement around x/L=0.87..1.02')
p.add_argument('--first-layer-thickness',type=float,default=.0002)
p.add_argument('--iterations',type=int,default=500)
p.add_argument('--reynolds',type=float,default=13218056.809580836)
a=p.parse_args();out=a.output
tail_geometry=' tailBox { type searchableBox; min (0.87 -0.04 -0.04); max (1.02 0.04 0.04); }' if a.tail_level else ''
tail_refinement=f' tailBox {{ mode inside; levels ((1e15 {a.tail_level})); }}' if a.tail_level else ''
if not 4 <= a.surface_level <= 7:p.error('surface level must be 4..7')
if not np.isfinite(a.first_layer_thickness) or a.first_layer_thickness <= 0:p.error('positive layer thickness required')
write_rans_case(None,out,reynolds=a.reynolds,iterations=a.iterations)
(out/'constant/triSurface').mkdir()
x=.5*(1-np.cos(np.linspace(0,np.pi,257)))
r=radius(x);r[[0,-1]]=0
azimuth=np.arange(96)*2*np.pi/96
points=np.stack(np.broadcast_arrays(x[:,None],r[:,None]*np.cos(azimuth),r[:,None]*np.sin(azimuth)),axis=2)
triangles=[]
for i in range(len(x)-1):
 for j in range(len(azimuth)):
  k=(j+1)%len(azimuth)
  for tri in [np.array([points[i,j],points[i+1,j],points[i+1,k]]),np.array([points[i,j],points[i+1,k],points[i,k]])]:
   normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=np.linalg.norm(normal)
   if norm<1e-18:continue
   center=tri.mean(0)
   if np.dot(normal[1:],center[1:])<0:tri=tri[[0,2,1]];normal=-normal
   triangles.append((normal/norm,tri))
with (out/'constant/triSurface/suboff.stl').open('w') as f:
 f.write('solid wall\n')
 for normal,tri in triangles:
  f.write('facet normal '+' '.join(f'{v:.16g}' for v in normal)+'\nouter loop\n')
  for point in tri:f.write('vertex '+' '.join(f'{v:.16g}' for v in point)+'\n')
  f.write('endloop\nendfacet\n')
 f.write('endsolid wall\n')
(out/'system/blockMeshDict').write_text(header('blockMeshDict')+'''
convertToMeters 1;
vertices ((-3 -2.625 -2.125) (4 -2.625 -2.125) (4 2.375 -2.125) (-3 2.375 -2.125)
          (-3 -2.625 0.875) (4 -2.625 0.875) (4 2.375 0.875) (-3 2.375 0.875));
blocks (hex (0 1 2 3 4 5 6 7) (28 20 12) simpleGrading (1 1 1));
edges ();
boundary
(
 inlet { type patch; faces ((0 4 7 3)); }
 outlet { type patch; faces ((1 2 6 5)); }
 farfield { type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }
);
mergePatchPairs ();
''')
(out/'system/snappyHexMeshDict').write_text(header('snappyHexMeshDict')+f'''
castellatedMesh true;
snap true;
addLayers true;
geometry
{{
 wall {{ type triSurfaceMesh; file "suboff.stl"; }}
 nearHull {{ type searchableBox; min (-0.1 -0.12 -0.12); max (1.1 0.12 0.12); }}
 wake {{ type searchableBox; min (0.7 -0.2 -0.2); max (2.5 0.2 0.2); }}
 {tail_geometry}
}}
castellatedMeshControls
{{
 maxLocalCells 2000000; maxGlobalCells 2000000; minRefinementCells 0;
 nCellsBetweenLevels 3;
 features ();
 refinementSurfaces {{ wall {{ level ({a.surface_level} {a.surface_level}); patchInfo {{ type wall; }} }} }}
 resolveFeatureAngle 30;
 refinementRegions
 {{
  nearHull {{ mode inside; levels ((1e15 4)); }}
  wake {{ mode inside; levels ((1e15 3)); }}
  {tail_refinement}
 }}
 locationInMesh (-2 0 0);
 allowFreeStandingZoneFaces true;
}}
snapControls
{{
 nSmoothPatch 3; tolerance 2; nSolveIter 30; nRelaxIter 5;
 nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false;
}}
addLayersControls
{{
 relativeSizes false;
 layers {{ wall {{ nSurfaceLayers 6; }} }}
 expansionRatio 1.15;
 firstLayerThickness {a.first_layer_thickness:.17g};
 minThickness {a.first_layer_thickness*.4:.17g};
 nGrow 0; featureAngle 60; slipFeatureAngle 30;
 nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
 maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90;
 nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
 #includeEtc "caseDicts/mesh/generation/meshQualityDict"
}}
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
''')
# Refine the volume before introducing the thin hull. Otherwise an initial
# absence of cell-centre intersection edges can terminate surface refinement.
background=(out/'system/snappyHexMeshDict').read_text()
background=background.replace('snap true;', 'snap false;').replace('addLayers true;', 'addLayers false;')
background=background.replace(' wall { type triSurfaceMesh; file "suboff.stl"; }', '')
background=re.sub(r' refinementSurfaces .*', ' refinementSurfaces {}', background)
(out/'system/backgroundRefinementDict').write_text(background)
(out/'provenance.json').write_text(json.dumps(dict(kind='OpenFOAM Foundation 12 AFF-1 independent snappy pilot',
 mother_profile_sha256=SOURCE_SHA256,stl_sha256=hashlib.sha256((out/'constant/triSurface/suboff.stl').read_bytes()).hexdigest(),
 surface_triangles=len(triangles),surface_level=a.surface_level,reynolds=a.reynolds,
 tail_level=a.tail_level,
 first_layer_thickness_L=a.first_layer_thickness,layers_requested=6,growth_requested=1.15,
 domain_L=[[-3,-2.625,-2.125],[4,2.375,.875]],turbulence_inlet=dict(intensity=.005,length_scale_L=.05),
 geometry='AFF-1 bare hull, zero incidence; fully turbulent; no explicit trip',
 literature_doi='10.47176/jafm.16.05.1513',literature_geometry='AFF-8',literature_version='OpenCFD v2112',
 literature_exact_reproduction=False,physical_benchmark_passed=False),indent=2)+'\n')
print(out)
