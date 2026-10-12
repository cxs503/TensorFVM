#!/usr/bin/env python3
"""Plot actual OpenFOAM cut cells, with cell-value pressure/velocity (no smoothing)."""
import argparse,gzip,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path,required=True);p.add_argument('--time',type=int,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--foam-wrapper',default='tensorfvm-foam12')
a=p.parse_args();case=a.case.resolve();out=a.output;out.mkdir(parents=True,exist_ok=True)
name='tensorfvmCutPlane'
(case/'system'/name).write_text('''tensorfvmCutPlane
{
 type surfaces;
 libs ("libsampling.so");
 writeControl writeTime;
 surfaceFormat vtk;
 fields (p U);
 interpolationScheme cell;
 surfaces
 (
  yNormal
  {
   type cutPlane;
   planeType pointAndNormal;
   point (0 0 0);
   normal (0 1 0);
   interpolate false;
  }
 );
}
''')
with (case/'log.tensorfvmCutPlane').open('w') as log:
 subprocess.run([a.foam_wrapper,'foamPostProcess','-case',str(case),'-func',name,'-time',str(a.time)],stdout=log,stderr=subprocess.STDOUT,check=True)
path=case/'postProcessing'/name/str(a.time)/'yNormal.vtk';text=path.read_text()
assert '\nASCII\n' in text
words=text.split();i=words.index('POINTS');n=int(words[i+1]);i+=3
points=np.asarray(words[i:i+3*n],dtype=float).reshape(n,3);i+=3*n
assert words[i]=='POLYGONS';nf=int(words[i+1]);i+=3;faces=[]
for _ in range(nf):
 n=int(words[i]);ids=np.asarray(words[i+1:i+1+n],dtype=int);faces.append(points[ids][:,[0,2]]);i+=1+n
assert words[i]=='CELL_DATA' and int(words[i+1])==nf;i+=2
assert words[i]=='FIELD';nfields=int(words[i+2]);i+=3;fields={}
for _ in range(nfields):
 field=words[i];components=int(words[i+1]);n=int(words[i+2]);i+=4
 fields[field]=np.asarray(words[i:i+n*components],dtype=float).reshape(n,components);i+=n*components
cp=2*fields['p'][:,0];speed=np.linalg.norm(fields['U'],axis=1)
fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
for ax,val,title,cmap,bounds in [(axes[0],cp,'Midplane pressure coefficient (cell values)','coolwarm',(-.35,1)),(axes[1],speed,'Midplane speed / U infinity (cell values)','viridis',(0,1.3))]:
 coll=PolyCollection(faces,array=val,cmap=cmap,edgecolors='none',antialiaseds=False,clim=bounds)
 ax.add_collection(coll);ax.set_title(title);fig.colorbar(coll,ax=ax,shrink=.8)
axes[2].add_collection(PolyCollection(faces,facecolors='none',edgecolors='#444444',linewidths=.25));axes[2].set_title('Actual volume mesh intersection; stern detail')
for ax in axes:ax.set(xlabel='x/L',ylabel='z/L',xlim=(-.1,1.5),ylim=(-.18,.18),aspect='equal')
axes[2].set(xlim=(.84,1.1),ylim=(-.055,.055))
fig.savefig(out/'midplane-fields.png',dpi=180);plt.close(fig)
with gzip.open(out/'midplane.vtk.gz','wb') as stream:stream.write(path.read_bytes())
(out/'midplane-metadata.json').write_text(json.dumps(dict(time=a.time,plane='y/L=0',interpolation='none; actual cell values on cut polygons',cut_polygons=nf,cp_min=float(cp.min()),cp_max=float(cp.max()),speed_min=float(speed.min()),speed_max=float(speed.max()),display_cp_limits=[-.35,1],display_speed_limits=[0,1.3]),indent=2)+'\n')
