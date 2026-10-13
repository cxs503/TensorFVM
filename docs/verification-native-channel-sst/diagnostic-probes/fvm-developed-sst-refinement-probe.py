import numpy as np
from tensorfvm.verification.native_developed_channel import developed_channel_rans,solve_developed_rans
for n in (16,32,64):
 m,s,b=developed_channel_rans(n)
 def progress(row):
  if row['iteration']%50==0:print(n,row['iteration'],row['momentum'],row['k'],row['omega'],row['scalar_absolute_ledger'],flush=True)
 r=solve_developed_rans(s,max_iterations=400,callback=progress)
 outlet=(m.neighbor<0)&(m.face_centers[:,0]>1-1e-12)
 bulk=float(r['flux'][outlet].sum());tau=-float(r['total_traction_integrated'][b['wall'],0].sum())
 print('RESULT',n,'converged',r['numerically_converged'],'bulk_flux',bulk,'bulk_cell',float(m.cell_volumes@r['velocity'][:,0]/m.cell_volumes.sum()),'tau',tau,'cf',2*tau/bulk**2,'cf_error',abs(2*tau/bulk**2-2*.0637309**2)/(2*.0637309**2),'ledgers',r['scalar_absolute_ledger'],'iterations',len(r['rans_history']),flush=True)
 np.savez_compressed(f'/tmp/fvm-developed-sst-n{n}.npz',velocity=r['velocity'],pressure=r['physical_pressure'],flux=r['flux'],k=r['k'],omega=r['omega'],cell_centers=m.cell_centers,cell_volumes=m.cell_volumes)
