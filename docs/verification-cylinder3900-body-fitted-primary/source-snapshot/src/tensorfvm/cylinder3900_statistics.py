"""Unsteady cylinder statistics with explicit insufficient-record gates."""
import numpy as np
from scipy.signal import welch

def statistics(force_history, diameter=1., velocity=1., discard_time=100., minimum_duration=500.):
    """discard_time is physical; minimum_duration and St use U/D scaling."""
    if not force_history:return {'eligible':False,'reason':'no samples','Cd_mean':None,'Cl_rms':None,'St':None}
    t=np.asarray([h['time'] for h in force_history]);keep=t>=discard_time
    y=np.asarray([[h['Cd'],h['Cl']]for h in force_history])[keep];tt=t[keep]
    if len(tt)<4:return {'eligible':False,'reason':'too few samples after startup discard','sample_count':len(tt),'Cd_mean':None,'Cl_rms':None,'St':None}
    dt=np.diff(tt);duration=float(tt[-1]-tt[0]);regular=np.max(abs(dt-dt.mean()))<=1e-6*dt.mean()
    if not regular:raise ValueError('Welch spectrum requires uniform time samples')
    cd=float(y[:,0].mean());lift=y[:,1]-y[:,1].mean()
    frequency,power=welch(lift,fs=1/dt.mean(),nperseg=min(len(tt),4096),detrend='linear')
    pos=frequency>0;peak=int(np.argmax(power[pos]));f=float(frequency[pos][peak]);st=f*diameter/velocity
    cycles=f*duration
    # A high-frequency startup peak is not a vortex-shedding measurement.
    spectrum_eligible=duration*velocity/diameter>=minimum_duration and cycles>=20
    chunks=np.array_split(y[:,0],10);block=np.asarray([x.mean()for x in chunks if len(x)])
    ci=1.96*float(block.std(ddof=1))/np.sqrt(len(block)) if len(block)>1 else None
    return dict(eligible=bool(spectrum_eligible),sample_count=len(tt),discard_time=discard_time,duration=duration,
                nondimensional_duration=duration*velocity/diameter,Cd_mean=cd,Cl_rms=float(np.sqrt((lift**2).mean())),
                St=st if spectrum_eligible else None,diagnostic_spectral_peak=st,
                frequency_resolution=float((frequency[1]-frequency[0])*diameter/velocity),estimated_peak_cycles=float(cycles),
                Cd_block_95_halfwidth=ci,block_ci_assumes_adequate_block_length=True,
                reason='statistics duration gate passed; grid/time independence and reference errors still required' if spectrum_eligible else 'short record: spectral peak is not a qualified shedding St')
