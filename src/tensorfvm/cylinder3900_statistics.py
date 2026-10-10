"""Unsteady cylinder statistics with actual spectral resolution gates."""
import numpy as np
from scipy.signal import welch
from scipy.stats import t as student_t

def statistics(force_history, diameter=1., velocity=1., discard_time=100., minimum_duration=500.,
               reference_strouhal=.215, relative_frequency_resolution=.03):
    """Physical discard_time; minimum_duration and spectral bins scale by U/D.

    Reference St only plans the record resolution. It never sets the measured
    peak. No zero padding is used: nondimensional bin spacing is D/(U*Tsegment).
    A resolvable spectrum is still not a physically qualified benchmark.
    """
    if min(diameter,velocity,minimum_duration,reference_strouhal,relative_frequency_resolution)<=0:raise ValueError('positive statistics scales required')
    if not force_history:return {'eligible':False,'reason':'no samples','Cd_mean':None,'Cl_rms':None,'St':None}
    t=np.asarray([h['time'] for h in force_history]);keep=t>=discard_time
    y=np.asarray([[h['Cd'],h['Cl']]for h in force_history])[keep];tt=t[keep]
    if len(tt)<4:return {'eligible':False,'duration_eligible':False,'St_spectrum_eligible':False,'reason':'too few samples after startup discard','sample_count':len(tt),'Cd_mean':None,'Cl_rms':None,'St':None}
    dt=np.diff(tt);duration=float(tt[-1]-tt[0]);regular=np.max(abs(dt-dt.mean()))<=1e-6*dt.mean()
    if not regular:raise ValueError('Welch spectrum requires uniform time samples')
    dtmean=float(dt.mean());cd=float(y[:,0].mean());lift=y[:,1]-y[:,1].mean();lift_rms=float(np.sqrt((lift**2).mean()))
    # Target half of the permitted full-bin spacing. Longer records use at
    # least half the entire record, rather than a fixed 4096 sample ceiling.
    target_bin=reference_strouhal*relative_frequency_resolution
    required=int(np.ceil(2*diameter/(velocity*dtmean*target_bin)))
    nperseg=min(len(tt),max(required,len(tt)//2));noverlap=nperseg//2
    frequency,power=welch(lift,fs=1/dtmean,nperseg=nperseg,noverlap=noverlap,nfft=nperseg,detrend='linear')
    scale=diameter/velocity;frequency_resolution=float((frequency[1]-frequency[0])*scale)
    signal_present=lift_rms>1e-12*max(1.,float(np.max(np.abs(y[:,1])))) and bool(np.max(power[1:])>0)
    f=float(frequency[1:][np.argmax(power[1:])])if signal_present else None
    st=f*scale if f is not None else None;cycles=f*duration if f is not None else 0.
    duration_ok=duration/scale>=minimum_duration;resolution_ok=frequency_resolution<=target_bin
    spectrum_ok=bool(signal_present and resolution_ok and cycles>=20)
    eligible=bool(duration_ok and spectrum_ok)
    chunks=np.array_split(y[:,0],10);block=np.asarray([x.mean()for x in chunks if len(x)])
    ci=float(student_t.ppf(.975,len(block)-1))*float(block.std(ddof=1))/np.sqrt(len(block)) if len(block)>1 else None
    block_samples=[len(x)for x in chunks if len(x)];block_durations=[n*dtmean for n in block_samples]
    reason='record and spectral resolution gates passed; stationarity, grid/time independence and reference errors still required'if eligible else 'insufficient duration, actual frequency resolution, nonzero lift spectrum, or resolved cycle count'
    return dict(eligible=eligible,duration_eligible=bool(duration_ok),St_spectrum_eligible=spectrum_ok,St_accuracy_qualified=False,
                sample_count=len(tt),discard_time=discard_time,discard_time_nondimensional=discard_time/scale,duration=duration,
                nondimensional_duration=duration/scale,Cd_mean=cd,Cl_rms=lift_rms,St=st if eligible else None,
                diagnostic_spectral_peak=st,frequency_resolution=frequency_resolution,frequency_resolution_limit=target_bin,
                frequency_resolution_gate_passed=bool(resolution_ok),segment_samples=nperseg,segment_duration=nperseg*dtmean,
                segment_nondimensional_duration=nperseg*dtmean/scale,welch_segments=1+(len(tt)-nperseg)//(nperseg-noverlap),
                zero_padding_used=False,signal_present=bool(signal_present),estimated_peak_cycles=float(cycles),
                reference_strouhal_for_resolution=reference_strouhal,Cd_block_95_halfwidth=ci,block_CI_distribution='Student t',block_CI_degrees_of_freedom=len(block)-1,block_samples=block_samples,block_duration=block_durations,block_ci_assumes_adequate_block_length=True,reason=reason)
