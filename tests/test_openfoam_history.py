import numpy as np
from tensorfvm.openfoam_history import convergence_audit,function_history,residual_history


def history(tmp_path,first_pressure=1e-7,cancel_components=False):
    for function,file in [('forceCoeffs','forceCoeffs.dat'),('forces','forces.dat')]:
        folder=tmp_path/'postProcessing'/function/'0';folder.mkdir(parents=True)
        values=np.zeros((220,6 if function=='forceCoeffs' else 13));values[:,0]=np.arange(1,221)
        if function=='forceCoeffs':values[:,2]=.1
        else:
            values[:,1]=.02;values[:,4]=.08
            if cancel_components:
                change=np.linspace(0,.001,len(values));values[:,1]+=change;values[:,4]-=change
        np.savetxt(folder/file,values)
    logs=[]
    for time in range(1,221):
        log=f'Time = {time}s\n'
        for field in ['Ux','Uy','Uz','p','k','omega']:
            value=first_pressure if field=='p' else 1e-7
            log+=f'Solving for {field}, Initial residual = {value}, Final residual = 1e-10\n'
        log+='Solving for p, Initial residual = 1e-12, Final residual = 1e-13\n'
        log+='time step continuity errors : sum local = 1e-10, global = 1e-12\n'
        logs.append(log)
    (tmp_path/'log.foamRun').write_text(''.join(logs)+'End\n')


def test_small_last_pressure_correction_does_not_hide_outer_residual(tmp_path):
    history(tmp_path,first_pressure=.01)
    assert residual_history(tmp_path)[220]['p']==.01
    audit=convergence_audit(tmp_path)
    assert audit['force_stable'] and not audit['numerically_converged']
    assert not audit['physical_benchmark_passed']


def test_constant_total_drag_does_not_hide_drifting_components(tmp_path):
    history(tmp_path,cancel_components=True)
    audit=convergence_audit(tmp_path)
    assert audit['residuals_passed'] and not audit['force_stable']
    assert not audit['numerically_converged']


def test_restart_deduplicates_times_and_requires_complete_residual_window(tmp_path):
    history(tmp_path)
    data,_=function_history(tmp_path,'forceCoeffs','forceCoeffs.dat')
    folder=tmp_path/'postProcessing/forceCoeffs/100';folder.mkdir()
    np.savetxt(folder/'forceCoeffs.dat',data[99:])
    merged,_=function_history(tmp_path,'forceCoeffs','forceCoeffs.dat')
    assert len(merged)==220
    assert convergence_audit(tmp_path)['numerically_converged']
    log=tmp_path/'log.foamRun';log.write_text(log.read_text().replace('Solving for k,','Solving for missing,'))
    assert not convergence_audit(tmp_path)['numerically_converged']
