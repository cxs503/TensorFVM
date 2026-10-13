"""Smoothed continuation Newton solve, accepted only at zero smoothing."""
import copy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .native_scalar_mmp3d import ScalarMMPConvergenceError


def solve_mmp_newton(scalar,volume_flux,**kwargs):
    linear=copy.copy(scalar);linear.mmp=None
    initial=linear.solve(volume_flux,**kwargs)  # validates data, independent initial guess
    state=np.r_[initial['cell_values'],initial['face_values']]
    N,F=scalar.N,scalar.F;v=scalar.mesh.cell_volumes;free=scalar.diffusion.free
    q=scalar._array(volume_flux,F,'volume flux')
    source=scalar._array(kwargs['source'],N,'source');sink=scalar._array(kwargs['sink'],N,'sink')
    invdt=0. if kwargs['dt'] is None else 1./kwargs['dt']
    old=np.zeros(N) if not invdt else scalar._array(kwargs['old_values'],N,'old scalar')
    incoming=scalar.boundary & (q<0);ids=np.flatnonzero(~incoming)
    upstream=np.where(q>=0,scalar.mesh.owner,scalar.mesh.neighbor)
    transport=sparse.coo_matrix((q[ids],(ids,upstream[ids])),shape=(F,N)).tocsr()
    adv_source=np.zeros(F)
    if incoming.any():adv_source[incoming]=q[incoming]*np.broadcast_to(np.asarray(kwargs['inflow_values'],float),(F,))[incoming]
    cells=scalar.D@transport+sparse.diags(v*(sink+invdt))
    extra=sparse.block_diag((cells,sparse.csr_matrix((F,F))),format='csr')
    gn=np.broadcast_to(np.asarray(kwargs['boundary_diffusive_flux'],float),(F,))
    forcing=scalar.diffusion._traction_load(gn[:,None])
    forcing[:N]+=v*(source+old*invdt)-scalar.D@adv_source
    prescribed=scalar.diffusion._prescribed(np.broadcast_to(np.asarray(kwargs['boundary_values'],float),(F,))[:,None])
    fixed_scale=max(np.linalg.norm((forcing-(scalar.diffusion.full_matrix+extra)@prescribed)[free]),1e-30)
    state_scale=max(float(np.max(np.abs(state))),float(np.max(np.abs(old))),1e-30)
    history=[];linear_residual=initial['linear_relative_residual']
    residual_kwargs={k:value for k,value in kwargs.items() if k not in ('dt','old_values')}

    def finish(failure=None):
        nonlocal state,linear_residual
        def evaluate(candidate):
            linear.diffusion,info=scalar.mmp.operator(candidate)
            result=linear._solution_result(candidate,q,source,sink,old,invdt,kwargs['inflow_values'],linear_residual)
            precise=np.r_[result['cell_equation_extended'],result['face_diffusion_reaction_extended']]
            return result,info,precise
        for refinement in range(9):
            r,info,precise=evaluate(state)
            audit=linear.steady_residual(state[:N],state[N:],q,**residual_kwargs)
            residual=audit['free_residual'].copy();residual[:N]+=v*invdt*(state[:N]-old)
            scale=audit['residual_scale']+np.linalg.norm(v*invdt*old)
            error=float(np.linalg.norm(residual)/max(scale,1e-30))
            r.update(info,nonlinear_history=history,current_nonlinear_residual=error,
                     scalar_diffusion_correction='nonlinear_mmp',nonlinear_solver='continuation_newton',
                     continuation_state_scale=state_scale,final_smoothing=float(smoothing),ledger_refinements=refinement)
            ledger=max(r['cell_balance_max'],r['interface_flux_defect_max'],abs(r['global_balance']),abs(r['work_balance']))
            if failure is None and error<1e-12 and ledger<1e-8 and history[-1]['change']<1e-12:
                return r
            if failure is not None or refinement==8:break
            # Mixed-precision residual refinement of the SAME zero-smoothing
            # equations. No value clipping or acceptance-threshold adjustment.
            _,tangent=scalar.mmp.nonlinear_system(state,jacobian=True)
            jac=(tangent+extra)[free][:,free].tocsc();rhs=-np.asarray(precise[free],float)
            step=splu(jac).solve(rhs)
            true_linear=float(np.linalg.norm(jac@step-rhs)/max(np.linalg.norm(rhs),1e-30))
            if not np.isfinite(step).all() or true_linear>1e-9:break
            norm=float(np.linalg.norm(precise[free]));candidates=[]
            for alpha in (1.,.5,.25,.125):
                candidate=state.copy();candidate[free]+=alpha*step
                candidate_result,_,candidate_residual=evaluate(candidate)
                candidate_norm=float(np.linalg.norm(candidate_residual[free]))
                candidate_ledger=max(candidate_result['cell_balance_max'],candidate_result['interface_flux_defect_max'],
                    abs(candidate_result['global_balance']),abs(candidate_result['work_balance']))
                if candidate_norm<norm:candidates.append((candidate_norm,alpha,candidate,candidate_ledger<1e-8))
            if not candidates:break
            after,alpha,candidate,_=min(candidates,key=lambda entry:(not entry[3],entry[0]))
            change=float(np.linalg.norm(candidate-state)/max(np.linalg.norm(candidate),state_scale))
            history.append(dict(iteration=len(history)+1,smoothing=0.,change=change,
                current_nonlinear_residual=error,accepted=False,refinement_accepted=True,
                extended_residual_before=norm,extended_residual_after=after,
                step_length=alpha,true_linear_relative_residual=true_linear))
            state=candidate;linear_residual=true_linear
        r.update(converged=False,failure=failure or 'original equation/ledger acceptance failed')
        raise ScalarMMPConvergenceError(r)

    for smoothing in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,0.):
        for iteration in range(80):
            matrix,tangent=scalar.mmp.nonlinear_system(state,smoothing=smoothing,state_scale=state_scale,jacobian=True)
            residual=(matrix+extra)@state-forcing;rhs=-residual[free]
            jac=(tangent+extra)[free][:,free].tocsc();step=splu(jac).solve(rhs)
            linear_residual=float(np.linalg.norm(jac@step-rhs)/max(np.linalg.norm(rhs),1e-30))
            if not np.isfinite(step).all():raise RuntimeError('nonfinite Newton increment rejected')
            merit=float(np.linalg.norm(rhs)/fixed_scale)
            change=float(np.linalg.norm(step)/max(np.linalg.norm(state[free]),state_scale))
            exact=scalar.mmp.nonlinear_system(state,smoothing=0.)
            row=dict(iteration=len(history)+1,smoothing=smoothing,stage_iteration=iteration+1,
                smoothed_relative_residual=merit,current_nonlinear_residual=float(np.linalg.norm(((exact+extra)@state-forcing)[free])/fixed_scale),
                change=change,true_linear_relative_residual=linear_residual,accepted=False)
            history.append(row)
            if linear_residual>1e-9:return finish('Newton linear true residual failed')
            if merit<(1e-14 if smoothing==0 else 1e-12) and change<1e-12:
                row['stage_converged']=True
                break
            for back in range(30):
                trial=state.copy();trial[free]+=2.**-back*step
                trial_matrix=scalar.mmp.nonlinear_system(trial,smoothing=smoothing,state_scale=state_scale)
                after=float(np.linalg.norm(((trial_matrix+extra)@trial-forcing)[free])/fixed_scale)
                if after<=(1-1e-4*2.**-back)*merit:
                    state=trial;row.update(accepted=True,step_length=2.**-back,merit_after=after,backtracks=back)
                    break
            else:return finish('continuation Newton line search failed')
        else:return finish('continuation Newton stage iteration limit')
    return finish()
