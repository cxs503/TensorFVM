"""Smoothed continuation Newton solve, accepted only at zero smoothing."""
import copy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .native_scalar_mmp3d import ScalarMMPConvergenceError


def solve_mmp_newton(scalar,volume_flux,*,initial_state=None,**kwargs):
    """Try an explicit state on the original equations, then fall back safely.

    A warm attempt must pass exactly the same residual, full increment and
    ledger gates as continuation. Its failure cannot change the supplied state.
    Invalid input or a malformed initial state is rejected, never hidden by a
    fallback. The existing cold continuation remains the default.
    """
    if initial_state is None:
        return _solve_mmp_newton(scalar,volume_flux,**kwargs)
    try:
        result=_solve_mmp_newton(scalar,volume_flux,initial_state=initial_state,**kwargs)
        result['warm_start']=dict(attempted=True,accepted=True,fallback_to_continuation=False)
        return result
    except RuntimeError as exc:
        # An unsmoothed tangent can be singular at the proposed state even
        # though the smoothed continuation path is well posed. Treat numerical
        # factorization/increment failures like a rejected nonlinear attempt.
        # Input ValueErrors continue to propagate without a fallback.
        diagnostic=getattr(exc,'scalar_result',{})
        attempt=dict(attempted=True,accepted=False,fallback_to_continuation=True,
            failure=diagnostic.get('failure') or f'{type(exc).__name__}: {exc}',
            history=diagnostic.get('nonlinear_history',[]))
        try:
            result=_solve_mmp_newton(scalar,volume_flux,**kwargs)
        except ScalarMMPConvergenceError as fallback:
            fallback.scalar_result['warm_start']=attempt
            raise
        result['warm_start']=attempt
        return result


def _solve_mmp_newton(scalar,volume_flux,*,initial_state=None,**kwargs):
    linear=copy.copy(scalar);linear.mmp=None
    initial=linear.solve(volume_flux,**kwargs)  # validates data, independent initial guess
    state=np.r_[initial['cell_values'],initial['face_values']].astype(np.longdouble)
    N,F=scalar.N,scalar.F;v=scalar.mesh.cell_volumes;free=scalar.free
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
    prescribed=scalar._prescribed(np.broadcast_to(np.asarray(kwargs['boundary_values'],float),(F,))[:,None])
    if initial_state is not None:
        candidate=np.asarray(initial_state)
        if candidate.shape!=(N+F,) or candidate.dtype.kind!='f' or not np.isfinite(candidate).all():
            raise ValueError('finite floating scalar initial state with shape (cells + faces,) required')
        fixed=np.setdiff1d(np.arange(N+F),free)
        if not np.array_equal(candidate[fixed],prescribed[fixed]):
            raise ValueError('scalar initial state must match prescribed face and cell values')
        state=candidate.astype(np.longdouble,copy=True)
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
            residual=audit['free_residual'].copy();cells=scalar.free_cells
            residual[:len(cells)]+=v[cells]*invdt*(state[cells]-old[cells])
            scale=audit['residual_scale']+np.linalg.norm(v*invdt*old)
            error=float(np.linalg.norm(residual)/max(scale,1e-30))
            r.update(info,nonlinear_history=history,current_nonlinear_residual=error,
                     scalar_diffusion_correction='nonlinear_mmp',nonlinear_solver='warm_start_newton' if initial_state is not None else 'continuation_newton',
                     continuation_state_scale=state_scale,final_smoothing=float(smoothing),ledger_refinements=refinement,
                     solution_precision_bits=int(np.finfo(state.dtype).nmant+1))
            ledger_passed=scalar.ledger_acceptable(r)
            if failure is None and error<1e-12 and ledger_passed and history[-1]['change']<1e-12:
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
                candidate=state.astype(np.longdouble);candidate[free]+=np.longdouble(alpha)*step.astype(np.longdouble)
                candidate_result,_,candidate_residual=evaluate(candidate)
                candidate_norm=float(np.linalg.norm(candidate_residual[free]))
                candidate_ledger=scalar.ledger_acceptable(candidate_result)
                if candidate_norm<norm:candidates.append((candidate_norm,alpha,candidate,candidate_ledger))
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

    def polish_fixed_point():
        """Unrelaxed original frozen-operator steps, with precise residuals.

        Near nonsmooth switches a Newton tangent can amplify roundoff even
        when the original equations are equilibrated. Certify two consecutive
        small ORIGINAL Picard updates, never a backtracked/clipped increment.
        The final original equation and all ledger gates still run in finish.
        """
        nonlocal state,linear_residual
        stable=0
        for iteration in range(8):
            frozen=copy.copy(linear);frozen.diffusion,_=scalar.mmp.operator(state)
            jac=(frozen.diffusion.full_matrix+extra)[free][:,free].tocsc();factor=splu(jac)
            candidate=state.astype(np.longdouble);worst_linear=0.
            for inner in range(3):
                r=frozen._solution_result(candidate,q,source,sink,old,invdt,kwargs['inflow_values'],linear_residual)
                precise=np.r_[r['cell_equation_extended'],r['face_diffusion_reaction_extended']]
                rhs=-np.asarray(precise[free],float);step=factor.solve(rhs)
                true_linear=float(np.linalg.norm(jac@step-rhs)/max(np.linalg.norm(rhs),1e-30))
                if not np.isfinite(step).all() or true_linear>1e-9:
                    return finish('fixed-point polish linear true residual failed')
                worst_linear=max(worst_linear,true_linear);candidate[free]+=step
            change=float(np.linalg.norm(candidate[free]-state[free])/max(np.linalg.norm(candidate[free]),state_scale))
            exact_residual=scalar.mmp.nonlinear_reaction(candidate)+extra@candidate-forcing
            error=float(np.linalg.norm(exact_residual[free])/fixed_scale)
            history.append(dict(iteration=len(history)+1,smoothing=0.,fixed_point_polish=True,
                change=change,current_nonlinear_residual=error,accepted=False,
                unrelaxed=True,linear_refinements=3,true_linear_relative_residual=worst_linear))
            state=candidate;linear_residual=worst_linear
            stable=stable+1 if change<1e-12 else 0
            if stable>=2:return finish()
        return finish('original fixed-point polish change failed')

    # Start with a broadly smoothed graph law. Jumping directly to 1e-3 can
    # strand the initial uncorrected diffusion guess near a singular tangent.
    # These are only continuation equations; acceptance still requires zero
    # smoothing and the unchanged original residual/update/flux gates.
    schedule=(0.,) if initial_state is not None else (1.,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,0.)
    for smoothing in schedule:
        for iteration in range(80):
            matrix,tangent=scalar.mmp.nonlinear_system(state,smoothing=smoothing,state_scale=state_scale,jacobian=True)
            residual=scalar.mmp.nonlinear_reaction(state,smoothing=smoothing,state_scale=state_scale)+extra@state-forcing
            rhs=-np.asarray(residual[free],float)
            jac=(tangent+extra)[free][:,free].tocsc();step=splu(jac).solve(rhs) if len(free) else np.empty(0)
            linear_residual=float(np.linalg.norm(jac@step-rhs)/max(np.linalg.norm(rhs),1e-30))
            if not np.isfinite(step).all():raise RuntimeError('nonfinite Newton increment rejected')
            merit=float(np.linalg.norm(rhs)/fixed_scale)
            change=float(np.linalg.norm(step)/max(np.linalg.norm(state[free]),state_scale))
            row=dict(iteration=len(history)+1,smoothing=smoothing,stage_iteration=iteration+1,
                smoothed_relative_residual=merit,current_nonlinear_residual=float(np.linalg.norm((scalar.mmp.nonlinear_reaction(state)+extra@state-forcing)[free])/fixed_scale),
                change=change,true_linear_relative_residual=linear_residual,accepted=False)
            history.append(row)
            if linear_residual>1e-9:return finish('Newton linear true residual failed')
            if merit<(1e-14 if smoothing==0 else 1e-12) and change<1e-12:
                row['stage_converged']=True
                break
            if smoothing==0 and merit<1e-14:
                return polish_fixed_point()
            for back in range(30):
                trial=state.copy();trial[free]+=2.**-back*step
                trial_residual=scalar.mmp.nonlinear_reaction(trial,smoothing=smoothing,state_scale=state_scale)+extra@trial-forcing
                after=float(np.linalg.norm(trial_residual[free])/fixed_scale)
                if after<=(1-1e-4*2.**-back)*merit:
                    state=trial;row.update(accepted=True,step_length=2.**-back,merit_after=after,backtracks=back)
                    break
            else:return finish('continuation Newton line search failed')
        else:return finish('continuation Newton stage iteration limit')
    return finish()
