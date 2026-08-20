from dataclasses import replace
import numpy as np
import pytest

from causal_opt.fmri import ConnROIData
from causal_opt.fmri_inference import (compute_bootstrap_edge_statistics,
    compute_edge_selection_stability,edge_results_dataframe,fdr_correct_edge_tests,
    paired_subject_bootstrap_case2,paired_subject_draws,pool_subject_draw)
from causal_opt.simulation.static_sem import EstimatorResult


def _state(lengths=(2,3,4),offset=0.,session="001"):
    ids=tuple(f"Subject{i+1:03d}" for i in range(len(lengths))); names=tuple(f"r{i}" for i in range(3))
    series={sid:np.column_stack((np.full(T,i+offset),np.arange(T),np.full(T,10+i))) for i,(sid,T) in enumerate(zip(ids,lengths))}
    return ConnROIData(series,names,np.arange(9).reshape(3,3),ids,session)


def _solver(X,fail=False,**kwargs):
    if fail: raise RuntimeError("mock failure")
    d=X.shape[1]; W=np.zeros((d,d)); W[0,1]=X[:,0].mean()/10
    return EstimatorResult(W0=W.copy(),W_raw=W.copy(),diagnostics={"optimizer_success":True,"h_raw":0.})


def test_draws_are_reproducible_paired_subject_ids():
    ids=("Subject001","Subject002","Subject003")
    assert np.array_equal(paired_subject_draws(ids,5,42),paired_subject_draws(ids,5,42))


def test_pool_draw_uses_complete_subjects_and_multiplicity():
    state=_state(); draw=np.array(["Subject002","Subject002","Subject001"]); pooled=pool_subject_draw(state.timeseries,draw)
    assert len(pooled)==3+3+2
    assert np.array_equal(pooled[:3],state.timeseries["Subject002"])
    assert np.array_equal(pooled[3:6],state.timeseries["Subject002"])


def test_paired_bootstrap_supports_different_state_lengths_and_raw_W(tmp_path):
    control=_state(); sdv=_state((3,4,5),offset=1.,session="002")
    result=paired_subject_bootstrap_case2(control,sdv,tmp_path,reps=4,seed=7,solver=_solver,model_kwargs={})
    assert result["successful"]==4 and result["failed"]==0
    assert result["W_control"].shape==(4,3,3) and result["W_sdv"].shape==(4,3,3)
    for record in result["records"]: assert np.array_equal(record["draw"],result["draws"][record["index"]])


def test_failed_replicates_are_counted_and_excluded(tmp_path):
    result=paired_subject_bootstrap_case2(_state(),_state(session="002"),tmp_path,reps=3,solver=_solver,model_kwargs={"fail":True})
    assert result["successful"]==0 and result["failed"]==3 and result["delta_W"].shape==(0,3,3)


def test_statistics_fdr_diagonal_and_stability_bounds():
    observed=np.array([[0,.3,-.2],[.1,0,.4],[.2,-.1,0.]])
    boot=np.stack([observed+s for s in (-.05,0,.05,.02)])
    stats=compute_bootstrap_edge_statistics(observed,boot)
    assert stats["ci_low"].shape==(3,3) and stats["ci_high"].shape==(3,3)
    assert np.isnan(np.diag(stats["p_boot"])).all() and np.isnan(np.diag(stats["q_fdr"])).all()
    assert np.nanmin(stats["p_boot"])>=0 and np.nanmax(stats["p_boot"])<=1
    stability=compute_edge_selection_stability(boot,boot+.1,[.1,.3])
    for values in stability["thresholds"].values():
        for matrix in values.values(): assert matrix.min()>=0 and matrix.max()<=1
    frame=edge_results_dataframe(["a","b","c"],observed,observed*2,stats,stability,.3)
    assert len(frame)==6


def test_bh_uses_all_342_offdiagonal_hypotheses():
    p=np.full((19,19),.5); np.fill_diagonal(p,np.nan); p[0,1]=.001
    q=fdr_correct_edge_tests(p)
    assert np.isnan(np.diag(q)).all(); assert q[0,1]==pytest.approx(.342)


def test_checkpoint_resume_matches_continuous_run(tmp_path):
    control=_state(); sdv=_state(session="002")
    full=paired_subject_bootstrap_case2(control,sdv,tmp_path/"full",reps=5,seed=9,solver=_solver)
    interrupted=paired_subject_bootstrap_case2(control,sdv,tmp_path/"resume",reps=5,seed=9,solver=_solver)
    (tmp_path/"resume"/"rep_000003.npz").unlink(); (tmp_path/"resume"/"rep_000004.npz").unlink()
    resumed=paired_subject_bootstrap_case2(control,sdv,tmp_path/"resume",reps=5,seed=9,resume=True,solver=_solver)
    assert np.array_equal(full["draws"],resumed["draws"])
    assert np.allclose(full["delta_W"],resumed["delta_W"])
