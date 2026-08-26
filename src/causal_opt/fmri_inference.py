"""Case 2 fMRI baselines and paired subject-bootstrap inference."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import nullcontext
from pathlib import Path
import json
import os
import warnings

import numpy as np
from tqdm.auto import tqdm

from .baselines.notears import fit_notears
from .methods.static_latent import fit_static_latent
from .simulation.static_sem import acyclicity


def fit_case2_baselines(X,lambda_w=.1,edge_threshold=.3,fci_alpha=.05,run_fci=True,notears_kwargs=None,fci_kwargs=None):
    """Fit the established official NOTEARS and optional FCI baselines."""
    X=np.asarray(X,float); centered=X-X.mean(0,keepdims=True)
    note=fit_notears(X,lambda1=lambda_w,loss_type="l2",w_threshold=0.,**(notears_kwargs or {}))
    raw=np.asarray(note.W0,float).copy(); thresholded=raw.copy(); thresholded[np.abs(thresholded)<edge_threshold]=0.
    residual=centered-centered@raw
    note.W_raw=raw; note.W0=thresholded
    note.diagnostics.update(h_raw=acyclicity(raw),h_thresholded=acyclicity(thresholded),edge_threshold=edge_threshold,
                            fit_loss=float(.5*np.sum(residual**2)/len(X)),residual_mse=float(np.mean(residual**2)),
                            directed_edges=int(np.count_nonzero(thresholded)),abs_weight_sum=float(np.abs(thresholded).sum()),W_fro=float(np.linalg.norm(thresholded)))
    fci=None
    if run_fci:
        from .baselines.fci import fit_fci
        fci=fit_fci(X,alpha=fci_alpha,**(fci_kwargs or {}))
        fci.diagnostics.update(pag_statistics(fci.graph),alpha=float(fci_alpha),settings=dict(fci_kwargs or {}))
    return {"notears":note,"fci":fci}


def pag_statistics(endpoints):
    """Summarize endpoint-coded PAG pairs without inventing weights."""
    E=np.asarray(endpoints,int); counts=dict(adjacencies=0,directed_orientations=0,bidirected_edges=0,partial_or_uncertain_edges=0)
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            a,b=E[i,j],E[j,i]
            if not (a or b): continue
            counts["adjacencies"]+=1
            if a==2 and b==2: counts["bidirected_edges"]+=1
            elif {a,b}=={1,2}: counts["directed_orientations"]+=1
            else: counts["partial_or_uncertain_edges"]+=1
    return counts


def pag_skeleton(endpoints):
    E=np.asarray(endpoints); return ((E!=0)|(E.T!=0)).astype(int)


def paired_subject_draws(subject_ids,reps,seed):
    ids=np.asarray(subject_ids,dtype=str); rng=np.random.default_rng(seed)
    return ids[rng.integers(0,len(ids),size=(int(reps),len(ids)))]


def pool_subject_draw(timeseries,draw):
    """Concatenate complete subject series, preserving draw order/multiplicity."""
    return np.vstack([np.asarray(timeseries[str(subject)]) for subject in draw])


def _converged(result,h_tol):
    d=result.diagnostics
    return bool(d.get("optimizer_success",False) and np.isfinite(d.get("h_raw",np.inf)) and d["h_raw"]<=h_tol and np.isfinite(result.W_raw).all())


def _fit_pair(index,draw,control_series,sdv_series,model_kwargs):
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[name]="1"
    try:
        Xc=pool_subject_draw(control_series,draw); Xs=pool_subject_draw(sdv_series,draw)
        context=nullcontext()
        try:
            from threadpoolctl import threadpool_limits
            context=threadpool_limits(limits=1)
        except ImportError: pass
        with context:
            rc=fit_static_latent(Xc,**model_kwargs); rs=fit_static_latent(Xs,**model_kwargs)
        h_tol=float(model_kwargs.get("h_tol",1e-8)); ok_c=_converged(rc,h_tol); ok_s=_converged(rs,h_tol)
        return index,draw,ok_c,ok_s,rc.W_raw if ok_c and ok_s else None,rs.W_raw if ok_c and ok_s else None,{"control":rc.diagnostics,"sdv":rs.diagnostics},None
    except Exception as exc:
        return index,draw,False,False,None,None,{},repr(exc)


def _save_replicate(folder,record):
    index,draw,ok_c,ok_s,Wc,Ws,diagnostics,error=record; final=folder/f"rep_{index:06d}.npz"; temp=folder/f"rep_{index:06d}.tmp.npz"
    valid=bool(ok_c and ok_s)
    np.savez_compressed(temp,draw=np.asarray(draw),converged_control=ok_c,converged_sdv=ok_s,valid=valid,
                        W_control=np.asarray(Wc) if valid else np.empty((0,0)),W_sdv=np.asarray(Ws) if valid else np.empty((0,0)),
                        delta_W=np.asarray(Ws)-np.asarray(Wc) if valid else np.empty((0,0)),diagnostics_json=json.dumps(diagnostics,default=lambda x:np.asarray(x).tolist()),error=error or "")
    temp.replace(final)


def paired_subject_bootstrap_case2(control,sdv,output_dir,reps=1000,seed=42,jobs=1,resume=False,checkpoint_every=10,model_kwargs=None,solver=None):
    """Run a deterministic, resumable paired subject bootstrap.

    Each completed replicate is an atomic checkpoint; ``checkpoint_every`` controls
    progress reporting frequency. A custom solver is supported for serial unit tests.
    """
    if control.subject_ids!=sdv.subject_ids: raise ValueError("paired bootstrap requires identical subject IDs")
    folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True); kwargs=dict(model_kwargs or {})
    draws=paired_subject_draws(control.subject_ids,reps,seed)
    meta={"reps":int(reps),"seed":int(seed),"subject_ids":list(control.subject_ids),"model_kwargs":kwargs,"roi_names":list(control.roi_names)}
    meta_path=folder/"bootstrap_meta.json"
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old!=meta: raise ValueError("existing bootstrap metadata does not match requested run")
    else: meta_path.write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    np.save(folder/"subject_draws.npy",draws)
    completed={int(p.stem.split("_")[1]) for p in folder.glob("rep_*.npz")} if resume else set()
    if completed and not resume: raise FileExistsError("bootstrap replicate files exist; pass resume=True or use a new directory")
    todo=[i for i in range(reps) if i not in completed]
    if solver is not None and jobs!=1: raise ValueError("custom solvers require jobs=1")
    def serial_record(i):
        if solver is None: return _fit_pair(i,draws[i],control.timeseries,sdv.timeseries,kwargs)
        try:
            rc=solver(pool_subject_draw(control.timeseries,draws[i]),**kwargs); rs=solver(pool_subject_draw(sdv.timeseries,draws[i]),**kwargs); h=float(kwargs.get("h_tol",1e-8)); oc=_converged(rc,h); os_=_converged(rs,h)
            return i,draws[i],oc,os_,rc.W_raw if oc and os_ else None,rs.W_raw if oc and os_ else None,{"control":rc.diagnostics,"sdv":rs.diagnostics},None
        except Exception as exc: return i,draws[i],False,False,None,None,{},repr(exc)
    done = len(completed)

    with tqdm(
        total=reps,
        initial=done,
        desc="Case 2 paired bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ) as pbar:

        if jobs == 1:
            for i in todo:
                record = serial_record(i)
                _save_replicate(folder, record)
                done += 1
                pbar.update(1)

        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = [
                    pool.submit(
                        _fit_pair,
                        i,
                        draws[i],
                        control.timeseries,
                        sdv.timeseries,
                        kwargs,
                    )
                    for i in todo
                ]

                for future in as_completed(futures):
                    _save_replicate(folder, future.result())
                    done += 1
                    pbar.update(1)
                    
    result=load_bootstrap_results(folder)
    if result["failure_rate"]>.2: warnings.warn(f"high bootstrap failure rate: {result['failure_rate']:.1%}",RuntimeWarning)
    return result


def load_bootstrap_results(output_dir):
    folder=Path(output_dir); meta=json.loads((folder/"bootstrap_meta.json").read_text()); records=[]
    for path in sorted(folder.glob("rep_*.npz")):
        with np.load(path,allow_pickle=False) as item:
            records.append({"index":int(path.stem.split("_")[1]),"draw":item["draw"],"valid":bool(item["valid"]),"converged_control":bool(item["converged_control"]),"converged_sdv":bool(item["converged_sdv"]),"W_control":item["W_control"],"W_sdv":item["W_sdv"],"delta_W":item["delta_W"],"error":str(item["error"])})
    valid=[r for r in records if r["valid"]]; d=len(meta["roi_names"]); empty=np.empty((0,d,d))
    requested=int(meta["reps"]); successful=len(valid); completed=len(records)
    return {"meta":meta,"records":records,"draws":np.load(folder/"subject_draws.npy",allow_pickle=False),"W_control":np.stack([r["W_control"] for r in valid]) if valid else empty,"W_sdv":np.stack([r["W_sdv"] for r in valid]) if valid else empty,"delta_W":np.stack([r["delta_W"] for r in valid]) if valid else empty,"requested":requested,"completed":completed,"successful":successful,"failed":completed-successful,"failure_rate":(completed-successful)/completed if completed else 0.}


def fdr_correct_edge_tests(p_values):
    """Benjamini-Hochberg adjustment over off-diagonal directed hypotheses."""
    p=np.asarray(p_values,float); q=np.full(p.shape,np.nan); mask=~np.eye(p.shape[0],dtype=bool)&np.isfinite(p)
    vals=p[mask]; order=np.argsort(vals); ranked=vals[order]; adjusted=ranked*len(ranked)/np.arange(1,len(ranked)+1); adjusted=np.minimum.accumulate(adjusted[::-1])[::-1]; restored=np.empty_like(adjusted); restored[order]=np.minimum(adjusted,1.); q[mask]=restored; return q


def compute_bootstrap_edge_statistics(delta_observed,delta_boot,alpha=.05):
    observed=np.asarray(delta_observed,float); boot=np.asarray(delta_boot,float)
    if boot.ndim!=3 or boot.shape[1:]!=observed.shape or len(boot)==0: raise ValueError("delta_boot must contain at least one matrix matching delta_observed")
    low,high=np.percentile(boot,[100*alpha/2,100*(1-alpha/2)],axis=0); deviation=boot-observed
    p=(1+np.sum(np.abs(deviation)>=np.abs(observed),axis=0))/(len(boot)+1); np.fill_diagonal(p,np.nan)
    q=fdr_correct_edge_tests(p); ci=(low>0)|(high<0); np.fill_diagonal(ci,False)
    return {"mean":boot.mean(0),"median":np.median(boot,axis=0),"ci_low":low,"ci_high":high,"p_boot":p,"q_fdr":q,"ci_excludes_zero":ci,"fdr_significant":q<alpha}


def compute_edge_selection_stability(W_control,W_sdv,thresholds=(.05,.1,.2,.3)):
    Wc=np.asarray(W_control,float); Ws=np.asarray(W_sdv,float)
    if Wc.shape!=Ws.shape or Wc.ndim!=3 or not len(Wc): raise ValueError("bootstrap W arrays must be nonempty and have equal shapes")
    out={"control_positive":(Wc>0).mean(0),"control_negative":(Wc<0).mean(0),"sdv_positive":(Ws>0).mean(0),"sdv_negative":(Ws<0).mean(0)}
    out["thresholds"]={float(t):{"control":(np.abs(Wc)>t).mean(0),"sdv":(np.abs(Ws)>t).mean(0)} for t in thresholds}; return out


def edge_results_dataframe(names,W_control,W_sdv,statistics,stability,threshold=.3):
    import pandas as pd
    names=list(names); rows=[]; sel=stability["thresholds"][float(threshold)]
    for i in range(len(names)):
        for j in range(len(names)):
            if i==j: continue
            rows.append({"source_index":i,"source_name":names[i],"target_index":j,"target_name":names[j],"W_control_observed":W_control[i,j],"W_sdv_observed":W_sdv[i,j],"delta_W_observed":W_sdv[i,j]-W_control[i,j],"bootstrap_mean_delta":statistics["mean"][i,j],"bootstrap_median_delta":statistics["median"][i,j],"ci_2.5":statistics["ci_low"][i,j],"ci_97.5":statistics["ci_high"][i,j],"p_boot":statistics["p_boot"][i,j],"q_fdr":statistics["q_fdr"][i,j],"ci_excludes_zero":statistics["ci_excludes_zero"][i,j],"fdr_significant":statistics["fdr_significant"][i,j],"control_selection_probability":sel["control"][i,j],"sdv_selection_probability":sel["sdv"][i,j],"control_positive_probability":stability["control_positive"][i,j],"sdv_positive_probability":stability["sdv_positive"][i,j]})
    frame=pd.DataFrame(rows); frame["abs_delta"]=frame.delta_W_observed.abs(); return frame.sort_values(["q_fdr","abs_delta"],ascending=[True,False]).drop(columns="abs_delta").reset_index(drop=True)


def node_reorganization(delta,names):
    import pandas as pd
    A=np.abs(np.asarray(delta,float)); incoming=A.sum(0); outgoing=A.sum(1)
    return pd.DataFrame({"roi":names,"incoming_change":incoming,"outgoing_change":outgoing,"total_change":incoming+outgoing}).sort_values("total_change",ascending=False).reset_index(drop=True)


def compare_case2_structures(proposed,notears,names,threshold=.3,fci_endpoints=None):
    """Return directed overlap and optional FCI skeleton support summaries."""
    P=np.abs(proposed)>=threshold; N=np.abs(notears)>=threshold; np.fill_diagonal(P,False); np.fill_diagonal(N,False)
    def labels(mask): return [f"{names[i]} -> {names[j]}" for i,j in np.argwhere(mask)]
    out={"both_proposed_and_notears":labels(P&N),"proposed_only":labels(P&~N),"notears_only":labels(N&~P)}
    if fci_endpoints is not None:
        skeleton=pag_skeleton(fci_endpoints).astype(bool); out["proposed_edges_with_fci_adjacency"]=labels(P&skeleton); out["proposed_edges_without_fci_adjacency"]=labels(P&~skeleton)
    return out
