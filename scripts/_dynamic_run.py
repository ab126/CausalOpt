"""Shared, sequential Case 3/4 experiment runner used by the CLI scripts."""
from __future__ import annotations
import hashlib,json,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from causal_opt.experiments.config import ExperimentConfig
from causal_opt.simulation import simulate_svar
from causal_opt.methods.dynamic_latent import fit_dynamic_latent
from causal_opt.evaluation.dynamic_metrics import dynamic_metrics,support_metrics
from causal_opt.evaluation.latent_metrics import latent_metrics

def _json(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    return repr(value)

def _write(path,value):
    path.write_text(json.dumps(value,indent=2,default=_json))

def _save_estimate(path,result):
    arrays={name:getattr(result,name) for name in ("W0","W_raw","W_lags","W_lags_raw","Z","L","C") if isinstance(getattr(result,name,None),np.ndarray)}
    np.savez_compressed(path,**arrays)

def _lpcmci_adjacencies(graph,p,d):
    graph=np.asarray(graph); contemporaneous=np.zeros((d,d),bool); lags=np.zeros((p,d,d),bool)
    if graph.ndim==3:
        contemporaneous=graph[:,:,0]!=""
        np.fill_diagonal(contemporaneous,False)
        for tau in range(1,min(p+1,graph.shape[2])): lags[tau-1]=graph[:,:,tau]!=""
    return contemporaneous,lags

def run_case(config_path,case,method="ours",output_dir="results",seed=None,smoke=False,resume=False):
    cfg=ExperimentConfig.load(config_path)
    if seed is not None: cfg.seed=int(seed)
    if smoke: cfg.n=min(cfg.n,200); cfg.d=min(cfg.d,5); cfg.s0=min(cfg.s0,5)
    payload=cfg.to_dict()|{"requested_method":method,"smoke":smoke}
    run_id=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    folder=Path(output_dir)/f"case{case}"/run_id; folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/"status.json"
    if resume and status_path.exists() and json.loads(status_path.read_text()).get("complete"):
        return folder,json.loads(status_path.read_text())
    _write(folder/"config.json",payload)
    data=simulate_svar(cfg.n,cfg.d,cfg.p,cfg.seed,s0=cfg.s0,k=cfg.k,**cfg.simulation)
    truth={name:value for name,value in asdict(data).items() if isinstance(value,np.ndarray)}
    np.savez_compressed(folder/"truth.npz",**truth)
    methods=(["ours","dynotears"] if case==3 and method=="all" else
             ["ours","lpcmci","liegeois"] if case==4 and method=="all" else [method])
    statuses={}
    for name in methods:
        target=folder/name; target.mkdir(exist_ok=True)
        if resume and (target/"status.json").exists() and json.loads((target/"status.json").read_text()).get("status")=="ok":
            statuses[name]="ok"; continue
        try:
            if name=="ours":
                result=fit_dynamic_latent(data.X,cfg.p,k=cfg.k,**cfg.method)
                metrics=dynamic_metrics(data.W0_true,data.W_lags_true,result.W0,result.W_lags,data.X,w0_threshold=0,wlag_threshold=0)
                metrics.update(h_raw=result.diagnostics["h_raw"],h_thresholded=result.diagnostics["h_thresholded"],runtime=result.runtime_seconds)
                if cfg.k:
                    metrics["latent"]=latent_metrics(data.C_true,result.C,data.L_true,result.L)
                    metrics["latent"].update(C_fro_norm=result.diagnostics["C_fro_norm"],factor_balance_ratio=result.diagnostics["factor_balance_ratio"],singular_values_true=np.linalg.svd(data.C_true,compute_uv=False),singular_values_est=np.linalg.svd(result.C,compute_uv=False))
                _save_estimate(target/"estimate.npz",result); _write(target/"history.json",result.diagnostics["history"]); _write(target/"diagnostics.json",result.diagnostics)
            elif name=="dynotears":
                from causal_opt.baselines.dynotears import fit_dynotears
                result=fit_dynotears(data.X,p=cfg.p,**cfg.baseline)
                metrics=dynamic_metrics(data.W0_true,data.W_lags_true,result.W0,result.W_lags,data.X,w0_threshold=0,wlag_threshold=0); metrics["runtime"]=result.runtime_seconds
                _save_estimate(target/"estimate.npz",result); (target/"native_result.txt").write_text(repr(result.native_result))
            elif name=="lpcmci":
                from causal_opt.baselines.lpcmci import fit_lpcmci
                result=fit_lpcmci(data.X,tau_max=cfg.p,**cfg.baseline)
                c,l=_lpcmci_adjacencies(result.graph,cfg.p,cfg.d)
                cm=support_metrics(data.W0_true,c,0); lm=support_metrics(data.W_lags_true,l,0)
                legitimate=("precision","recall","f1","fdr","fpr","nnz")
                metrics={"contemporaneous_adjacency":{q:cm[q] for q in legitimate},
                         "lagged_adjacency":{q:lm[q] for q in legitimate},
                         "metric_note":"Adjacency-only diagnostics; no DAG SHD or weight comparison is applied to LPCMCI's DPAG.",
                         "endpoint_summary":result.diagnostics["endpoint_symbol_counts"],"runtime":result.runtime_seconds}
                np.savez_compressed(target/"native_result.npz",**{k:v for k,v in result.native_result.items() if isinstance(v,np.ndarray)}); _write(target/"native_result.json",result.native_result)
            elif name=="liegeois":
                options=cfg.baseline.copy(); options.pop("pc_alpha",None)
                required={"matlab_entrypoint","implementation_dir"}
                if not required.issubset(options): raise RuntimeError("Liégeois unavailable: configure matlab_entrypoint and implementation_dir")
                from causal_opt.baselines.liegeois import fit_liegeois
                result=fit_liegeois(data.X,**options); metrics={"runtime":result.runtime_seconds,"native_only":True}; _save_estimate(target/"estimate.npz",result)
            else: raise ValueError(f"unsupported method {name}")
            _write(target/"metrics.json",metrics); _write(target/"status.json",{"status":"ok"}); statuses[name]="ok"
        except Exception as exc:
            failure={"status":"unavailable" if isinstance(exc,(ImportError,RuntimeError)) else "failed","exception":repr(exc),"traceback":traceback.format_exc()}
            _write(target/"status.json",failure); statuses[name]=failure["status"]
    status={"complete":True,"methods":statuses,"run_id":run_id}; _write(status_path,status)
    return folder,status
