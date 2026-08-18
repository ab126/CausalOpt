from __future__ import annotations
import csv,json,traceback
from dataclasses import asdict,is_dataclass
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .config import ExperimentConfig
from ..simulation import simulate_static_sem,simulate_static_latent_sem,simulate_svar
from ..methods.static_latent import fit_static_latent
from ..methods.dynamic_latent import fit_dynamic_latent
from ..evaluation.dag_metrics import dag_metrics
from ..evaluation.dynamic_metrics import dynamic_metrics
from ..evaluation.latent_metrics import latent_metrics

def _arrays(obj):
    d=asdict(obj) if is_dataclass(obj) else vars(obj)
    return {k:v for k,v in d.items() if isinstance(v,np.ndarray)}
def _json(obj):
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,np.ndarray): return obj.tolist()
    raise TypeError(type(obj).__name__)

def run_experiment(config:ExperimentConfig,results_root="results",run_id=None):
    run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder=Path(results_root)/f"case_{config.case}"/run_id; folder.mkdir(parents=True,exist_ok=True)
    (folder/"config.json").write_text(json.dumps(config.to_dict(),indent=2))
    try:
        if config.case==1:
            data=simulate_static_sem(config.n,config.d,config.s0,config.seed,**config.simulation)
            est=fit_static_latent(data.X,k=0,**config.method); metrics=dag_metrics(data.W_true,est.W0,X=data.X)
        elif config.case==2:
            data=simulate_static_latent_sem(config.n,config.d,config.s0,config.k,config.seed,**config.simulation)
            est=fit_static_latent(data.X,k=config.k,**config.method); metrics=dag_metrics(data.W_true,est.W0,X=data.X)
            metrics.update(latent_metrics(data.C_true,est.C,data.L_true,est.L))
        else:
            data=simulate_svar(config.n,config.d,config.p,config.seed,s0=config.s0,k=config.k,**config.simulation)
            est=fit_dynamic_latent(data.X,config.p,k=config.k,**config.method)
            metrics=dynamic_metrics(data.W0_true,data.W_lags_true,est.W0,est.W_lags,data.X)
            if config.k: metrics["latent"]=latent_metrics(data.C_true,est.C,data.L_true,est.L)
        metrics["runtime"]=est.runtime_seconds; metrics["status"]="ok"
        np.savez_compressed(folder/"truth.npz",**_arrays(data)); np.savez_compressed(folder/"estimate.npz",**_arrays(est))
        (folder/"metrics.json").write_text(json.dumps(metrics,indent=2,default=_json))
    except Exception as exc:
        metrics={"status":"failed","exception":repr(exc),"traceback":traceback.format_exc()}
        (folder/"metrics.json").write_text(json.dumps(metrics,indent=2))
    summary=Path(results_root)/"summary.csv"; flat={"run_id":run_id,"case":config.case,"seed":config.seed,"status":metrics["status"],"path":str(folder)}
    exists=summary.exists()
    with summary.open("a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=flat); writer.writeheader() if not exists else None; writer.writerow(flat)
    return folder,metrics
