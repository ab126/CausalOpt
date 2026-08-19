"""Official CausalNex DYNOTEARS through an isolated compatible Python."""
from __future__ import annotations
import json,shutil,subprocess,time,uuid
from pathlib import Path
import numpy as np
from ..simulation.static_sem import EstimatorResult

def _default_python():
    root=Path(__file__).resolve().parents[3]
    candidate=root/".venv-dynotears"/"Scripts"/"python.exe"
    return candidate if candidate.exists() else None

def _stack_lags(X,p):
    X=np.asarray(X,float)
    if X.ndim!=2 or p<1 or p>=len(X): raise ValueError("require 2-D X and 1 <= p < T")
    return X[p:],np.concatenate([X[p-tau:len(X)-tau] for tau in range(1,p+1)],axis=1)

def _edges_to_matrices(edges,d,p):
    W0=np.zeros((d,d)); Wlags=np.zeros((p,d,d))
    for edge in edges:
        source,destination,weight=edge["source"],edge["destination"],float(edge["weight"])
        si,sl=str(source).rsplit("_lag",1); di,dl=str(destination).rsplit("_lag",1)
        lag=int(sl)-int(dl)
        if int(dl)!=0: continue
        if lag==0: W0[int(si),int(di)]=weight
        elif 1<=lag<=p: Wlags[lag-1,int(si),int(di)]=weight
    return W0,Wlags

def fit_dynotears(X,p=1,backend="subprocess",python_executable=None,timeout=600,
                  preserve_temp=False,lambda_w=.1,lambda_a=.1,max_iter=100,
                  h_tol=1e-8,w_threshold=0.,**kwargs):
    if backend!="subprocess": raise ValueError("only backend='subprocess' is supported")
    X=np.asarray(X,float); _stack_lags(X,p)
    python_path=Path(python_executable) if python_executable else _default_python()
    if python_path is None or not python_path.exists(): raise ImportError("DYNOTEARS side environment unavailable; run scripts/setup_dynotears_env.ps1 or pass python_executable")
    root=Path(__file__).resolve().parents[3]; worker=root/"scripts"/"baseline_workers"/"run_dynotears.py"
    td=root/".baseline_tmp"/f"dynotears-{uuid.uuid4().hex}"; td.mkdir(parents=True)
    inp=td/"input.npz"; config_path=td/"config.json"; output=td/"output.npz"; metadata_path=td/"metadata.json"
    np.savez_compressed(inp,X=X); config={"p":int(p),"lambda_w":lambda_w,"lambda_a":lambda_a,"max_iter":max_iter,"h_tol":h_tol,"w_threshold":w_threshold,**kwargs}; config_path.write_text(json.dumps(config))
    command=[str(python_path),str(worker),"--input",str(inp),"--config",str(config_path),"--output",str(output),"--metadata",str(metadata_path)]
    started=time.perf_counter(); proc=subprocess.run(command,capture_output=True,text=True,timeout=timeout); wall=time.perf_counter()-started
    if proc.returncode: raise RuntimeError(f"DYNOTEARS subprocess failed ({proc.returncode}): {proc.stderr.strip()}")
    with np.load(output) as values:
        W0=np.asarray(values["W0"],float); Wlags=np.asarray(values["W_lags"],float)
        edge_source=values["edge_source"].tolist(); edge_destination=values["edge_destination"].tolist(); edge_weight=values["edge_weight"].tolist()
    metadata=json.loads(metadata_path.read_text())
    if W0.shape!=(X.shape[1],X.shape[1]) or Wlags.shape!=(p,X.shape[1],X.shape[1]) or not np.isfinite(W0).all() or not np.isfinite(Wlags).all(): raise RuntimeError("invalid DYNOTEARS worker output")
    native={"edges":metadata["edges"],"edge_source":edge_source,"edge_destination":edge_destination,"edge_weight":edge_weight}
    diagnostics={**metadata,"command":command,"stdout":proc.stdout,"stderr":proc.stderr,"wall_runtime_seconds":wall}
    if preserve_temp: diagnostics["temporary_directory"]=str(td)
    else: shutil.rmtree(td)
    return EstimatorResult(W0=W0,W_lags=Wlags,native_result=native,runtime_seconds=wall,diagnostics=diagnostics)

dynotears=fit_dynotears
