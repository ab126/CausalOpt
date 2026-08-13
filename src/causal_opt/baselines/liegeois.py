"""Wrapper for the authors' MATLAB sparse-plus-low-rank AR implementation."""
from __future__ import annotations
import json, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from scipy.io import savemat, loadmat
from ..simulation.static_sem import EstimatorResult

def fit_liegeois(X, matlab_entrypoint: str, implementation_dir: str,
                 matlab_executable: str = "matlab", extra_options=None):
    """Exchange MAT files with MATLAB; no baseline algorithm is implemented here."""
    start=time.perf_counter(); extra_options=extra_options or {}
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); inp=td/"input.mat"; out=td/"output.mat"; savemat(inp,{"X":np.asarray(X)})
        opts=json.dumps(extra_options).replace("'","''")
        command=(f"addpath('{Path(implementation_dir).as_posix()}'); "
                 f"{matlab_entrypoint}('{inp.as_posix()}','{out.as_posix()}','{opts}');")
        proc=subprocess.run([matlab_executable,"-batch",command],capture_output=True,text=True)
        if proc.returncode: raise RuntimeError(f"MATLAB baseline failed: {proc.stderr}")
        native=loadmat(out,squeeze_me=True)
    return EstimatorResult(W_lags=native.get("sparse"),C=native.get("low_rank"),native_result=native,
        runtime_seconds=time.perf_counter()-start,diagnostics={"baseline":"Liegeois MATLAB","stdout":proc.stdout})
