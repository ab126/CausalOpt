"""Thin adapter around the vendored, unmodified official NOTEARS code."""
import importlib.util
from pathlib import Path
import time
from ..simulation.static_sem import EstimatorResult

def _linear_module():
    path=Path(__file__).resolve().parents[3]/"notears"/"notears"/"linear.py"
    spec=importlib.util.spec_from_file_location("causal_opt_reference_linear",path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def fit_notears(X,lambda1=.1,loss_type="l2",**kwargs):
    start=time.perf_counter()
    W=_linear_module().notears_linear(X,lambda1,loss_type,**kwargs)
    return EstimatorResult(W0=W,runtime_seconds=time.perf_counter()-start,
                           diagnostics={"baseline":"official-notears"})

notears=fit_notears
