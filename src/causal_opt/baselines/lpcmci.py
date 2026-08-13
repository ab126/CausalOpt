import time
import numpy as np
from ..simulation.static_sem import EstimatorResult

def fit_lpcmci(X,tau_max=1,pc_alpha=.05,**kwargs):
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.lpcmci import LPCMCI
    except ImportError as e: raise ImportError("LPCMCI requires optional package 'tigramite'") from e
    start=time.perf_counter(); dataframe=pp.DataFrame(np.asarray(X)); alg=LPCMCI(dataframe=dataframe,cond_ind_test=ParCorr())
    native=alg.run_lpcmci(tau_max=tau_max,pc_alpha=pc_alpha,**kwargs)
    return EstimatorResult(graph=native.get("graph"),native_result=native,
                           runtime_seconds=time.perf_counter()-start,diagnostics={"baseline":"Tigramite LPCMCI"})
