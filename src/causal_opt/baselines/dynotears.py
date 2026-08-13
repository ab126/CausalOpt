import time
import numpy as np
from ..simulation.static_sem import EstimatorResult

def fit_dynotears(X,p=1,**kwargs):
    try: from causalnex.structure.dynotears import from_numpy_dynamic
    except ImportError as e: raise ImportError("DYNOTEARS requires optional package 'causalnex'; use an isolated environment if needed") from e
    start=time.perf_counter(); graph=from_numpy_dynamic(np.asarray(X),p=p,**kwargs); d=X.shape[1]
    W0=np.zeros((d,d)); lags=np.zeros((p,d,d))
    for u,v,data in graph.edges(data=True):
        def parse(node):
            if isinstance(node,tuple): return int(node[0]),int(node[1])
            parts=str(node).rsplit("_lag",1); return int(parts[0]),int(parts[1]) if len(parts)>1 else 0
        i,li=parse(u); j,lj=parse(v); lag=li-lj
        if lag==0: W0[i,j]=data.get("weight",1.)
        elif 1<=lag<=p: lags[lag-1,i,j]=data.get("weight",1.)
    return EstimatorResult(W0=W0,W_lags=lags,graph=graph,native_result=graph,
                           runtime_seconds=time.perf_counter()-start,diagnostics={"baseline":"CausalNex DYNOTEARS"})
