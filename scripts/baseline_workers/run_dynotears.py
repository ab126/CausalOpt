"""Isolated official CausalNex DYNOTEARS worker (Python 3.8--3.10)."""
from __future__ import annotations
import argparse,json,platform,time
from pathlib import Path
import numpy as np

def aligned_design(X,p):
    X=np.asarray(X,float)
    if X.ndim!=2 or p<1 or p>=len(X): raise ValueError("require 2-D X and 1 <= p < T")
    current=X[p:]
    lags=np.concatenate([X[p-tau:len(X)-tau] for tau in range(1,p+1)],axis=1)
    return current,lags

def parse_node(node):
    name=str(node); variable,lag=name.rsplit("_lag",1)
    return int(variable),int(lag)

def edges_to_matrices(edges,d,p):
    W0=np.zeros((d,d)); Wlags=np.zeros((p,d,d))
    for source,destination,weight in edges:
        i,source_lag=parse_node(source); j,destination_lag=parse_node(destination)
        lag=source_lag-destination_lag
        if destination_lag!=0: continue
        if lag==0: W0[i,j]=float(weight)
        elif 1<=lag<=p: Wlags[lag-1,i,j]=float(weight)
    return W0,Wlags

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--input",required=True); parser.add_argument("--config",required=True); parser.add_argument("--output",required=True); parser.add_argument("--metadata",required=True); args=parser.parse_args()
    import causalnex
    from causalnex.structure.dynotears import from_numpy_dynamic
    X=np.load(args.input)["X"]; config=json.loads(Path(args.config).read_text()); p=int(config.pop("p"))
    current,lags=aligned_design(X,p); started=time.perf_counter()
    graph=from_numpy_dynamic(current,lags,**config); runtime=time.perf_counter()-started
    edges=[(str(u),str(v),float(data.get("weight",1.))) for u,v,data in graph.edges(data=True)]
    W0,Wlags=edges_to_matrices(edges,X.shape[1],p)
    np.savez_compressed(args.output,W0=W0,W_lags=Wlags,
                        edge_source=np.asarray([e[0] for e in edges]),edge_destination=np.asarray([e[1] for e in edges]),edge_weight=np.asarray([e[2] for e in edges]))
    metadata={"backend":"official CausalNex subprocess","causalnex_version":causalnex.__version__,"python_version":platform.python_version(),"runtime_seconds":runtime,"p":p,"d":X.shape[1],"edges":[{"source":a,"destination":b,"weight":c} for a,b,c in edges]}
    Path(args.metadata).write_text(json.dumps(metadata,indent=2))
if __name__=="__main__": main()
