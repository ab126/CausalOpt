import time
import numpy as np
from ..simulation.static_sem import EstimatorResult

def _encode(graph):
    nodes=graph.get_nodes(); d=len(nodes); out=np.zeros((d,d),int)
    mapping={"TAIL":1,"ARROW":2,"CIRCLE":3}
    for edge in graph.get_graph_edges():
        i=nodes.index(edge.get_node1()); j=nodes.index(edge.get_node2())
        out[i,j]=mapping.get(str(edge.get_endpoint1()).upper(),3)
        out[j,i]=mapping.get(str(edge.get_endpoint2()).upper(),3)
    return out

def fit_fci(X,alpha=.05,**kwargs):
    try: from causallearn.search.ConstraintBased.FCI import fci
    except ImportError as e: raise ImportError("FCI requires the optional 'causal-learn' package") from e
    start=time.perf_counter(); graph,edges=fci(np.asarray(X),independence_test_method="fisherz",alpha=alpha,**kwargs)
    return EstimatorResult(graph=_encode(graph),native_result=graph,runtime_seconds=time.perf_counter()-start,
                           diagnostics={"edges":edges,"baseline":"causal-learn FCI"})

fci_baseline=fit_fci
