import numpy as np

def pag_endpoint_metrics(true_endpoints,estimated_endpoints):
    """Compare endpoint codes (0 none, 1 tail, 2 arrow, 3 circle)."""
    a=np.asarray(true_endpoints); b=np.asarray(estimated_endpoints)
    if a.shape!=b.shape: raise ValueError("endpoint arrays must have equal shapes")
    mask=(a!=0)|(b!=0)
    return {"endpoint_accuracy":float(np.mean(a[mask]==b[mask])) if mask.any() else 1.,
            "endpoint_errors":int(np.sum((a!=b)&mask)),"adjacency_errors":int(np.sum((a!=0)!=(b!=0))//2)}
