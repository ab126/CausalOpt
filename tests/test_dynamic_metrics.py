import numpy as np
from causal_opt.evaluation.dynamic_metrics import dynamic_metrics,support_metrics

def test_lag_metrics_per_lag_and_aggregate():
    W0=np.zeros((3,3)); W0[0,1]=1
    lags=np.zeros((2,3,3)); lags[0,0,0]=.5; lags[1,2,1]=-.4
    m=dynamic_metrics(W0,lags,W0,lags,threshold=0)
    assert m["W0"]["f1"]==1 and all(x["f1"]==1 for x in m["lags"])
    assert m["lag_aggregate"]["f1"]==1 and m["lag_aggregate"]["shd"]==0

def test_support_orientation_matters():
    a=np.zeros((1,3,3)); a[0,0,2]=1; b=np.zeros_like(a); b[0,2,0]=1
    assert support_metrics(a,b,0)["shd"]==2
