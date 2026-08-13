import numpy as np
from .dag_metrics import dag_metrics

def dynamic_metrics(W0_true,Wlags_true,W0_est,Wlags_est,X=None):
    out={"W0":dag_metrics(W0_true,W0_est)}
    out["lags"]=[dag_metrics(a,b) for a,b in zip(Wlags_true,Wlags_est)]
    out["lag_aggregate"]=dag_metrics(Wlags_true.reshape(-1,Wlags_true.shape[-1]),Wlags_est.reshape(-1,Wlags_est.shape[-1]))
    if X is not None:
        p=len(Wlags_est); Y=X[p:]; pred=Y@W0_est
        for q in range(1,p+1): pred+=X[p-q:len(X)-q]@Wlags_est[q-1]
        out["prediction_mse"]=float(np.mean((Y-pred)**2))
    return out
