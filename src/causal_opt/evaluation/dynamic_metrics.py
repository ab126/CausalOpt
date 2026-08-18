import numpy as np
from .dag_metrics import dag_metrics

def support_metrics(truth,estimate,threshold=.3):
    truth=np.asarray(truth,float); estimate=np.asarray(estimate,float)
    if truth.shape!=estimate.shape: raise ValueError("truth and estimate shapes differ")
    T=np.abs(truth)>0; P=np.abs(estimate)>threshold
    tp=int(np.sum(T&P)); fp=int(np.sum(~T&P)); fn=int(np.sum(T&~P)); tn=int(np.sum(~T&~P))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    return {"fdr":float(fp/max(tp+fp,1)),"tpr":float(recall),"recall":float(recall),
      "fpr":float(fp/max(fp+tn,1)),"precision":float(precision),
      "f1":float(2*precision*recall/max(precision+recall,1e-15)),
      "shd":int(np.sum(T!=P)),"nnz":int(P.sum()),
      "weight_rmse":float(np.sqrt(np.mean((estimate-truth)**2)))}

def dynamic_metrics(W0_true,Wlags_true,W0_est,Wlags_est,X=None,threshold=.3,w0_threshold=None,wlag_threshold=None):
    Wlags_true=np.asarray(Wlags_true); Wlags_est=np.asarray(Wlags_est)
    t0=threshold if w0_threshold is None else w0_threshold; tl=threshold if wlag_threshold is None else wlag_threshold
    out={"W0":dag_metrics(W0_true,W0_est,threshold=t0)}
    out["lags"]=[support_metrics(a,b,tl) for a,b in zip(Wlags_true,Wlags_est)]
    out["lag_aggregate"]=support_metrics(Wlags_true,Wlags_est,tl)
    if X is not None:
        X=np.asarray(X); p=len(Wlags_est); Y=X[p:]; pred=Y@W0_est
        for q in range(1,p+1): pred+=X[p-q:len(X)-q]@Wlags_est[q-1]
        out["prediction_mse"]=float(np.mean((Y-pred)**2))
    return out
