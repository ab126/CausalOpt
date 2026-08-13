from __future__ import annotations
import numpy as np
from ..simulation.static_sem import acyclicity


def dag_metrics(W_true, W_est, threshold=.3, X=None):
    W_true=np.asarray(W_true); W_est=np.asarray(W_est)
    T=np.abs(W_true)>0; P=np.abs(W_est)>threshold
    np.fill_diagonal(T,False); np.fill_diagonal(P,False)
    tp=np.sum(T&P); fp=np.sum(~T&P); fn=np.sum(T&~P); tn=np.sum(~T&~P)-len(T)
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    # SHD counts reversal once rather than as one missing plus one extra.
    skeleton_true=T|T.T; skeleton_pred=P|P.T
    extra=np.sum(np.triu(skeleton_pred&~skeleton_true,1))
    missing=np.sum(np.triu(skeleton_true&~skeleton_pred,1))
    reversals=sum(P[j,i] and T[i,j] and not P[i,j] for i in range(len(T)) for j in range(i+1,len(T)))
    m={"fdr":float(fp/max(tp+fp,1)),"tpr":float(recall),"recall":float(recall),
       "fpr":float(fp/max(fp+tn,1)),"precision":float(precision),
       "f1":float(2*precision*recall/max(precision+recall,1e-15)),
       "shd":int(extra+missing+reversals),"nnz":int(P.sum()),
       "weight_rmse":float(np.sqrt(np.mean((W_est-W_true)**2))),"h":acyclicity(W_est)}
    if X is not None: m["residual_mse"]=float(np.mean((X-X@W_est)**2))
    return m


evaluate_dag = dag_metrics
