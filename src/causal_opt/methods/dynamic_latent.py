"""Dynamic sparse structural estimator with optional low-rank innovations."""
from __future__ import annotations
import time
import numpy as np
from ..simulation.static_sem import EstimatorResult, acyclicity
from .static_latent import _fit_w


def lagged_views(X, p):
    Y = X[p:]
    return Y, [X[p-q:len(X)-q] for q in range(1, p+1)]


def fit_dynamic_latent(X, p: int, k: int = 0, lambda_0: float = .1,
                       lambda_lag: float = .1, lambda_L: float = .1,
                       max_iter: int = 20, h_tol: float = 1e-8,
                       w_threshold: float = .3, random_state: int = 0):
    started = time.perf_counter(); X = np.asarray(X, float)
    Y, lags = lagged_views(X, p); d = X.shape[1]
    Wlags = np.zeros((p, d, d)); C = np.zeros_like(Y); prev = np.inf
    W0=np.zeros((d,d)); Z=L=None
    for it in range(max_iter):
        lag_effect = sum((lags[q]@Wlags[q] for q in range(p)), np.zeros_like(Y))
        W0, sol, _ = _fit_w(Y, Y-lag_effect-C, lambda_0, W0, 30, h_tol)
        if k:
            U,s,Vt=np.linalg.svd(Y-Y@W0-lag_effect,full_matrices=False)
            root=np.sqrt(np.maximum(s[:k]-lambda_L,0.)); Z=U[:,:k]*root; L=Vt[:k].T*root; C=Z@L.T
        target = Y-Y@W0-C
        if p:
            D = np.concatenate(lags, axis=1)
            # Proximal-gradient lasso for unrestricted lag coefficients.
            B = np.concatenate(list(Wlags), axis=0)
            step = 1.0/(np.linalg.norm(D, 2)**2/len(Y)+1e-12)
            for _ in range(100):
                G = D.T@(D@B-target)/len(Y)
                Q = B-step*G
                B = np.sign(Q)*np.maximum(np.abs(Q)-step*lambda_lag, 0.)
            Wlags = B.reshape(p, d, d)
        residual = target - (D@B if p else 0.)
        obj = .5*np.sum(residual**2)/len(Y)+lambda_0*np.abs(W0).sum()+lambda_lag*np.abs(Wlags).sum()
        if abs(prev-obj) < 1e-7: break
        prev = obj
    W0[np.abs(W0)<w_threshold]=0.; Wlags[np.abs(Wlags)<w_threshold]=0.
    return EstimatorResult(W0=W0,W_lags=Wlags,Z=Z,L=L,C=C if k else None,
        runtime_seconds=time.perf_counter()-started,
        diagnostics={"objective":float(obj),"h":acyclicity(W0),"iterations":it+1,
                     "optimizer_success":bool(sol.success)})


dynamic_latent = fit_dynamic_latent
