"""Gradient-based static DAG estimator with an optional low-rank term."""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from ..simulation.static_sem import EstimatorResult, acyclicity


def _fit_w(X, target, lambda_w, initial, max_iter, h_tol):
    n, d = X.shape
    bounds = [(0., 0.) if i == j else (0., None)
              for _ in range(2) for i in range(d) for j in range(d)]
    w = np.r_[np.maximum(initial, 0).ravel(), np.maximum(-initial, 0).ravel()]
    rho, alpha, h = 1., 0., np.inf
    last = None
    def unpack(v): return (v[:d*d] - v[d*d:]).reshape(d, d)
    for outer in range(max_iter):
        while rho < 1e16:
            def fun(v):
                W = unpack(v); R = target - X @ W
                E = expm(W*W); hv = np.trace(E)-d
                G = -X.T@R/n + (rho*hv+alpha)*(E.T*W*2)
                val = .5*np.sum(R*R)/n + lambda_w*np.sum(v) + .5*rho*hv*hv + alpha*hv
                return val, np.r_[(G+lambda_w).ravel(), (-G+lambda_w).ravel()]
            last = minimize(fun, w, jac=True, method="L-BFGS-B", bounds=bounds)
            candidate = unpack(last.x); hnew = acyclicity(candidate)
            if hnew <= .25*h or rho >= 1e15: break
            rho *= 10
        w, h = last.x, hnew
        alpha += rho*h
        if h <= h_tol: break
    return unpack(w), last, outer + 1


def fit_static_latent(X, k: int = 0, lambda_w: float = .1,
                      lambda_l: float = .1, max_iter: int = 30,
                      inner_max_iter: int = 100, h_tol: float = 1e-8,
                      w_threshold: float = .3, random_state: int = 0):
    X = np.asarray(X, float); n, d = X.shape
    if k < 0 or k > min(n, d): raise ValueError("invalid latent rank")
    started = time.perf_counter(); W = np.zeros((d, d)); C = np.zeros_like(X)
    Z = L = None; history = []
    for iteration in range(max(1, max_iter if k else 1)):
        W, sol, outer = _fit_w(X, X-C, lambda_w, W, inner_max_iter, h_tol)
        if k:
            U, s, Vt = np.linalg.svd(X-X@W, full_matrices=False)
            shrunk = np.maximum(s[:k]-lambda_l, 0.)
            root = np.sqrt(shrunk)
            Z, L = U[:, :k]*root, Vt[:k].T*root
            Cnew = Z@L.T
            delta = np.linalg.norm(Cnew-C)/(np.linalg.norm(C)+1e-12); C = Cnew
        else: delta = 0.
        obj = .5*np.sum((X-X@W-C)**2)/n + lambda_w*np.abs(W).sum()
        if k: obj += .5*lambda_l*(np.sum(Z*Z)+np.sum(L*L))
        history.append(float(obj))
        if delta < 1e-7: break
    W[np.abs(W) < w_threshold] = 0.; np.fill_diagonal(W, 0.)
    return EstimatorResult(W0=W, Z=Z, L=L, C=C if k else None,
        runtime_seconds=time.perf_counter()-started,
        diagnostics={"objective": history[-1], "objective_history": history,
                     "h": acyclicity(W), "finite": bool(np.isfinite(history[-1])),
                     "iterations": iteration+1, "optimizer_success": bool(sol.success)})


static_latent = fit_static_latent
