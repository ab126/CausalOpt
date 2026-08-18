"""Joint NOTEARS estimator for dynamic Gaussian systems with latent factors."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from ..simulation.static_sem import EstimatorResult
from .static_latent import _acyclicity, _initial_factors

def lagged_views(X,p):
    X=np.asarray(X)
    if X.ndim!=2 or not isinstance(p,(int,np.integer)) or p<1 or p>=len(X): raise ValueError("require 2-D X and 1 <= p < T")
    return X[p:],np.stack([X[p-t:len(X)-t] for t in range(1,p+1)])

def _pack_parameters(W0p,W0n,Wlp,Wln,Z=None,L=None):
    parts=[np.asarray(a).ravel() for a in (W0p,W0n,Wlp,Wln)]
    if (Z is None)!=(L is None): raise ValueError("Z and L must both be present or absent")
    if Z is not None: parts.extend((np.asarray(Z).ravel(),np.asarray(L).ravel()))
    return np.concatenate(parts).astype(float,copy=False)

def _unpack_parameters(v,m,d,p,k):
    v=np.asarray(v,float); a=d*d; b=p*a; expected=2*a+2*b+(m*k+d*k if k else 0)
    if v.ndim!=1 or v.size!=expected: raise ValueError(f"expected length-{expected} parameters")
    out=[v[:a].reshape(d,d),v[a:2*a].reshape(d,d),v[2*a:2*a+b].reshape(p,d,d),v[2*a+b:2*a+2*b].reshape(p,d,d)]
    if not k: return (*out,None,None)
    o=2*a+2*b
    return (*out,v[o:o+m*k].reshape(m,k),v[o+m*k:].reshape(d,k))

def _bounds(m,d,p,k):
    b=[(0.,0.) if i==j else (0.,None) for _ in range(2) for i in range(d) for j in range(d)]
    return b+[(0.,None)]*(2*p*d*d)+[(None,None)]*(m*k+d*k)

def _objective(v,X0,Xlags,k,lambda_0,lambda_lag,lambda_latent,rho,alpha):
    m,d=X0.shape; p=len(Xlags); W0p,W0n,Wlp,Wln,Z,L=_unpack_parameters(v,m,d,p,k)
    W0,Wlags=W0p-W0n,Wlp-Wln
    R=X0-X0@W0-np.einsum("tmi,tij->mj",Xlags,Wlags)
    if k: R-=Z@L.T
    fit=.5*np.sum(R*R)/m; s0=lambda_0*np.sum(W0p+W0n); sl=lambda_lag*np.sum(Wlp+Wln)
    latent=.5*lambda_latent*(np.sum(Z*Z)+np.sum(L*L)) if k else 0.
    h,gh=_acyclicity(W0); g0=-(X0.T@R)/m+(alpha+rho*h)*gh
    gl=-np.einsum("tmi,mj->tij",Xlags,R)/m
    gZ=-(R@L)/m+lambda_latent*Z if k else None; gL=-(R.T@Z)/m+lambda_latent*L if k else None
    value=fit+s0+sl+latent+alpha*h+.5*rho*h*h
    grad=_pack_parameters(g0+lambda_0,-g0+lambda_0,gl+lambda_lag,-gl+lambda_lag,gZ,gL)
    if not np.isfinite(value) or not np.isfinite(grad).all(): raise FloatingPointError("non-finite dynamic objective or gradient")
    return float(value),grad

def fit_dynamic_latent(X,p,k=0,lambda_0=.1,lambda_lag=.1,lambda_latent=.1,max_outer_iter=100,h_tol=1e-8,
 rho_init=1.,rho_max=1e16,rho_multiplier=10.,h_progress_ratio=.25,w0_threshold=.3,wlag_threshold=.3,
 initialization="svd_residual",W0_init=None,W_lags_init=None,random_state=None,center=True,return_raw=True,
 verbose=False,inner_max_iter=None,lambda_L=None,max_iter=None,w_threshold=None):
    started=time.perf_counter(); X=np.asarray(X,float)
    if X.ndim!=2 or X.shape[1]<=1 or len(X)<=2 or not np.isfinite(X).all(): raise ValueError("X must be a finite 2-D time series")
    T,d=X.shape
    if not isinstance(p,(int,np.integer)) or p<1 or p>=T: raise ValueError("require 1 <= p < T")
    if not isinstance(k,(int,np.integer)) or k<0 or k>d: raise ValueError("k must be between 0 and d")
    if lambda_L is not None: lambda_latent=lambda_L
    if max_iter is not None: max_outer_iter=max_iter
    if w_threshold is not None: w0_threshold=wlag_threshold=w_threshold
    if min(lambda_0,lambda_lag,lambda_latent,w0_threshold,wlag_threshold)<0: raise ValueError("penalties/thresholds must be nonnegative")
    if max_outer_iter<=0 or rho_init<=0 or rho_max<=0 or rho_multiplier<=1 or not 0<h_progress_ratio<1: raise ValueError("invalid optimizer controls")
    means=X.mean(0,keepdims=True) if center else np.zeros((1,d)); Xc=X.copy()-means; X0,Xlags=lagged_views(Xc,p); m=len(X0)
    W0=np.zeros((d,d)) if W0_init is None else np.asarray(W0_init,float).copy()
    Wlags=np.zeros((p,d,d)) if W_lags_init is None else np.asarray(W_lags_init,float).copy()
    if W0.shape!=(d,d) or Wlags.shape!=(p,d,d) or not np.isfinite(W0).all() or not np.isfinite(Wlags).all(): raise ValueError("invalid initial matrices")
    np.fill_diagonal(W0,0.)
    if initialization=="dynamic_then_svd" and (W0_init is None or W_lags_init is None): raise ValueError("dynamic_then_svd requires explicit starts")
    W0p,W0n=np.maximum(W0,0),np.maximum(-W0,0); Wlp,Wln=np.maximum(Wlags,0),np.maximum(-Wlags,0)
    if k:
        R0=X0-X0@W0-np.einsum("tmi,tij->mj",Xlags,Wlags); mode="svd_residual" if initialization=="dynamic_then_svd" else initialization
        Z,L=_initial_factors(R0,k,mode,np.random.default_rng(random_state))
    else: Z=L=None
    v=_pack_parameters(W0p,W0n,Wlp,Wln,Z,L); bounds=_bounds(m,d,p,k)
    if len(bounds)!=v.size: raise RuntimeError("bounds length mismatch")
    rho,alpha,hprev=float(rho_init),0.,np.inf; history=[]; attempts=[]; total_nit=total_nfev=0; sol=None
    for outer in range(max_outer_iter):
        while True:
            options={} if inner_max_iter is None else {"maxiter":int(inner_max_iter)}
            sol=minimize(_objective,v,args=(X0,Xlags,k,lambda_0,lambda_lag,lambda_latent,rho,alpha),method="L-BFGS-B",jac=True,bounds=bounds,options=options)
            v=sol.x; total_nit+=int(sol.nit); total_nfev+=int(sol.nfev)
            W0p,W0n,Wlp,Wln,Z,L=_unpack_parameters(v,m,d,p,k); W0,Wlags=W0p-W0n,Wlp-Wln; h,_=_acyclicity(W0)
            attempts.append({"outer_iter":outer+1,"rho":rho,"h":h,"status":int(sol.status),"message":str(sol.message),"nit":int(sol.nit),"nfev":int(sol.nfev)})
            if h<=h_progress_ratio*hprev or rho>=rho_max: break
            rho=min(rho*rho_multiplier,rho_max)
        R=X0-X0@W0-np.einsum("tmi,tij->mj",Xlags,Wlags)-(Z@L.T if k else 0.)
        fit=float(.5*np.sum(R*R)/m); s0=float(lambda_0*np.sum(W0p+W0n)); sl=float(lambda_lag*np.sum(Wlp+Wln)); lat=float(.5*lambda_latent*(np.sum(Z*Z)+np.sum(L*L))) if k else 0.; aug=fit+s0+sl+lat+alpha*h+.5*rho*h*h
        row={"outer_iter":outer+1,"rho":rho,"alpha":alpha,"h":h,"fit_loss":fit,"sparse_0_penalty":s0,"sparse_lag_penalty":sl,"latent_penalty":lat,"augmented_objective":float(aug),"optimizer_success":bool(sol.success),"optimizer_nit":int(sol.nit),"optimizer_nfev":int(sol.nfev)}
        if k:
            C=Z@L.T; row.update(Z_fro_norm=float(np.linalg.norm(Z)),L_fro_norm=float(np.linalg.norm(L)),C_fro_norm=float(np.linalg.norm(C)),effective_rank_C=int(np.linalg.matrix_rank(C)))
        history.append(row)
        if verbose: print(f"outer={outer+1} rho={rho:.1e} h={h:.3e} fit={fit:.6g}")
        hprev=h; alpha+=rho*h
        if h<=h_tol or rho>=rho_max: break
    W0raw,Wlagsraw=W0.copy(),Wlags.copy(); W0est,Wlagsest=W0raw.copy(),Wlagsraw.copy(); W0est[np.abs(W0est)<w0_threshold]=0.; Wlagsest[np.abs(Wlagsest)<wlag_threshold]=0.
    C=Z@L.T if k else None; ht,_=_acyclicity(W0est); zn=float(np.linalg.norm(Z)) if k else 0.; ln=float(np.linalg.norm(L)) if k else 0.; cn=float(np.linalg.norm(C)) if k else 0.; balance=zn/max(ln,1e-15) if k else None
    warnings=[]
    if h>h_tol: warnings.append("acyclicity_not_converged")
    if np.linalg.norm(Wlagsest)==0: warnings.append("lag_collapse")
    if np.linalg.norm(W0est)==0: warnings.append("W0_collapse")
    if k and cn<=1e-10: warnings.append("latent_collapse")
    if k and np.linalg.matrix_rank(C)!=k: warnings.append("rank_mismatch")
    if k and not 1e-3<=balance<=1e3: warnings.append("factor_scale_imbalance")
    diag={"objective":float(aug),"fit_loss":fit,"sparse_0_penalty":s0,"sparse_lag_penalty":sl,"latent_penalty":lat,"h_raw":float(h),"h_thresholded":float(ht),"rho":rho,"peak_rho":rho,"alpha":alpha,"outer_iterations":outer+1,"total_inner_iterations":total_nit,"total_function_evaluations":total_nfev,"optimizer_success":bool(sol.success),"optimizer_status":int(sol.status),"optimizer_message":str(sol.message),"gradient_norm":float(np.linalg.norm(sol.jac)),"history":history,"inner_attempt_history":attempts,"column_means":means.ravel(),"number_optimization_variables":v.size,"T":T,"d":d,"p":p,"k":k,"Z_fro_norm":zn,"L_fro_norm":ln,"C_fro_norm":cn,"factor_balance_ratio":balance,"effective_rank_C":int(np.linalg.matrix_rank(C)) if k else 0,"warnings":warnings}
    return EstimatorResult(W0=W0est,W_raw=W0raw if return_raw else None,W_lags=Wlagsest,W_lags_raw=Wlagsraw if return_raw else None,Z=Z,L=L,C=C,runtime_seconds=time.perf_counter()-started,diagnostics=diag)

dynamic_latent=fit_dynamic_latent
