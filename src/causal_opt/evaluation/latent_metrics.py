import numpy as np

def projection_matrix(L):
    L=np.asarray(L,float)
    return L@np.linalg.pinv(L.T@L)@L.T

def reconstruction_error(C_true,C_est):
    return float(np.linalg.norm(C_est-C_true)/max(np.linalg.norm(C_true),1e-15))

def subspace_error(L_true,L_est):
    return float(np.linalg.norm(projection_matrix(L_est)-projection_matrix(L_true),"fro")/np.sqrt(2.0))

def effective_rank(C,tol=None):
    s=np.linalg.svd(C,compute_uv=False)
    if tol is None: tol=max(C.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    return int(np.sum(s>tol))

def latent_metrics(C_true,C_est,L_true,L_est):
    return {"c_reconstruction_error":reconstruction_error(C_true,C_est),
            "effective_rank":effective_rank(C_est),"loading_subspace_error":subspace_error(L_true,L_est)}
