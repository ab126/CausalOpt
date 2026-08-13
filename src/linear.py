import numpy as np
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def causal_opt_linear(mat_x, lambda1, loss_type, max_iter=100, h_tol=1e-8, tau_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 for the case with latent variables using augmented Lagrangian.

    Args:
        mat_x (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        tau_max (float): exit if tau >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        mat_w_est (np.ndarray): [d, d] estimated DAG
    """

    n, d = mat_x.shape
    w_est, tau, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        mat_x = mat_x - np.mean(mat_x, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while tau < tau_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                tau *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += tau * h
        if h <= h_tol or tau >= tau_max:
            break
    mat_w_est = _adj(w_est)
    mat_w_est[np.abs(mat_w_est) < w_threshold] = 0
    return mat_w_est

    pass
