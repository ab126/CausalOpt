"""Joint gradient-based static DAG estimator with a low-rank latent term."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from ..simulation.static_sem import EstimatorResult


def _pack_parameters(W_pos, W_neg, Z=None, L=None):
    """Pack split adjacency and optional latent factors into one vector."""
    parts = [np.asarray(W_pos).ravel(), np.asarray(W_neg).ravel()]
    if (Z is None) != (L is None):
        raise ValueError("Z and L must either both be present or both be absent")
    if Z is not None:
        parts.extend((np.asarray(Z).ravel(), np.asarray(L).ravel()))
    return np.concatenate(parts).astype(float, copy=False)


def _unpack_parameters(parameters, n, d, k):
    """Unpack a parameter vector into arrays with exact known shapes."""
    parameters = np.asarray(parameters, dtype=float)
    expected = 2 * d * d + (n * k + d * k if k else 0)
    if parameters.ndim != 1 or parameters.size != expected:
        raise ValueError(f"expected a length-{expected} parameter vector")
    offset = d * d
    W_pos = parameters[:offset].reshape(d, d)
    W_neg = parameters[offset:2 * offset].reshape(d, d)
    if not k:
        return W_pos, W_neg, None, None
    start = 2 * offset
    Z = parameters[start:start + n * k].reshape(n, k)
    L = parameters[start + n * k:].reshape(d, k)
    return W_pos, W_neg, Z, L


def _acyclicity(W):
    """Return NOTEARS' smooth acyclicity constraint and its gradient."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    E = expm(W * W)
    if not np.all(np.isfinite(E)):
        raise FloatingPointError("non-finite matrix exponential in acyclicity constraint")
    return float(np.trace(E) - W.shape[0]), E.T * (2.0 * W)


def _objective(parameters, X, k, lambda_w, lambda_latent, rho, alpha):
    """Evaluate the augmented-Lagrangian objective and analytic gradient."""
    n, d = X.shape
    W_pos, W_neg, Z, L = _unpack_parameters(parameters, n, d, k)
    W = W_pos - W_neg
    R = X - X @ W
    if k:
        R = R - Z @ L.T
    fit_loss = 0.5 * np.sum(R * R) / n
    sparse_penalty = lambda_w * np.sum(W_pos + W_neg)
    latent_penalty = (0.5 * lambda_latent
                      * (np.sum(Z * Z) + np.sum(L * L))) if k else 0.0
    h, grad_h = _acyclicity(W)
    grad_W = -(X.T @ R) / n + (alpha + rho * h) * grad_h
    grad_W_pos = grad_W + lambda_w
    grad_W_neg = -grad_W + lambda_w
    if k:
        grad_Z = -(R @ L) / n + lambda_latent * Z
        grad_L = -(R.T @ Z) / n + lambda_latent * L
    else:
        grad_Z = grad_L = None
    value = fit_loss + sparse_penalty + latent_penalty + alpha * h + 0.5 * rho * h * h
    gradient = _pack_parameters(grad_W_pos, grad_W_neg, grad_Z, grad_L)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite static latent objective or gradient")
    return float(value), gradient


def _initial_factors(residual, k, initialization, rng):
    n, d = residual.shape
    if initialization in {"svd_residual", "notears_then_svd"}:
        U, singular_values, Vt = np.linalg.svd(residual, full_matrices=False)
        rank = min(k, singular_values.size)
        roots = np.sqrt(np.maximum(singular_values[:rank], 0.0))
        Z, L = np.zeros((n, k)), np.zeros((d, k))
        Z[:, :rank], L[:, :rank] = U[:, :rank] * roots, Vt[:rank].T * roots
        return Z, L
    if initialization == "random":
        return rng.normal(scale=1e-2, size=(n, k)), rng.normal(scale=1e-2, size=(d, k))
    raise ValueError("for k > 0 initialization must be 'svd_residual', 'random', or 'notears_then_svd'")


def _bounds(n, d, k):
    result = [(0.0, 0.0) if i == j else (0.0, None)
              for _ in range(2) for i in range(d) for j in range(d)]
    result.extend([(None, None)] * (n * k + d * k))
    return result


def _fit_w(X, target, lambda_w, initial, max_iter, h_tol):
    """Compatibility helper for the existing dynamic estimator's W update."""
    n, d = X.shape
    parameters = _pack_parameters(np.maximum(initial, 0.0),
                                  np.maximum(-initial, 0.0))
    bounds = _bounds(n, d, 0)
    rho, alpha, h_prev = 1.0, 0.0, np.inf
    solution = None

    def objective(values):
        Wp, Wn, _, _ = _unpack_parameters(values, n, d, 0)
        W = Wp - Wn
        residual = target - X @ W
        h, grad_h = _acyclicity(W)
        grad_W = -(X.T @ residual) / n + (alpha + rho * h) * grad_h
        value = (0.5 * np.sum(residual * residual) / n
                 + lambda_w * np.sum(Wp + Wn) + alpha * h + 0.5 * rho * h * h)
        return value, _pack_parameters(grad_W + lambda_w, -grad_W + lambda_w)

    for outer in range(max_iter):
        while rho < 1e16:
            solution = minimize(objective, parameters, method="L-BFGS-B",
                                jac=True, bounds=bounds)
            parameters = solution.x
            Wp, Wn, _, _ = _unpack_parameters(parameters, n, d, 0)
            h_new, _ = _acyclicity(Wp - Wn)
            if h_new <= 0.25 * h_prev:
                break
            rho *= 10.0
        h_prev = h_new
        alpha += rho * h_new
        if h_new <= h_tol or rho >= 1e16:
            break
    Wp, Wn, _, _ = _unpack_parameters(parameters, n, d, 0)
    return Wp - Wn, solution, outer + 1


def fit_static_latent(
    X,
    k=0,
    lambda_w=0.1,
    lambda_latent=0.1,
    max_outer_iter=100,
    h_tol=1e-8,
    rho_init=1.0,
    rho_max=1e16,
    rho_multiplier=10.0,
    h_progress_ratio=0.25,
    w_threshold=0.3,
    initialization="svd_residual",
    random_state=None,
    standardize=False,
    return_raw=True,
    verbose=False,
    W_init: Optional[np.ndarray] = None,
    inner_max_iter=None,
    # Backwards-compatible aliases used by the initial experiment configs.
    lambda_l=None,
    max_iter=None,
):
    """Fit ``X = XW + ZL.T + noise`` subject to the NOTEARS DAG constraint."""
    started = time.perf_counter()
    X_input = np.asarray(X, dtype=float)
    if X_input.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    n, d = X_input.shape
    if n <= 1 or d <= 1:
        raise ValueError("X must contain more than one row and column")
    if not np.all(np.isfinite(X_input)):
        raise ValueError("X must contain only finite values")
    if not isinstance(k, (int, np.integer)) or k < 0 or k > d:
        raise ValueError("k must be an integer between 0 and d")
    if lambda_l is not None:
        lambda_latent = lambda_l
    if max_iter is not None:
        max_outer_iter = max_iter
    if lambda_w < 0 or lambda_latent < 0:
        raise ValueError("regularization parameters must be nonnegative")
    if max_outer_iter <= 0 or rho_init <= 0 or rho_max <= 0 or rho_multiplier <= 1:
        raise ValueError("invalid augmented-Lagrangian controls")
    if not 0 < h_progress_ratio < 1 or w_threshold < 0:
        raise ValueError("h_progress_ratio must be in (0,1) and w_threshold nonnegative")

    means = X_input.mean(axis=0, keepdims=True)
    Xc = X_input.copy() - means
    scales = np.ones((1, d))
    if standardize:
        scales = Xc.std(axis=0, keepdims=True)
        if np.any(scales == 0):
            raise ValueError("cannot standardize a constant column")
        Xc /= scales

    if W_init is None:
        W0 = np.zeros((d, d))
    else:
        W0 = np.asarray(W_init, dtype=float).copy()
        if W0.shape != (d, d) or not np.all(np.isfinite(W0)):
            raise ValueError(f"W_init must be a finite {(d, d)} matrix")
        np.fill_diagonal(W0, 0.0)
    if initialization == "zeros" and k:
        raise ValueError("zeros initialization is only supported for k=0")
    if initialization == "notears_then_svd" and W_init is None:
        warm = fit_static_latent(X_input, k=0, lambda_w=lambda_w,
                                 max_outer_iter=max_outer_iter, h_tol=h_tol,
                                 rho_init=rho_init, rho_max=rho_max,
                                 rho_multiplier=rho_multiplier,
                                 h_progress_ratio=h_progress_ratio,
                                 w_threshold=0.0, standardize=standardize,
                                 return_raw=True, inner_max_iter=inner_max_iter)
        W0 = warm.diagnostics["W_raw"].copy()

    W_pos, W_neg = np.maximum(W0, 0.0), np.maximum(-W0, 0.0)
    np.fill_diagonal(W_pos, 0.0)
    np.fill_diagonal(W_neg, 0.0)
    if k:
        Z, L = _initial_factors(Xc - Xc @ W0, k, initialization,
                                np.random.default_rng(random_state))
    else:
        Z = L = None
    parameters = _pack_parameters(W_pos, W_neg, Z, L)
    bounds = _bounds(n, d, k)
    if len(bounds) != parameters.size:
        raise RuntimeError("internal bounds/parameter length mismatch")

    rho, alpha, h_prev = float(rho_init), 0.0, np.inf
    history, messages, statuses, inner_nits, inner_nfevs = [], [], [], [], []
    total_nit = total_nfev = 0
    last_solution = None
    for outer_iter in range(max_outer_iter):
        while True:
            options = {} if inner_max_iter is None else {"maxiter": int(inner_max_iter)}
            solution = minimize(_objective, parameters,
                                args=(Xc, k, lambda_w, lambda_latent, rho, alpha),
                                method="L-BFGS-B", jac=True, bounds=bounds, options=options)
            last_solution = solution
            parameters = solution.x  # always warm-start the next solve
            total_nit += int(solution.nit)
            total_nfev += int(solution.nfev)
            messages.append(str(solution.message))
            statuses.append(int(solution.status))
            inner_nits.append(int(solution.nit))
            inner_nfevs.append(int(solution.nfev))
            Wp, Wn, Z, L = _unpack_parameters(parameters, n, d, k)
            h_new, _ = _acyclicity(Wp - Wn)
            if h_new <= h_progress_ratio * h_prev or rho >= rho_max:
                break
            rho = min(rho * rho_multiplier, rho_max)

        W = Wp - Wn
        R = Xc - Xc @ W - (Z @ L.T if k else 0.0)
        fit_loss = float(0.5 * np.sum(R * R) / n)
        sparse_penalty = float(lambda_w * np.sum(Wp + Wn))
        latent_penalty = float(0.5 * lambda_latent *
                               (np.sum(Z * Z) + np.sum(L * L))) if k else 0.0
        primal_objective = fit_loss + sparse_penalty + latent_penalty
        objective = primal_objective + alpha * h_new + 0.5 * rho * h_new * h_new
        history.append({"outer_iter": outer_iter + 1, "rho": rho, "alpha": alpha,
                        "h": h_new, "objective": objective, "fit_loss": fit_loss,
                        "sparse_penalty": sparse_penalty,
                        "latent_penalty": latent_penalty,
                        "optimizer_success": bool(solution.success),
                        "optimizer_nit": int(solution.nit)})
        if verbose:
            print(f"outer={outer_iter + 1} rho={rho:.1e} h={h_new:.3e} obj={objective:.6g}")
        h_prev = h_new
        alpha += rho * h_new
        if h_new <= h_tol or rho >= rho_max:
            break

    W_raw = W.copy()
    W_est = W_raw.copy()
    W_est[np.abs(W_est) < w_threshold] = 0.0
    C = Z @ L.T if k else None
    h_thresholded, _ = _acyclicity(W_est)
    latent_norms = ({"effective_rank_C": int(np.linalg.matrix_rank(C)),
                     "Z_fro_norm": float(np.linalg.norm(Z)),
                     "L_fro_norm": float(np.linalg.norm(L)),
                     "C_fro_norm": float(np.linalg.norm(C))} if k else
                    {"effective_rank_C": 0, "Z_fro_norm": 0.0,
                     "L_fro_norm": 0.0, "C_fro_norm": 0.0})
    diagnostics = {
        "objective": float(objective), "primal_objective": float(primal_objective),
        "fit_loss": fit_loss,
        "sparse_penalty": sparse_penalty, "latent_penalty": latent_penalty,
        "h_raw": float(h_new), "h_thresholded": float(h_thresholded),
        "h": float(h_thresholded), "rho": rho, "alpha": alpha,
        "outer_iterations": outer_iter + 1, "total_inner_iterations": total_nit,
        "total_function_evaluations": total_nfev,
        "optimizer_success": bool(last_solution.success),
        "optimizer_messages": messages, "optimizer_statuses": statuses,
        "optimizer_nits": inner_nits, "optimizer_nfevs": inner_nfevs,
        "history": history,
        "column_means": means.ravel(), "column_scales": scales.ravel(),
        "finite": True, **latent_norms,
    }
    if return_raw:
        diagnostics["W_raw"] = W_raw
    return EstimatorResult(W0=W_est, W_raw=W_raw if return_raw else None,
                           Z=Z, L=L, C=C,
                           runtime_seconds=time.perf_counter() - started,
                           diagnostics=diagnostics)


static_latent = fit_static_latent
