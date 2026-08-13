"""Static SEMs with observed latent-root confounding."""
from __future__ import annotations
import numpy as np
from .static_sem import StaticSEMData, _notears_utils, _numpy_seed


def _loading_matrix(rng, d, k, density, min_children, weight_ranges):
    L = np.zeros((d, k))
    for q in range(k):
        m = max(min_children, int(round(d * density)))
        m = min(d, m)
        children = rng.choice(d, size=m, replace=False)
        lo, hi = weight_ranges[rng.integers(len(weight_ranges))]
        L[children, q] = rng.uniform(lo, hi, size=m)
    return L


def simulate_static_latent_sem(n: int, d_obs: int, s0: int, k: int, seed: int,
                               graph_type: str = "ER", sem_type: str = "gauss",
                               latent_mode: str = "root_confounders",
                               latent_density: float = .3,
                               min_latent_children: int = 2,
                               loading_ranges=((-2., -.5), (.5, 2.))):
    if k < 0 or k > d_obs + k:
        raise ValueError("k must be nonnegative")
    utils, rng = _notears_utils(), np.random.default_rng(seed + 7919)
    with _numpy_seed(seed):
        if latent_mode == "root_confounders":
            B_obs = utils.simulate_dag(d_obs, s0, graph_type)
            W_obs = utils.simulate_parameter(B_obs)
            L = _loading_matrix(rng, d_obs, k, latent_density,
                                min_latent_children, loading_ranges) if k else None
            W_full = np.zeros((d_obs + k, d_obs + k))
            W_full[:d_obs, :d_obs] = W_obs
            if k:
                W_full[d_obs:, :d_obs] = L.T
            X_full = utils.simulate_linear_sem(W_full, n, sem_type)
            latent_idx = np.arange(d_obs, d_obs + k, dtype=int)
        elif latent_mode == "random_hidden_nodes":
            total = d_obs + k
            B_full = utils.simulate_dag(total, s0 + 2 * k, graph_type)
            W_full = utils.simulate_parameter(B_full)
            latent_idx = np.sort(rng.choice(total, k, replace=False))
            observed = np.setdiff1d(np.arange(total), latent_idx)
            X_full0 = utils.simulate_linear_sem(W_full, n, sem_type)
            X_full = np.column_stack((X_full0[:, observed], X_full0[:, latent_idx]))
            P = np.r_[observed, latent_idx]
            W_full = W_full[np.ix_(P, P)]
            W_obs = W_full[:d_obs, :d_obs]
            B_obs = W_obs != 0
            L = W_full[d_obs:, :d_obs].T if k else None
            latent_idx = np.arange(d_obs, total)
        else:
            raise ValueError("unknown latent_mode")
    X = X_full[:, :d_obs]
    Z = X_full[:, d_obs:] if k else None
    C = Z @ L.T if k else None
    E = X - X @ W_obs - C if k else X - X @ W_obs
    return StaticSEMData(X, W_obs, np.asarray(B_obs, int), int(seed), Z, L, C,
                         W_full, latent_idx if k else None, E)
