"""Stable structural VAR simulation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .static_sem import _notears_utils, _numpy_seed


@dataclass
class TimeSeriesSEMData:
    X: np.ndarray
    W0_true: np.ndarray
    W_lags_true: np.ndarray
    seed: int
    spectral_radius: float
    condition_number: float
    innovations: Optional[np.ndarray] = None
    Z_true: Optional[np.ndarray] = None
    L_true: Optional[np.ndarray] = None
    C_true: Optional[np.ndarray] = None
    Z_aligned_true: Optional[np.ndarray] = None


def companion_spectral_radius(W0, W_lags):
    p, d, _ = W_lags.shape
    if p == 0:
        return 0.0
    inv = np.linalg.inv(np.eye(d) - W0)
    A = [W_lags[q] @ inv for q in range(p)]
    comp = np.zeros((p*d, p*d))
    comp[:d, :] = np.concatenate([a.T for a in A], axis=1)
    if p > 1:
        comp[d:, :-d] = np.eye((p-1)*d)
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


def simulate_svar(T: int, d: int, p: int, seed: int, s0: int | None = None,
                  lag_sparsity: float = .15, burn_in: int = 200,
                  stability_radius: float = .95, noise_scale: float = 1.,
                  store_innovations: bool = False, k: int = 0,
                  latent_mode: str = "iid", phi_z: float = .5,
                  latent_density: float = .3, latent_process: str | None = None,
                  min_latent_children: int = 2,
                  loading_ranges=((-1.5, -.5), (.5, 1.5)),
                  condition_number_max: float = 1e8) -> TimeSeriesSEMData:
    if latent_process is not None: latent_mode = latent_process
    if p < 0 or k < 0 or k > d or T <= p or burn_in < 0:
        raise ValueError("require p,k >= 0 and T > p")
    if not 0 < stability_radius < 1 or noise_scale <= 0:
        raise ValueError("stability_radius must be in (0,1) and noise_scale positive")
    if latent_mode not in {"iid", "ar1"}:
        raise ValueError("latent_process must be 'iid' or 'ar1'")
    if latent_mode == "ar1" and abs(phi_z) >= 1:
        raise ValueError("AR(1) latent process must be stationary")
    utils, rng = _notears_utils(), np.random.default_rng(seed)
    with _numpy_seed(seed):
        B0 = utils.simulate_dag(d, d if s0 is None else s0, "ER")
        W0 = utils.simulate_parameter(B0)
    condition_number = float(np.linalg.cond(np.eye(d) - W0))
    if not np.isfinite(condition_number) or condition_number > condition_number_max:
        raise RuntimeError("I-W0 is pathologically ill-conditioned")
    Wlags = np.zeros((p, d, d))
    for q in range(p):
        mask = rng.random((d, d)) < lag_sparsity
        Wlags[q] = mask * rng.uniform(-.4, .4, (d, d))
    radius = companion_spectral_radius(W0, Wlags)
    for _ in range(100):
        if radius < stability_radius:
            break
        Wlags *= min(.95, stability_radius / max(radius, 1e-12) * .95)
        radius = companion_spectral_radius(W0, Wlags)
    if radius >= stability_radius:
        raise RuntimeError("could not stabilize lag matrices")
    L = None
    if k:
        L = np.zeros((d, k))
        for q in range(k):
            count=max(min_latent_children,min(d,round(d*latent_density)))
            if count>d: raise ValueError("min_latent_children cannot exceed d")
            children = rng.choice(d, count, replace=False)
            interval=loading_ranges[int(rng.integers(len(loading_ranges)))]
            L[children, q] = rng.uniform(interval[0],interval[1],len(children))
    total = T + burn_in
    X = np.zeros((total, d)); Z = np.zeros((total, k)) if k else None
    E = rng.normal(scale=noise_scale, size=(total, d))
    inv = np.linalg.inv(np.eye(d) - W0)
    for t in range(total):
        if k:
            eta = rng.normal(size=k)
            Z[t] = eta if latent_mode == "iid" or t == 0 else phi_z * Z[t-1] + eta
        rhs = E[t].copy()
        for q in range(1, min(p, t) + 1):
            rhs += X[t-q] @ Wlags[q-1]
        if k:
            rhs += Z[t] @ L.T
        X[t] = rhs @ inv
    sl = slice(burn_in, total)
    Xout=X[sl]; Zout=Z[sl] if k else None
    Zaligned=Zout[p:] if k else None
    C = Zaligned @ L.T if k else None
    if not np.isfinite(X[sl]).all():
        raise FloatingPointError("non-finite simulated samples")
    return TimeSeriesSEMData(Xout, W0, Wlags, int(seed), radius, condition_number,
                             E[sl] if store_innovations else None,
                             Zout, L, C, Zaligned)


def simulate_dynamic_latent_sem(*args, **kwargs):
    return simulate_svar(*args, **kwargs)
