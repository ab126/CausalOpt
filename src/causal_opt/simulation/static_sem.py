"""Static linear-SEM simulation containers and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import contextlib
import importlib.util
import random
from pathlib import Path

import numpy as np


@dataclass
class StaticSEMData:
    X: np.ndarray
    W_true: np.ndarray
    B_true: np.ndarray
    seed: int
    Z_true: Optional[np.ndarray] = None
    L_true: Optional[np.ndarray] = None
    C_true: Optional[np.ndarray] = None
    W_full: Optional[np.ndarray] = None
    latent_indices: Optional[np.ndarray] = None
    innovations: Optional[np.ndarray] = None


@dataclass
class EstimatorResult:
    W0: Optional[np.ndarray] = None
    W_lags: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    L: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    graph: Any = None
    native_result: Any = None
    runtime_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _notears_utils():
    root = Path(__file__).resolve().parents[3]
    path = root / "notears" / "notears" / "utils.py"
    spec = importlib.util.spec_from_file_location("causal_opt_reference_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference NOTEARS utilities at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _numpy_seed(seed: int):
    """Isolate legacy global RNG use required by the official utilities."""
    state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)
        random.setstate(py_state)


def simulate_static_sem(n: int, d: int, s0: int, seed: int,
                        graph_type: str = "ER", sem_type: str = "gauss",
                        noise_scale=None) -> StaticSEMData:
    utils = _notears_utils()
    with _numpy_seed(seed):
        B = utils.simulate_dag(d, s0, graph_type)
        W = utils.simulate_parameter(B)
        X = utils.simulate_linear_sem(W, n, sem_type, noise_scale)
    return StaticSEMData(X=X, W_true=W, B_true=B.astype(int), seed=int(seed))


def is_dag(W: np.ndarray, threshold: float = 0.0) -> bool:
    A = np.abs(np.asarray(W)) > threshold
    indegree = A.sum(axis=0).astype(int)
    stack = list(np.flatnonzero(indegree == 0))
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        for j in np.flatnonzero(A[i]):
            indegree[j] -= 1
            if indegree[j] == 0:
                stack.append(int(j))
    return seen == A.shape[0]


def acyclicity(W: np.ndarray) -> float:
    from scipy.linalg import expm
    W = np.asarray(W, dtype=float)
    return float(np.trace(expm(W * W)) - W.shape[0])
