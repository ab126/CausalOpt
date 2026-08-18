import numpy as np
import pytest

from causal_opt.methods.static_latent import (
    _acyclicity,
    _bounds,
    _objective,
    _pack_parameters,
    _unpack_parameters,
    fit_static_latent,
)
from causal_opt.simulation import is_dag, simulate_static_sem


@pytest.mark.parametrize("k", [0, 2])
def test_pack_unpack_round_trip(k):
    rng = np.random.default_rng(4)
    n, d = 8, 4
    Wp, Wn = rng.random((d, d)), rng.random((d, d))
    Z = rng.normal(size=(n, k)) if k else None
    L = rng.normal(size=(d, k)) if k else None
    values = _unpack_parameters(_pack_parameters(Wp, Wn, Z, L), n, d, k)
    expected = (Wp, Wn, Z, L)
    for actual, wanted in zip(values, expected):
        assert actual is wanted is None or np.array_equal(actual, wanted)


def test_acyclicity_empty_cycle_and_gradient():
    empty = np.zeros((4, 4))
    assert _acyclicity(empty)[0] == 0.0
    cyclic = empty.copy()
    cyclic[0, 1], cyclic[1, 2], cyclic[2, 0] = 0.7, -0.8, 0.6
    h, gradient = _acyclicity(cyclic)
    assert h > 0
    eps = 1e-6
    numeric = np.empty_like(cyclic)
    for index in np.ndindex(cyclic.shape):
        plus, minus = cyclic.copy(), cyclic.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = (_acyclicity(plus)[0] - _acyclicity(minus)[0]) / (2 * eps)
    assert np.max(np.abs(gradient - numeric)) < 1e-7


def test_all_joint_objective_gradients():
    rng = np.random.default_rng(7)
    n, d, k = 8, 4, 2
    X = rng.normal(size=(n, d))
    Wp, Wn = rng.uniform(0.05, 0.2, size=(2, d, d))
    Z, L = rng.normal(scale=0.2, size=(n, k)), rng.normal(scale=0.2, size=(d, k))
    parameters = _pack_parameters(Wp, Wn, Z, L)
    args = (X, k, 0.13, 0.17, 1.7, 0.3)
    _, analytic = _objective(parameters, *args)
    eps = 1e-6
    numeric = np.empty_like(parameters)
    for i in range(parameters.size):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric[i] = (_objective(plus, *args)[0] - _objective(minus, *args)[0]) / (2 * eps)
    sizes = [d * d, d * d, n * k, d * k]
    offset = 0
    for size in sizes:
        relative = np.linalg.norm(analytic[offset:offset + size] - numeric[offset:offset + size]) / max(
            1.0, np.linalg.norm(analytic[offset:offset + size]), np.linalg.norm(numeric[offset:offset + size]))
        assert relative < 1e-6
        offset += size


def test_bounds_fix_diagonal_and_leave_factors_free():
    n, d, k = 8, 4, 2
    bounds = _bounds(n, d, k)
    assert len(bounds) == 2 * d * d + n * k + d * k
    for block in range(2):
        for i in range(d):
            assert bounds[block * d * d + i * d + i] == (0.0, 0.0)
    assert all(bound == (None, None) for bound in bounds[2 * d * d:])


def test_input_validation_and_nonmutating_centering():
    X = np.arange(20.0).reshape(5, 4)
    original = X.copy()
    result = fit_static_latent(X, k=0, max_outer_iter=1, inner_max_iter=2)
    assert np.array_equal(X, original)
    assert np.array_equal(result.diagnostics["column_means"], original.mean(axis=0))
    with pytest.raises(ValueError):
        fit_static_latent(np.ones(4))
    with pytest.raises(ValueError):
        fit_static_latent(np.full((4, 3), np.nan))
    with pytest.raises(ValueError):
        fit_static_latent(np.ones((4, 3)), k=4)
    with pytest.raises(ValueError):
        fit_static_latent(np.ones((4, 3)), k=1, initialization="zeros")


def test_k_zero_smoke_is_dag_and_raw_diagonal_is_bounded():
    data = simulate_static_sem(200, 5, 5, 1)
    result = fit_static_latent(data.X, k=0, lambda_w=0.1,
                               max_outer_iter=20, inner_max_iter=200)
    assert result.Z is result.L is result.C is None
    assert np.array_equal(np.diag(result.W_raw), np.zeros(5))
    assert result.diagnostics["h_raw"] < 1e-6
    assert is_dag(result.W0)


def test_latent_initialization_does_not_collapse():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 4))
    result = fit_static_latent(X, k=2, lambda_w=0.1, lambda_latent=0.1,
                               max_outer_iter=2, inner_max_iter=30, random_state=3)
    assert result.C.shape == X.shape
    assert np.linalg.norm(result.C) > 1e-6
    assert result.diagnostics["effective_rank_C"] > 0
