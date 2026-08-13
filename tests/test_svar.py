import numpy as np
import pytest
from causal_opt.simulation import simulate_svar,is_dag

@pytest.mark.parametrize("p",[1,2,3])
def test_svar_stable_finite_and_exact(p):
    a=simulate_svar(120,5,p,4,burn_in=50,store_innovations=True)
    b=simulate_svar(120,5,p,4,burn_in=50,store_innovations=True)
    assert np.array_equal(a.X,b.X); assert is_dag(a.W0_true)
    assert a.spectral_radius<.95 and np.isfinite(a.X).all()
    for t in range(p,120):
        r=a.X[t]-a.X[t]@a.W0_true-a.innovations[t]
        for q in range(1,p+1): r-=a.X[t-q]@a.W_lags_true[q-1]
        assert np.max(np.abs(r))<1e-9

def test_dynamic_latent_and_p_zero():
    a=simulate_svar(40,4,0,3,k=0); assert a.W_lags_true.shape==(0,4,4); assert a.Z_true is None
    b=simulate_svar(40,4,1,3,k=2); assert b.C_true.shape==(40,4); assert np.linalg.matrix_rank(b.C_true)<=2
