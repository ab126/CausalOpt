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
    b=simulate_svar(40,4,1,3,k=2); assert b.C_true.shape==(39,4); assert b.Z_aligned_true.shape==(39,2); assert np.linalg.matrix_rank(b.C_true)<=2

def test_different_seed_and_latent_modes_and_dimensions():
    a=simulate_svar(60,5,2,3,k=2,min_latent_children=3,latent_process="iid")
    b=simulate_svar(60,5,2,4,k=2,min_latent_children=3,latent_process="ar1",phi_z=.6)
    assert not np.array_equal(a.X,b.X) and a.W_lags_true.shape==(2,5,5)
    assert np.all(np.count_nonzero(a.L_true,axis=0)>=3) and a.C_true.shape==(58,5)
    assert np.isfinite(a.condition_number) and np.max(np.abs(np.diag(a.W0_true)))==0
    with pytest.raises(ValueError): simulate_svar(20,4,1,1,k=1,latent_process="ar1",phi_z=1.)

def test_latent_structural_equation_exact():
    a=simulate_svar(80,5,2,7,k=2,burn_in=40,store_innovations=True,min_latent_children=2)
    errors=[]
    for t in range(2,len(a.X)):
        r=a.X[t]-a.X[t]@a.W0_true-a.Z_true[t]@a.L_true.T-a.innovations[t]
        for q in range(1,3): r-=a.X[t-q]@a.W_lags_true[q-1]
        errors.append(np.max(np.abs(r)))
    assert max(errors)<1e-9
