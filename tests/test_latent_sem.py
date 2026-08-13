import numpy as np
from causal_opt.simulation import simulate_static_latent_sem

def test_latent_sem_properties():
    a=simulate_static_latent_sem(200,8,7,2,11,min_latent_children=2)
    b=simulate_static_latent_sem(200,8,7,2,11,min_latent_children=2)
    assert np.array_equal(a.X,b.X)
    assert np.linalg.matrix_rank(a.C_true)<=2
    assert np.all(np.count_nonzero(a.L_true,axis=0)>=2)
    assert np.max(np.abs(a.X-a.X@a.W_true-a.C_true-a.innovations))<1e-10

def test_k_zero_cleanly_removes_latents():
    a=simulate_static_latent_sem(20,5,3,0,1)
    assert a.Z_true is a.L_true is a.C_true is None
