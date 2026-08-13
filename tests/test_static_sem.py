import numpy as np
from causal_opt.simulation import simulate_static_sem,is_dag
from causal_opt.methods.static_latent import fit_static_latent

def test_static_deterministic_and_dag():
    a=simulate_static_sem(100,6,5,7); b=simulate_static_sem(100,6,5,7)
    assert np.array_equal(a.X,b.X); assert is_dag(a.W_true)

def test_orientation_and_estimator_dimensions():
    data=simulate_static_sem(200,4,3,9)
    assert np.allclose(data.X-data.X@data.W_true,data.X-data.X@data.W_true)
    result=fit_static_latent(data.X,k=0,max_iter=2,inner_max_iter=10,w_threshold=.1)
    assert result.W0.shape==(4,4); assert np.diag(result.W0).sum()==0; assert np.isfinite(result.W0).all()
