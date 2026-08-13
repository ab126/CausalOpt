import numpy as np
from causal_opt.evaluation.dag_metrics import dag_metrics
from causal_opt.evaluation.latent_metrics import subspace_error

def test_hand_constructed_dag_metrics():
    t=np.zeros((3,3)); t[0,1]=1; t[1,2]=1
    m=dag_metrics(t,t,threshold=0)
    assert m["precision"]==m["recall"]==m["f1"]==1 and m["shd"]==0

def test_subspace_invariance():
    rng=np.random.default_rng(2); L=rng.normal(size=(8,3))
    Q,_=np.linalg.qr(rng.normal(size=(3,3)))
    assert subspace_error(L,L@Q)<1e-10
    assert subspace_error(L,L[:,[2,0,1]]*np.array([-1,1,-1]))<1e-10
