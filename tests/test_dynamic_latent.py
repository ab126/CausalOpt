import numpy as np
import pytest
from causal_opt.methods.dynamic_latent import (_bounds,_objective,_pack_parameters,
    _unpack_parameters,fit_dynamic_latent,lagged_views)
from causal_opt.methods.static_latent import _acyclicity

@pytest.mark.parametrize("p,k",[(1,0),(2,0),(1,2),(3,2)])
def test_pack_round_trip(p,k):
    rng=np.random.default_rng(2); m,d=9,4
    arrays=[rng.random((d,d)),rng.random((d,d)),rng.random((p,d,d)),rng.random((p,d,d))]
    Z=rng.normal(size=(m,k)) if k else None; L=rng.normal(size=(d,k)) if k else None
    got=_unpack_parameters(_pack_parameters(*arrays,Z,L),m,d,p,k)
    for a,b in zip(got,(*arrays,Z,L)): assert a is b is None or np.array_equal(a,b)

def test_lag_alignment_p1_p2():
    X=np.arange(18).reshape(6,3)
    X0,L=lagged_views(X,1); assert np.array_equal(X0,X[1:]); assert np.array_equal(L[0],X[:-1])
    X0,L=lagged_views(X,2); assert np.array_equal(X0,X[2:]); assert np.array_equal(L[0],X[1:-1]); assert np.array_equal(L[1],X[:-2])

def test_bounds_fix_only_W0_diagonal():
    m,d,p,k=10,4,2,2; b=_bounds(m,d,p,k)
    assert len(b)==2*d*d+2*p*d*d+m*k+d*k
    for block in range(2):
        for i in range(d): assert b[block*d*d+i*d+i]==(0.,0.)
    assert all(x==(0.,None) for x in b[2*d*d:2*d*d+2*p*d*d])

@pytest.mark.parametrize("T,p,k",[(12,1,0),(14,2,2)])
def test_dynamic_gradient(T,p,k):
    rng=np.random.default_rng(7); d=4; X=rng.normal(size=(T,d)); X0,Xl=lagged_views(X,p); m=len(X0)
    arrays=[rng.uniform(.05,.2,(d,d)),rng.uniform(.05,.2,(d,d)),rng.uniform(.05,.2,(p,d,d)),rng.uniform(.05,.2,(p,d,d))]
    Z=rng.normal(scale=.2,size=(m,k)) if k else None; L=rng.normal(scale=.2,size=(d,k)) if k else None
    v=_pack_parameters(*arrays,Z,L); args=(X0,Xl,k,.13,.11,.17,1.7,.3); analytic=_objective(v,*args)[1]
    eps=1e-6; numeric=np.empty_like(v)
    for i in range(v.size):
        plus=v.copy(); minus=v.copy(); plus[i]+=eps; minus[i]-=eps
        numeric[i]=(_objective(plus,*args)[0]-_objective(minus,*args)[0])/(2*eps)
    assert np.max(np.abs(analytic-numeric))<1e-7
    sizes=[d*d,d*d]+[d*d]*p+[d*d]*p+([m*k,d*k] if k else [])
    offset=0
    for size in sizes:
        rel=np.linalg.norm(analytic[offset:offset+size]-numeric[offset:offset+size])/max(1.,np.linalg.norm(analytic[offset:offset+size]),np.linalg.norm(numeric[offset:offset+size]))
        assert rel<1e-7; offset+=size

def test_outputs_raw_threshold_and_latent_determinism():
    rng=np.random.default_rng(4); X=rng.normal(size=(80,4))
    a=fit_dynamic_latent(X,1,k=2,max_outer_iter=2,inner_max_iter=20,random_state=9)
    b=fit_dynamic_latent(X,1,k=2,max_outer_iter=2,inner_max_iter=20,random_state=9)
    assert a.W0.shape==(4,4) and a.W_lags.shape==(1,4,4) and a.C.shape==(79,4)
    assert np.array_equal(np.diag(a.W_raw),np.zeros(4)); assert np.array_equal(a.Z,b.Z)
    assert not np.shares_memory(a.W0,a.W_raw) and not np.shares_memory(a.W_lags,a.W_lags_raw)
    z=fit_dynamic_latent(X,1,k=0,max_outer_iter=1,inner_max_iter=5)
    assert z.Z is z.L is z.C is None

def test_acyclicity_gradient_shared_helper():
    rng=np.random.default_rng(3); W=rng.normal(scale=.2,size=(4,4)); eps=1e-6; h,g=_acyclicity(W); num=np.empty_like(W)
    for idx in np.ndindex(W.shape):
        a=W.copy(); b=W.copy(); a[idx]+=eps; b[idx]-=eps; num[idx]=(_acyclicity(a)[0]-_acyclicity(b)[0])/(2*eps)
    assert np.max(np.abs(g-num))<1e-7
