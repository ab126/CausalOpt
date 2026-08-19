import shutil
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.integration
def test_dynotears_optional_smoke():
    from causal_opt.baselines.dynotears import fit_dynotears
    python=Path(__file__).resolve().parents[1]/".venv-dynotears"/"Scripts"/"python.exe"
    if not python.exists(): pytest.skip("isolated DYNOTEARS environment unavailable")
    result=fit_dynotears(np.random.default_rng(1).normal(size=(50,3)),p=1)
    assert result.W0.shape==(3,3) and result.W_lags.shape==(1,3,3)
    assert "edges" in result.native_result and result.diagnostics["causalnex_version"]=="0.12.1"

@pytest.mark.parametrize("p",[1,2])
def test_dynotears_lag_stacking(p):
    from causal_opt.baselines.dynotears import _stack_lags
    X=np.arange(24).reshape(8,3); current,lags=_stack_lags(X,p)
    assert np.array_equal(current,X[p:])
    assert np.array_equal(lags,np.concatenate([X[p-q:len(X)-q] for q in range(1,p+1)],axis=1))

def test_dynotears_edge_orientation_and_self_lag():
    from causal_opt.baselines.dynotears import _edges_to_matrices
    edges=[{"source":"0_lag0","destination":"1_lag0","weight":.7},
           {"source":"2_lag1","destination":"0_lag0","weight":-.4},
           {"source":"1_lag2","destination":"1_lag0","weight":.3}]
    W0,Wlags=_edges_to_matrices(edges,3,2)
    assert W0[0,1]==.7 and Wlags[0,2,0]==-.4 and Wlags[1,1,1]==.3

def test_dynotears_unavailable_environment():
    from causal_opt.baselines.dynotears import fit_dynotears
    with pytest.raises(ImportError,match="side environment unavailable"):
        fit_dynotears(np.ones((10,3)),p=1,python_executable="definitely-missing-python.exe")

def test_dynotears_worker_argument_handling():
    import subprocess,sys
    worker=Path(__file__).resolve().parents[1]/"scripts"/"baseline_workers"/"run_dynotears.py"
    proc=subprocess.run([sys.executable,str(worker),"--help"],capture_output=True,text=True)
    assert proc.returncode==0 and "--metadata" in proc.stdout

@pytest.mark.integration
def test_lpcmci_optional_smoke():
    pytest.importorskip("tigramite")
    from causal_opt.baselines.lpcmci import fit_lpcmci
    result=fit_lpcmci(np.random.default_rng(1).normal(size=(50,3)),tau_max=1)
    assert result.native_result is not None

@pytest.mark.integration
def test_liegeois_requires_matlab_and_repo():
    from causal_opt.baselines.liegeois import matlab_engine_available
    if not matlab_engine_available(): pytest.skip("MATLAB Engine for Python is not configured")
    repo=Path(__file__).resolve().parents[1]/"external"/"SparseLowRankIdentification"
    if not repo.exists(): pytest.skip("authors' repository is not checked out")
    from causal_opt.baselines.liegeois import fit_liegeois
    result=fit_liegeois(np.random.default_rng(1).normal(size=(80,3)),p=1,backend="matlab_engine",repo_path=repo,maxiter=5,compute_relative_duality_gap=False)
    xs=result.native_result["x_sol"]
    expected={"L":(6,6),"S":(3,3,2),"Omega":(3,3),"Delta":(6,6)}
    for name in ("L","S","Omega","Delta"):
        value=np.asarray(xs[name])
        assert value.shape==expected[name]
        assert np.isfinite(value).all()
    assert np.array_equal(np.asarray(xs["Omega"])!=0,np.asarray(result.graph)!=0)
    assert _git_status(repo)==""

@pytest.mark.integration
def test_liegeois_subprocess_optional_smoke():
    from causal_opt.baselines.liegeois import matlab_subprocess_available
    if not matlab_subprocess_available(): pytest.skip("MATLAB executable is unavailable")
    repo=Path(__file__).resolve().parents[1]/"external"/"SparseLowRankIdentification"
    if not repo.exists(): pytest.skip("authors' repository is not checked out")
    from causal_opt.baselines.liegeois import fit_liegeois
    result=fit_liegeois(np.random.default_rng(1).normal(size=(80,3)),p=1,
        backend="matlab_subprocess",repo_path=repo,maxiter=5,
        compute_relative_duality_gap=False,timeout=300)
    assert all(name in result.native_result["x_sol"] for name in ("L","S","Omega","Delta","h"))
    assert result.diagnostics["return_code"]==0
    assert result.diagnostics["matlab_version"]

def test_liegeois_engine_unavailable_is_explicit():
    from causal_opt.baselines.liegeois import matlab_engine_available
    if matlab_engine_available(): pytest.skip("MATLAB Engine is configured")
    from causal_opt.baselines.liegeois import fit_liegeois
    repo=Path(__file__).resolve().parents[1]/"external"/"SparseLowRankIdentification"
    with pytest.raises(ImportError,match="MATLAB Engine"):
        fit_liegeois(np.ones((20,3)),p=1,backend="matlab_engine",repo_path=repo)

def test_liegeois_subprocess_unavailable_is_explicit():
    from causal_opt.baselines.liegeois import fit_liegeois
    repo=Path(__file__).resolve().parents[1]/"external"/"SparseLowRankIdentification"
    with pytest.raises(ImportError,match="MATLAB executable"):
        fit_liegeois(np.ones((20,3)),p=1,backend="matlab_subprocess",repo_path=repo,
                     matlab_executable="definitely-missing-matlab.exe")

def test_liegeois_upstream_provenance_matches_checkout():
    from causal_opt.baselines.liegeois import UPSTREAM_COMMIT,_repo_commit
    repo=Path(__file__).resolve().parents[1]/"external"/"SparseLowRankIdentification"
    if not repo.exists(): pytest.skip("authors' repository is not checked out")
    assert _repo_commit(repo)==UPSTREAM_COMMIT

def _git_status(repo):
    import subprocess
    return subprocess.run(["git","-C",str(repo),"status","--porcelain"],
                          capture_output=True,text=True,check=True).stdout.strip()
