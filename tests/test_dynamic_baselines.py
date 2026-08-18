import shutil
import numpy as np
import pytest

@pytest.mark.integration
def test_dynotears_optional_smoke():
    pytest.importorskip("causalnex")
    from causal_opt.baselines.dynotears import fit_dynotears
    result=fit_dynotears(np.random.default_rng(1).normal(size=(50,3)),p=1)
    assert result.W0.shape==(3,3) and result.W_lags.shape==(1,3,3)

@pytest.mark.integration
def test_lpcmci_optional_smoke():
    pytest.importorskip("tigramite")
    from causal_opt.baselines.lpcmci import fit_lpcmci
    result=fit_lpcmci(np.random.default_rng(1).normal(size=(50,3)),tau_max=1)
    assert result.native_result is not None

@pytest.mark.integration
def test_liegeois_requires_matlab_and_repo():
    if shutil.which("matlab") is None: pytest.skip("MATLAB is not configured")
    pytest.skip("requires an explicitly configured authors' repository and entrypoint")
