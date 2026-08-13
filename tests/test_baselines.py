import numpy as np
import pytest
from causal_opt.baselines.notears import fit_notears
from causal_opt.simulation import simulate_static_sem,is_dag

def test_notears_smoke():
    data=simulate_static_sem(200,5,5,1)
    result=fit_notears(data.X,lambda1=.1,max_iter=10)
    assert result.W0.shape==(5,5); assert is_dag(result.W0)

@pytest.mark.integration
@pytest.mark.parametrize("module",["causallearn","causalnex","tigramite"])
def test_optional_baseline_dependencies(module):
    pytest.importorskip(module)
