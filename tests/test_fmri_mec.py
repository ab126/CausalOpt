"""Structural conversion and plot semantics; no model or bootstrap fitting."""
from itertools import permutations, product

import numpy as np
import pytest

pytest.importorskip("causallearn")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from causal_opt.fmri_mec import (
    dag_to_cpdag, cpdag_transitions, sex_transition_comparison, mec_output_path,
    draw_cpdag_edges,
)
from causal_opt import fmri_plotting as plotting


def matrix(edges, n=3):
    W = np.zeros((n, n))
    for i, j in edges:
        W[i, j] = 1
    return W


def test_chain_fork_and_complete_triangle():
    chain = matrix([(0, 1), (1, 2)])
    fork = matrix([(1, 0), (1, 2)])
    expected = -(chain + chain.T).astype(int)
    np.testing.assert_array_equal(dag_to_cpdag(chain), expected)
    np.testing.assert_array_equal(dag_to_cpdag(fork), expected)
    triangle = matrix([(0, 1), (1, 2), (0, 2)])
    np.testing.assert_array_equal(dag_to_cpdag(triangle), np.eye(3, dtype=int) - 1)


def test_collider_encoding_and_roi_permutations():
    collider = matrix([(0, 1), (2, 1)])
    expected = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])
    np.testing.assert_array_equal(dag_to_cpdag(collider), expected)
    for perm in permutations(range(3)):
        ix = np.ix_(perm, perm)
        np.testing.assert_array_equal(dag_to_cpdag(collider[ix], names=[str(i) for i in perm]), expected[ix])


def test_selected_cycle_invalid_inputs_and_no_mutation():
    W = matrix([(0, 1), (1, 2), (2, 0)])
    W[2, 0] = .1
    original = W.copy()
    dag_to_cpdag(W, .3)
    with pytest.raises(ValueError, match="A -> B -> C -> A"):
        dag_to_cpdag(W, .05, names=["A", "B", "C"])
    np.testing.assert_array_equal(W, original)
    for invalid in (np.zeros((2, 3)), np.array([[np.nan]]), np.ones(3)):
        with pytest.raises(ValueError, match="finite square"):
            dag_to_cpdag(invalid)
    W = np.diag([2., 3., 4.])
    W[0, 1] = 1e-14
    assert not dag_to_cpdag(W, 0).any()


def test_all_four_node_dags_against_enumerated_equivalence_classes():
    """Independent oracle: common skeleton + unshielded colliders defines MEC."""
    import networkx as nx
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    groups = {}
    for codes in product((0, 1, 2), repeat=len(pairs)):
        edges = [(i, j) if code == 1 else (j, i) for (i, j), code in zip(pairs, codes) if code]
        W = matrix(edges, 4)
        if not nx.is_directed_acyclic_graph(nx.from_numpy_array(W, create_using=nx.DiGraph)):
            continue
        skeleton = tuple((W + W.T).astype(bool).ravel())
        colliders = tuple((i, k, j) for i, j in pairs for k in range(4)
                          if W[i, k] and W[j, k] and not (W[i, j] or W[j, i]))
        groups.setdefault((skeleton, colliders), []).append(W)
    for representatives in groups.values():
        expected = np.zeros((4, 4), dtype=int)
        for i, j in pairs:
            if not (representatives[0][i, j] or representatives[0][j, i]):
                continue
            if all(W[i, j] for W in representatives):
                expected[i, j], expected[j, i] = -1, 1
            elif all(W[j, i] for W in representatives):
                expected[i, j], expected[j, i] = 1, -1
            else:
                expected[i, j] = expected[j, i] = -1
        for W in representatives:
            np.testing.assert_array_equal(dag_to_cpdag(W), expected)


def test_transitions_keep_endpoint_histories_and_sex_comparison():
    # Explicit endpoint pairs exercise transitions without inventing a delta DAG.
    old = np.zeros((6, 6), dtype=int)
    new = old.copy()
    statuses = [((0, 0), (-1, -1), "gained"),
                ((-1, 1), (0, 0), "lost"),
                ((-1, -1), (-1, -1), "unchanged"),
                ((-1, 1), (1, -1), "reversal"),
                ((-1, 1), (-1, -1), "reversibility")]
    for j, (before, after, _) in enumerate(statuses, 1):
        old[0, j], old[j, 0] = before
        new[0, j], new[j, 0] = after
    changes = cpdag_transitions(old, new)
    assert [r["category"] for r in changes] == [r[2] for r in statuses]
    assert all(r["category"] == "same_history" for r in sex_transition_comparison(changes, changes))
    # Same category (reversal) but opposite history must remain distinct.
    flipped = [dict(changes[3], before=(1, -1), after=(-1, 1))]
    assert sex_transition_comparison([changes[3]], flipped)[0]["category"] == "different_history"


def test_one_artist_per_cpdag_adjacency():
    fig, ax = plt.subplots()
    E = dag_to_cpdag(matrix([(0, 1), (1, 2), (0, 2)]))
    draw_cpdag_edges(ax, E, np.array([[0, 0], [1, 0], [0, 1]]))
    assert len(ax.patches) == 3
    assert all(type(p.get_arrowstyle()).__name__ == "Curve" for p in ax.patches)
    plt.close(fig)


def test_raw_contrast_never_converted_and_context_required(monkeypatch):
    from causal_opt import fmri_mec
    control = matrix([(0, 1)])
    sdv = matrix([(1, 0)])
    # This contrast contains a 2-cycle, although both constituent states are DAGs.
    contrast = sdv - control
    xyz = np.array([[0, 0, 0], [20, 0, 20], [-20, 0, 20]])
    calls = []
    original = fmri_mec.dag_to_cpdag
    def spy(W, *args, **kwargs):
        calls.append(np.array(W))
        return original(W, *args, **kwargs)
    monkeypatch.setattr(fmri_mec, "dag_to_cpdag", spy)
    monkeypatch.setattr(plotting, "_draw_brain_outline", lambda *a, **k: None)
    with pytest.raises(ValueError, match="state_matrices"):
        plotting.plot_anatomical_directed_difference(contrast, xyz, ["A", "B", "C"])
    assert not calls
    fig, ax = plotting.plot_anatomical_graph_difference(
        control, sdv, xyz, ["A", "B", "C"], min_abs_change=100,
        significant_mask=np.ones((3, 3), bool),
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], control)
    np.testing.assert_array_equal(calls[1], sdv)
    assert len(ax.patches) == 1  # unchanged reversible adjacency, ignoring delta filter
    assert not any(t.get_text() == "*" for t in ax.texts)
    plt.close(fig)


def test_mec_ignores_coefficient_significance_and_legacy_keeps_it(monkeypatch, tmp_path):
    captures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kwargs: captures.append((self, path)))
    W = matrix([(0, 1), (1, 2)])
    sig = np.ones((3, 3), bool)
    path = tmp_path / "graph.png"
    plotting.plot_graph_comparison(W, W, ["A", "B", "C"], path, significant_control=sig)
    fig, written = captures[-1]
    assert written == tmp_path / "graph_mec.png"
    assert all(p.get_linewidth() == 2 for ax in fig.axes for p in ax.patches)
    assert not any("bootstrap" in t.get_text() for t in fig.findobj(matplotlib.text.Text))
    plotting.plot_graph_comparison(W, W, ["A", "B", "C"], path,
                                  significant_control=sig, show_mec=False)
    fig, written = captures[-1]
    assert written == path
    assert any("bootstrap p_unc" in t.get_text() for t in fig.texts)
    assert any(t.arrow_patch.get_linewidth() == 3 for t in fig.axes[0].texts if getattr(t, "arrow_patch", None))
    assert mec_output_path(tmp_path / "graph_mec.png") == tmp_path / "graph_mec.png"


def test_legacy_anatomical_significance_below_threshold_is_preserved(monkeypatch):
    monkeypatch.setattr(plotting, "_draw_brain_outline", lambda *a, **k: None)
    W = matrix([(0, 1)]) * .01
    sig = W != 0
    fig, ax = plotting.plot_anatomical_directed_difference(
        W, np.array([[0, 0, 0], [20, 0, 20], [-20, 0, 20]]), ["A", "B", "C"],
        show_mec=False, min_abs_change=.3, significant_mask=sig,
    )
    assert len(ax.patches) == 1
    assert any(t.get_text() == "*" for t in ax.texts)
    plt.close(fig)
