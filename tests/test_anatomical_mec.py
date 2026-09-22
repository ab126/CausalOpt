"""Coefficient annotations must follow CPDAG glyphs, not unordered pairs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from causal_opt import fmri_plotting as plotting


@pytest.mark.parametrize("before,after", [
    ((0, 0), (-1, 1)), ((-1, 1), (0, 0)),
    ((-1, 1), (-1, 1)), ((-1, -1), (-1, -1)),
    ((-1, 1), (1, -1)), ((-1, 1), (-1, -1)),
    ((-1, -1), (1, -1)),
])
def test_transition_annotations(monkeypatch, before, after):
    old = np.zeros((3, 3), int)
    new = old.copy()
    old[0, 1], old[1, 0] = before
    new[0, 1], new[1, 0] = after
    monkeypatch.setattr(plotting, "state_cpdags", lambda *args: [old, new])
    monkeypatch.setattr(plotting, "_draw_brain_outline", lambda *args, **kw: None)
    delta = np.zeros((3, 3))
    delta[0, 1], delta[1, 0] = -.2, .8
    sig = np.zeros((3, 3), bool)
    sig[0, 1] = True
    xyz = np.array([[-30, 0, 0], [30, 0, 30], [0, 0, 60]])
    fig, ax = plotting.plot_anatomical_directed_difference(
        delta, xyz, ["1. PAG 1", "2. PAG 2", "3. Isolated"],
        state_matrices=(old, new), significant_mask=sig, title="Custom title",
    )
    changed = before != after and (0, 0) not in (before, after)
    statuses = [after, before] if changed else [before if after == (0, 0) else after]
    directed = [s for s in statuses if s != (-1, -1)]
    maximum = max([.2 if s == (-1, 1) else .8 for s in directed] + [1e-12])
    assert len(ax.patches) == len(statuses)
    for patch, status in zip(ax.patches, statuses):
        width = 2 if status == (-1, -1) else 1.5 + 5 * (.2 if status == (-1, 1) else .8) / maximum
        assert patch.get_linewidth() == pytest.approx(width)
        assert (type(patch.get_arrowstyle()).__name__ == "Curve") == (status == (-1, -1))
    assert [p.get_linestyle() for p in ax.patches] == (["-", "--"] if changed else ["-"])
    stars = [t for t in ax.texts if t.get_text() == "*"]
    assert len(stars) == statuses.count((-1, 1))
    if stars:
        start, end = xyz[0, [0, 2]], xyz[1, [0, 2]]
        vector = end - start
        length = np.linalg.norm(vector)
        perpendicular = np.array([-vector[1], vector[0]]) / length
        lane = (1 if after == (-1, 1) else -1) if changed else 0
        expected = (start + end) / 2 + (2.5 * lane - .05 * length - 1.25) * perpendicular
        np.testing.assert_allclose(stars[0].get_position(), expected)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    category = plotting.cpdag_transitions(old, new)[0]["category"]
    from matplotlib.colors import to_rgba
    assert all(p.get_edgecolor() == to_rgba(plotting.TRANSITION_STYLES[category][0], .82)
               for p in ax.patches)
    assert set(labels) == {
        plotting._ANATOMICAL_TRANSITION_LABELS[category],
        *(["SDV", "Control"] if changed else []),
        *(["Direction resolved"] if directed else []),
        *(["Direction unresolved"] if (-1, -1) in statuses else []),
    }
    assert ax.get_title() == "Custom title"
    assert not len(ax.get_xticks()) and not len(ax.get_yticks())
    assert "PAG 1" in [t.get_text() for t in ax.texts]
    assert ax.collections[0].get_facecolors()[2, 3] == .2
    fig.canvas.draw()
    plt.close(fig)


def test_empty_comparison_has_no_legends_or_star_caption(monkeypatch):
    empty = np.zeros((2, 2))
    monkeypatch.setattr(plotting, "state_cpdags", lambda *args: [empty, empty])
    monkeypatch.setattr(plotting, "_draw_brain_outline", lambda *args, **kw: None)
    fig, ax = plotting.plot_anatomical_directed_difference(
        empty, [[-20, 0, 0], [20, 0, 20]], ["A", "B"],
        state_matrices=(empty, empty),
    )
    assert not fig.legends and not ax.patches
    assert "Stars" not in fig.texts[0].get_text()
    assert ax.get_title() == "Bladder state: SDV vs Control"
    fig.canvas.draw()
    plt.close(fig)


def test_zero_width_and_two_reversal_stars():
    records = [dict(pair=(0, 1), before=(-1, 1), after=(1, -1), category="reversal")]
    fig, ax = plt.subplots()
    plotting._draw_anatomical_transitions(
        ax, records, np.array([[-30, 0], [30, 30]]),
        np.zeros((2, 2)), np.ones((2, 2), bool),
    )
    assert all(p.get_linewidth() == 1.5 for p in ax.patches)
    assert len(ax.texts) == 2
    assert ax.texts[0].get_position() != ax.texts[1].get_position()
    fig.canvas.draw()
    plt.close(fig)
