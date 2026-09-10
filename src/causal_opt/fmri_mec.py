"""CPDAGs of selected observed directed structures, not latent-model PAGs.

Weights use W[i, j] = i -> j. Endpoints use causal-learn's encoding:
E[i, j] is the endpoint at i: -1 = tail, +1 = arrow, 0 = absent.
Thus i -> j is (-1, +1), and i -- j is (-1, -1).
"""
from pathlib import Path

import numpy as np


def mec_output_path(path, show_mec=True):
    """Keep structural figures separate, including direct notebook callers."""
    if path is None or not show_mec:
        return path
    path = Path(path)
    return path if path.stem.endswith("_mec") else path.with_name(path.stem + "_mec" + path.suffix)


def dag_to_cpdag(weights, dag_threshold=0.3, *, names=None, zero_tol=1e-12):
    """Convert the complete threshold-selected DAG without changing inputs.

    Select off-diagonal abs(W) >= dag_threshold and abs(W) > zero_tol.
    Thresholding changes the selected structure and can change its MEC.
    No coefficient significance mask enters this structural conversion.
    """
    import networkx as nx

    W = np.array(weights, dtype=float, copy=True)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or not np.isfinite(W).all():
        raise ValueError("DAG weights must be a finite square matrix")
    if not np.isfinite(dag_threshold) or dag_threshold < 0 or not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("dag_threshold and zero_tol must be finite and nonnegative")
    labels = tuple(map(str, names)) if names is not None else tuple(map(str, range(len(W))))
    if len(labels) != len(W):
        raise ValueError("ROI names must match the matrix order and size")
    adjacency = (np.abs(W) >= dag_threshold) & (np.abs(W) > zero_tol)
    np.fill_diagonal(adjacency, False)
    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        description = " -> ".join([labels[cycle[0][0]]] + [labels[j] for _, j in cycle])
        raise ValueError(
            f"Selected structure is cyclic at dag_threshold={dag_threshold:g}: {description}. "
            "CPDAG conversion requires a DAG; no edges were removed. "
            "Check the fitted state matrix/threshold or use show_mec=False."
        )
    try:
        from causallearn.graph.Dag import Dag
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.DAG2CPDAG import dag2cpdag
    except ImportError as exc:
        raise ImportError("MEC plots require causal-learn: python -m pip install causal-learn; "
                          "or select --no-show-mec for the original weighted plots") from exc
    # Unique internal names avoid accidental merging of duplicate display labels.
    nodes = [GraphNode(f"ROI_{i}") for i in range(len(W))]
    dag = Dag(nodes)
    for i, j in np.argwhere(adjacency):
        dag.add_directed_edge(nodes[i], nodes[j])
    endpoints = np.array(dag2cpdag(dag).graph, dtype=np.int8, copy=True)
    if not np.array_equal(endpoints != 0, adjacency | adjacency.T):
        raise RuntimeError("DAG2CPDAG returned an unexpected skeleton")
    for i, j in zip(*np.triu_indices(len(W), 1)):
        if (int(endpoints[i, j]), int(endpoints[j, i])) not in ((0, 0), (-1, -1), (-1, 1), (1, -1)):
            raise RuntimeError("DAG2CPDAG returned a non-CPDAG endpoint encoding")
    return endpoints


def cpdag_transitions(before, after):
    """One record per unordered pair, retaining explicit before/after endpoints."""
    if before.shape != after.shape:
        raise ValueError("State CPDAGs must have matching ROI order and shape")
    records = []
    for i, j in zip(*np.triu_indices(len(before), 1)):
        old = (int(before[i, j]), int(before[j, i]))
        new = (int(after[i, j]), int(after[j, i]))
        if old == new == (0, 0):
            continue
        if old == (0, 0):
            category = "gained"
        elif new == (0, 0):
            category = "lost"
        elif old == new:
            category = "unchanged"
        elif old != (-1, -1) and new != (-1, -1):
            category = "reversal"
        else:
            category = "reversibility"
        records.append(dict(pair=(i, j), before=old, after=new, category=category))
    return records


def state_cpdags(state_matrices, dag_threshold, names):
    if state_matrices is None or len(state_matrices) not in (2, 4):
        raise ValueError(
            "show_mec=True requires state_matrices=(Control, SDV) or "
            "(Male Control, Male SDV, Female Control, Female SDV). "
            "A raw difference/interaction matrix is not a DAG. "
            "Pass constituent state matrices or use show_mec=False."
        )
    graphs = [dag_to_cpdag(W, dag_threshold, names=names) for W in state_matrices]
    if any(g.shape != graphs[0].shape for g in graphs):
        raise ValueError("Constituent states must have the same ROI order and size")
    return graphs


def sex_transition_comparison(male, female):
    """Compare exact ordered endpoint histories on unordered ROI pairs."""
    m = {r["pair"]: r for r in male}
    f = {r["pair"]: r for r in female}
    records = []
    for pair in sorted(m.keys() | f.keys()):
        if pair not in m:
            category = "female_only"
        elif pair not in f:
            category = "male_only"
        elif (m[pair]["before"], m[pair]["after"]) == (f[pair]["before"], f[pair]["after"]):
            category = "same_history"
        else:
            category = "different_history"
        records.append(dict(pair=pair, category=category))
    return records


TRANSITION_STYLES = {
    "gained": ("#009E73", "Adjacency gained"),
    "lost": ("#D55E00", "Adjacency lost"),
    "unchanged": ("#999999", "Retained: same endpoints"),
    "reversal": ("#CC79A7", "Retained: compelled reversal"),
    "reversibility": ("#0072B2", "Retained: directed ↔ reversible"),
}
SEX_STYLES = {
    "same_history": ("#999999", "Same endpoint history"),
    "different_history": ("#7B3294", "Different endpoint history"),
    "male_only": ("#E69F00", "Adjacency history only in men"),
    "female_only": ("#009E73", "Adjacency history only in women"),
}
MEC_DESCRIPTION = "CPDAG of the selected observed directed structure"


def orientation_legend():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color="#444444", lw=2, label="A → B: compelled"),
            Line2D([], [], color="#444444", lw=2, label="A — B: reversible (not reciprocal)")]


def category_legend(styles=TRANSITION_STYLES):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=color, lw=2.2, label=label) for color, label in styles.values()]


def draw_cpdag_edges(ax, endpoints, positions, *, shrink=12):
    """Exactly one artist per adjacency; never two arrows for reversible edges."""
    for i, j in zip(*np.triu_indices(len(endpoints), 1)):
        status = (int(endpoints[i, j]), int(endpoints[j, i]))
        if status != (0, 0):
            _draw_endpoint_edge(ax, positions[i], positions[j], status, "#444444", shrink=shrink)


def _draw_endpoint_edge(ax, start, end, status, color, *, shrink=12, linestyle="-", rad=0):
    from matplotlib.patches import FancyArrowPatch
    if status == (1, -1):
        start, end = end, start
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-" if status == (-1, -1) else "-|>",
        mutation_scale=19, linewidth=2, color=color, linestyle=linestyle,
        shrinkA=shrink, shrinkB=shrink, connectionstyle=f"arc3,rad={rad}", zorder=2,
    )
    ax.add_patch(arrow)


def draw_transition_edges(ax, records, positions, *, shrink=12):
    """Current endpoints, or former endpoints for a lost edge.

    Changed endpoint histories additionally show the old edge dashed, curved
    away from the solid new edge. This distinguishes reversal from loss of a
    compelled orientation without suggesting a reciprocal state edge.
    """
    for record in records:
        i, j = record["pair"]
        old, new = record["before"], record["after"]
        category = record["category"]
        color = TRANSITION_STYLES[category][0]
        if category in ("reversal", "reversibility"):
            _draw_endpoint_edge(ax, positions[i], positions[j], old, color,
                                shrink=shrink, linestyle="--", rad=0.14)
        status = old if category == "lost" else new
        _draw_endpoint_edge(ax, positions[i], positions[j], status, color,
                            shrink=shrink, linestyle="--" if category == "lost" else "-")


def draw_sex_history_edges(ax, records, positions, *, shrink=12):
    for record in records:
        i, j = record["pair"]
        _draw_endpoint_edge(ax, positions[i], positions[j], (-1, -1),
                            SEX_STYLES[record["category"]][0], shrink=shrink)


def circular_nodes(ax, names, *, fontsize=12):
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    ax.scatter(*positions.T, s=650, color="#d9eaf7", edgecolors="#333333", zorder=3)
    for position, name in zip(positions, names):
        ax.text(*position, name, ha="center", va="center", fontsize=fontsize, zorder=4)
    ax.set(xlim=(-1.28, 1.28), ylim=(-1.16, 1.16), aspect="equal")
    ax.axis("off")
    return positions


def plot_circular_mec(states, names, path, threshold, *, titles=None, sex=False,
                      fontsize_scale=1.2, dpi=180):
    import matplotlib.pyplot as plt
    from causal_opt.fmri_plotting import _scale_figure_fonts

    graphs = state_cpdags(states, threshold, names)
    fig, axes = plt.subplots(2 if sex else 1, 3 if sex else 2,
                             figsize=(24, 14) if sex else (16, 8), squeeze=False)
    fig.subplots_adjust(left=.025, right=.975, bottom=.19 if sex else .15,
                        top=.89, wspace=.06, hspace=.26) # TODO: adjust vertical spacing
    for row in range(len(axes)):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            pos = circular_nodes(ax, names, fontsize=9 * fontsize_scale)
            if col < 2:
                draw_cpdag_edges(ax, graphs[row * 2 + col], pos)
                title = (("Male", "Female")[row] + " — " + ("Control", "SDV")[col]
                         if sex else (titles or ("Control", "SDV"))[col].replace("causal DAG", "CPDAG"))
            else:
                draw_transition_edges(ax, cpdag_transitions(graphs[row * 2], graphs[row * 2 + 1]), pos)
                title = ("Male", "Female")[row] + " — Control → SDV transitions"
            ax.set_title(title, fontsize=14 * fontsize_scale, pad=12)
    fig.suptitle(MEC_DESCRIPTION, fontsize=20)
    handles = orientation_legend() + (category_legend() if sex else [])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .052),
               ncol=3 if sex else 2, frameon=False, fontsize=11)
    note = ("Transitions: solid = SDV endpoints; dashed = Control endpoints on lost/changed edges.\n"
            if sex else "")
    fig.text(.5, .012, note + "Compelled within the selected MEC, not statistical confidence. "
             f"State threshold |W| ≥ {threshold:g}.", ha="center", fontsize=11)
    _scale_figure_fonts(fig)
    fig.savefig(mec_output_path(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
