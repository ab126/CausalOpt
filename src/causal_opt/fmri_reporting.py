"""Small report-building utilities shared by fMRI notebooks."""
from pathlib import Path

from .fmri import latent_similarity
from .fmri_plotting import (
    plot_graph_comparison,
    plot_matrix_comparison,
    plot_dynamic_matrix_comparison,
)
from .fmri_mec import mec_output_path


def plot_static_report(control, sdv, names, output, *, threshold=0.3, show_mec=True):
    """Write descriptive static plots and return their paths in display order."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"static_{kind}_comparison.png" for kind in ("W", "latent", "graph")]
    paths[2] = mec_output_path(paths[2], show_mec)
    plot_matrix_comparison(control.W0, sdv.W0, names,
                           ("Control W", "SDV W", "SDV - Control"), paths[0])
    plot_matrix_comparison(latent_similarity(control), latent_similarity(sdv), names,
                           ("Control LL^T", "SDV LL^T", "SDV - Control"), paths[1], True)
    plot_graph_comparison(control.W0, sdv.W0, names, paths[2], threshold=threshold, show_mec=show_mec)
    return paths


def plot_dynamic_report(control, sdv, names, output, *, threshold=0.3):
    """Write contemporaneous, per-lag, and latent plots for any fitted lag order."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for label, left, right in [("W0", control.W0, sdv.W0)] + [
        (f"Wlag{tau + 1}", left, right)
        for tau, (left, right) in enumerate(zip(control.W_lags, sdv.W_lags))
    ]:
        path = output / f"dynamic_{label}_comparison.png"
        plot_matrix_comparison(left, right, names,
                               (f"Control {label}", f"SDV {label}", "SDV - Control"), path)
        paths.append(path)
    path = output / "dynamic_latent_comparison.png"
    plot_matrix_comparison(latent_similarity(control), latent_similarity(sdv), names,
                           ("Control LL^T", "SDV LL^T", "SDV - Control"), path, True)
    paths.append(path)
    if len(control.W_lags) == 1:
        path = output / "dynamic_W0_W1_comparison.png"
        plot_dynamic_matrix_comparison(control.W0, sdv.W0, control.W_lags[0],
                                       sdv.W_lags[0], threshold, names, path)
        paths.append(path)
    return paths


def display_figures(paths):
    """Display generated figures in Jupyter without notebook-local helpers."""
    from IPython.display import Image, display

    for path in paths:
        display(Image(filename=str(path)))
