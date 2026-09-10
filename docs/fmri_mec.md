# fMRI structural and weighted graph views

Both summary scripts default to `--show-mec`. Install the CPDAG conversion
dependency once in the Python environment used on Yoda:

```bash
python -m pip install -r requirements-fmri-mec.txt
```

From the repository root, regenerate cached-result summaries:

```bash
# Default structural (MEC) mode; --show-mec can also be supplied explicitly.
MPLBACKEND=Agg python scripts/summarize_fmri_case2_bootstrap.py results/fmri_control_sdv
MPLBACKEND=Agg python scripts/summarize_fmri_sex_bootstrap.py results/fmri_control_sdv/sex_analysis

# Original weighted-DAG/coefficient views.
MPLBACKEND=Agg python scripts/summarize_fmri_case2_bootstrap.py results/fmri_control_sdv --no-show-mec
MPLBACKEND=Agg python scripts/summarize_fmri_sex_bootstrap.py results/fmri_control_sdv/sex_analysis --no-show-mec
```

These scripts load fitted matrices and bootstrap checkpoints. They recompute
summary statistics, but never fit a model or optimize a bootstrap replicate.
Anatomical plots read coordinates from the CONN Session 1 archive in the sibling
`fmri_connectivity/data/mat_files/rs_sessions_r03_healthy` directory. Set
`FMRI_SESSION1_ZIP` to override it (the sex script also accepts `--session1-zip`).
The sex summary accepts `--output-dir` for a separate plot destination.

## Interpretation

Each state is the **CPDAG of the selected observed directed structure**.
The selected adjacency has `abs(W[i,j]) >= --edge-threshold` (default 0.3),
excluding the diagonal and numerical zeros (`abs(W) <= 1e-12`). The convention
is `W[i,j] = i → j`. Changing this threshold changes the selected structure and
can change its MEC. A selected cycle raises an error with an example cycle;
the conversion never removes edges to manufacture a DAG.

Conversion uses causal-learn's
[`dag2cpdag`](https://github.com/py-why/causal-learn/blob/main/causallearn/utils/DAG2CPDAG.py).
The endpoint matrix has -1 for a tail, +1 for an arrowhead, and 0 for absence:
`E[i,j] = -1, E[j,i] = +1` means `i → j`, while two tails mean `i — j`.
The tests independently verify equivalence classes for all four-node DAGs.

State panels use neutral arrows for compelled edges and plain lines for
reversible edges, once per unordered pair. Undirected edges are not reciprocal
causation. “Compelled” is relative to the selected graph's equivalence class,
not statistical confidence. These figures show conditional-independence
equivalence; stronger functional or distributional assumptions may identify
additional orientations. The fitted model has latent structure, but these
CPDAGs are **not** the observed-variable PAG of that full latent model. Loading
correlations are not converted into latent graph edges. Existing FCI PAG plots
remain PAGs.

Control→SDV structural panels compare separately converted state CPDAGs:

* Green: adjacency gained.
* Orange: adjacency lost.
* Gray: retained with unchanged endpoints.
* Pink: retained with a compelled reversal.
* Blue: retained with a directed/reversible transition.

Solid edges show SDV endpoints. Dashed edges show former Control endpoints for
lost or changed edges. A changed pair can therefore show two temporal snapshots;
the transition panel is a comparison, not itself a CPDAG. No difference matrix
or union is converted as if it were a DAG.

The sex anatomical comparison shows male and female transitions side by side,
plus an unordered-pair comparison of their exact `(Control endpoints, SDV
endpoints)` histories. Its plain lines indicate same/different histories or
adjacency history in only one sex; they do not encode orientations. This is a
descriptive comparison of CPDAG transitions, not a numerical or statistically
tested sex × state interaction.

MEC figures have no coefficient-significance stars, signed weights, or
weight-scaled widths. A significant below-threshold coefficient is not added
to a CPDAG. Coefficient heatmaps, ROI rankings, bootstrap distributions, and
their significance tests remain fitted-model summaries in both modes.

## Outputs

Case 2 structural figures:

* `static_graph_comparison_mec.png`
* `case2_sdv_minus_control_coronal_mec.png`

Sex structural figures:

* `sex_static_dag_comparison_mec.png`
* `sex_interaction_anatomical_mec.png`
* `male_sdv_control_anatomical_mec.png`
* `female_sdv_control_anatomical_mec.png`

`--no-show-mec` uses the corresponding legacy filenames without `_mec`.
Other summary figures and statistics retain their existing names in both modes.

## Plotting API

`plot_graph_comparison`, `plot_sex_dag_comparison`,
`plot_anatomical_graph_difference`, `plot_anatomical_directed_difference`, and
the `plot_static_report` wrapper expose keyword-only `show_mec=True`.
The anatomical functions distinguish `dag_threshold=0.3` (state structure) from
`min_abs_change` (weighted contrast filtering). At the CLI, `--min-abs-change`
defaults to `--edge-threshold` for legacy compatibility and is ignored in MEC mode.

For a difference-only anatomical call in MEC mode, supply
`state_matrices=(Control, SDV)` or
`state_matrices=(MaleControl, MaleSDV, FemaleControl, FemaleSDV)`.
Missing context raises a useful error. `show_mec=False` uses the original
weighted difference and significance overlays. Direct plotting calls also add
the `_mec` suffix automatically; `plot_static_report` returns the suffixed path.
