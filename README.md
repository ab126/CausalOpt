# CausalOpt : Causal Discovery with Latent Low-Rank Confounding

CausalOpt is primarily a research implementation of causal discovery with latent low-rank confounding. It estimates sparse directed structure jointly with a low-rank latent component in static and dynamic linear systems. The repository contains synthetic experiments for four progressively harder cases and one real-world application to fMRI connectivity data.

> [!IMPORTANT]
> This repository does **not** distribute patient information or raw fMRI archives. Keep all source `.zip` files and subject-level data local; the ignore rules exclude ZIP archives and conventional local data directories. Before publishing derived results, verify that they are aggregate, de-identified, and permitted by the data-use agreement for the source dataset.

## Four experimental cases

| Case | Setting | Latent confounding | Primary comparison |
| --- | --- | --- | --- |
| 1 | Static linear SEM | No (`p=0`, `k=0`) | NOTEARS |
| 2 | Static linear SEM | Yes (`p=0`, `k>0`) | FCI |
| 3 | Dynamic linear SEM | No (`p>0`, `k=0`) | DYNOTEARS |
| 4 | Dynamic linear SEM | Yes (`p>0`, `k>0`) | LPCMCI and the Liéois sparse-plus-low-rank AR method |

The overarching model formulation is

```text
X₀ = X₀ W₀ + Σ_{τ=1}^p X_τ W_τ + Z Lᵀ + noise.
```

Here, `p` is the autoregressive lag order and `k` is the assumed latent rank. The general model uses a contemporaneous structural matrix `W₀`, lagged structural matrices `W₁, ..., Wₚ,` and a rank-k latent contribution `ZLᵀ`. Setting `p=0` gives the static model, while setting `k=0` removes latent confounding. The contemporaneous matrix `W₀` is constrained to be acyclic using the smooth NOTEARS constraint.

## Stable installation

Python 3.10 is recommended. Run the commands from the repository root in a fresh virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy networkx pyyaml matplotlib jupyter pytest
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pytest -q
```

On macOS or Linux, activate with `source .venv/bin/activate` and set `export PYTHONPATH="$PWD/src"`.

The core Case 2 and Case 4 methods require only NumPy and SciPy; PyYAML is used by the experiment drivers, and Matplotlib/Jupyter are used by the notebooks. Baselines have intentionally separate optional dependencies:

- FCI: `python -m pip install causal-learn`
- LPCMCI: `python -m pip install tigramite`
- DYNOTEARS: use a separate Python 3.10 environment because the pinned CausalNex stack is older:

  ```powershell
  py -3.10 -m venv .venv-dynotears
  .\.venv-dynotears\Scripts\python.exe -m pip install --upgrade pip
  .\.venv-dynotears\Scripts\python.exe -m pip install -r requirements-dynotears.txt
  ```

- Liéois: requires MATLAB and a checkout of the reference implementation at the commit validated by the adapter:

  ```powershell
  git clone https://github.com/CyclotronResearchCentre/SparseLowRankIdentification.git external/SparseLowRankIdentification
  git -C external/SparseLowRankIdentification checkout f9ca591039fb5c6a3288108e371ba328c7284e48
  ```

## Toy runs: Cases 2 and 4

These commands simulate data; they do not require or read fMRI files.

Case 2 fits a static DAG together with a rank-2 latent component:

```powershell
python scripts/run_case2_static_latent.py --smoke --seed 2 --output-dir results
```

Case 4 fits contemporaneous and lagged structure together with a rank-2 latent component. The smoke configuration keeps the example small:

```powershell
python scripts/run_case4_dynamic_latent.py --smoke --method ours --seed 1 --output-dir results
```

Remove `--smoke` and pass `--config configs/case2.yaml` or `--config configs/case4.yaml` for the full configured simulations. Results are written below `results/`, which is excluded from version control. To run a seed range, use the common driver, for example `python scripts/run_grid.py --case 2 --seed-start 1 --seed-end 10`.

## Notebooks

- [`01_static_notears.ipynb`](notebooks/01_static_notears.ipynb) is the Case 1 static, fully observed debugger. It checks the `k=0` specialization against the official [NOTEARS](https://github.com/xunzheng/notears) baseline on shared simulated data.
- [`02_static_latent_fci.ipynb`](notebooks/02_static_latent_fci.ipynb) covers Cases 1 and 2. It first verifies the no-latent limit against NOTEARS, then estimates a static DAG and low-rank latent term for Case 2 and compares appropriate graph information with [FCI from causal-learn](https://github.com/py-why/causal-learn). FCI returns a PAG, so unresolved endpoints should retain their native interpretation.
- [`03_dynamic_dynotears.ipynb`](notebooks/03_dynamic_dynotears.ipynb) is the Case 3 time-series debugger. It fits the dynamic `k=0` model and compares contemporaneous and lagged recovery with the official [DYNOTEARS implementation in CausalNex](https://github.com/mckinsey/causalnex) on the same realization.
- [`04_dynamic_latent.ipynb`](notebooks/04_dynamic_latent.ipynb) is the Case 4 latent time-series debugger. It compares the proposed directed structural/latent-factor estimate with [LPCMCI in Tigramite](https://github.com/jakobrunge/tigramite) and the [Liéois sparse-plus-low-rank AR reference implementation](https://github.com/CyclotronResearchCentre/SparseLowRankIdentification). Because these methods return different mathematical objects, the notebook reports only semantically valid comparisons rather than a universal DAG ranking.

The separate [`fmri_control_sdv_causal_analysis.ipynb`](notebooks/fmri_control_sdv_causal_analysis.ipynb) applies the static and dynamic latent methods to de-identified ROI-level fMRI time series. It is an application notebook, not one of the four synthetic cases. Users must supply authorized data locally; never commit archives, subject identifiers, acquisition metadata containing identifiers, or other patient information.

## Citation

There is not yet a DOI or archived release for this repository. Until one is available, cite the repository and the exact commit used:

```bibtex
@software{causalopt,
  title        = {CausalOpt: Causal Discovery with Latent Low-Rank Confounding},
  author       = {Arda Bayer},
  url          = {https://github.com/ab126/CausalOpt},
  year         = {2026}
}
```

Please also cite the methodological works corresponding to any baselines used in a result.

## Related repositories and works

- [NOTEARS code](https://github.com/xunzheng/notears) and Zheng et al., [“DAGs with NO TEARS: Continuous Optimization for Structure Learning”](https://arxiv.org/abs/1803.01422), NeurIPS 2018.
- [CausalNex](https://github.com/mckinsey/causalnex) and Pamfil et al., [“DYNOTEARS: Structure Learning from Time-Series Data”](https://arxiv.org/abs/2002.00498), AISTATS 2020.
- [causal-learn](https://github.com/py-why/causal-learn), which supplies the FCI baseline adapter.
- [Tigramite](https://github.com/jakobrunge/tigramite), which supplies the LPCMCI baseline and documents its time-series PAG semantics.
- [SparseLowRankIdentification](https://github.com/CyclotronResearchCentre/SparseLowRankIdentification), the original MATLAB sparse-plus-low-rank autoregressive reference used by the Liéois adapter.

## License and third-party code

This project is licensed under the terms in [`LICENSE`](LICENSE). It includes/adapts code from [NOTEARS](https://github.com/xunzheng/notears), which is licensed under Apache License 2.0. Optional baselines remain subject to their respective licenses and citation requirements.
