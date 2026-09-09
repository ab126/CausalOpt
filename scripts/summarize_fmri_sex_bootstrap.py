"""Regenerate sex node and static DAG plots from saved results only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_opt.fmri_inference import (
    compute_sex_node_statistics, load_bootstrap_results, node_reorganization,
    compute_bootstrap_nonzero_statistics, compute_bootstrap_edge_statistics,
)
from causal_opt.fmri_plotting import plot_sex_node_reorganization, plot_sex_dag_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sex_dir", type=Path, nargs="?",
                        default=Path("results/fmri_control_sdv/sex_analysis"))
    parser.add_argument("--output-dir", type=Path,
                        help="optional separate destination for the PNGs and CSV")
    parser.add_argument("--edge-threshold", type=float, default=0.3,
                        help="minimum absolute edge weight/change shown in DAG panels")
    args = parser.parse_args(argv)
    with np.load(args.sex_dir / "sex_change_matrices.npz", allow_pickle=False) as saved:
        male, female = saved["delta_male"], saved["delta_female"]
        names = saved["display_names"].astype(str)
    boots = [load_bootstrap_results(args.sex_dir / f"bootstrap_{sex}")
             for sex in ("male", "female")]
    for sex, boot in zip(("male", "female"), boots):
        print(f"{sex}: {boot['successful']} valid / {boot['completed']} completed "
              f"/ {boot['requested']} requested replicates")
    if boots[0]["meta"]["roi_names"] != boots[1]["meta"]["roi_names"]:
        raise ValueError("Male/female bootstrap ROI ordering differs")
    # Saved fits establish that checkpoint ROI order matches the plotted matrices.
    fitted = {}
    significance = {}
    for sex, boot in zip(("male", "female"), boots):
        for state in ("control", "sdv"):
            with np.load(args.sex_dir / f"static_{sex}_{state}.npz", allow_pickle=False) as fit:
                if (list(fit["roi_names"].astype(str)) != boot["meta"]["roi_names"]
                        or not np.array_equal(fit["display_names"].astype(str), names)):
                    raise ValueError("Saved fit and bootstrap ROI ordering differs")
                fitted[f"{sex}_{state}"] = fit["W"].copy()
                significance[f"{sex}_{state}_sig"] = compute_bootstrap_nonzero_statistics(
                    fit["W_raw"], boot[f"W_{state}"],
                )["p_boot"] < 0.05
        delta = male if sex == "male" else female
        significance[f"{sex}_delta_sig"] = compute_bootstrap_edge_statistics(
            delta, boot["delta_W"],
        )["p_boot"] < 0.05
    stats = compute_sex_node_statistics(male, female, *boots, names)
    count = int(stats["valid_bootstrap_pairs"].iloc[0])
    print(f"Common valid replicates: {count}; minimum attainable p = {1/(count+1):.6g}")
    out = args.output_dir or args.sex_dir
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "sex_node_reorganization_statistics.csv", index=False)
    plot_sex_node_reorganization(
        node_reorganization(male, names), node_reorganization(female, names),
        out / "sex_node_reorganization.png", node_statistics=stats,
    )
    plot_sex_dag_comparison(
        fitted["male_control"], fitted["male_sdv"],
        fitted["female_control"], fitted["female_sdv"], names,
        out / "sex_static_dag_comparison.png",
        threshold=args.edge_threshold, **significance,
    )
    print(f"Saved node plot, static DAG comparison, and node statistics to {out}")


if __name__ == "__main__":
    main()
