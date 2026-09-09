"""Regenerate the sex node plot and inference table from saved results only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_opt.fmri_inference import (
    compute_sex_node_statistics, load_bootstrap_results, node_reorganization,
)
from causal_opt.fmri_plotting import plot_sex_node_reorganization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sex_dir", type=Path, nargs="?",
                        default=Path("results/fmri_control_sdv/sex_analysis"))
    parser.add_argument("--output-dir", type=Path,
                        help="optional separate destination for the PNG and CSV")
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
    for sex, boot in zip(("male", "female"), boots):
        for state in ("control", "sdv"):
            with np.load(args.sex_dir / f"static_{sex}_{state}.npz", allow_pickle=False) as fit:
                if (list(fit["roi_names"].astype(str)) != boot["meta"]["roi_names"]
                        or not np.array_equal(fit["display_names"].astype(str), names)):
                    raise ValueError("Saved fit and bootstrap ROI ordering differs")
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
    print(f"Saved plot and node statistics to {out}")


if __name__ == "__main__":
    main()
