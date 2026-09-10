"""Regenerate sex node and static DAG plots from saved results only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_opt.fmri_inference import (
    compute_sex_node_statistics, load_bootstrap_results, node_reorganization,
    compute_bootstrap_nonzero_statistics, compute_bootstrap_edge_statistics,
)
from causal_opt.fmri_plotting import (
    plot_sex_node_reorganization, plot_sex_dag_comparison,
    plot_anatomical_directed_difference,
)
from causal_opt.fmri_data import load_conn_roi_zip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sex_dir", type=Path, nargs="?",
                        default=Path("results/fmri_control_sdv/sex_analysis"))
    parser.add_argument("--output-dir", type=Path,
                        help="optional separate destination for the PNGs and CSV")
    parser.add_argument("--edge-threshold", type=float, default=0.3,
                        help="state |W| threshold defining the selected DAGs; changing it can change the MEC")
    parser.add_argument("--show-mec", action=argparse.BooleanOptionalAction, default=True,
                        help="CPDAG structures/transitions in _mec figures (default); --no-show-mec restores weighted DAGs and coefficient significance")
    parser.add_argument("--min-abs-change", type=float, default=None,
                        help="weighted anatomical contrast threshold only (default: edge-threshold); ignored in MEC mode")
    parser.add_argument("--session1-zip", type=Path, default=Path(os.getenv(
        "FMRI_SESSION1_ZIP", ROOT.parent / "fmri_connectivity/data/mat_files/rs_sessions_r03_healthy/roi_rs_sessions_Session1.zip")),
        help="CONN Session 1 archive used only for anatomical ROI coordinates")
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
    raw_fitted = {}
    significance = {}
    for sex, boot in zip(("male", "female"), boots):
        for state in ("control", "sdv"):
            with np.load(args.sex_dir / f"static_{sex}_{state}.npz", allow_pickle=False) as fit:
                if (list(fit["roi_names"].astype(str)) != boot["meta"]["roi_names"]
                        or not np.array_equal(fit["display_names"].astype(str), names)):
                    raise ValueError("Saved fit and bootstrap ROI ordering differs")
                fitted[f"{sex}_{state}"] = fit["W"].copy()
                raw_fitted[f"{sex}_{state}"] = fit["W_raw"].copy()
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
    displayed = raw_fitted if args.show_mec else fitted
    plot_sex_dag_comparison(
        displayed["male_control"], displayed["male_sdv"],
        displayed["female_control"], displayed["female_sdv"], names,
        out / "sex_static_dag_comparison.png",
        threshold=args.edge_threshold, show_mec=args.show_mec, **significance,
    )
    coordinates = load_conn_roi_zip(args.session1_zip, expected_session="001")
    if tuple(coordinates.roi_names) != tuple(boots[0]["meta"]["roi_names"]):
        raise ValueError("Coordinate ROI order differs from the saved models")
    with np.load(args.sex_dir / "sex_bootstrap_statistics.npz", allow_pickle=False) as saved:
        interaction_sig = saved["interaction_p"] < .05
    import matplotlib.pyplot as plt
    raw_states = tuple(raw_fitted[k] for k in (
        "male_control", "male_sdv", "female_control", "female_sdv"))
    common = dict(
        view="coronal", show_mec=args.show_mec, dag_threshold=args.edge_threshold,
        min_abs_change=args.edge_threshold if args.min_abs_change is None else args.min_abs_change,
    )
    fig, _ = plot_anatomical_directed_difference(
        female - male, coordinates.roi_xyz, names, out / "sex_interaction_anatomical.png",
        state_matrices=raw_states, significant_mask=interaction_sig,
        title="Sex × Bladder-State Interaction\n(Female SDV − Control) − (Male SDV − Control)",
        positive_color="darkorange", negative_color="purple",
        positive_label="Larger / more positive SDV change in women",
        negative_label="Larger / more positive SDV change in men",
        magnitude_label=r"Arrow width $\propto |\Delta W_{Female}-\Delta W_{Male}|$", **common,
    )
    plt.close(fig)
    for sex, delta, states in (("male", male, raw_states[:2]), ("female", female, raw_states[2:])):
        fig, _ = plot_anatomical_directed_difference(
            delta, coordinates.roi_xyz, names, out / f"{sex}_sdv_control_anatomical.png",
            state_matrices=states, significant_mask=significance[f"{sex}_delta_sig"],
            title=f"{sex.title()}: SDV − Control",
            mec_title=f"{sex.title()}: Control → SDV CPDAG transitions",
            positive_color="tab:red", negative_color="tab:blue",
            positive_label="Increase in SDV", negative_label="Decrease in SDV", **common,
        )
        plt.close(fig)
    print(f"Saved node statistics and {'CPDAG (_mec)' if args.show_mec else 'weighted DAG'} plots to {out}")


if __name__ == "__main__":
    main()
