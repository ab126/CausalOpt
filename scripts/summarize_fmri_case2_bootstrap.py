"""Recompute Case 2 inference products exclusively from saved fits/checkpoints."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from causal_opt.fmri import (plot_bootstrap_edge_significance,plot_bootstrap_edge_stability,
    plot_bootstrap_edge_intervals,plot_key_edge_bootstrap_distributions,
    plot_node_reorganization,plot_nominal_bootstrap_edge_intervals, plot_graph_comparison, plot_anatomical_graph_difference)
from causal_opt.fmri_data import load_conn_roi_zip
from causal_opt.fmri_inference import (compute_bootstrap_edge_statistics,
    compute_edge_selection_stability,edge_results_dataframe,load_bootstrap_results,node_reorganization,
    compute_bootstrap_nonzero_statistics)


def _load_static(path):
    with np.load(path,allow_pickle=False) as item: return np.asarray(item["W_raw"]),tuple(map(str,item["display_names"]))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir",type=Path)
    ap.add_argument("--edge-threshold",type=float,default=.3,
                    help="state |W| threshold defining each selected DAG; changing it can change the MEC")
    ap.add_argument("--min-abs-change",type=float,default=None,
                    help="weighted anatomical |SDV-Control| threshold only (default: edge-threshold); ignored in MEC mode")
    ap.add_argument("--show-mec", action=argparse.BooleanOptionalAction, default=True,
                    help="CPDAG structures/transitions in _mec figures (default); --no-show-mec restores weighted DAGs and coefficient significance")
    ap.add_argument("--stability-thresholds",type=float,nargs="+",default=[.05,.1,.2,.3])
    args=ap.parse_args(argv)
    out=args.output_dir; Wc,names=_load_static(out/"static_control.npz"); Ws,names_sdv=_load_static(out/"static_sdv.npz")
    if names!=names_sdv: raise ValueError("saved static ROI ordering differs")
    boot=load_bootstrap_results(out/"case2_bootstrap")
    if boot["completed"]!=boot["requested"]: print(f"WARNING: cache incomplete ({boot['completed']}/{boot['requested']}); statistics use {boot['successful']} currently valid checkpoints")
    if not boot["successful"]: raise ValueError("no valid paired bootstrap replicates")
    delta=Ws-Wc; stats=compute_bootstrap_edge_statistics(delta,boot["delta_W"]); stability=compute_edge_selection_stability(boot["W_control"],boot["W_sdv"],args.stability_thresholds)

    control_stats = compute_bootstrap_nonzero_statistics(
        Wc,
        boot["W_control"],
    )

    sdv_stats = compute_bootstrap_nonzero_statistics(
        Ws,
        boot["W_sdv"],
    )
    control_sig = control_stats["p_boot"] < .05
    sdv_sig = sdv_stats["p_boot"] < .05

    frame=edge_results_dataframe(names,Wc,Ws,stats,stability,args.edge_threshold); frame["abs_delta_W"]=frame.delta_W_observed.abs()
    frame.to_csv(out/"case2_bootstrap_edge_results.csv",index=False)
    nominal=frame[frame.p_boot<.05].sort_values(["p_boot","abs_delta_W"],ascending=[True,False]); fdr=frame[frame.q_fdr<.05].sort_values(["q_fdr","abs_delta_W"],ascending=[True,False])
    report_columns={"source_name":"source","target_name":"target","W_control_observed":"W_control","W_sdv_observed":"W_sdv","delta_W_observed":"delta_W","ci_2.5":"ci_2.5","ci_97.5":"ci_97.5","p_boot":"p_boot","q_fdr":"q_fdr","ci_excludes_zero":"CI excludes zero","control_selection_probability":"Control edge-selection probability","sdv_selection_probability":"SDV edge-selection probability"}
    nominal[list(report_columns)].rename(columns=report_columns).to_csv(out/"case2_bootstrap_nominal_edges.csv",index=False); fdr.to_csv(out/"case2_bootstrap_fdr_edges.csv",index=False)
    nodes=node_reorganization(delta,names); nodes.to_csv(out/"case2_node_reorganization.csv",index=False)
    focus=frame[(frame.source_name=="L Insula")&(frame.target_name=="PAG1")]
    if len(focus)!=1: raise ValueError("anatomical mapping must contain exactly L Insula -> PAG1")
    row=focus.iloc[0]; focus_summary={"source":"L Insula","target":"PAG1","Control W":row.W_control_observed,"SDV W":row.W_sdv_observed,"SDV-Control delta W":row.delta_W_observed,"bootstrap mean delta":row.bootstrap_mean_delta,"95% CI":[row["ci_2.5"],row["ci_97.5"]],"bootstrap p":row.p_boot,"FDR q":row.q_fdr,"Control selection probability":row.control_selection_probability,"SDV selection probability":row.sdv_selection_probability,"nominal_statement":"Nominally significant Control-to-SDV change (p < 0.05)." if row.p_boot<.05 else "No nominally significant Control-to-SDV change (p >= 0.05).","fdr_statement":"Survives BH-FDR correction." if row.q_fdr<.05 else "Does not survive BH-FDR correction."}
    summary={"requested":boot["requested"],"completed":boot["completed"],"valid":boot["successful"],"failed":boot["failed"],"nominal_p_lt_0.05":len(nominal),"fdr_q_lt_0.05":len(fdr),"left_insula_to_pag1":focus_summary,"largest_absolute_changes":frame.nlargest(5,"abs_delta_W")[["source_name","target_name","delta_W_observed","p_boot","q_fdr"]].to_dict("records"),"ci_pvalue_note":"The 95% CI is a percentile-bootstrap interval; p_boot is computed separately from the centered bootstrap distribution. CI exclusion and p_boot < 0.05 are related but not guaranteed numerically equivalent."}
    (out/"case2_bootstrap_summary.json").write_text(json.dumps(summary,indent=2,default=float),encoding="utf-8")

    # Plots
    plot_bootstrap_edge_significance(delta,stats,names,out/"case2_bootstrap_delta_significance.png", "Observed raw SDV - Control W"); plot_bootstrap_edge_stability(stability,names,args.edge_threshold,out/"case2_bootstrap_edge_stability.png"); plot_bootstrap_edge_intervals(frame,out/"case2_bootstrap_edge_intervals.png"); plot_nominal_bootstrap_edge_intervals(frame,out/"case2_bootstrap_nominal_edge_intervals.png"); plot_key_edge_bootstrap_distributions(frame,boot["delta_W"],out/"case2_bootstrap_key_edge_distributions.png"); plot_node_reorganization(nodes,out/"case2_node_reorganization.png")

    plot_bootstrap_edge_significance(
        Wc,
        control_stats,
        names,
        out / "case2_bootstrap_control_significance.png",
        "Control directed edges — uncorrected bootstrap p < 0.05",
    )

    plot_bootstrap_edge_significance(
        Ws,
        sdv_stats,
        names,
        out / "case2_bootstrap_sdv_significance.png",
        "SDV directed edges — uncorrected bootstrap p < 0.05",
    )
    
    plot_graph_comparison(
        Wc,
        Ws,
        names,
        out / "static_graph_comparison.png",
        args.edge_threshold,
        significant_control=control_sig,
        significant_sdv=sdv_sig,
        show_mec=args.show_mec,
    )

    # Anatomical graph difference plot. TODO: Modularize
    # Load the original BN19 coordinates; this does not fit any models.
    data_dir = (
        ROOT.parent
        / "fmri_connectivity"
        / "data"
        / "mat_files"
        / "rs_sessions_r03_healthy"
    )
    session1_zip = Path(
        os.getenv(
            "FMRI_SESSION1_ZIP",
            str(data_dir / "roi_rs_sessions_Session1.zip"),
        )
    )
    control = load_conn_roi_zip(session1_zip, expected_session="001")

    # Verify that coordinates follow the saved model's ROI order.
    with np.load(out / "static_control.npz", allow_pickle=False) as saved:
        saved_roi_names = tuple(map(str, saved["roi_names"]))

    if control.roi_names != saved_roi_names:
        raise ValueError("Coordinate ROI order differs from saved model.")

    delta_sig = np.asarray(stats["p_boot"]) < 0.05
    np.fill_diagonal(delta_sig, False)

    fig, ax = plot_anatomical_graph_difference(
        Wc,
        Ws,
        control.roi_xyz,
        names,
        output_path=out / "case2_sdv_minus_control_coronal.png",
        view="coronal",
        min_abs_change=args.edge_threshold if args.min_abs_change is None else args.min_abs_change,
        dag_threshold=args.edge_threshold,
        show_mec=args.show_mec,
        significant_mask=delta_sig,
        title="SDV − Control: Coronal View",
        figsize=(12, 10),
        dpi=300,
    )
    plt.close(fig)

    print(f"Valid paired bootstrap replicates: {boot['successful']}"); print(f"Nominal p < 0.05 edges: {len(nominal)}"); print(f"FDR q < 0.05 edges: {len(fdr)}"); print(json.dumps(focus_summary,indent=2,default=float))


if __name__=="__main__": main()
