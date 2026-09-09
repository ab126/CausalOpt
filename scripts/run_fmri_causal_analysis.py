"""Fit CausalOpt Case 2 and Case 4 models to paired CONN sessions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT.parent / "fmri_connectivity" / "data" / "mat_files" / "rs_sessions_r03_healthy"
SESSION1 = Path(os.getenv("FMRI_SESSION1_ZIP", DATA / "roi_rs_sessions_Session1.zip"))
SESSION2 = Path(os.getenv("FMRI_SESSION2_ZIP", DATA / "roi_rs_sessions_Session2.zip"))

patient_sex = {
    2: 'Female',
    3: 'Female',
    4: 'Female',
    5: 'Female',
    6: 'Female',
    7: 'Male',
    8: 'Male',
    9: 'Female',
    10: 'Male',
    11: 'Male',
    12: 'Male',
    13: 'Female',
    14: 'Male',
    15: 'Female',
    16: 'Male',
    17: 'Female',
    18: 'Male',
    19: 'Female',
    20: 'Male',
}

from causal_opt.fmri import (build_multisubject_lagged_data,build_static_fmri_matrix,
    compare_matrices,data_summary,latent_similarity,load_fmri_state_data,model_statistics,
    load_roi_display_names,plot_bootstrap_edge_significance,plot_bootstrap_edge_intervals,
    plot_bootstrap_edge_stability,plot_graph_comparison,plot_matrix_comparison,
    plot_key_edge_bootstrap_distributions,plot_node_reorganization,plot_pag_comparison,plot_skeleton_comparison,
    plot_two_matrix_comparison,save_result,top_matrix_changes,
    # Sex analysis
    subset_conn_roi_data,
    plot_sex_dag_comparison,
    plot_sex_change_heatmaps,
    plot_sex_node_reorganization,
    plot_anatomical_directed_difference,
    )
from causal_opt.fmri_inference import (compute_bootstrap_edge_statistics,
    compare_case2_structures,compute_edge_selection_stability,edge_results_dataframe,fit_case2_baselines,
    node_reorganization,pag_skeleton,paired_subject_bootstrap_case2,
    # Sex analysis
    compute_bootstrap_nonzero_statistics,
    compute_sex_node_statistics,
    combine_sex_interaction_bootstraps,)
from causal_opt.methods.dynamic_latent import fit_dynamic_latent
from causal_opt.methods.static_latent import fit_static_latent
from causal_opt.simulation.static_sem import EstimatorResult


def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "session1_zip",
        type=Path,
        nargs="?",
        default=SESSION1,
        help=f"Session 1 ZIP (default: {SESSION1})",
    )
    ap.add_argument(
        "session2_zip",
        type=Path,
        nargs="?",
        default=SESSION2,
        help=f"Session 2 ZIP (default: {SESSION2})",
    )
    ap.add_argument("--output-dir",type=Path,default=Path("results/fmri_control_sdv")); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--k",type=int,default=2); ap.add_argument("--p",type=int,default=1); ap.add_argument("--edge-threshold",type=float,default=.3)
    ap.add_argument("--lambda-w",type=float,default=.1); ap.add_argument("--lambda-0",type=float,default=.1); ap.add_argument("--lambda-lag",type=float,default=.05)
    ap.add_argument("--static-lambda-latent",type=float,default=.1); ap.add_argument("--dynamic-lambda-latent",type=float,default=.02)
    ap.add_argument("--max-outer-iter",type=int,default=100); ap.add_argument("--inner-max-iter",type=int); ap.add_argument("--h-tol",type=float,default=1e-8)
    ap.add_argument("--case2-baselines",action="store_true",help="fit official NOTEARS and FCI on the static data")
    ap.add_argument("--reuse-static-results",action="store_true",help="load existing converged static_control/static_sdv NPZ files")
    ap.add_argument("--skip-fci",action="store_true"); ap.add_argument("--fci-alpha",type=float,default=.05); ap.add_argument("--fci-depth",type=int,default=-1); ap.add_argument("--roi-labels-mat",type=Path)
    ap.add_argument("--bootstrap-reps",type=int,default=0); ap.add_argument("--bootstrap-seed",type=int,default=42); ap.add_argument("--bootstrap-jobs",type=int,default=1)
    ap.add_argument("--bootstrap-resume",action="store_true"); ap.add_argument("--bootstrap-checkpoint-every",type=int,default=10); ap.add_argument("--stability-thresholds",type=float,nargs="+",default=[.05,.1,.2,.3])
    group=ap.add_mutually_exclusive_group(); group.add_argument("--static-only",action="store_true"); group.add_argument("--dynamic-only",action="store_true")

    ap.add_argument(
        "--sex-analysis",
        action="store_true",
        help="fit static Case 2 models separately in men and women",
    )

    ap.add_argument(
        "--reuse-sex-results",
        action="store_true",
        help="reuse saved sex-specific static fits instead of refitting",
    )

    ap.add_argument(
        "--sex-bootstrap-reps",
        type=int,
        default=0,
        help="paired-subject bootstrap replicates within each sex; 0 disables sex bootstrap",
    )

    ap.add_argument(
        "--sex-bootstrap-seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--sex-bootstrap-jobs",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--sex-bootstrap-resume",
        action="store_true",
        help="resume existing male/female bootstrap checkpoints",
    )

    ap.add_argument(
        "--sex-bootstrap-checkpoint-every",
        type=int,
        default=10,
    )
    return ap


def _print(label,value): print(f"{label}: {json.dumps(value,default=lambda x:np.asarray(x).tolist())}")


def _load_estimator(path):
    with np.load(path,allow_pickle=False) as item:
        diagnostics=json.loads(str(item["diagnostics_json"])); W=np.asarray(item["W"]); raw=np.asarray(item["W_raw"]); Z=np.asarray(item["Z"]); L=np.asarray(item["L"])
        return EstimatorResult(W0=W,W_raw=raw,Z=Z,L=L,C=Z@L.T,runtime_seconds=float(item["runtime_seconds"]),diagnostics=diagnostics)


def main(argv=None):
    args=parser().parse_args(argv); args.output_dir.mkdir(parents=True,exist_ok=True)
    control,sdv=load_fmri_state_data(args.session1_zip,args.session2_zip)
    labels_path=args.roi_labels_mat
    if labels_path is None:
        candidate=args.session1_zip.parent/"labels"/"Bladder Network 19_labels.mat"
        labels_path=candidate if candidate.exists() else None
    display=load_roi_display_names(labels_path) if labels_path else tuple(f"BN19-{i:02d}" for i in range(1,20))
    _print("control data",data_summary(control,"control / empty bladder")); _print("SDV data",data_summary(sdv,"full bladder / SDV"))
    common=dict(k=args.k,max_outer_iter=args.max_outer_iter,h_tol=args.h_tol,inner_max_iter=args.inner_max_iter,random_state=args.seed,verbose=True)
    summary={"configuration":vars(args).copy(),"data":{"control":data_summary(control,"control"),"sdv":data_summary(sdv,"sdv")}}
    summary["configuration"]["session1_zip"]=str(args.session1_zip); summary["configuration"]["session2_zip"]=str(args.session2_zip); summary["configuration"]["output_dir"]=str(args.output_dir)
    if not args.dynamic_only:
        Xc,Xs=build_static_fmri_matrix(control),build_static_fmri_matrix(sdv); _print("static shapes",{"control":Xc.shape,"sdv":Xs.shape})
        cached=(args.output_dir/"static_control.npz",args.output_dir/"static_sdv.npz")
        if args.reuse_static_results:
            if not all(path.exists() for path in cached): raise FileNotFoundError("--reuse-static-results requires static_control.npz and static_sdv.npz")
            rc,rs=(_load_estimator(cached[0]),_load_estimator(cached[1]))
        else:
            rc=fit_static_latent(Xc,lambda_w=args.lambda_w,lambda_latent=args.static_lambda_latent,w_threshold=args.edge_threshold,**common); rs=fit_static_latent(Xs,lambda_w=args.lambda_w,lambda_latent=args.static_lambda_latent,w_threshold=args.edge_threshold,**common)
        Sc,Ss=latent_similarity(rc),latent_similarity(rs)
        save_result(args.output_dir/"static_control.npz",rc,control.roi_names,display,control.subject_ids,{"X_shape":Xc.shape}); save_result(args.output_dir/"static_sdv.npz",rs,sdv.roi_names,display,sdv.subject_ids,{"X_shape":Xs.shape})
        plot_matrix_comparison(rc.W0,rs.W0,display,("Control W","SDV W","SDV - Control"),args.output_dir/"static_W_comparison.png")
        plot_matrix_comparison(Sc,Ss,display,("Control LL^T","SDV LL^T","SDV - Control"),args.output_dir/"static_latent_comparison.png",True)
        plot_graph_comparison(rc.W0,rs.W0,display,args.output_dir/"static_graph_comparison.png",args.edge_threshold)
        summary["static"]={"control":model_statistics(rc.W0,Sc,rc.diagnostics["h_thresholded"]),"sdv":model_statistics(rs.W0,Ss,rs.diagnostics["h_thresholded"]),"top_W_changes":top_matrix_changes(rc.W0,rs.W0,display),"top_latent_changes":top_matrix_changes(Sc,Ss,display)}
        _print("static summary",summary["static"])
        if args.case2_baselines:
            fci_kwargs={"depth":args.fci_depth,"show_progress":False}
            (args.output_dir/"case2_baseline_config.json").write_text(json.dumps({"lambda_w":args.lambda_w,"edge_threshold":args.edge_threshold,"fci_alpha":args.fci_alpha,"fci_settings":fci_kwargs,"note":"FCI depth=-1 is unrestricted; nonnegative depths are explicitly truncated integration runs."},indent=2),encoding="utf-8")
            bc=fit_case2_baselines(Xc,args.lambda_w,args.edge_threshold,args.fci_alpha,not args.skip_fci,fci_kwargs=fci_kwargs); bs=fit_case2_baselines(Xs,args.lambda_w,args.edge_threshold,args.fci_alpha,not args.skip_fci,fci_kwargs=fci_kwargs)
            for state,result in (("control",bc),("sdv",bs)):
                note=result["notears"]; np.savez_compressed(args.output_dir/f"notears_{state}.npz",W=note.W0,W_raw=note.W_raw,roi_names=control.roi_names,display_names=display,diagnostics_json=json.dumps(note.diagnostics))
                if result["fci"] is not None: np.savez_compressed(args.output_dir/f"fci_{state}.npz",endpoints=result["fci"].graph,roi_names=control.roi_names,display_names=display,diagnostics_json=json.dumps(result["fci"].diagnostics,default=str))
            plot_two_matrix_comparison(rc.W0,bc["notears"].W0,display,("Latent Case 2 Control W","Standard NOTEARS Control W"),args.output_dir/"static_control_proposed_vs_notears.png")
            plot_two_matrix_comparison(rs.W0,bs["notears"].W0,display,("Latent Case 2 SDV W","Standard NOTEARS SDV W"),args.output_dir/"static_sdv_proposed_vs_notears.png")
            plot_matrix_comparison(bc["notears"].W0,bs["notears"].W0,display,("Control NOTEARS","SDV NOTEARS","SDV - Control"),args.output_dir/"notears_W_comparison.png")
            plot_graph_comparison(rc.W0,bc["notears"].W0,display,args.output_dir/"static_control_proposed_vs_notears_graph.png",args.edge_threshold,("Proposed Control","NOTEARS Control"))
            plot_graph_comparison(rs.W0,bs["notears"].W0,display,args.output_dir/"static_sdv_proposed_vs_notears_graph.png",args.edge_threshold,("Proposed SDV","NOTEARS SDV"))
            baseline_rows=[]
            for method,state,result in (("Proposed latent Case 2","Control",rc),("Proposed latent Case 2","SDV",rs),("Standard NOTEARS","Control",bc["notears"]),("Standard NOTEARS","SDV",bs["notears"])):
                baseline_rows.append({"Method":method,"State":state,"Directed edges":int(np.count_nonzero(result.W0)),"Other/PAG edges":"N/A","h(W)":result.diagnostics.get("h_thresholded"),"Fit loss":result.diagnostics.get("fit_loss")})
            if bc["fci"] is not None:
                plot_pag_comparison(bc["fci"].graph,bs["fci"].graph,display,args.output_dir/"fci_pag_comparison.png"); plot_skeleton_comparison(pag_skeleton(bc["fci"].graph),pag_skeleton(bs["fci"].graph),display,args.output_dir/"fci_skeleton_comparison.png")
                for state,result in (("Control",bc["fci"]),("SDV",bs["fci"])): baseline_rows.append({"Method":"FCI","State":state,"Directed edges":result.diagnostics["directed_orientations"],"Other/PAG edges":result.diagnostics["adjacencies"]-result.diagnostics["directed_orientations"],"h(W)":"N/A","Fit loss":"N/A"})
            import pandas as pd
            pd.DataFrame(baseline_rows).to_csv(args.output_dir/"case2_baseline_summary.csv",index=False); summary["case2_baselines"]=baseline_rows
            structural={"control":compare_case2_structures(rc.W0,bc["notears"].W0,display,args.edge_threshold,bc["fci"].graph if bc["fci"] else None),"sdv":compare_case2_structures(rs.W0,bs["notears"].W0,display,args.edge_threshold,bs["fci"].graph if bs["fci"] else None)}
            (args.output_dir/"case2_structure_comparison.json").write_text(json.dumps(structural,indent=2),encoding="utf-8")
        if args.bootstrap_reps:
            model_kwargs=dict(common,lambda_w=args.lambda_w,lambda_latent=args.static_lambda_latent,w_threshold=args.edge_threshold,verbose=False)
            boot=paired_subject_bootstrap_case2(control,sdv,args.output_dir/"case2_bootstrap",args.bootstrap_reps,args.bootstrap_seed,args.bootstrap_jobs,args.bootstrap_resume,args.bootstrap_checkpoint_every,model_kwargs)
            stats=compute_bootstrap_edge_statistics(rs.W_raw-rc.W_raw,boot["delta_W"]); stability=compute_edge_selection_stability(boot["W_control"],boot["W_sdv"],args.stability_thresholds)
            frame=edge_results_dataframe(display,rc.W_raw,rs.W_raw,stats,stability,args.edge_threshold); frame.to_csv(args.output_dir/"case2_bootstrap_edge_results.csv",index=False); frame[frame.fdr_significant].to_csv(args.output_dir/"case2_bootstrap_fdr_edges.csv",index=False)
            nodes=node_reorganization(rs.W_raw-rc.W_raw,display); nodes.to_csv(args.output_dir/"case2_node_reorganization.csv",index=False)
            plot_bootstrap_edge_significance(rs.W_raw-rc.W_raw,stats,display,args.output_dir/"case2_bootstrap_delta_significance.png", "Observed raw SDV - Control W"); plot_bootstrap_edge_stability(stability,display,args.edge_threshold,args.output_dir/"case2_bootstrap_edge_stability.png"); plot_bootstrap_edge_intervals(frame,args.output_dir/"case2_bootstrap_edge_intervals.png"); plot_node_reorganization(nodes,args.output_dir/"case2_node_reorganization.png"); plot_key_edge_bootstrap_distributions(frame,boot["delta_W"],args.output_dir/"case2_bootstrap_key_edge_distributions.png")
            summary["bootstrap"]={k:boot[k] for k in ("requested","completed","successful","failed","failure_rate")}; summary["bootstrap"]["fdr_supported_edges"]=int(frame.fdr_significant.sum()); _print("bootstrap summary",summary["bootstrap"])

        if args.sex_analysis:
            sex_out = args.output_dir / "sex_analysis"
            sex_out.mkdir(parents=True, exist_ok=True)

            # Convert integer patient IDs to the SubjectNNN convention used by ConnROIData.
            sex_by_subject = {
                f"Subject{patient_id - 1:03d}": sex
                for patient_id, sex in patient_sex.items()
            }

            print("Analyzed subjects:", control.subject_ids)
            print("Sex mapping:", sex_by_subject)

            missing = [
                subject
                for subject in control.subject_ids
                if subject not in sex_by_subject
            ]

            extra = [
                subject
                for subject in sex_by_subject
                if subject not in control.subject_ids
            ]

            if missing or extra:
                raise ValueError(
                    "patient_sex / CONN subject mapping mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            male_ids = tuple(
                subject for subject in control.subject_ids
                if sex_by_subject[subject] == "Male"
            )
            female_ids = tuple(
                subject for subject in control.subject_ids
                if sex_by_subject[subject] == "Female"
            )
            if len(male_ids) != 9 or len(female_ids) != 10:
                raise ValueError(
                    "Unexpected sex split: "
                    f"{len(male_ids)} Male, {len(female_ids)} Female. "
                    "Check patient_sex-to-CONN subject mapping."
                )

            print(
                f"Sex analysis: {len(male_ids)} Male, "
                f"{len(female_ids)} Female"
            )

            male_control = subset_conn_roi_data(control, male_ids)
            male_sdv = subset_conn_roi_data(sdv, male_ids)
            female_control = subset_conn_roi_data(control, female_ids)
            female_sdv = subset_conn_roi_data(sdv, female_ids)

            sex_data_summary = {
                "Male": {
                    "n": len(male_ids),
                    "subjects": list(male_ids),
                    "control": data_summary(male_control, "Male Control"),
                    "sdv": data_summary(male_sdv, "Male SDV"),
                },
                "Female": {
                    "n": len(female_ids),
                    "subjects": list(female_ids),
                    "control": data_summary(female_control, "Female Control"),
                    "sdv": data_summary(female_sdv, "Female SDV"),
                },
            }

            (sex_out / "sex_subjects.json").write_text(
                json.dumps(sex_data_summary, indent=2),
                encoding="utf-8",
            )

            # ---------------------------------------------------------------
            # Point estimates
            # ---------------------------------------------------------------

            sex_paths = {
                "male_control": sex_out / "static_male_control.npz",
                "male_sdv": sex_out / "static_male_sdv.npz",
                "female_control": sex_out / "static_female_control.npz",
                "female_sdv": sex_out / "static_female_sdv.npz",
            }

            if args.reuse_sex_results:
                missing_paths = [
                    path for path in sex_paths.values()
                    if not path.exists()
                ]
                if missing_paths:
                    raise FileNotFoundError(
                        "--reuse-sex-results requires all four sex-specific fits; "
                        f"missing: {missing_paths}"
                    )

                r_mc = _load_estimator(sex_paths["male_control"])
                r_ms = _load_estimator(sex_paths["male_sdv"])
                r_fc = _load_estimator(sex_paths["female_control"])
                r_fs = _load_estimator(sex_paths["female_sdv"])

            else:
                X_mc = build_static_fmri_matrix(male_control)
                X_ms = build_static_fmri_matrix(male_sdv)
                X_fc = build_static_fmri_matrix(female_control)
                X_fs = build_static_fmri_matrix(female_sdv)

                sex_common = dict(
                    k=args.k,
                    max_outer_iter=args.max_outer_iter,
                    h_tol=args.h_tol,
                    inner_max_iter=args.inner_max_iter,
                    random_state=args.seed,
                    verbose=True,
                )

                fit_kwargs = dict(
                    lambda_w=args.lambda_w,
                    lambda_latent=args.static_lambda_latent,
                    w_threshold=args.edge_threshold,
                    **sex_common,
                )

                print("Fitting Male Control...")
                r_mc = fit_static_latent(X_mc, **fit_kwargs)

                print("Fitting Male SDV...")
                r_ms = fit_static_latent(X_ms, **fit_kwargs)

                print("Fitting Female Control...")
                r_fc = fit_static_latent(X_fc, **fit_kwargs)

                print("Fitting Female SDV...")
                r_fs = fit_static_latent(X_fs, **fit_kwargs)

                save_result(
                    sex_paths["male_control"],
                    r_mc,
                    control.roi_names,
                    display,
                    male_ids,
                    {"sex": "Male", "state": "Control", "X_shape": X_mc.shape},
                )
                save_result(
                    sex_paths["male_sdv"],
                    r_ms,
                    sdv.roi_names,
                    display,
                    male_ids,
                    {"sex": "Male", "state": "SDV", "X_shape": X_ms.shape},
                )
                save_result(
                    sex_paths["female_control"],
                    r_fc,
                    control.roi_names,
                    display,
                    female_ids,
                    {"sex": "Female", "state": "Control", "X_shape": X_fc.shape},
                )
                save_result(
                    sex_paths["female_sdv"],
                    r_fs,
                    sdv.roi_names,
                    display,
                    female_ids,
                    {"sex": "Female", "state": "SDV", "X_shape": X_fs.shape},
                )

            # Raw matrices for quantitative comparison/inference.
            W_mc = r_mc.W_raw
            W_ms = r_ms.W_raw
            W_fc = r_fc.W_raw
            W_fs = r_fs.W_raw

            delta_male = W_ms - W_mc
            delta_female = W_fs - W_fc

            # Positive = bladder-state change is more positive/larger in women.
            interaction = delta_female - delta_male

            np.savez_compressed(
                sex_out / "sex_change_matrices.npz",
                W_male_control=W_mc,
                W_male_sdv=W_ms,
                W_female_control=W_fc,
                W_female_sdv=W_fs,
                delta_male=delta_male,
                delta_female=delta_female,
                sex_by_state_interaction=interaction,
                display_names=np.asarray(display),
            )

            # Additional plots
            male_nodes = node_reorganization(delta_male, display)
            female_nodes = node_reorganization(delta_female, display)

            male_nodes.to_csv(
                sex_out / "male_node_reorganization.csv",
                index=False,
            )
            female_nodes.to_csv(
                sex_out / "female_node_reorganization.csv",
                index=False,
            )

            def edge_mask(W):
                mask = np.abs(W) >= args.edge_threshold
                np.fill_diagonal(mask, False)
                return mask

            def overlap_stats(A, B):
                intersection = A & B
                union = A | B
                return {
                    "edges_a": int(A.sum()),
                    "edges_b": int(B.sum()),
                    "shared_edges": int(intersection.sum()),
                    "a_only": int((A & ~B).sum()),
                    "b_only": int((B & ~A).sum()),
                    "jaccard": (
                        float(intersection.sum() / union.sum())
                        if union.sum()
                        else 1.0
                    ),
                }

            sex_overlap = {
                "control_male_vs_female": overlap_stats(
                    edge_mask(r_mc.W0),
                    edge_mask(r_fc.W0),
                ),
                "sdv_male_vs_female": overlap_stats(
                    edge_mask(r_ms.W0),
                    edge_mask(r_fs.W0),
                ),
                "reorganization": {
                    "male_frobenius": float(np.linalg.norm(delta_male)),
                    "female_frobenius": float(np.linalg.norm(delta_female)),
                    "male_abs_sum": float(np.abs(delta_male).sum()),
                    "female_abs_sum": float(np.abs(delta_female).sum()),
                    "interaction_frobenius": float(np.linalg.norm(interaction)),
                },
            }

            (sex_out / "sex_graph_summary.json").write_text(
                json.dumps(sex_overlap, indent=2),
                encoding="utf-8",
            )

            male_stats = None
            female_stats = None
            interaction_stats = None
            male_control_stats = None
            male_sdv_stats = None
            female_control_stats = None
            female_sdv_stats = None

            if args.sex_bootstrap_reps:
                sex_model_kwargs = dict(
                    k=args.k,
                    max_outer_iter=args.max_outer_iter,
                    h_tol=args.h_tol,
                    inner_max_iter=args.inner_max_iter,
                    random_state=args.seed,
                    lambda_w=args.lambda_w,
                    lambda_latent=args.static_lambda_latent,
                    w_threshold=args.edge_threshold,
                    verbose=False,
                )

                male_boot = paired_subject_bootstrap_case2(
                    male_control,
                    male_sdv,
                    sex_out / "bootstrap_male",
                    reps=args.sex_bootstrap_reps,
                    seed=args.sex_bootstrap_seed,
                    jobs=args.sex_bootstrap_jobs,
                    resume=args.sex_bootstrap_resume,
                    checkpoint_every=args.sex_bootstrap_checkpoint_every,
                    model_kwargs=sex_model_kwargs,
                )

                # Different seed so the two independent groups do not receive
                # mechanically identical bootstrap-index sequences.
                female_boot = paired_subject_bootstrap_case2(
                    female_control,
                    female_sdv,
                    sex_out / "bootstrap_female",
                    reps=args.sex_bootstrap_reps,
                    seed=args.sex_bootstrap_seed + 1,
                    jobs=args.sex_bootstrap_jobs,
                    resume=args.sex_bootstrap_resume,
                    checkpoint_every=args.sex_bootstrap_checkpoint_every,
                    model_kwargs=sex_model_kwargs,
                )

                male_stats = compute_bootstrap_edge_statistics(
                    delta_male,
                    male_boot["delta_W"],
                )

                female_stats = compute_bootstrap_edge_statistics(
                    delta_female,
                    female_boot["delta_W"],
                )

                # Within-state nonzero tests, useful for thick DAG edges.
                male_control_stats = compute_bootstrap_nonzero_statistics(
                    W_mc,
                    male_boot["W_control"],
                )
                male_sdv_stats = compute_bootstrap_nonzero_statistics(
                    W_ms,
                    male_boot["W_sdv"],
                )
                female_control_stats = compute_bootstrap_nonzero_statistics(
                    W_fc,
                    female_boot["W_control"],
                )
                female_sdv_stats = compute_bootstrap_nonzero_statistics(
                    W_fs,
                    female_boot["W_sdv"],
                )

                interaction_boot, common_indices = (
                    combine_sex_interaction_bootstraps(
                        male_boot,
                        female_boot,
                    )
                )

                interaction_stats = compute_bootstrap_edge_statistics(
                    interaction,
                    interaction_boot,
                )

                np.savez_compressed(
                    sex_out / "sex_bootstrap_statistics.npz",
                    male_delta_p=male_stats["p_boot"],
                    male_delta_q=male_stats["q_fdr"],
                    female_delta_p=female_stats["p_boot"],
                    female_delta_q=female_stats["q_fdr"],
                    interaction_p=interaction_stats["p_boot"],
                    interaction_q=interaction_stats["q_fdr"],
                    common_bootstrap_indices=np.asarray(common_indices),
                )

                import pandas as pd

                def interaction_frame(
                    names,
                    observed,
                    stats,
                ):
                    rows = []

                    for i, source in enumerate(names):
                        for j, target in enumerate(names):
                            if i == j:
                                continue

                            rows.append({
                                "source_index": i,
                                "source_name": source,
                                "target_index": j,
                                "target_name": target,
                                "interaction_observed": observed[i, j],
                                "bootstrap_mean": stats["mean"][i, j],
                                "ci_2.5": stats["ci_low"][i, j],
                                "ci_97.5": stats["ci_high"][i, j],
                                "p_boot": stats["p_boot"][i, j],
                                "q_fdr": stats["q_fdr"][i, j],
                                "fdr_significant": stats["fdr_significant"][i, j],
                            })

                    frame = pd.DataFrame(rows)
                    frame["abs_interaction"] = (
                        frame["interaction_observed"].abs()
                    )

                    return frame.sort_values(
                        ["q_fdr", "abs_interaction"],
                        ascending=[True, False],
                    )

                interaction_df = interaction_frame(
                    display,
                    interaction,
                    interaction_stats,
                )

                interaction_df.to_csv(
                    sex_out / "sex_interaction_edge_results.csv",
                    index=False,
                )

                interaction_df[
                    interaction_df.p_boot < 0.05
                ].to_csv(
                    sex_out / "sex_interaction_nominal_edges.csv",
                    index=False,
                )

                interaction_df[
                    interaction_df.q_fdr < 0.05
                ].to_csv(
                    sex_out / "sex_interaction_fdr_edges.csv",
                    index=False,
                )

                male_stability = compute_edge_selection_stability(
                    male_boot["W_control"],
                    male_boot["W_sdv"],
                    args.stability_thresholds,
                )

                female_stability = compute_edge_selection_stability(
                    female_boot["W_control"],
                    female_boot["W_sdv"],
                    args.stability_thresholds,
                )

                male_frame = edge_results_dataframe(
                    display,
                    W_mc,
                    W_ms,
                    male_stats,
                    male_stability,
                    args.edge_threshold,
                )

                female_frame = edge_results_dataframe(
                    display,
                    W_fc,
                    W_fs,
                    female_stats,
                    female_stability,
                    args.edge_threshold,
                )

                male_frame.to_csv(
                    sex_out / "male_state_change_edge_results.csv",
                    index=False,
                )
                female_frame.to_csv(
                    sex_out / "female_state_change_edge_results.csv",
                    index=False,
                )

                male_frame[male_frame.p_boot < .05].to_csv(
                    sex_out / "male_state_change_nominal_edges.csv",
                    index=False,
                )
                female_frame[female_frame.p_boot < .05].to_csv(
                    sex_out / "female_state_change_nominal_edges.csv",
                    index=False,
                )

                # Bootstrap Summary
                sex_bootstrap_summary = {
                    "male": {
                        key: male_boot[key]
                        for key in (
                            "requested",
                            "completed",
                            "successful",
                            "failed",
                            "failure_rate",
                        )
                    },
                    "female": {
                        key: female_boot[key]
                        for key in (
                            "requested",
                            "completed",
                            "successful",
                            "failed",
                            "failure_rate",
                        )
                    },
                    "interaction_valid_pairs": len(common_indices),
                    "male_nominal_state_changes": int(
                        np.nansum(male_stats["p_boot"] < 0.05)
                    ),
                    "female_nominal_state_changes": int(
                        np.nansum(female_stats["p_boot"] < 0.05)
                    ),
                    "interaction_nominal_edges": int(
                        np.nansum(
                            interaction_stats["p_boot"] < 0.05
                        )
                    ),
                    "interaction_fdr_edges": int(
                        np.nansum(
                            interaction_stats["q_fdr"] < 0.05
                        )
                    ),
                }

                (
                    sex_out / "sex_bootstrap_summary.json"
                ).write_text(
                    json.dumps(
                        sex_bootstrap_summary,
                        indent=2,
                        default=float,
                    ),
                    encoding="utf-8",
                )

            if args.sex_bootstrap_reps:
                mc_sig = male_control_stats["p_boot"] < 0.05
                ms_sig = male_sdv_stats["p_boot"] < 0.05
                fc_sig = female_control_stats["p_boot"] < 0.05
                fs_sig = female_sdv_stats["p_boot"] < 0.05

                dm_sig = male_stats["p_boot"] < 0.05
                df_sig = female_stats["p_boot"] < 0.05
            else:
                mc_sig = ms_sig = fc_sig = fs_sig = None
                dm_sig = df_sig = None

            # Plots
            plot_sex_dag_comparison(
                r_mc.W0,
                r_ms.W0,
                r_fc.W0,
                r_fs.W0,
                display,
                sex_out / "sex_static_dag_comparison.png",
                threshold=args.edge_threshold,
                male_control_sig=mc_sig,
                male_sdv_sig=ms_sig,
                female_control_sig=fc_sig,
                female_sdv_sig=fs_sig,
                male_delta_sig=dm_sig,
                female_delta_sig=df_sig,
            )

            plot_sex_change_heatmaps(
                delta_male,
                delta_female,
                interaction,
                display,
                sex_out / "sex_state_change_heatmaps.png",
                male_stats=male_stats,
                female_stats=female_stats,
                interaction_stats=interaction_stats,
            )

            node_stats = None
            if args.sex_bootstrap_reps:
                node_stats = compute_sex_node_statistics(
                    delta_male, delta_female, male_boot, female_boot, display,
                )
                node_stats.to_csv(
                    sex_out / "sex_node_reorganization_statistics.csv", index=False,
                )
            plot_sex_node_reorganization(
                male_nodes,
                female_nodes,
                sex_out / "sex_node_reorganization.png",
                node_statistics=node_stats,
            )

            plot_anatomical_directed_difference(
                interaction,
                control.roi_xyz,
                display,
                sex_out / "sex_interaction_anatomical.png",
                view="coronal",
                min_abs_change=args.edge_threshold,
                title=(
                    "Sex × Bladder-State Interaction\n"
                    "(Female SDV − Control) − "
                    "(Male SDV − Control)"
                ),
                positive_color="darkorange",
                negative_color="purple",
                positive_label=(
                    "Larger / more positive SDV change in women"
                ),
                negative_label=(
                    "Larger / more positive SDV change in men"
                ),
                magnitude_label=(
                    r"Arrow width $\propto "
                    r"|\Delta W_{Female}-\Delta W_{Male}|$"
                ),
            )

            plot_anatomical_directed_difference(
                delta_male,
                control.roi_xyz,
                display,
                sex_out / "male_sdv_control_anatomical.png",
                view="coronal",
                min_abs_change=args.edge_threshold,
                title="Male: SDV − Control",
                positive_color="tab:red",
                negative_color="tab:blue",
                positive_label="Increase in SDV",
                negative_label="Decrease in SDV",
            )

            plot_anatomical_directed_difference(
                delta_female,
                control.roi_xyz,
                display,
                sex_out / "female_sdv_control_anatomical.png",
                view="coronal",
                min_abs_change=args.edge_threshold,
                title="Female: SDV − Control",
                positive_color="tab:red",
                negative_color="tab:blue",
                positive_label="Increase in SDV",
                negative_label="Decrease in SDV",
            )

    if not args.static_only:
        X0c,Xlc,_=build_multisubject_lagged_data(control,args.p); X0s,Xls,_=build_multisubject_lagged_data(sdv,args.p); _print("dynamic shapes",{"control_X0":X0c.shape,"control_Xlags":Xlc.shape,"sdv_X0":X0s.shape,"sdv_Xlags":Xls.shape})
        dc=fit_dynamic_latent([control.timeseries[s] for s in control.subject_ids],args.p,lambda_0=args.lambda_0,lambda_lag=args.lambda_lag,lambda_latent=args.dynamic_lambda_latent,w0_threshold=args.edge_threshold,wlag_threshold=args.edge_threshold,**common)
        ds=fit_dynamic_latent([sdv.timeseries[s] for s in sdv.subject_ids],args.p,lambda_0=args.lambda_0,lambda_lag=args.lambda_lag,lambda_latent=args.dynamic_lambda_latent,w0_threshold=args.edge_threshold,wlag_threshold=args.edge_threshold,**common)
        Sc,Ss=latent_similarity(dc),latent_similarity(ds)
        save_result(args.output_dir/"dynamic_control.npz",dc,control.roi_names,display,control.subject_ids,{"X0_shape":X0c.shape,"Xlags_shape":Xlc.shape}); save_result(args.output_dir/"dynamic_sdv.npz",ds,sdv.roi_names,display,sdv.subject_ids,{"X0_shape":X0s.shape,"Xlags_shape":Xls.shape})
        plot_matrix_comparison(dc.W0,ds.W0,display,("Control W0","SDV W0","SDV - Control W0"),args.output_dir/"dynamic_W0_comparison.png")
        for tau in range(args.p): plot_matrix_comparison(dc.W_lags[tau],ds.W_lags[tau],display,(f"Control W{tau+1}",f"SDV W{tau+1}",f"SDV - Control W{tau+1}"),args.output_dir/f"dynamic_Wlag{tau+1}_comparison.png")
        plot_matrix_comparison(Sc,Ss,display,("Control LL^T","SDV LL^T","SDV - Control"),args.output_dir/"dynamic_latent_comparison.png",True)
        def dynstats(r,S):
            out=model_statistics(r.W0,S,r.diagnostics["h_thresholded"]); out.update(lagged_edges=[int(np.count_nonzero(w)) for w in r.W_lags],lagged_abs_sums=[float(np.abs(w).sum()) for w in r.W_lags]); return out
        summary["dynamic"]={"control":dynstats(dc,Sc),"sdv":dynstats(ds,Ss),"top_W0_changes":top_matrix_changes(dc.W0,ds.W0,display),"top_lag_changes":{str(t+1):top_matrix_changes(dc.W_lags[t],ds.W_lags[t],display) for t in range(args.p)},"top_latent_changes":top_matrix_changes(Sc,Ss,display)}
        _print("dynamic summary",summary["dynamic"])
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__=="__main__": main()
