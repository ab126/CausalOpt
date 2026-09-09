"""fMRI model summaries and backward-compatible data/plotting imports.

New code can import loaders from ``fmri_data`` and plots from ``fmri_plotting``.
"""
from __future__ import annotations

import json
import numpy as np

from .fmri_data import (
    ConnROIData,
    build_multisubject_lagged_data,
    build_static_fmri_matrix,
    data_summary,
    load_conn_roi_zip,
    load_conn_subject_timeseries,
    load_fmri_state_data,
    load_roi_display_names,
    load_tsv_state,
    load_paired_tsv_states,
    pair_fmri_states,
    subset_conn_roi_data,
    validate_paired_states,
)
from .fmri_plotting import (
    PLOT_FONT_SCALE,
    plot_matrix_comparison,
    plot_dynamic_matrix_comparison,
    plot_graph_comparison,
    plot_two_matrix_comparison,
    plot_pag_comparison,
    plot_skeleton_comparison,
    plot_bootstrap_edge_significance,
    plot_bootstrap_edge_stability,
    plot_bootstrap_edge_intervals,
    plot_nominal_bootstrap_edge_intervals,
    plot_node_reorganization,
    plot_key_edge_bootstrap_distributions,
    BLADDER19_LABELS,
    plot_anatomical_directed_difference,
    plot_anatomical_graph_difference,
    plot_sex_dag_comparison,
    plot_sex_change_heatmaps,
    plot_sex_node_reorganization,
)


def latent_similarity(result):
    return np.zeros((result.W0.shape[0],)*2) if result.L is None else result.L@result.L.T


def report_effective_latent_rank(result, state, eps=1e-3):
    """Print latent-factor norms and return the effective rank at ``eps``."""
    L = np.zeros((result.W0.shape[0], 0)) if result.L is None else np.asarray(result.L)
    col_norms = np.linalg.norm(L, axis=0)
    rank = int(np.sum(col_norms > eps))
    print(f"{state}: k_max={L.shape[1]}, k_hat={rank}, epsilon={eps:g}")
    print(f"  column norms={np.array2string(col_norms, precision=6)}")
    print(f"  ||L||_F={np.linalg.norm(L):.6e}, ||LL^T||_F={np.linalg.norm(L @ L.T):.6e}")
    return rank


def compare_matrices(control,sdv):
    a=np.asarray(control,float); b=np.asarray(sdv,float)
    if a.shape!=b.shape: raise ValueError("comparison matrices must have equal shapes")
    return b-a


def model_statistics(W,S,h,threshold=0.0):
    mask=np.abs(W)>threshold; off=~np.eye(S.shape[0],dtype=bool)
    return {"directed_edges":int(mask.sum()),"directed_abs_sum":float(np.abs(W[mask]).sum()),"directed_abs_mean":float(np.abs(W[mask]).mean()) if mask.any() else 0.0,"W_fro":float(np.linalg.norm(W)),"h":float(h),"latent_fro":float(np.linalg.norm(S)),"latent_offdiag_abs_mean":float(np.abs(S[off]).mean())}


def top_matrix_changes(control,sdv,names,n=10,exclude_diagonal=True):
    delta=compare_matrices(control,sdv); mask=np.ones(delta.shape,bool)
    if exclude_diagonal: np.fill_diagonal(mask,False)
    flat=np.argwhere(mask); order=np.argsort(np.abs(delta[mask]))[::-1][:n]
    return [{"source":names[i],"target":names[j],"control":float(control[i,j]),"sdv":float(sdv[i,j]),"change":float(delta[i,j])} for i,j in flat[order]]


def save_result(path,result,roi_names,display_names,subject_ids,extra=None):
    payload={"W":result.W0,"W_raw":result.W_raw,"Z":result.Z,"L":result.L,"S_latent":latent_similarity(result),"roi_names":np.asarray(roi_names),"display_names":np.asarray(display_names),"subject_ids":np.asarray(subject_ids),"runtime_seconds":result.runtime_seconds,"diagnostics_json":json.dumps(result.diagnostics,default=lambda x:np.asarray(x).tolist())}
    if result.W_lags is not None: payload.update(W0=result.W0,W_lags=result.W_lags,W_lags_raw=result.W_lags_raw)
    if extra: payload.update(extra)
    np.savez_compressed(path,**payload)


