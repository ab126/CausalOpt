"""CONN ROI loading and Control-versus-SDV model comparison utilities."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json
import re
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

from scipy.io import loadmat

ROI_RE = re.compile(r"^Bladder Network 19\.cluster(\d{3})$")
FILE_RE = re.compile(r"ROI_Subject(\d+)_Session(\d{3})\.mat$", re.I)


@dataclass(frozen=True)
class ConnROIData:
    timeseries: dict[str, np.ndarray]
    roi_names: tuple[str, ...]
    roi_xyz: np.ndarray
    subject_ids: tuple[str, ...]
    session: str


def _mat_strings(value):
    out=[]
    for item in np.asarray(value,dtype=object).ravel():
        while isinstance(item,np.ndarray) and item.size==1: item=item.item()
        out.append(str(item))
    return out


def load_conn_subject_timeseries(mat, expected_session=None):
    """Extract the numerically ordered custom 19-ROI atlas from one CONN MAT dict."""
    for key in ("names","data","xyz"):
        if key not in mat: raise ValueError(f"CONN MAT is missing {key!r}")
    names=_mat_strings(mat["names"]); data=np.asarray(mat["data"],dtype=object).ravel(); xyz=np.asarray(mat["xyz"],dtype=object).ravel()
    if not (len(names)==len(data)==len(xyz)): raise ValueError("names/data/xyz lengths differ")
    selected=[]
    for name,values,coords in zip(names,data,xyz):
        match=ROI_RE.fullmatch(name)
        if match: selected.append((int(match.group(1)),name,np.asarray(values,float).ravel(),np.asarray(coords,float).ravel()))
    selected.sort(key=lambda row:row[0])
    if [row[0] for row in selected]!=list(range(1,20)): raise ValueError("expected exactly Bladder Network 19 clusters 001..019")
    lengths={len(row[2]) for row in selected}
    if len(lengths)!=1: raise ValueError("the 19 ROI series do not have identical lengths")
    X=np.column_stack([row[2] for row in selected])
    if not np.isfinite(X).all(): raise ValueError("ROI time series contain NaN or Inf")
    coords=np.vstack([row[3] for row in selected])
    return X,tuple(row[1] for row in selected),coords


def load_conn_roi_zip(path,expected_session):
    """Read a Session001/002 CONN archive directly and preserve all subject IDs."""
    path=Path(path); timeseries={}; reference_names=None; reference_xyz=None
    with zipfile.ZipFile(path) as archive:
        members=sorted(n for n in archive.namelist() if n.lower().endswith(".mat"))
        if not members: raise ValueError(f"no MAT files in {path}")
        for member in members:
            match=FILE_RE.search(Path(member).name)
            if not match: continue
            subject=f"Subject{int(match.group(1)):03d}"; session=match.group(2)
            if session!=expected_session: raise ValueError(f"{member} is Session{session}, expected Session{expected_session}")
            if subject in timeseries: raise ValueError(f"duplicate {subject} in {path}")
            X,names,coords=load_conn_subject_timeseries(loadmat(BytesIO(archive.read(member))))
            if reference_names is None: reference_names,reference_xyz=names,coords
            elif names!=reference_names: raise ValueError(f"ROI order differs for {subject}")
            elif not np.allclose(coords,reference_xyz,rtol=1e-6,atol=1e-5,equal_nan=False): raise ValueError(f"ROI coordinates differ for {subject}")
            timeseries[subject]=X
    ids=tuple(sorted(timeseries,key=lambda s:int(re.search(r"\d+",s).group())))
    if not ids: raise ValueError(f"no ROI_Subject*_Session{expected_session}.mat files in {path}")
    return ConnROIData({sid:timeseries[sid] for sid in ids},reference_names,reference_xyz,ids,expected_session)


def validate_paired_states(control,sdv):
    if control.session!="001" or sdv.session!="002": raise ValueError("state mapping must be Session001=control and Session002=SDV")
    if control.subject_ids!=sdv.subject_ids: raise ValueError(f"unpaired subjects: control={control.subject_ids}, SDV={sdv.subject_ids}")
    if control.roi_names!=sdv.roi_names: raise ValueError("ROI ordering differs between states")
    if not np.allclose(control.roi_xyz,sdv.roi_xyz,rtol=1e-6,atol=1e-5): raise ValueError("ROI coordinates differ between states")


def load_fmri_state_data(session1_zip,session2_zip):
    control=load_conn_roi_zip(session1_zip,"001"); sdv=load_conn_roi_zip(session2_zip,"002"); validate_paired_states(control,sdv); return control,sdv


def build_static_fmri_matrix(data): return np.vstack([data.timeseries[s] for s in data.subject_ids])


def build_multisubject_lagged_data(data,p,center=True):
    from .methods.dynamic_latent import multisubject_lagged_views
    return multisubject_lagged_views([data.timeseries[s] for s in data.subject_ids],p,center=center)


def data_summary(data,label):
    lengths=np.array([len(data.timeseries[s]) for s in data.subject_ids])
    return {"state":label,"subjects":len(lengths),"rois":len(data.roi_names),"T_min":int(lengths.min()),"T_median":float(np.median(lengths)),"T_max":int(lengths.max()),"pooled_rows":int(lengths.sum())}


def load_roi_display_names(labels_mat):
    """Load anatomical labels indexed 1..19 from a CONN ``roi_info`` MAT file."""
    mat=loadmat(labels_mat,squeeze_me=True,struct_as_record=False)
    if "roi_info" not in mat: raise ValueError("labels MAT does not contain roi_info")
    mapping={int(item.number):str(item.label) for item in np.atleast_1d(mat["roi_info"]).ravel()}
    if set(mapping)!=set(range(1,20)): raise ValueError("roi_info must map exactly ROI numbers 1..19")
    return tuple(mapping[i] for i in range(1,20))


def latent_similarity(result):
    return np.zeros((result.W0.shape[0],)*2) if result.L is None else result.L@result.L.T


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


def plot_matrix_comparison(control,sdv,names,titles,path,zero_diagonal=False):
    import matplotlib.pyplot as plt
    a=np.array(control,copy=True); b=np.array(sdv,copy=True)
    if zero_diagonal: np.fill_diagonal(a,0); np.fill_diagonal(b,0)
    delta=b-a; state=max(float(np.max(np.abs([a,b]))),1e-12); diff=max(float(np.max(np.abs(delta))),1e-12)
    fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    for ax,m,title,lim in zip(axes,(a,b,delta),titles,(state,state,diff)):
        im=ax.imshow(m,cmap="RdBu_r",vmin=-lim,vmax=lim); ax.set_title(title); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax,shrink=.75)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_graph_comparison(control,sdv,names,path,threshold=.3,titles=("Control causal DAG","SDV causal DAG")):
    import matplotlib.pyplot as plt
    angles=np.linspace(0,2*np.pi,len(names),endpoint=False); positions=np.column_stack((np.cos(angles),np.sin(angles))); limit=max(np.max(np.abs(control)),np.max(np.abs(sdv)),1e-12)
    fig,axes=plt.subplots(1,2,figsize=(16,8),constrained_layout=True)
    for ax,W,title in zip(axes,(control,sdv),titles):
        ax.scatter(positions[:,0],positions[:,1],s=650,c="#d9eaf7",edgecolors="#333",zorder=3)
        for i,(x,y) in enumerate(positions): ax.text(x,y,names[i],ha="center",va="center",fontsize=7,zorder=4)
        for i,j in np.argwhere(np.abs(W)>=threshold):
            color="#b2182b" if W[i,j]>0 else "#2166ac"; width=.5+3*abs(W[i,j])/limit
            start=positions[i]*.91; end=positions[j]*.91
            ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="-|>",color=color,lw=width,shrinkA=8,shrinkB=8,connectionstyle="arc3,rad=.08"),zorder=2)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.axis("off")
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_two_matrix_comparison(left,right,names,titles,path):
    import matplotlib.pyplot as plt
    limit=max(float(np.max(np.abs([left,right]))),1e-12); fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,m,title in zip(axes,(left,right),titles):
        im=ax.imshow(m,cmap="RdBu_r",vmin=-limit,vmax=limit); ax.set_title(title); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax,shrink=.75)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_pag_comparison(control,sdv,names,path):
    """Plot endpoint-coded PAGs; tail, arrow, and circle endpoints remain distinct."""
    import matplotlib.pyplot as plt
    angles=np.linspace(0,2*np.pi,len(names),endpoint=False); pos=np.column_stack((np.cos(angles),np.sin(angles)))
    fig,axes=plt.subplots(1,2,figsize=(16,8),constrained_layout=True)
    for ax,E,title in zip(axes,(control,sdv),("FCI Control PAG","FCI SDV PAG")):
        ax.scatter(pos[:,0],pos[:,1],s=650,c="#f1e5ac",edgecolors="#333",zorder=3)
        for i,(x,y) in enumerate(pos): ax.text(x,y,names[i],ha="center",va="center",fontsize=7,zorder=4)
        for i in range(len(E)):
            for j in range(i+1,len(E)):
                a,b=int(E[i,j]),int(E[j,i])
                if not (a or b): continue
                start,end=pos[i]*.91,pos[j]*.91; ax.plot([start[0],end[0]],[start[1],end[1]],color="#555",lw=1,zorder=1)
                vector=end-start; unit=vector/max(np.linalg.norm(vector),1e-12)
                for point,code,direction in ((start,a,unit),(end,b,-unit)):
                    if code==3: ax.scatter(*point,s=28,facecolors="white",edgecolors="#222",zorder=2)
                    elif code==2: ax.annotate("",xy=point,xytext=point+direction*.09,arrowprops=dict(arrowstyle="-|>",color="#222",lw=1),zorder=2)
                    elif code==1: ax.plot([point[0]-unit[1]*.025,point[0]+unit[1]*.025],[point[1]+unit[0]*.025,point[1]-unit[0]*.025],color="#222",lw=1.5,zorder=2)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.axis("off")
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_skeleton_comparison(control,sdv,names,path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,m,title in zip(axes,(control,sdv),("FCI Control adjacency","FCI SDV adjacency")):
        im=ax.imshow(m,cmap="Greys",vmin=0,vmax=1); ax.set_title(title); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_bootstrap_delta_significance(delta,statistics,names,path):
    import matplotlib.pyplot as plt
    limit=max(float(np.max(np.abs(delta))),1e-12); fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True); im=ax.imshow(delta,cmap="RdBu_r",vmin=-limit,vmax=limit)
    ci=np.argwhere(statistics["ci_excludes_zero"]&~statistics["fdr_significant"]); fdr=np.argwhere(statistics["fdr_significant"])
    if len(ci): ax.scatter(ci[:,1],ci[:,0],marker="o",facecolors="none",edgecolors="black",s=35,label="95% CI excludes 0")
    if len(fdr): ax.scatter(fdr[:,1],fdr[:,0],marker="*",c="#ffd700",edgecolors="black",s=80,label="FDR q < .05")
    ax.set_title("Observed raw SDV - Control W"); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax); 
    if len(ci) or len(fdr): ax.legend(loc="upper left",bbox_to_anchor=(1.12,1))
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_bootstrap_edge_stability(stability,names,threshold,path):
    import matplotlib.pyplot as plt
    values=stability["thresholds"][float(threshold)]; fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,key,title in zip(axes,("control","sdv"),("Control edge-selection probability","SDV edge-selection probability")):
        im=ax.imshow(values[key],cmap="viridis",vmin=0,vmax=1); ax.set_title(f"{title} (|W| > {threshold:g})"); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_bootstrap_edge_intervals(frame,path,top_n=15):
    import matplotlib.pyplot as plt
    selected=frame[frame.fdr_significant].copy()
    if selected.empty: selected=frame.iloc[np.argsort(np.abs(frame.delta_W_observed.to_numpy()))[::-1][:top_n]].copy()
    else: selected=selected.iloc[:top_n]
    selected=selected.iloc[::-1]; labels=[f"{r.source_name} -> {r.target_name}  (q={r.q_fdr:.3g})" for r in selected.itertuples()]
    y=np.arange(len(selected)); x=selected.delta_W_observed.to_numpy(); lo=x-selected["ci_2.5"].to_numpy(); hi=selected["ci_97.5"].to_numpy()-x
    fig,ax=plt.subplots(figsize=(11,max(5,.42*len(selected))),constrained_layout=True); ax.errorbar(x,y,xerr=np.vstack((lo,hi)),fmt="o",color="#2166ac",ecolor="#555",capsize=3); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,labels); ax.set_xlabel("Observed SDV - Control raw W (95% percentile CI)"); ax.set_title("Strongest directed edge differences")
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_nominal_bootstrap_edge_intervals(frame,path,focus_source="L Insula",focus_target="PAG1",max_edges=15):
    """Forest plot for nominal edges, always including and marking a focus edge."""
    import matplotlib.pyplot as plt
    import pandas as pd
    nominal=frame[frame.p_boot<.05].sort_values(["p_boot","abs_delta_W"],ascending=[True,False])
    selected=nominal.head(max_edges).copy() if len(nominal)>max_edges else nominal.copy()
    focus=frame[(frame.source_name==focus_source)&(frame.target_name==focus_target)]
    if not focus.empty and not ((selected.source_name==focus_source)&(selected.target_name==focus_target)).any(): selected=pd.concat([selected,focus.iloc[[0]]],ignore_index=True)
    if selected.empty: return []
    selected=selected.iloc[::-1].reset_index(drop=True); fig,ax=plt.subplots(figsize=(12,max(4,.48*len(selected))),constrained_layout=True)
    for y,row in selected.iterrows():
        color="#b2182b" if bool(row.fdr_significant) else "#2166ac"; marker="D" if row.source_name==focus_source and row.target_name==focus_target else "o"
        ax.hlines(y,row["ci_2.5"],row["ci_97.5"],color=color,lw=2); ax.plot(row.delta_W_observed,y,marker=marker,color=color,markersize=7)
    labels=[f"{r.source_name} -> {r.target_name}  p={r.p_boot:.3g}, q={r.q_fdr:.3g}" for r in selected.itertuples()]
    ax.set_yticks(range(len(selected)),labels); ax.axvline(0,color="black",lw=1); ax.set_xlabel("Observed SDV - Control raw W (95% percentile CI)"); ax.set_title("Nominal bootstrap edge differences (red = BH-FDR; diamond = L Insula -> PAG1)")
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); return labels


def plot_node_reorganization(frame,path,top_n=12):
    import matplotlib.pyplot as plt
    view=frame.head(top_n).iloc[::-1]; fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True); ax.barh(view.roi,view.total_change,color="#4c78a8"); ax.set_xlabel("sum absolute incoming + outgoing change"); ax.set_title("State-dependent ROI reorganization (descriptive)"); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_key_edge_bootstrap_distributions(frame,delta_boot,path,terms=("PAG","Insula","PMC","Motor Area","Cerebell"),max_edges=6):
    """Plot strongest observed differences touching automatically matched key ROIs."""
    import matplotlib.pyplot as plt
    mask=frame.source_name.str.contains("|".join(terms),case=False,regex=True)|frame.target_name.str.contains("|".join(terms),case=False,regex=True)
    selected=frame[mask].copy(); selected["magnitude"]=selected.delta_W_observed.abs(); selected=selected.nlargest(max_edges,"magnitude")
    if selected.empty: return []
    fig,axes=plt.subplots(len(selected),1,figsize=(9,2.3*len(selected)),squeeze=False,constrained_layout=True)
    labels=[]
    for ax,(_,row) in zip(axes.ravel(),selected.iterrows()):
        values=delta_boot[:,int(row.source_index),int(row.target_index)]; label=f"{row.source_name} -> {row.target_name}"; labels.append(label)
        ax.hist(values,bins=min(30,max(5,len(values)//2)),color="#8da0cb",alpha=.8); ax.axvline(0,color="black",lw=1); ax.axvline(row.delta_W_observed,color="#b2182b",lw=2,label="observed"); ax.axvline(row["ci_2.5"],color="#555",ls="--"); ax.axvline(row["ci_97.5"],color="#555",ls="--",label="95% CI"); ax.set_title(f"{label} (q={row.q_fdr:.3g})"); ax.legend(fontsize=8)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); return labels

# Anatomical SDV difference plot
BLADDER19_LABELS = (
    "Cerebellum 1",
    "Cerebellum 2",
    "Cerebellum 3",
    "DLPFC 1",
    "DLPFC 2",
    "Medial Frontal",
    "PAG 1",
    "PAG 2",
    "PAG 3",
    "PMC 1",
    "PMC 2",
    "SMA 1",
    "SMA 2",
    "dACC",
    "IFG",
    "L Insula",
    "mPFC",
    "R Insula",
    "Thalamus",
)

def _draw_axial_brain_outline(ax):
    """Draw a schematic axial brain boundary in MNI x-y space."""
    brain = Ellipse(
        (0, -12),
        width=155,
        height=185,
        facecolor="0.97",
        edgecolor="0.45",
        linewidth=2.2,
        zorder=0,
    )
    ax.add_patch(brain)

    # Approximate interhemispheric fissure.
    ax.plot(
        [0, 0],
        [-98, 72],
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )

def _draw_brain_outline(ax, view):
    """Draw a simple schematic brain boundary for an MNI projection."""

    if view == "coronal":
        # x-z plane, viewed along y
        brain = Ellipse(
            (0, 15),
            width=155,
            height=125,
            facecolor="0.97",
            edgecolor="0.45",
            linewidth=2.0,
            zorder=0,
        )
        ax.add_patch(brain)

        # Approximate interhemispheric fissure
        ax.plot(
            [0, 0], [-45, 78],
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
            zorder=1,
        )

    elif view == "sagittal":
        # y-z plane, viewed along x
        brain = Ellipse(
            (-5, 15),
            width=145,
            height=125,
            facecolor="0.97",
            edgecolor="0.45",
            linewidth=2.0,
            zorder=0,
        )
        ax.add_patch(brain)

    elif view == "axial":
        # x-y plane, viewed along z
        brain = Ellipse(
            (0, -12),
            width=155,
            height=185,
            facecolor="0.97",
            edgecolor="0.45",
            linewidth=2.0,
            zorder=0,
        )
        ax.add_patch(brain)

    else:
        raise ValueError(
            "view must be 'coronal', 'sagittal', or 'axial'"
        )


def _draw_directed_difference_edges(
    ax,
    delta_w,
    x,
    y,
    min_abs_change=0.0,
):
    """Draw directed edges for a difference adjacency matrix."""
    delta_w = np.asarray(delta_w, dtype=float)

    edge_mask = np.abs(delta_w) > min_abs_change

    if not np.any(edge_mask):
        return

    max_abs_delta = np.max(np.abs(delta_w[edge_mask]))

    for i in range(delta_w.shape[0]):
        for j in range(delta_w.shape[1]):
            dw = delta_w[i, j]

            if i == j or abs(dw) <= min_abs_change:
                continue

            # Red = relationship increases / becomes more positive in SDV.
            # Blue = relationship decreases / becomes more negative in SDV.
            edge_color = "tab:red" if dw > 0 else "tab:blue"

            linewidth = 1.5 + 5.0 * abs(dw) / max_abs_delta

            start = (x[i], y[i])
            end = (x[j], y[j])

            # Opposite curvature directions help separate overlapping arrows.
            rad = 0.10 if i < j else -0.10

            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=linewidth,
                color=edge_color,
                alpha=0.78,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=13,
                shrinkB=13,
                zorder=2,
            )
            ax.add_patch(arrow)


def plot_anatomical_graph_difference(
    W_control,
    W_sdv,
    roi_xyz,
    roi_labels,
    output_path=None,
    *,
    view="coronal",
    min_abs_change=0.0,
    title="SDV − Control",
    figsize=(10, 9),
    dpi=300,
):
    """
    Plot SDV-Control directed-graph differences in an axial MNI projection.

    Parameters
    ----------
    W_control : ndarray, shape (d, d)
        Directed adjacency matrix for Control.

    W_sdv : ndarray, shape (d, d)
        Directed adjacency matrix for SDV.

    roi_xyz : ndarray, shape (d, 3)
        ROI MNI coordinates.

    roi_labels : sequence of str
        Anatomical labels corresponding to roi_xyz.

    output_path : path-like or None
        Optional path for saving the figure.

    min_abs_change : float
        Only plot edges with |W_sdv - W_control| above this value.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    W_control = np.asarray(W_control, dtype=float)
    W_sdv = np.asarray(W_sdv, dtype=float)
    roi_xyz = np.asarray(roi_xyz, dtype=float)

    if W_control.shape != W_sdv.shape:
        raise ValueError("Control and SDV matrices must have identical shape.")

    d = W_control.shape[0]

    if W_control.shape != (d, d):
        raise ValueError("W matrices must be square.")

    if roi_xyz.shape != (d, 3):
        raise ValueError(
            f"roi_xyz must have shape ({d}, 3), got {roi_xyz.shape}."
        )

    if len(roi_labels) != d:
        raise ValueError(
            f"Expected {d} ROI labels, got {len(roi_labels)}."
        )

    delta_w = W_sdv - W_control

    # Choose MNI projection
    if view == "coronal":
        # Looking along y-axis: left/right vs inferior/superior
        x = roi_xyz[:, 0]
        y = roi_xyz[:, 2]

        xlabel = "MNI x (Left ←  → Right)"
        ylabel = "MNI z"
        xlim = (-82, 82)
        ylim = (-50, 85)

    elif view == "sagittal":
        # Looking along x-axis: posterior/anterior vs inferior/superior
        x = roi_xyz[:, 1]
        y = roi_xyz[:, 2]

        xlabel = "MNI y (Posterior ←  → Anterior)"
        ylabel = "MNI z"
        xlim = (-75, 75)
        ylim = (-50, 85)

    elif view == "axial":
        x = roi_xyz[:, 0]
        y = roi_xyz[:, 1]

        xlabel = "MNI x (Left ←  → Right)"
        ylabel = "MNI y"
        xlim = (-82, 82)
        ylim = (-75, 75)

    else:
        raise ValueError(
            "view must be 'coronal', 'sagittal', or 'axial'"
        )

    fig, ax = plt.subplots(figsize=figsize)

    _draw_brain_outline(ax, view)

    _draw_directed_difference_edges(
        ax,
        delta_w,
        x,
        y,
        min_abs_change=min_abs_change,
    )

    # ROI nodes.
    ax.scatter(
        x,
        y,
        s=260,
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
    )

    # Anatomical labels.
    for idx, (xi, yi, label) in enumerate(zip(x, y, roi_labels)):
        if xi < -10:
            dx = -4
            ha = "right"
        else:
            dx = 4
            ha = "left"

        ax.annotate(
            f"{idx + 1}. {label}",
            (xi, yi),
            xytext=(dx, 3),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9,
            zorder=4,
        )

    ax.set_title(title, fontsize=16, pad=16)

    ax.text(
        -78,
        -59,
        "Arrow width ∝ |ΔW|",
        fontsize=10,
        va="top",
    )
    

    legend_handles = [
        Line2D(
            [0], [0],
            color="tab:red",
            linewidth=3,
            label="Increase in SDV",
        ),
        Line2D(
            [0], [0],
            color="tab:blue",
            linewidth=3,
            label="Decrease in SDV",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    # Leave room below the axes for the legend.
    fig.subplots_adjust(bottom=0.16)

    ax.set_title(title, fontsize=15, pad=12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig, ax
