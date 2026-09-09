"""Matrix, network, anatomical, and inference plots for fMRI analyses."""
from __future__ import annotations

from functools import lru_cache
from textwrap import fill
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import nibabel as nib

BLADDER19_LABELS = (
    "Cerebellum 1",
    "Cerebellum 2",
    "Cerebellum 3",
    "DLPFC 1",
    "DLPFC 2",
    "MFG",
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


PLOT_FONT_SCALE = 1.5


_ROI_ABBREVIATIONS = {
    "Dorsolateral Prefrontal Cortex": "DLPFC",
    "Medial Frontal Gyrus": "MFG",
    "Supplementary Motor Area": "SMA",
    "Dorsal Anterior Cingulate Cortex": "dACC",
    "Inferior Frontal Gyrus": "IFG",
    "Medial Prefrontal Cortex": "mPFC",
}


def abbreviate_roi_label(label):
    """Shorten display text only, preserving ROI numbers and side labels."""
    label = str(label)
    for full, short in _ROI_ABBREVIATIONS.items():
        pattern = r"\b" + r"\s+".join(full.split()) + r"\b"
        label = re.sub(pattern, short, label, flags=re.IGNORECASE)
    return label


def _scale_figure_fonts(fig, scale=PLOT_FONT_SCALE):
    """Scale all text in a Matplotlib figure by a common factor."""
    from matplotlib.text import Text

    scale = float(scale)
    if scale <= 0:
        raise ValueError("font scale must be positive")

    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() * scale)

    return fig


def plot_matrix_comparison(control,sdv,names,titles,path,zero_diagonal=False):
    names = [abbreviate_roi_label(label) for label in names]
    import matplotlib.pyplot as plt
    a=np.array(control,copy=True); b=np.array(sdv,copy=True)
    if zero_diagonal: np.fill_diagonal(a,0); np.fill_diagonal(b,0)
    delta=b-a; state=max(float(np.max(np.abs([a,b]))),1e-12); diff=max(float(np.max(np.abs(delta))),1e-12)
    fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    for ax,m,title,lim in zip(axes,(a,b,delta),titles,(state,state,diff)):
        im=ax.imshow(m,cmap="RdBu_r",vmin=-lim,vmax=lim); ax.set_title(title); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax,shrink=.75)
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_dynamic_matrix_comparison(
    W0_control,
    W0_sdv,
    W1_control,
    W1_sdv,
    display_threshold,
    roi_labels,
    output_path,
):
    """
    Plot contemporaneous W0 and lag-1 W1 comparisons as a 2 x 3 figure.

    Rows:
        1. W0: contemporaneous effects
        2. W1: lag-1 effects

    Columns:
        1. Control
        2. SDV
        3. SDV - Control

    Matrices use the convention W[source, target].
    """
    roi_labels = [abbreviate_roi_label(label) for label in roi_labels]

    matrices = [W0_control, W0_sdv, W1_control, W1_sdv]
    shapes = {np.asarray(matrix).shape for matrix in matrices}

    if len(shapes) != 1:
        raise ValueError(f"All matrices must have the same shape; got {shapes}.")

    d = W0_control.shape[0]
    if len(roi_labels) != d:
        raise ValueError(
            f"Expected {d} ROI labels, but received {len(roi_labels)}."
        )

    def threshold_for_display(matrix):
        displayed = np.asarray(matrix, dtype=float).copy()
        displayed[np.abs(displayed) < display_threshold] = 0.0
        return displayed

    row_data = []

    for row_name, control, sdv in [
        (r"Contemporaneous $W_0$", W0_control, W0_sdv),
        (r"Lag-1 $W_1$", W1_control, W1_sdv),
    ]:
        control_display = threshold_for_display(control)
        sdv_display = threshold_for_display(sdv)

        # Compute the difference from the displayed state matrices.
        difference_display = sdv_display - control_display

        row_data.append(
            (
                row_name,
                [
                    control_display,
                    sdv_display,
                    difference_display,
                ],
            )
        )

    panel_titles = [
        ["Control $W_0$", "SDV $W_0$", "SDV - Control $W_0$"],
        ["Control $W_1$", "SDV $W_1$", "SDV - Control $W_1$"],
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        constrained_layout=True,
    )

    ticks = np.arange(d)

    for row_index, (row_name, displayed_matrices) in enumerate(row_data):
        # Use one scale within each row, but allow W0 and W1 to differ.
        vmax = max(
            np.max(np.abs(matrix))
            for matrix in displayed_matrices
        )
        vmax = max(vmax, 1e-12)

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        for column_index, matrix in enumerate(displayed_matrices):
            ax = axes[row_index, column_index]

            # Direct plotting means rows are sources and columns are targets.
            image = ax.imshow(
                matrix,
                cmap="RdBu_r",
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )

            ax.set_title(panel_titles[row_index][column_index], fontsize=13)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # Show target labels on the bottom row.
            if row_index == 1:
                ax.set_xticklabels(
                    roi_labels,
                    rotation=60,
                    ha="right",
                    fontsize=9,
                )
            else:
                ax.set_xticklabels([])

            # Show source labels only in the first column.
            if column_index == 0:
                ax.set_yticklabels(roi_labels, fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(length=0)

        colorbar = fig.colorbar(
            image,
            ax=axes[row_index, :].tolist(),
            fraction=0.025,
            pad=0.015,
            shrink=0.90,
        )
        colorbar.set_label(f"{row_name} weight", fontsize=10)

    fig.supxlabel("Target ROI", fontsize=12)
    fig.supylabel("Source ROI", fontsize=12)
    fig.suptitle(
        "Dynamic directed-network comparison",
        fontsize=15,
    )
    _scale_figure_fonts(fig)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_graph_comparison(control,sdv,names,path,threshold=.3,
                          titles=("Control causal DAG","SDV causal DAG"), significant_control=None, significant_sdv=None, fontsize_scale=1.2):
    """Plot two directed graphs as a 1x2 figure, with optional significance highlighting."""
    
    names = [abbreviate_roi_label(label) for label in names]

    angles=np.linspace(0,2*np.pi,len(names),endpoint=False); positions=np.column_stack((np.cos(angles),np.sin(angles))); limit=max(np.max(np.abs(control)),np.max(np.abs(sdv)),1e-12)
    fig,axes=plt.subplots(1,2,figsize=(16,8),constrained_layout=True)
    for i, (ax,W,title) in enumerate(zip(axes,(control,sdv),titles)):
        significant_mask = significant_control if i == 0 else significant_sdv
        ax.scatter(positions[:,0],positions[:,1],s=650,c="#d9eaf7",edgecolors="#333",zorder=3)
        for i,(x,y) in enumerate(positions):
            label = fill(names[i], width=14, break_long_words=False)
            ax.text(x,y,label,ha="center",va="center",fontsize=8*fontsize_scale if len(names[i]) > 14 else 9*fontsize_scale, zorder=4)
        for i,j in np.argwhere(np.abs(W)>=threshold):
            color="#b2182b" if W[i,j]>0 else "#2166ac";
            # width=.5+3*abs(W[i,j])/limit
            width=1.
            alpha = 0.35
            if significant_mask is not None and significant_mask[i, j]:
                width = 3.0
                alpha = 1.0
            start=positions[i]*.91; end=positions[j]*.91
            ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="-|>",color=color,lw=width,shrinkA=8,shrinkB=8,connectionstyle="arc3,rad=.08"),zorder=2)
        ax.set_title(title, fontsize=12 * fontsize_scale); ax.set_aspect("equal"); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.axis("off")
    if significant_control is not None or significant_sdv is not None:
        fig.text(0.5,0.02,"Thick edge: bootstrap p_unc < 0.05\nThin edge: inferred edge without nominal significance",ha="center",va="bottom",fontsize=10 * fontsize_scale,
                 bbox=dict(facecolor="white",edgecolor="0.7",alpha=0.85,pad=0.4))
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_two_matrix_comparison(left,right,names,titles,path):
    names = [abbreviate_roi_label(label) for label in names]
    import matplotlib.pyplot as plt
    limit=max(float(np.max(np.abs([left,right]))),1e-12); fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,m,title in zip(axes,(left,right),titles):
        im=ax.imshow(m,cmap="RdBu_r",vmin=-limit,vmax=limit); ax.set_title(title); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax,shrink=.75)
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_pag_comparison(control,sdv,names,path):
    """Plot endpoint-coded PAGs; tail, arrow, and circle endpoints remain distinct."""
    names = [abbreviate_roi_label(label) for label in names]
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
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_skeleton_comparison(control,sdv,names,path):
    names = [abbreviate_roi_label(label) for label in names]
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,m,title in zip(axes,(control,sdv),("FCI Control adjacency","FCI SDV adjacency")):
        im=ax.imshow(m,cmap="Greys",vmin=0,vmax=1); ax.set_title(title); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7)
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_bootstrap_edge_significance(W,statistics,names,path, title="", alpha=0.05,):
    names = [abbreviate_roi_label(label) for label in names]
    import matplotlib.pyplot as plt
    limit=max(float(np.max(np.abs(W))),1e-12); fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True); im=ax.imshow(W,cmap="RdBu_r",vmin=-limit,vmax=limit)
    ci = None
    #ci=np.argwhere(statistics["ci_excludes_zero"]&~statistics["fdr_significant"]); 
    fdr=np.argwhere(statistics["fdr_significant"])
    sig = np.argwhere(statistics["p_boot"] < alpha)
    #if len(ci): ax.scatter(ci[:,1],ci[:,0],marker="o",facecolors="none",edgecolors="black",s=35,label="95% CI excludes 0")
    if len(sig):
        ax.scatter(sig[:,1],sig[:,0],marker="o",facecolors="none",edgecolors="black",s=30,label=f"p < {alpha} (Uncorrected)")
    if len(fdr):
        ax.scatter(fdr[:,1],fdr[:,0],marker="*",c="#ffd700",edgecolors="black",s=80,label=f"FDR q < {alpha}")
    ax.set_title(title)

    ax.set_xticks(
        range(len(names)),
        names,
        rotation=90,
        fontsize=5,
    )
    ax.set_yticks(
        range(len(names)),
        names,
        fontsize=5,
    )

    # Significance legend to the right of the heatmap.
    if len(sig) or len(fdr):
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.10, 1.0),
            borderaxespad=0.0,
        )

    # Colorbar directly below the significance legend.
    # Coordinates are relative to the heatmap axis:
    # [left, bottom, width, height].
    cax = ax.inset_axes([
        1.14,   # horizontal position
        0.08,   # bottom
        0.07,   # width
        0.72,   # height
    ])

    cbar = fig.colorbar(
        im,
        cax=cax,
    )

    cbar.ax.tick_params(labelsize=5)
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_bootstrap_edge_stability(stability,names,threshold,path):
    names = [abbreviate_roi_label(label) for label in names]
    import matplotlib.pyplot as plt
    values=stability["thresholds"][float(threshold)]; fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,key,title in zip(axes,("control","sdv"),("Control edge-selection probability","SDV edge-selection probability")):
        im=ax.imshow(values[key],cmap="viridis",vmin=0,vmax=1); ax.set_title(f"{title} (|W| > {threshold:g})"); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7); ax.set_yticks(range(len(names)),names,fontsize=7); fig.colorbar(im,ax=ax)
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def plot_bootstrap_edge_intervals(frame,path,top_n=15):
    import matplotlib.pyplot as plt
    selected=frame[frame.fdr_significant].copy()
    if selected.empty: selected=frame.iloc[np.argsort(np.abs(frame.delta_W_observed.to_numpy()))[::-1][:top_n]].copy()
    else: selected=selected.iloc[:top_n]
    selected=selected.iloc[::-1]; labels=[f"{abbreviate_roi_label(r.source_name)} -> {abbreviate_roi_label(r.target_name)}  (q={r.q_fdr:.3g})" for r in selected.itertuples()]
    y=np.arange(len(selected)); x=selected.delta_W_observed.to_numpy(); lo=x-selected["ci_2.5"].to_numpy(); hi=selected["ci_97.5"].to_numpy()-x
    fig,ax=plt.subplots(figsize=(11,max(5,.42*len(selected))),constrained_layout=True); ax.errorbar(x,y,xerr=np.vstack((lo,hi)),fmt="o",color="#2166ac",ecolor="#555",capsize=3); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,labels); ax.set_xlabel("Observed SDV - Control raw W (95% percentile CI)"); ax.set_title("Strongest directed edge differences")
    _scale_figure_fonts(fig)
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
    labels=[f"{abbreviate_roi_label(r.source_name)} -> {abbreviate_roi_label(r.target_name)}  p={r.p_boot:.3g}, q={r.q_fdr:.3g}" for r in selected.itertuples()]
    ax.set_yticks(range(len(selected)),labels); ax.axvline(0,color="black",lw=1); ax.set_xlabel("Observed SDV - Control raw W (95% percentile CI)"); ax.set_title("Nominal bootstrap edge differences (red = BH-FDR; diamond = L Insula -> PAG1)")
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); return labels


def plot_node_reorganization(frame, path, top_n=12):
    """Plot the top N ROIs with the largest total absolute weight change."""

    view = (
        frame.sort_values("total_change", ascending=False)
        .head(top_n)
        .iloc[::-1]
        .copy()
    )
    if view.empty:
        return

    # Shorten display labels without changing the underlying ROI names.
    replacements = {
        "Dorsal Anterior Cingulate Cortex": "dACC",
        "Dorsolateral Prefrontal Cortex": "DLPFC",
        "Medial Prefrontal Cortex": "mPFC",
        "Supplementary Motor Area": "SMA",
        "Inferior Frontal Gyrus": "IFG",
        "L Insula": "Left insula",
        "R Insula": "Right insula",
    }

    def short_label(name):
        for full, short in replacements.items():
            name = name.replace(full, short)
        return name

    labels = [short_label(str(name)) for name in view.roi]
    y = np.arange(len(view))
    incoming = view.incoming_change.to_numpy()
    outgoing = view.outgoing_change.to_numpy()
    totals = incoming + outgoing

    fig, ax = plt.subplots(
        figsize=(9, max(4.2, 0.43 * len(view) + 1.5)),
        constrained_layout=True,
    )

    ax.barh(
        y, incoming, height=0.65,
        color="#4C78A8", label="Incoming",
    )
    ax.barh(
        y, outgoing, left=incoming, height=0.65,
        color="#E5A34D", label="Outgoing",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)

    # Totals at the ends of the stacked bars.
    xmax = max(float(totals.max()), 1e-6)
    for yi, total in zip(y, totals):
        ax.text(
            total + 0.018 * xmax, yi, f"{total:.2f}",
            va="center", fontsize=9, color="#374151",
        )

    ax.set_xlim(0, xmax * 1.14)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.yaxis.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", pad=8)

    ax.set_xlabel(
        r"Total absolute weight change, $\sum |\Delta W|$",
        fontsize=10, labelpad=10,
    )
    ax.set_title(
        "ROI reorganization",
        loc="left", fontsize=13, fontweight="bold", pad=48,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9,
        borderaxespad=0,
    )

    _scale_figure_fonts(fig)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_key_edge_bootstrap_distributions(frame,delta_boot,path,terms=("PAG","Insula","PMC","Motor Area","Cerebell"),max_edges=6):
    """Plot strongest observed differences touching automatically matched key ROIs."""
    import matplotlib.pyplot as plt
    mask=frame.source_name.str.contains("|".join(terms),case=False,regex=True)|frame.target_name.str.contains("|".join(terms),case=False,regex=True)
    selected=frame[mask].copy(); selected["magnitude"]=selected.delta_W_observed.abs(); selected=selected.nlargest(max_edges,"magnitude")
    if selected.empty: return []
    fig,axes=plt.subplots(len(selected),1,figsize=(9,2.3*len(selected)),squeeze=False,constrained_layout=True)
    labels=[]
    for ax,(_,row) in zip(axes.ravel(),selected.iterrows()):
        values=delta_boot[:,int(row.source_index),int(row.target_index)]; label=f"{abbreviate_roi_label(row.source_name)} -> {abbreviate_roi_label(row.target_name)}"; labels.append(label)
        ax.hist(values,bins=min(30,max(5,len(values)//2)),color="#8da0cb",alpha=.8); ax.axvline(0,color="black",lw=1); ax.axvline(row.delta_W_observed,color="#b2182b",lw=2,label="observed"); ax.axvline(row["ci_2.5"],color="#555",ls="--"); ax.axvline(row["ci_97.5"],color="#555",ls="--",label="95% CI"); ax.set_title(f"{label} (q={row.q_fdr:.3g})"); ax.legend(fontsize=8)
    _scale_figure_fonts(fig)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig); return labels


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


@lru_cache(maxsize=1)
def _load_mni_background():
    """Load a canonical MNI152 anatomical template and brain mask."""
    from nilearn.datasets import (
        load_mni152_template,
        load_mni152_brain_mask,
    )

    anat_img = nib.as_closest_canonical(
        load_mni152_template(resolution=2)
    )
    mask_img = nib.as_closest_canonical(
        load_mni152_brain_mask(resolution=2)
    )

    anat = np.asarray(anat_img.get_fdata(), dtype=float)
    mask = np.asarray(mask_img.get_fdata(), dtype=float) > 0

    return anat, mask, anat_img.affine


def _axis_world_coords(length, affine, axis):
    """World-coordinate values (mm) along one image axis."""
    ijk = np.zeros((length, 3), dtype=float)
    ijk[:, axis] = np.arange(length, dtype=float)
    xyz = nib.affines.apply_affine(affine, ijk)
    return xyz[:, axis]


def _draw_mni_slice_background(ax, view, roi_xyz=None, alpha=0.16):
    """
    Draw a faint MNI152 anatomical slice and brain contour on the
    given axes, using a slice near the ROI cloud.
    """
    anat, mask, affine = _load_mni_background()

    nx, ny, nz = anat.shape
    x_mm = _axis_world_coords(nx, affine, 0)
    y_mm = _axis_world_coords(ny, affine, 1)
    z_mm = _axis_world_coords(nz, affine, 2)

    if roi_xyz is None:
        roi_xyz = np.zeros((0, 3), dtype=float)
    else:
        roi_xyz = np.asarray(roi_xyz, dtype=float)

    if view == "coronal":
        # x-z slice at representative y
        plane_mm = float(np.median(roi_xyz[:, 1])) if roi_xyz.size else 0.0
        j = int(np.argmin(np.abs(y_mm - plane_mm)))

        anat2d = anat[:, j, :].T
        mask2d = mask[:, j, :].T.astype(float)
        extent = [x_mm.min(), x_mm.max(), z_mm.min(), z_mm.max()]

    elif view == "sagittal":
        # y-z slice at representative x
        plane_mm = float(np.median(roi_xyz[:, 0])) if roi_xyz.size else 0.0
        i = int(np.argmin(np.abs(x_mm - plane_mm)))

        anat2d = anat[i, :, :].T
        mask2d = mask[i, :, :].T.astype(float)
        extent = [y_mm.min(), y_mm.max(), z_mm.min(), z_mm.max()]

    elif view == "axial":
        # x-y slice at representative z
        plane_mm = float(np.median(roi_xyz[:, 2])) if roi_xyz.size else 0.0
        k = int(np.argmin(np.abs(z_mm - plane_mm)))

        anat2d = anat[:, :, k].T
        mask2d = mask[:, :, k].T.astype(float)
        extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()]

    else:
        raise ValueError("view must be 'coronal', 'sagittal', or 'axial'")

    # Only show anatomy inside the brain mask
    anat2d = np.ma.masked_where(mask2d <= 0, anat2d)

    ax.imshow(
        anat2d,
        cmap="gray",
        extent=extent,
        origin="lower",
        alpha=alpha,
        zorder=0,
    )

    ax.contour(
        mask2d,
        levels=[0.5],
        colors=["0.45"],
        linewidths=1.6,
        extent=extent,
        origin="lower",
        zorder=1,
    )

    # Keep a subtle midline for coronal/axial
    if view in {"coronal", "axial"}:
        ax.axvline(
            0,
            linestyle="--",
            linewidth=0.8,
            alpha=0.22,
            color="0.45",
            zorder=1,
        )


def _draw_brain_outline_simple(ax, view):
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


def _draw_brain_outline(ax, view, roi_xyz=None):
    """Draw an MNI background when available, else fall back to a schematic outline."""
    try:
        _draw_mni_slice_background(ax, view, roi_xyz=roi_xyz)
        return
    except Exception as exc:
        print(
            f"[plot] MNI background unavailable ({exc}); "
            f"falling back to schematic outline."
        )

    if view == "coronal":
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

        ax.plot(
            [0, 0], [-45, 78],
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
            zorder=1,
        )

    elif view == "sagittal":
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
    positive_color="tab:red",
    negative_color="tab:blue",
    significant_mask=None,
):
    """
    Draw directed difference edges.

    An edge is displayed if either:
      - |delta_w| > min_abs_change, or
      - significant_mask is True for that edge.

    Nominally significant edges are marked with an asterisk.
    """
    delta_w = np.asarray(delta_w, dtype=float)

    if significant_mask is None:
        significant_mask = np.zeros(delta_w.shape, dtype=bool)
    else:
        significant_mask = np.asarray(significant_mask, dtype=bool)

        if significant_mask.shape != delta_w.shape:
            raise ValueError(
                "significant_mask must have the same shape as delta_w"
            )

    # Keep descriptively large edges AND all significant edges.
    edge_mask = (
        (np.abs(delta_w) > min_abs_change)
        | significant_mask
    )
    np.fill_diagonal(edge_mask, False)

    if not np.any(edge_mask):
        return

    max_abs_delta = max(
        float(np.max(np.abs(delta_w[edge_mask]))),
        1e-12,
    )

    for i in range(delta_w.shape[0]):
        for j in range(delta_w.shape[1]):

            if not edge_mask[i, j]:
                continue

            dw = delta_w[i, j]

            edge_color = (
                positive_color if dw > 0
                else negative_color
            )

            linewidth = (
                1.5
                + 5.0 * abs(dw) / max_abs_delta
            )

            start = np.asarray(
                (x[i], y[i]),
                dtype=float,
            )
            end = np.asarray(
                (x[j], y[j]),
                dtype=float,
            )

            rad = 0.10 if i < j else -0.10

            # Separate reciprocal edges so neither direction is hidden.
            if edge_mask[j, i]:
                direction = end - start
                length = max(float(np.linalg.norm(direction)), 1e-12)
                perpendicular = np.asarray(
                    [-direction[1], direction[0]]
                ) / length

                offset = 2.5 * perpendicular
                start = start + offset
                end = end + offset

            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=24,
                linewidth=linewidth,
                color=edge_color,
                alpha=0.82,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=13,
                shrinkB=13,
                zorder=2.6 if significant_mask[i, j] else 2,
            )
            ax.add_patch(arrow)

            # Mark p_unc < 0.05 edges.
            if significant_mask[i, j]:
                midpoint = 0.5 * (start + end)

                direction = end - start
                length = max(
                    float(np.linalg.norm(direction)),
                    1e-12,
                )

                perpendicular = np.asarray(
                    [-direction[1], direction[0]]
                ) / length

                # Actual midpoint of matplotlib's arc3 quadratic curve.
                curve_midpoint = (
                    midpoint
                    - 0.5 * rad * length * perpendicular
                )

                # Small visual separation from the arrow itself.
                star_position = (
                    curve_midpoint
                    - 1.25 * np.sign(rad) * perpendicular
                )

                ax.text(
                    star_position[0],
                    star_position[1],
                    "*",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="black",
                    zorder=5,
                )


def plot_anatomical_directed_difference(
    delta_w,
    roi_xyz,
    roi_labels,
    output_path=None,
    *,
    view="coronal",
    min_abs_change=0.0,
    significant_mask=None,
    title="Directed-network difference",
    positive_color="tab:orange",
    negative_color="tab:purple",
    positive_label="Positive difference",
    negative_label="Negative difference",
    magnitude_label=r"Arrow width $\propto |\Delta W|$",
    figsize=(12, 10),
    dpi=300,
):
    roi_labels = [abbreviate_roi_label(label) for label in roi_labels]
    delta_w = np.asarray(delta_w, dtype=float)
    roi_xyz = np.asarray(roi_xyz, dtype=float)

    if significant_mask is not None:
        significant_mask = np.asarray(
            significant_mask,
            dtype=bool,
        )

        if significant_mask.shape != delta_w.shape:
            raise ValueError(
                "significant_mask must have the same "
                "shape as delta_w"
            )

    d = delta_w.shape[0]

    if delta_w.shape != (d, d):
        raise ValueError("delta_w must be square.")

    if roi_xyz.shape != (d, 3):
        raise ValueError(
            f"roi_xyz must have shape ({d}, 3), got {roi_xyz.shape}"
        )

    if len(roi_labels) != d:
        raise ValueError(
            f"Expected {d} ROI labels, got {len(roi_labels)}"
        )

    if view == "coronal":
        x = roi_xyz[:, 0]
        y = roi_xyz[:, 2]
        xlabel = "MNI x (Left \u2190  \u2192 Right)"
        ylabel = "MNI z"
        xlim = (-82, 82)
        ylim = (-50, 85)

    elif view == "sagittal":
        x = roi_xyz[:, 1]
        y = roi_xyz[:, 2]
        xlabel = "MNI y (Posterior \u2190  \u2192 Anterior)"
        ylabel = "MNI z"
        xlim = (-75, 75)
        ylim = (-50, 85)

    elif view == "axial":
        x = roi_xyz[:, 0]
        y = roi_xyz[:, 1]
        xlabel = "MNI x (Left \u2190  \u2192 Right)"
        ylabel = "MNI y"
        xlim = (-82, 82)
        ylim = (-75, 75)

    else:
        raise ValueError(
            "view must be 'coronal', 'sagittal', or 'axial'"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # _draw_brain_outline_simple(ax, view)
    _draw_brain_outline(ax, view, roi_xyz=roi_xyz)

    _draw_directed_difference_edges(
        ax,
        delta_w,
        x,
        y,
        min_abs_change=min_abs_change,
        positive_color=positive_color,
        negative_color=negative_color,
        significant_mask=significant_mask,
    )

    ax.scatter(
        x,
        y,
        s=260,                 
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )

    for idx, (xi, yi, label) in enumerate(zip(x, y, roi_labels)):
        if xi < -10:
            dx, ha = -5, "right"
        else:
            dx, ha = 5, "left"

        ax.annotate(
            f"{idx + 1}. {label}",
            (xi, yi),
            xytext=(dx, 4),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9,             
            zorder=4,
        )

    ax.set_title(title, fontsize=18, pad=18)

    ax.text(
        xlim[0] + 4,
        ylim[0] - 8,
        magnitude_label,
        fontsize=10,
        va="top",
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color=positive_color,
            linewidth=4,
            label=positive_label,
        ),
        Line2D(
            [0], [0],
            color=negative_color,
            linewidth=4,
            label=negative_label,
        ),
    ]

    if (
        significant_mask is not None
        and np.any(significant_mask)
    ):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=r"$*$",
                linestyle="None",
                color="black",
                markersize=12,
                label=r"Uncorrected bootstrap $p<0.05$",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(bottom=0.18)
    _scale_figure_fonts(fig)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, ax


def plot_anatomical_graph_difference(
    W_control,
    W_sdv,
    roi_xyz,
    roi_labels,
    output_path=None,
    *,
    view="coronal",
    min_abs_change=0.0,
    significant_mask=None,
    title="SDV − Control",
    figsize=(12, 10),
    dpi=300,
):
    return plot_anatomical_directed_difference(
        np.asarray(W_sdv) - np.asarray(W_control),
        roi_xyz,
        roi_labels,
        output_path,
        view=view,
        min_abs_change=min_abs_change,
        significant_mask=significant_mask,
        title=title,
        positive_color="tab:red",
        negative_color="tab:blue",
        positive_label="Increase in SDV",
        negative_label="Decrease in SDV",
        figsize=figsize,
        dpi=dpi,
    )


def plot_sex_dag_comparison(
    W_male_control,
    W_male_sdv,
    W_female_control,
    W_female_sdv,
    names,
    output_path,
    threshold=0.3,
    *,
    male_control_sig=None,
    male_sdv_sig=None,
    female_control_sig=None,
    female_sdv_sig=None,
    male_delta_sig=None,
    female_delta_sig=None,
    dpi=300,
    fontsize_scale=1.4
):
    """Report-ready 2x3 sex/state directed-network comparison."""
    names = [abbreviate_roi_label(label) for label in names]

    matrices = [
        [W_male_control, W_male_sdv, W_male_sdv - W_male_control],
        [W_female_control, W_female_sdv, W_female_sdv - W_female_control],
    ]

    sigs = [
        [male_control_sig, male_sdv_sig, male_delta_sig],
        [female_control_sig, female_sdv_sig, female_delta_sig],
    ]

    titles = [
        ["Male — Control", "Male — SDV", "Male — SDV − Control"],
        ["Female — Control", "Female — SDV", "Female — SDV − Control"],
    ]

    legend = [
        Line2D([0], [0], color="#b2182b", lw=3, label="Positive directed weight/change"),
        Line2D([0], [0], color="#2166ac", lw=3, label="Negative directed weight/change"),
        Line2D([0], [0], color="0.2", lw=4, label=r"Thick: bootstrap $p_{unc}<0.05$"),
        Line2D([0], [0], color="0.2", lw=1.5, alpha=0.4, label=r"Thin: $p_{unc}\geq0.05$ or bootstrap not run"),
    ]


    angles = np.linspace(
        0,
        2 * np.pi,
        len(names),
        endpoint=False,
    )
    positions = np.column_stack(
        (np.cos(angles), np.sin(angles))
    )

    fig, axes = plt.subplots(
        2, 3,
        figsize=(24, 13.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.075,   # Space for the legend
        top=0.925,     # Space for the overall title
        wspace=0.04,
        hspace=0.10,
    )

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            W = np.asarray(matrices[row][col], float)
            sig = sigs[row][col]

            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                s=900,
                facecolor="#d9eaf7",
                edgecolors="#333333",
                linewidths=1.3,
                zorder=3,
            )

            for node, (x, y) in enumerate(positions):
                ax.text(
                    x,
                    y,
                    names[node],
                    ha="center",
                    va="center",
                    fontsize=10.5 * fontsize_scale,       # old 7 * 1.5
                    zorder=4,
                )

            for i, j in np.argwhere(np.abs(W) >= threshold):
                if i == j:
                    continue

                color = (
                    "#b2182b" if W[i, j] > 0
                    else "#2166ac"
                )

                significant = (
                    sig is not None
                    and bool(sig[i, j])
                )

                width = 4.0 if significant else 1.6
                alpha = 1.0 if significant else 0.38

                start = positions[i] * 0.90
                end = positions[j] * 0.90

                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=width,
                        alpha=alpha,
                        shrinkA=10,
                        shrinkB=10,
                        connectionstyle="arc3,rad=.08",
                    ),
                    zorder=2,
                )

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.28, 1.28)  # Retain room for long horizontal labels
            ax.set_ylim(-1.12, 1.12)  # Remove excess space above/below the nodes

            ax.set_title(
                titles[row][col],
                fontsize=18 * fontsize_scale,
                y=1.0,
                pad=2,
            )
            ax.axis("off")

    fig.suptitle(
        "Sex-Stratified Static Directed Networks and Bladder-State Reorganization",
        fontsize=23 * fontsize_scale,
        y=0.985,
    )

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=14 * fontsize_scale,
        bbox_to_anchor=(0.5, 0.015),
        borderaxespad=0,
        columnspacing=1.3,
        handletextpad=0.6,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_sex_change_heatmaps(
    delta_male,
    delta_female,
    interaction,
    names,
    output_path,
    *,
    male_stats=None,
    female_stats=None,
    interaction_stats=None,
    alpha=0.05,
    dpi=300,
):
    names = [abbreviate_roi_label(label) for label in names]
    matrices = [
        np.asarray(delta_male, float),
        np.asarray(delta_female, float),
        np.asarray(interaction, float),
    ]

    statistics = [
        male_stats,
        female_stats,
        interaction_stats,
    ]

    titles = [
        "Male: SDV − Control",
        "Female: SDV − Control",
        "Sex × State Interaction\nFemale ΔW − Male ΔW",
    ]

    state_limit = max(
        np.max(np.abs(matrices[0])),
        np.max(np.abs(matrices[1])),
        1e-12,
    )
    interaction_limit = max(
        np.max(np.abs(matrices[2])),
        1e-12,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8.5),
        constrained_layout=True,
    )

    for index, (ax, matrix, stats, title) in enumerate(
        zip(axes, matrices, statistics, titles)
    ):
        limit = (
            state_limit if index < 2
            else interaction_limit
        )

        image = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )

        if stats is not None:
            nominal = np.argwhere(
                stats["p_boot"] < alpha
            )
            fdr = np.argwhere(
                stats["q_fdr"] < alpha
            )

            if len(nominal):
                ax.scatter(
                    nominal[:, 1],
                    nominal[:, 0],
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.8,
                    s=75,
                )

            if len(fdr):
                ax.scatter(
                    fdr[:, 1],
                    fdr[:, 0],
                    marker="*",
                    c="gold",
                    edgecolors="black",
                    linewidths=0.8,
                    s=150,
                )

        ax.set_title(title, fontsize=19)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            names,
            rotation=70,
            ha="right",
            fontsize=10.5,
        )

        if index == 0:
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(
                names,
                fontsize=10.5,
            )
            ax.set_ylabel("Source ROI", fontsize=16)
        else:
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([])

        ax.set_xlabel("Target ROI", fontsize=16)

        cbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.82,
            pad=0.025,
        )
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label(
            "Directed weight difference",
            fontsize=13,
        )

    fig.suptitle(
        "Sex-Stratified Bladder-State Changes in the Static Directed Network",
        fontsize=23,
    )

    legend = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.8,
            markersize=9,
            label=r"Uncorrected bootstrap $p<0.05$",
        ),
        Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=13,
            label=r"BH-FDR $q<0.05$",
        ),
    ]

    if any(stats is not None for stats in statistics):
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=14,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_sex_node_reorganization(
    male_nodes,
    female_nodes,
    output_path,
    *,
    node_statistics=None,
    dpi=300,
    fontsize_scale=1.2
):
    """Plot node scores; optional ROI-keyed bootstrap p-values mark midpoints."""

    male = male_nodes.set_index("roi")
    female = female_nodes.set_index("roi")

    names = list(
        (
            male["total_change"]
            + female["total_change"]
        )
        .sort_values()
        .index
    )

    y = np.arange(len(names))
    male_values = male.loc[names, "total_change"].to_numpy()
    female_values = female.loc[names, "total_change"].to_numpy()
    if node_statistics is not None:
        stats = node_statistics.set_index("roi", verify_integrity=True)
        p_values = stats.loc[names, "p_boot"].to_numpy(dtype=float)
        if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
            raise ValueError("node p-values must be finite and between 0 and 1")
    else:
        p_values = np.ones(len(names))

    fig, ax = plt.subplots(
        figsize=(12, 10),
        constrained_layout=True,
    )

    for yi, m, f, p in zip(y, male_values, female_values, p_values):
        ax.plot(
            [m, f],
            [yi, yi],
            color="0.75",
            lw=2,
            zorder=1,
        )
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if stars:
            ax.text(
                (m + f) / 2, yi, stars, ha="center", va="center",
                fontsize=16 * fontsize_scale, fontweight="bold", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
            )

    ax.scatter(
        male_values,
        y,
        s=90,
        marker="o",
        label="Male",
        zorder=2,
    )

    ax.scatter(
        female_values,
        y,
        s=100,
        marker="D",
        label="Female",
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([abbreviate_roi_label(name) for name in names], fontsize=13 * fontsize_scale)

    ax.set_xlabel(
        r"Total ROI reorganization: "
        r"$\sum |\Delta W_{in}|+\sum |\Delta W_{out}|$",
        fontsize=16 * fontsize_scale,
    )

    ax.set_title(
        "ROI-Level Bladder-State Reorganization by Sex",
        fontsize=21 * fontsize_scale,
        pad=14,
    )

    ax.tick_params(axis="x", labelsize=13 * fontsize_scale)
    ax.legend(fontsize=15 * fontsize_scale, frameon=False)
    if node_statistics is not None:
        fig.supxlabel(
            "Female − Male: two-sided, uncorrected bootstrap p-values\n"
            "* p < 0.05    ** p < 0.01    *** p < 0.001",
            fontsize=11 * fontsize_scale,
        )

    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


