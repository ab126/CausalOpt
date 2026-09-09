"""Load and pair ROI timeseries; preserve subject and atlas identity."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import csv
import json
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat

FILE_RE = re.compile(r"ROI_Subject(\d+)_Session(\d{3})\.mat$", re.I)


ROI_RE = re.compile(r"^Bladder Network 19\.cluster(\d{3})$")


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


def validate_paired_states(control, sdv):
    """Require identical ordered subjects and atlas metadata; never drop data."""
    if control.session != "001" or sdv.session != "002":
        raise ValueError("state mapping must be Session001=control and Session002=SDV")
    if not control.subject_ids or not sdv.subject_ids:
        raise ValueError("paired states must contain at least one subject")
    if control.subject_ids != sdv.subject_ids:
        missing_control = sorted(set(sdv.subject_ids) - set(control.subject_ids))
        missing_sdv = sorted(set(control.subject_ids) - set(sdv.subject_ids))
        raise ValueError(
            f"unpaired subjects: missing control={missing_control}, missing SDV={missing_sdv}. "
            "Subject ordering must also match. Use pair_fmri_states(..., pairing='intersection') "
            "to explicitly retain complete pairs."
        )
    if control.roi_names != sdv.roi_names:
        raise ValueError("ROI ordering differs between states")
    if control.roi_xyz.shape != sdv.roi_xyz.shape or not np.allclose(
        control.roi_xyz, sdv.roi_xyz, rtol=1e-6, atol=1e-5
    ):
        raise ValueError("ROI coordinates differ between states")


def pair_fmri_states(control, sdv, *, pairing="strict"):
    """Validate all subjects, or explicitly retain and report complete pairs.

    Intersection pairing preserves control order and does not mutate either input.
    Different scan lengths are allowed; subjects are paired by identity, not row.
    """
    if pairing not in {"strict", "intersection"}:
        raise ValueError("pairing must be 'strict' or 'intersection'")
    if pairing == "intersection":
        common = tuple(s for s in control.subject_ids if s in sdv.subject_ids)
        if not common:
            raise ValueError("no paired subjects shared by control and SDV")
        dropped_control = tuple(s for s in control.subject_ids if s not in common)
        dropped_sdv = tuple(s for s in sdv.subject_ids if s not in common)
        control = subset_conn_roi_data(control, common)
        sdv = subset_conn_roi_data(sdv, common)
        validate_paired_states(control, sdv)
        if dropped_control or dropped_sdv:
            warnings.warn(
                f"Retaining {len(common)} paired subjects; excluded control={dropped_control}, "
                f"SDV={dropped_sdv} (opposite state missing).",
                UserWarning, stacklevel=2,
            )
    else:
        validate_paired_states(control, sdv)
    return control, sdv


def load_fmri_state_data(session1_zip,session2_zip):
    control=load_conn_roi_zip(session1_zip,"001"); sdv=load_conn_roi_zip(session2_zip,"002"); validate_paired_states(control,sdv); return control,sdv


def load_tsv_state(root, atlas_labels, session, session_code=None):
    """Load one numeric-subject BIDS ROI TSV per subject/state.

    Accept paths or strings. Columns must match across subjects and the label
    JSON; coordinates follow TSV column order. Multiple runs are rejected so
    they cannot silently overwrite each other or introduce artificial lags.
    """
    codes = {"control": "001", "sdv": "002"}
    if session not in codes:
        raise ValueError("session must be 'control' or 'sdv'")
    if session_code is None:
        session_code = codes[session]
    if session_code != codes[session]:
        raise ValueError(f"{session} must map to Session{codes[session]}")

    root = Path(root)
    files = sorted(root.glob(f"sub-*/ses-{session}/*timeseries.tsv"))
    if not files:
        raise FileNotFoundError(f"No ROI timeseries TSVs for session {session!r} under {root}")

    with Path(atlas_labels).open(encoding="utf-8") as stream:
        labels = json.load(stream)
    coord_by_name = {}
    for item in labels.values():
        name = item["roi_name"]
        if name in coord_by_name:
            raise ValueError(f"Duplicate ROI label {name!r} in {atlas_labels}")
        centers = np.asarray(item["centers_mni"], dtype=float)
        if centers.shape != (1, 3) or not np.isfinite(centers).all():
            raise ValueError(f"ROI {name!r} must have exactly one finite 3D MNI center")
        coord_by_name[name] = centers[0]

    timeseries = {}
    source_paths = {}
    roi_names = None
    for path in files:
        # Parse the immediate subject directory, never an ancestor named sub-*.
        match = re.fullmatch(r"sub-(\d+)", path.parent.parent.name)
        if match is None:
            raise ValueError(f"Expected numeric subject directory for {path}")
        subject = f"Subject{int(match.group(1)):03d}"
        entities = dict(re.findall(r"(?:^|_)(sub|ses)-([^_]+)", path.name))
        if (entities.get("sub") != match.group(1)
                or entities.get("ses") != session):
            raise ValueError(f"Filename subject/session does not match its directories: {path}")
        if subject in timeseries:
            raise ValueError(
                f"Multiple timeseries for {subject} / {session}: {source_paths[subject]} and {path}. "
                "Select one run upstream; runs cannot be silently overwritten."
            )
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            names = tuple(next(reader, []))
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(names):
                    raise ValueError(f"Malformed TSV row {line_number} in {path}: expected {len(names)} values")
        if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError(f"Empty or duplicate ROI column names in {path}")
        if set(names) != set(coord_by_name):
            raise ValueError(
                f"ROI labels differ in {path}: missing={sorted(set(coord_by_name) - set(names))}, "
                f"unknown={sorted(set(names) - set(coord_by_name))}"
            )
        if roi_names is None:
            roi_names = names
        elif names != roi_names:
            raise ValueError(f"ROI order differs in {path}")
        try:
            X = pd.read_csv(path, sep="\t", encoding="utf-8-sig").to_numpy(dtype=float)
        except (ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"Invalid numeric timeseries in {path}: {exc}") from exc
        if X.shape[0] == 0 or X.shape[1] != len(names):
            raise ValueError(f"Empty or malformed timeseries in {path}")
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite values in {path}")
        timeseries[subject] = X
        source_paths[subject] = path

    subject_ids = tuple(sorted(timeseries, key=lambda s: int(s.removeprefix("Subject"))))
    return ConnROIData(
        timeseries={s: timeseries[s] for s in subject_ids},
        roi_names=roi_names,
        roi_xyz=np.vstack([coord_by_name[name] for name in roi_names]),
        subject_ids=subject_ids,
        session=session_code,
    )


def load_paired_tsv_states(root, atlas_labels, *, pairing="strict"):
    """Load control/SDV TSVs and apply an explicit subject-pairing policy."""
    control = load_tsv_state(root, atlas_labels, "control")
    sdv = load_tsv_state(root, atlas_labels, "sdv")
    return pair_fmri_states(control, sdv, pairing=pairing)


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


def subset_conn_roi_data(data, subject_ids):
    """Return a ConnROIData object restricted to selected subjects."""

    subject_ids = tuple(subject_ids)

    missing = [
        subject for subject in subject_ids
        if subject not in data.timeseries
    ]
    if missing:
        raise ValueError(
            f"subjects not present in {data.session}: {missing}"
        )

    return ConnROIData(
        timeseries={
            subject: data.timeseries[subject]
            for subject in subject_ids
        },
        roi_names=data.roi_names,
        roi_xyz=np.asarray(data.roi_xyz).copy(),
        subject_ids=subject_ids,
        session=data.session,
    )


