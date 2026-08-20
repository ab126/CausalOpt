from io import BytesIO
import zipfile
import numpy as np
import pytest
from scipy.io import savemat
from causal_opt.fmri import (build_multisubject_lagged_data,build_static_fmri_matrix,
    compare_matrices,load_conn_roi_zip,validate_paired_states)


def _mat_bytes(order,T=5,extra=True):
    names=[f"Bladder Network 19.cluster{i:03d}" for i in order]; data=[np.arange(T)+100*i for i in order]; xyz=[np.array([i,i+1,i+2]) for i in order]
    if extra: names.insert(2,"DefaultMode.MPFC"); data.insert(2,np.zeros(T)); xyz.insert(2,np.zeros(3))
    cells={}
    for key,values in (("names",names),("data",data),("xyz",xyz)):
        cells[key]=np.empty((1,len(values)),dtype=object)
        for i,value in enumerate(values): cells[key][0,i]=value
    stream=BytesIO(); savemat(stream,cells); return stream.getvalue()


def _zip(path,session,lengths=(5,7),order=tuple(range(19,0,-1))):
    with zipfile.ZipFile(path,"w") as z:
        for subject,T in enumerate(lengths,1): z.writestr(f"ROI_Subject{subject:03d}_Session{session}.mat",_mat_bytes(order,T))


def test_loader_sorts_filters_and_parses_subjects(tmp_path):
    path=tmp_path/"s1.zip"; _zip(path,"001"); data=load_conn_roi_zip(path,"001")
    assert data.roi_names==tuple(f"Bladder Network 19.cluster{i:03d}" for i in range(1,20))
    assert data.subject_ids==("Subject001","Subject002"); assert data.timeseries["Subject001"].shape==(5,19); assert data.timeseries["Subject002"].shape==(7,19)


def test_exactly_19_required(tmp_path):
    path=tmp_path/"bad.zip"; _zip(path,"001",lengths=(5,),order=tuple(range(1,19)))
    with pytest.raises(ValueError,match="exactly"): load_conn_roi_zip(path,"001")


def test_pairing_static_pooling_and_differences(tmp_path):
    a=tmp_path/"a.zip"; b=tmp_path/"b.zip"; _zip(a,"001"); _zip(b,"002")
    control=load_conn_roi_zip(a,"001"); sdv=load_conn_roi_zip(b,"002"); validate_paired_states(control,sdv)
    assert build_static_fmri_matrix(control).shape==(12,19); assert compare_matrices(np.zeros((19,19)),np.ones((19,19))).shape==(19,19)
    c=tmp_path/"c.zip"; _zip(c,"002",lengths=(5,)); unpaired=load_conn_roi_zip(c,"002")
    with pytest.raises(ValueError,match="unpaired"): validate_paired_states(control,unpaired)


def test_subject_safe_lags(tmp_path):
    path=tmp_path/"s1.zip"; _zip(path,"001",lengths=(3,4)); data=load_conn_roi_zip(path,"001")
    X0,Xlags,_=build_multisubject_lagged_data(data,1,center=False)
    assert X0.shape==(5,19) and Xlags.shape==(1,5,19)
    assert np.array_equal(X0[:,0],[101,102,101,102,103]); assert np.array_equal(Xlags[0,:,0],[100,101,100,101,102])
