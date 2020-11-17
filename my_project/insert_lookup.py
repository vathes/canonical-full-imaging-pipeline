from my_project import subject, imaging
import numpy as np


# ========== Insert new "Subject" ===========

subjects = [{'subject': 'subjectname', 'sex': 'F', 'subject_birth_date': '2020-05-06 15:20:01'}]

subject.Subject.insert(subjects, skip_duplicates=True)

# ========== Insert new "ProcessingParamSet" for Suite2p ===========
params = np.load('./ops.npy', allow_pickle=True).item()

imaging.ProcessingParamSet.insert_new_params(
    'suite2p', 0, 'Ca-imaging analysis with Suite2p using default Suite2p ops', params)

# ========== Insert new "ProcessingParamSet" for CaImAn ===========
imaging.ProcessingParamSet.insert_new_params(
    'caiman', 1, 'Ca-imaging analysis with CaImAn using CaImAn params', 
    dict(processing_method='caiman', task_mode='load'))