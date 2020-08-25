from my_project.init_imaging import subject, imaging
import numpy as np


# ========== Insert new "Subject" ===========

subjects = [{'subject': '82951', 'sex': 'F', 'subject_birth_date': '2020-05-06 15:20:01'},
            {'subject': '90853', 'sex': 'M', 'subject_birth_date': '2019-07-14 08:40:01'},
            {'subject': '91706', 'sex': 'F', 'subject_birth_date': '2019-04-11 01:12:01'},
            {'subject': '92696', 'sex': 'F', 'subject_birth_date': '2019-03-03 12:15:01'}]

subject.Subject.insert(subjects, skip_duplicates=True)


# ========== Insert new "ParamSet" for Suite2p ===========
param_set_name = 'default_suite2p'
params = ops = np.load('./ops.npy', allow_pickle=True).item()

imaging.Suite2pParamSet.insert_new_params(param_set_name, params)


# ========== Insert new "ProcessingParamSet" for Suite2p ===========
imaging.ProcessingParamSet.insert1({'processing_method': 'suite2p',
                                    'paramset_idx': 0,
                                    'paramset_desc': 'Ca-imaging analysis with Suite2p using default Suite2p ops'})
imaging.ProcessingParamSet.Suite2p.insert1({'processing_method': 'suite2p',
                                            'paramset_idx': 0,
                                            'param_set_name': 'default_suite2p'})
