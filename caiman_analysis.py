import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

#%% Declare file paths
caiman_loader_dir   = '~/canonical-imaging/img_loaders'
session_dir         = ''
input_files         = [session_dir+'/.tif']
output_file         = session_dir+'/caiman/analysis_results.hdf5'

#%%
import sys
sys.path.insert(1, caiman_loader_dir)
import caiman_loader

#%% Define parameters
# Dataset dependent parameters
fr                  = 40            # imaging rate [frames per second]
decay_time          = 0.4           # length of a typical transient [s]
is3D                = False

# Motion correction parameters
if is3D:
    strides         = (48, 48, 1)   # start a new patch for pw-rigid motion correction every x pixels
    overlaps        = (24, 24, 1)   # overlap between patches (size of patch strides+overlaps) [pixels]
    max_shifts      = (6, 6, 1)     # maximum allowed rigid shifts [pixels]
else:
    strides         = (48, 48)
    overlaps        = (24, 24)
    max_shifts      = (6,6)
max_deviation_rigid = 3             # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid            = True          # flag for performing non-rigid motion correction

# Source extraction and deconvolution parameters
p                   = 1             # order of the autoregressive system
gnb                 = 2             # number of global background components
merge_thr           = 0.85          # merging threshold, max correlation allowed
rf                  = 15            # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf         = 6             # amount of overlap between the patches [pixels]
K                   = 4             # number of components per patch
if is3D:
    gSig            = [4, 4, 1]     # expected half size of neurons [pixels]
else:
    gSig            = [4, 4]
method_init         = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub                = 1             # spatial subsampling during initialization
tsub                = 1             # temporal subsampling during intialization
rolling_sum         = True
only_init           = True

# Component evaluation parameters
min_SNR             = 2.0           # signal to noise ratio for accepting a component
rval_thr            = 0.85          # space correlation threshold for accepting a component
cnn_thr             = 0.99          # threshold for CNN based classifier
cnn_lowest          = 0.1           # neurons with cnn probability lower than this value are rejected
use_cnn             = True

#%% Create parameters dictionary
opts_dict = {'fnames': input_files,
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'is3D': is3D,
            'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'gSig': gSig,
            'method_init': method_init,
            'rolling_sum': rolling_sum,
            'only_init': only_init,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr, 
            'min_SNR': min_SNR,
            'rval_thr': rval_thr,
            'use_cnn': use_cnn,
            'min_cnn_thr': cnn_thr,
            'cnn_lowest': cnn_lowest}

#%% Convert TIFF file to multipage format
# https://caiman.readthedocs.io/en/master/On_file_types_and_sizes.html
def convert_tiff(input_file):
    m			=	cm.load(input_file)#,is3D=True)		# Load file
    T			=	m.shape[0]							# Total number of frames for the file
    L			=	1000								# Individual file length
    fileparts	=	input_file.split('.')

    for t in np.arange(0,T,L):
        sys.stdout.write('\rFile length: %d, Total frames: %d, Current frame: %d' % (L,T,t))
        sys.stdout.flush()
        m[t:t+L].save((".").join(fileparts[:-1]) + '_' + str(t//L) + '.' + fileparts[-1])

#%% CaImAn analysis example workflow
def execute_caiman_analysis(input_files, output_file, opts_dict):
    #%%
    logging.basicConfig(format = "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",level = logging.WARNING)

    #%% Display original movie
    m_orig = cm.load_movie_chain(input_files)
    ds_ratio = 0.2
    m_orig.resize(1, 1, ds_ratio).play(q_max=99.5, fr=fr, magnification=2)

    #%% Create parameters object
    opts = params.CNMFParams(params_dict=opts_dict)

    #%% Start a cluster for parallel processing 
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    #%% Create a motion correction object with the parameters specified
    mc = MotionCorrect(input_files, dview=dview, **opts.get_group('motion'))

    #%% Run piecewise-rigid motion correction
    mc.motion_correct(save_movie=True)

    #%% Load motion corrected file
    m_els = cm.load(mc.fname_tot_els)
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0

    #%% Display motion corrected movie
    m_orig = cm.load_movie_chain(input_files)
    ds_ratio = 0.2
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie, m_els.resize(1, 1, ds_ratio)], axis=2).play(fr=fr, gain=1, magnification=2, offset=0)

    #%% Memory mapping in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0, dview=dview)

    # Load file (format: T x X x Y)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    #%% Restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    #%% Run CNMF
    cnm = cnmf.CNMF(n_processes, params=opts)#, dview=dview)
    cnm = cnm.fit(images)

    #%% Plot all components
    Cn = cm.local_correlations(images.transpose(1,2,0))
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours_nb(img=Cn)

    #%% Re-run seeded CNMF on accepted patches to refine and perform deconvolution 
    cnm2 = cnm #cnm.refit(images, dview=dview)

    #%% Evaluate components
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    #%% Plot all components
    cnm2.estimates.plot_contours_nb(img=Cn, idx=cnm2.estimates.idx_components)

    #%% Plot accepted components
    cnm2.estimates.nb_view_components(img=Cn, idx=cnm2.estimates.idx_components)

    #%% Plot rejected components
    if len(cnm2.estimates.idx_components_bad) > 0:
        cnm2.estimates.nb_view_components(img=Cn, idx=cnm2.estimates.idx_components_bad)
    else:
        print("No components were rejected.")

    #%% Extract DF/F
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    #%% Select high quality components
    cnm2.estimates.select_components(use_object=True)

    #%% Plot final results
    cnm2.estimates.nb_view_components(img=Cn, denoised_color='red')

    #%% Save results
    cnm2.save(output_file)

    #%% Stop cluster and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    #%% Display final movie
    cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2, magnification=2, bpx=border_to_0, include_bck=False)

    #%%
    denoised = cm.movie(cnm2.estimates.A.dot(cnm2.estimates.C) + \
                        cnm2.estimates.b.dot(cnm2.estimates.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])

    #%% Save motion correction shifts and summary images
    caiman_loader.save_mc(mc=mc, caiman_fp=output_file)

#%% Reshape 5 dimensional ScanImage TIFF files
caiman_loader.process_scanimage_tiff(input_files, session_dir)

#%% Run CaImAn analysis
execute_caiman_analysis(input_files, output_file, opts_dict)