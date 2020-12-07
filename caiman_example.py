#%% 
import scanreader
from img_loaders import caiman_loader, parse_scanimage_header

#%% Declare file paths
session_dir         = ''
input_file_combined = session_dir+'/.tif'
input_file_split    = [session_dir+'/.tif']
output_file         = session_dir+'/caiman/analysis_results.hdf5'

#%% Reshape 5 dimensional ScanImage TIFF file and save new TIFF file for each channel
caiman_loader.split_tiff_channels(input_file_combined, session_dir)

#%% Split TIFF files each with a certain number of frames
caiman_loader.split_tiff_frames(input_file_combined)

#%% Define parameters
# Get frame rate from ScanImage header
scan                = scanreader.read_scan(input_file_combined)
header              = parse_scanimage_header(scan)
frame_rate          = int(header['SI_hRoiManager_scanFrameRate'])

# Dataset dependent parameters
fr                  = frame_rate    # imaging rate [frames per second]
decay_time          = 0.4           # length of a typical transient [s]
is3D                = False
dxy                 = (2., 2.)      # spatial resolution in x and y [um per pixel]
max_shift_um        = (12., 12.)    # maximum shift [um]
patch_motion_um     = (100., 100.)  # patch size for non-rigid correction [um]

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
if is3D:
    use_cnn         = False
else:
    use_cnn         = True

# Create parameters dictionary
opts_dict = {'fnames': input_file_split,
            'fr': fr,
            'decay_time': decay_time,
            'dxy': dxy,
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

#%% CaImAn analysis example function
def caiman_analysis(opts_dict, output_file):
    """
    Example CaImAn workflow
    """

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
    from caiman.summary_images import local_correlations_movie_offline

    #%%
    logging.basicConfig(format = "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",level = logging.WARNING)

    #%% Display original movie
    m_orig = cm.load_movie_chain(opts_dict['fnames'], is3D=opts_dict['is3D'])
    
    if opts_dict['is3D']:
        z_layer = 0
        m_orig = m_orig[:,:,:,z_layer]

    ds_ratio = 0.2
    m_orig.resize(1, 1, ds_ratio).play(q_max=99.5, fr=opts_dict['fr'], magnification=2)

    #%% Create parameters object
    opts = params.CNMFParams(params_dict=opts_dict)

    #%% Start a cluster for parallel processing 
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    if opts_dict['is3D']:
        base_name = os.path.splitext(os.path.basename(opts_dict['fnames'][0]))[0]
        fname = cm.save_memmap(opts_dict['fnames'], base_name=base_name, is_3D=opts_dict['is3D'], order='C')
    else:
        fname = opts_dict['fnames']

    #%% Create a motion correction object with parameters specified
    mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))

    #%% Run piecewise-rigid motion correction
    mc.motion_correct(save_movie=True)

    #%% Display motion corrected movie
    m_orig = cm.load_movie_chain(opts_dict['fnames'], is3D=opts_dict['is3D'])
    m_els = cm.load(mc.fname_tot_els, is3D=opts_dict['is3D'])

    if opts_dict['is3D']:
        z_layer = 0
        m_orig = m_orig[:,:,:,z_layer]
        m_els = m_els[:,:,:,z_layer]

    ds_ratio = 0.2
    cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie, m_els.resize(1, 1, ds_ratio)], axis=2).play(fr=opts_dict['fr'], gain=1, q_max=99.5, magnification=2, offset=0)

    #%% Memory mapping in order 'C'
    base_name_mc = os.path.splitext(os.path.basename(mc.fname_tot_els[0]))[0]
    if opts_dict['pw_rigid']:
        base_name_mc = base_name_mc + '_els_'
    else:
        base_name_mc = base_name_mc + '_rig_'
    
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    
    fname_new = cm.save_memmap(mc.mmap_file, base_name=base_name_mc, order='C', border_to_0=border_to_0, dview=dview)

    # Load file (format: T x X x Y)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    #%% Restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    #%% Run CNMF
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    #%% Plot all components
    if not opts_dict['is3D']:
        Cn = cm.local_correlations(images.transpose(1,2,0))
        Cn[np.isnan(Cn)] = 0
        cnm.estimates.plot_contours_nb(img=Cn)

    #%% Re-run seeded CNMF on accepted patches to refine and perform deconvolution 
    cnm2 = cnm #cnm.refit(images, dview=dview)

    #%% Evaluate components
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    #%% Extract DF/F
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    #%% Select high quality components
    cnm2.estimates.select_components(use_object=True)

    #%% Plot all components
    if not opts_dict['is3D']:
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    #%% Plot accepted component traces
    if opts_dict['is3D']:
        cnm2.estimates.nb_view_components_3d(image_type='max', dims=dims, axis=2)
    else:
        cnm2.estimates.nb_view_components(images, img=Cn, idx=cnm2.estimates.idx_components)

    #%% Plot rejected component traces
    if not opts_dict['is3D'] and len(cnm2.estimates.idx_components_bad) > 0:
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)

    #%% Save results
    if not opts_dict['is3D']:
        cnm2.estimates.Cn = Cn
    cnm2.save(output_file)

    #%% Stop cluster and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    #%% Display final movie
    if not opts_dict['is3D']:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2, magnification=2, bpx=border_to_0, include_bck=False)

    #%% Reconstruct denoised movie
    if not opts_dict['is3D']:
        denoised = cm.movie(cnm2.estimates.A.dot(cnm2.estimates.C) + \
                        cnm2.estimates.b.dot(cnm2.estimates.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])

    #%% Save motion correction shifts and summary images
    save_mc(mc=mc, caiman_fp=output_file, is3D=opts_dict['is3D'])

#%% Run CaImAn analysis
caiman_analysis(opts_dict, output_file)