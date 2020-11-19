import datajoint as dj
import pathlib
import numpy as np
import pandas as pd
import re
from datetime import datetime

import scanreader
from img_loaders import get_scanimage_acq_time


def get_imaging_root_data_dir():
    data_dir = dj.config.get('custom', {}).get('imaging_data_dir', None)
    return pathlib.Path(data_dir) if data_dir else None


def get_scan_image_files(scan_key):
    # Folder structure: root / subject / session / .tif (raw)
    data_dir = get_imaging_root_data_dir()
    subj_dir = data_dir / scan_key['subject']
    if subj_dir.exists():
        sess_dirs = set([fp.parent for fp in subj_dir.glob('*/*.tif')])
        for sess_folder in sess_dirs:
            tiff_filepaths = [fp.as_posix() for fp in (subj_dir / sess_folder).glob('*.tif')]

            try:  # attempt to read .tif as a scanimage file
                scan = scanreader.read_scan(tiff_filepaths)
            except Exception as e:
                print(f'ScanImage loading error: {tiff_filepaths[0]}\n{str(e)}')
                scan = None

            if scan is not None:
                recording_time = get_scanimage_acq_time(scan)

                recording_time_diff = abs((scan_key['session_datetime'] - recording_time).total_seconds())
                if recording_time_diff <= 120:  # safeguard that
                    return tiff_filepaths


def get_suite2p_dir(processing_task_key):
    # Folder structure: root / subject / session / suite2p / plane / ops.npy

    tiff_filepaths = get_scan_image_files(processing_task_key)
    sess_folder = pathlib.Path(tiff_filepaths[0]).parent

    suite2p_dirs = set([fp.parent.parent for fp in sess_folder.rglob('*ops.npy')])

    if len(suite2p_dirs) != 1:
        raise FileNotFoundError(f'Error searching for Suite2p output directory - Found {suite2p_dirs}')

    return suite2p_dirs.pop()


def get_caiman_dir(processing_task_key):
    # Folder structure: root / subject / session / caiman / *.hdf5

    tiff_filepaths = get_scan_image_files(processing_task_key)
    sess_dir = pathlib.Path(tiff_filepaths[0]).parent
    caiman_dir = sess_dir / 'caiman'

    if not caiman_dir.exists():
        raise FileNotFoundError('CaImAn directory not found at {}'.format(caiman_dir))

    return caiman_dir