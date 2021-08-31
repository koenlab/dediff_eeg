"""
Script: study_06_compute_tf_morlet.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

#####---Import Libraries---#####
import sys
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import numpy as np
import json
from collections import OrderedDict

from mne import read_epochs
from mne.time_frequency import tfr_morlet
import mne

from study_config import (deriv_dir, task)
from functions import get_sub_list

scenes = []
objects = []

# Define which reference to use
ref = 'avg'    

# Frequecies to estimate
freqs = np.arange(3,51,1) # Estimate 4-50Hz in linear intervals
n_cycles = freqs / 2 # of cycles

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    print(f'Creating TF data (Morlet) for task-{task} data for {sub}')
    print(f'  Derivatives Folder: {deriv_path}')

    # Load epochs
    epochs_fif_file = deriv_path / f'{sub}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file, preload=True, verbose=False)

    # Compute TFR Power for hits
    hit_power = tfr_morlet(epochs["study_n_responses==1 and test_resp in [5,6]"],
                           freqs=freqs, n_cycles=n_cycles, return_itc=False, n_jobs=4)
    hit_power.crop(tmin=-.5, tmax=1.5).apply_baseline((-.4, -.2), mode='logratio')
    hit_power.save(deriv_path / f'{sub}_task-study_cond-hit65_tfr.h5',
                   overwrite=True, verbose=True)

    # Compute TFR Power for hits
    miss_power = tfr_morlet(epochs["study_n_responses==1 and test_resp in [1,2,3,4]"],
                            freqs=freqs, n_cycles=n_cycles, return_itc=False, n_jobs=4)
    miss_power.crop(tmin=-.5, tmax=1.5).apply_baseline((-.4, -.2), mode='logratio')
    miss_power.save(deriv_path / f'{sub}_task-study_cond-miss65_tfr.h5',
                    overwrite=True, verbose=True)

    # Compute SME Power
    sme_power = mne.combine_evoked([hit_power, miss_power], weights=[1, -1])
    sme_power.save(deriv_path / f'{sub}_task-study_cond-sme65_tfr.h5',
                   overwrite=True, verbose=True)