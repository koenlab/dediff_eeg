"""
Script: study_06_compute_tf_morlet.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

# Import Libraries
import sys
import os

os.chdir(os.path.split(__file__)[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import numpy as np
import json

from mne import read_epochs
from mne.time_frequency import tfr_morlet

from study_config import (deriv_dir, task)
from functions import get_sub_list

scenes = []
objects = []
smos = []

# Define which reference to use
ref = 'mastoids'

# Frequecies to estimate
freqs = np.arange(3, 36, 1)  # Estimate 4-50Hz in linear intervals
n_cycles = freqs

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    print(f'Creating TF data (Morlet) for task-{task} data for {sub}')
    print(f'  Derivatives Folder: {deriv_path}')

    # Load epochs
    epochs_fif_file = deriv_path / \
        f'{sub}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file).drop_channels(
        ['TP9', 'TP10', 'FT9', 'FT10', 'VEOG', 'HEOG'])

    # Run TFR
    power = tfr_morlet(epochs, freqs, n_cycles, use_fft=True, decim=2,
                       return_itc=False, average=False)

    # Save the power
    tfr_file = deriv_path / f'{sub}_task-{task}_ref-{ref}_tfr.h5'
    power.save(tfr_file, overwrite=True)
