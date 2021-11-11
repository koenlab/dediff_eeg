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
from mne.time_frequency import (tfr_array_morlet, EpochsTFR)

from study_config import (deriv_dir, task)
from functions import get_sub_list

scenes = []
objects = []
smos = []

# Define which reference to use
ref = 'mastoids'

# Frequecies to estimate
freqs = np.arange(3, 51, 1)  # Estimate 4-50Hz in linear intervals
freqs = np.log10(np.logspace(3, 50, num=20))
n_cycles = freqs / 2  # of cycles

# Conditions and queries to look at
queries = {
        'all': "study_n_responses==1 and test_resp in [1,2,3,4,5,6]",
        'scene': "category=='scene' and study_n_responses==1 and test_resp in [1,2,3,4,5,6]",
        'object': "category=='object' and study_n_responses==1 and test_resp in [1,2,3,4,5,6]",
        'hit65': "study_n_responses==1 and test_resp in [5,6]",
        'miss65': "study_n_responses==1 and test_resp in [1,2,3,4]",
        'scenehit65': "category=='scene' and study_n_responses==1 and test_resp in [5,6]",
        'scenemiss65': "category=='scene' and study_n_responses==1 and test_resp in [1,2,3,4]",
        'objecthit65': "category=='object' and study_n_responses==1 and test_resp in [5,6]",
        'objectmiss65': "category=='object' and study_n_responses==1 and test_resp in [1,2,3,4]",
        'hit6': "study_n_responses==1 and test_resp in [6]",
        'miss6': "study_n_responses==1 and test_resp in [1,2,3,4,5]",
        'scenehit6': "category=='scene' and study_n_responses==1 and test_resp in [6]",
        'scenemiss6': "category=='scene' and study_n_responses==1 and test_resp in [1,2,3,4,5]",
        'objecthit6': "category=='object' and study_n_responses==1 and test_resp in [6]",
        'objectmiss6': "category=='object' and study_n_responses==1 and test_resp in [1,2,3,4,5]"
    }

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

    # Estimate TFR
    powers = []
    for comment, query in queries.items():

        # Get the data and pad it
        tmp_epochs = epochs[query]
        epoch_data = tmp_epochs.get_data()
        pad_before, pad_after = np.array([.50, .50]) / (1 / epochs.info['sfreq'])
        pad_width = ((0, 0), (0, 0), (int(pad_before), int(pad_after)))
        epoch_data = np.pad(epoch_data, pad_width, mode='symmetric')

        # Compute power (averaged)
        tfr_data = tfr_array_morlet(
            epoch_data, epochs.info['sfreq'], freqs=freqs,
            n_cycles=n_cycles, use_fft=True, output='power',
            n_jobs=4)
        tfr_data = tfr_data[:, :, :, int(pad_before):]
        tfr_data = tfr_data[:, :, :, :-int(pad_after)]
        power = EpochsTFR(
            epochs.info, tfr_data, epochs.times, freqs,
            method='morlet', events=tmp_epochs.events,
            event_id=tmp_epochs.event_id, metadata=tmp_epochs.metadata,
            verbose=True).average()

        # Crop and apply baseline to TFR
        power.crop(tmin=-.5, tmax=1.5)
        power.comment = comment
        powers.append(power)

        # Save the power
        tfr_file = deriv_path / f'{sub}_task-{task}_ref-{ref}_method-morlet_cond-{comment}_tfr.h5'
        power.save(tfr_file, overwrite=True)

        # Make JSON
        json_info = {
            'Description': 'TFR Power from Morlet Wavelets',
            'sfreq': power.info['sfreq'],
            'reference': 'average',
            'tmin': power.times.min(),
            'tmax': power.times.max(),
            'freqs': freqs.tolist(),
            'n_cycles': n_cycles.tolist()
        }
        json_file = deriv_path / f'{sub}_task-{task}_ref-{ref}_method-morlet_cond-{comment}_tfr.json'
        with open(json_file, 'w') as outfile:
            json.dump(json_info, outfile, indent=4)
