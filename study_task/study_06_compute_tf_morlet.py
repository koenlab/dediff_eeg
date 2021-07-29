"""
Script: study_06_compute_tf_morlet.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

#####---Import Libraries---#####
from sys import modules

import numpy as np
import json
from collections import OrderedDict

from mne import read_epochs
from mne.time_frequency import (tfr_morlet, tfr_array_morlet, EpochsTFR)
import mne

from study_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

scenes = []
objects = []
smos = []

# Define which reference to use
ref = 'avg'    

# Frequecies to estimate
freqs = np.arange(3,51,1) # Estimate 4-50Hz in linear intervals
n_cycles = 6 # of cycles

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
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Creating TF data (Morlet) for task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')

    # Load epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file)

    # Estiamte TFR
    for comment, query in queries.items():
        
        # Get the data and pad it
        tmp_epochs = epochs[query]
        epoch_data = tmp_epochs.get_data()
        pad_before, pad_after = np.array([1, 1]) / (1/epochs.info['sfreq'])
        pad_width = ((0,0),(0,0),(int(pad_before), int(pad_after)))
        epoch_data = np.pad(epoch_data, pad_width, mode='symmetric')
        
        # Compute power (averaged)
        tfr_data = tfr_array_morlet(epoch_data, epochs.info['sfreq'], freqs=freqs, 
                                    n_cycles=n_cycles, use_fft=True, output='power', 
                                    n_jobs=1)
        tfr_data = tfr_data[:,:,:,int(pad_before):]
        tfr_data = tfr_data[:,:,:,:-int(pad_before)]
        power = EpochsTFR(epochs.info, tfr_data, epochs.times, freqs,
                            method='morlet', 
                            events=epochs.events,
                            event_id=epochs.event_id,
                            metadata=epochs.metadata,
                            verbose=True).average()

        # Crop and apply baseline to TFR
        power.crop(tmin=-.5, tmax=1.5).apply_baseline((-.5,-.2), mode='logratio')
        power.comment = comment

        # Save the power
        tfr_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_method-morlet_cond-{comment}_tfr.h5'
        power.save(tfr_file, overwrite=True)

        # Make JSON
        json_info = {
            'Description': 'TFR Power from Morlet Wavelets',
            'baseline': dict(twin=(-.5,-.2), mode='logratio', sort_keys=True), 
            'sfreq': power.info['sfreq'],
            'reference': 'average',
            'tmin': power.times.min(),
            'tmax': power.times.max(),
            'freqs': freqs.tolist(),
            'n_cycles': n_cycles
        }
        json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_method-morlet_cond-{comment}_tfr.json'
        with open(json_file, 'w') as outfile:
            json.dump(json_info, outfile, indent=4)
