"""
Script: study_05_compute_tf_hilbert.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import json

from mne import read_epochs
from mne.time_frequency import EpochsTFR
import mne

from study_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

# Define which reference to use
ref = 'avg'    

#####---Define Frequency Bands---#####
freq_bands = OrderedDict(
    theta = dict(l_freq=4, h_freq=7, l_trans_bandwidth=2, h_trans_bandwidth=2),
    alpha = dict(l_freq=8, h_freq=12, l_trans_bandwidth=2, h_trans_bandwidth=2),
    lowbeta = dict(l_freq=13, h_freq=17, l_trans_bandwidth=2, h_trans_bandwidth=2),
    highbeta = dict(l_freq=18, h_freq=25, l_trans_bandwidth=2, h_trans_bandwidth=2),
    gamma = dict(l_freq=30,h_freq=50, l_trans_bandwidth=2, h_trans_bandwidth=2)
)

scenes = []
objects = []
smos = []

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Creating TF data (Hilbert Transform) for task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')

    # Load epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file)

    # amp output array
    n_trials, n_chans, n_times = epochs._data.shape
    n_freqs = len(freq_bands.keys())
    tf_shape = (n_trials, n_chans, n_freqs, n_times)
    amp = np.zeros(tf_shape)
    phase = np.zeros(tf_shape)
    center_f = np.zeros(n_freqs)
    
    # Apply badpass filter
    for i, (band, opts) in enumerate(freq_bands.items()):
        
        # Make copy of epochs and filter
        epochs_band = epochs.copy()
        epochs_band.filter(opts['l_freq'], opts['h_freq'], 
                            l_trans_bandwidth=opts['l_trans_bandwidth'], 
                            h_trans_bandwidth=opts['h_trans_bandwidth'], 
                            pad = 'symmetric', 
                            n_jobs=4)
        
        # Apply hilbert
        epochs_hilb = epochs_band.copy()
        epochs_hilb.apply_hilbert(picks=['eeg'], n_jobs=4)
        
        # Compute amp and phase
        amp[:,:,i,:] = np.abs(epochs_hilb._data.real)
        phase[:,:,i,:] = np.angle(epochs_hilb._data)
        
        # Compute center frequency
        center_f[i] = np.mean([opts['l_freq'], opts['h_freq']])

    # Make an EpochsTFR using center frequency as the point in each band
    amp_epochs = EpochsTFR(epochs.info, amp, epochs.times, center_f,
                        method='hilbert', 
                        events=epochs.events,
                        event_id=epochs.event_id,
                        metadata=epochs.metadata,
                        verbose=True)
    amp_epochs.crop(tmin=-.5, tmax=1.5)

    # Save the amp
    amp_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_mode-hilbert_value-amp_tfr.h5'
    amp_epochs.save(amp_file, overwrite=True)

    # Make JSON
    json_info = {
        'Description': 'TFR amplitude from Hilbert Transform',
        'baseline': dict(twin='n/a', mode='n/a', sort_keys=True), 
        'sfreq': amp_epochs.info['sfreq'],
        'reference': 'average',
        'bandpass_filters': dict(freq_bands, sort_keys=True),
        'freq_centers': list(center_f),
        'tmin': amp_epochs.times.min(),
        'tmax': amp_epochs.times.max()
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_mode-hilbert_value-amp_tfr.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    
    
    # Make an EpochsTFR using center frequency as the point in each band
    phase_epochs = EpochsTFR(epochs.info, phase, epochs.times, center_f,
                      method='hilbert', 
                      events=epochs.events,
                      event_id=epochs.event_id,
                      metadata=epochs.metadata,
                      verbose=True)
    phase_epochs.crop(tmin=-.5, tmax=1.5)
    
    # Save the phase
    phase_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_mode-hilbert_value-phase_tfr.h5'
    phase_epochs.save(phase_file, overwrite=True)
    
    # Make JSON
    json_info = {
        'Description': 'TFR Phase from Hilbert Transform',
        'baseline': dict(twin='n/a', mode='n/a', sort_keys=True), 
        'sfreq': phase_epochs.info['sfreq'],
        'reference': 'average',
        'bandpass_filters': dict(freq_bands, sort_keys=True),
        'freq_centers': list(center_f),
        'tmin': phase_epochs.times.min(),
        'tmax': phase_epochs.times.max()
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_mode-hilbert_value-phase_tfr.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
