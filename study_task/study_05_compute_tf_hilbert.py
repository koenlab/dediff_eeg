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
import mne

from study_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

#####---Define Frequency Bands---#####
freq_bands = OrderedDict(
    #theta=dict(l_freq=4,h_freq=7,l_trans_bandwidth=2, h_trans_bandwidth=2),
    #alpha=dict(l_freq=8,h_freq=12,l_trans_bandwidth=2, h_trans_bandwidth=2),
    lowbeta=dict(l_freq=13,h_freq=17,l_trans_bandwidth=2, h_trans_bandwidth=2),
    highbeta=dict(l_freq=18,h_freq=25,l_trans_bandwidth=2, h_trans_bandwidth=2),
    gamma=dict(l_freq=30,h_freq=50,l_trans_bandwidth=2, h_trans_bandwidth=2)
)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Creating evoked for task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    # Load epochs
    ref = 'avg'
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file)
    
    # Apply badpass filter
    for band, opts in freq_bands.items():
        
        # Make copy of epochs and filter
        epochs_band = epochs.copy()
        epochs_band.filter(opts['l_freq'], opts['h_freq'], 
                           l_trans_bandwidth=opts['l_trans_bandwidth'], 
                           h_trans_bandwidth=opts['h_trans_bandwidth'])
        
        scene_hilb = epochs_band['scene'].copy()
        scene_hilb.subtract_evoked()
        scene_hilb.apply_hilbert(picks=['eeg'], n_jobs=4).crop(tmin=-.5,tmax=.6)
        scene_amp = scene_hilb.copy()
        scene_amp._data = np.abs(scene_amp._data)**2
        scene_amp.apply_baseline((-.3,0))
        scene = scene_amp.average()
        scene.plot_joint()
        
        object_hilb = epochs_band['object'].copy()
        object_hilb.subtract_evoked()
        object_hilb.apply_hilbert(picks=['eeg'], n_jobs=4).crop(tmin=-.5,tmax=.6)
        object_amp = object_hilb.copy()
        object_amp._data = np.abs(object_amp._data)**2
        object_amp.apply_baseline((-.3,0))
        objects = object_amp.average()
        objects.plot_joint()
        
        diff = mne.combine_evoked([scene,objects], weights=[1,-1])
        diff.plot_joint()
        mne.viz.plot_compare_evokeds([scene,objects],axes='topo')
        
       