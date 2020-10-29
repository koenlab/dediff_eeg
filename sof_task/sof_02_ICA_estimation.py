"""
Script: 02_sof_ICA_estimation.py
Creator: Joshua D. Koen
Description: This script handles ICA estimation for the SOF task. 
Events are adjust with Photosensor and data/events resampled (written to file)
Blinks at stimulus onset are flagged. 
AutoReject is used to highlight potentially problematic epochs. 
Epochs are visually inspected. 
FastICA is used to estimate ICs
EOG ICs are flagged using the ica.find_bad_eogs() method (corelation)
ICs are then manually flagged. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from mne.io import read_raw_fif
from mne import events_from_annotations
from mne.preprocessing import ICA
from mne.time_frequency import psd_welch
import mne

from autoreject import (AutoReject, get_rejection_threshold)

from sof_config import bids_dir, deriv_dir, event_dict, task, preprocess_options

# Ask for subject IDs to analyze
print('What IDs are being preprocessed?')
print('(Enter multiple values separated by a comma; e.g., 101,102)')
sub_list = input('Enter IDs: ')
sub_list = sub_list.split(',')
print(sub_list)

for sub in sub_list:

    ### STEP 0: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_string = f'sub-{sub}'
    bids_path = bids_dir / sub_string
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  BIDS Folder: {bids_path}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: LOAD DATA AND UPDATE EVENTS
    # Load Raw EEG data from derivatives folder
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-import_raw.fif.gz'
    raw = read_raw_fif(raw_fif_file, preload=True)

    # Read events from annotations
    events, event_id = mne.events_from_annotations(raw, event_id=event_dict)

    ## Adjust events with photosensor
    # Loop through and group markers and photosensor data
    # Changes happen inplace
    print('Adjusting event onsets with photosensor')
    threshold = .85
    min_dif = .005 # 5 ms difference
    events_adjusted = 0
    for event in events:
        eeg, times = raw.copy().pick_channels(['Photosensor']).get_data(
            start=event[0]-20,stop=event[0]+51,return_times=True)
        latencies = (times * raw.info['sfreq']).astype(int)
        cutoff = threshold * eeg.max()
        mask = np.where(eeg > cutoff)[1][0]
        psensor_onset = latencies[mask]
        if (np.abs(event[0] - psensor_onset)/raw.info['sfreq']) > min_dif:
            event[0] = psensor_onset
            events_adjusted += 1            
    print(f'  {events_adjusted} events were shifted')
            
    ## Remove Photosensor from channels
    raw.drop_channels('Photosensor')

    ## Resample events and raw to 250Hz (yes this causes jitter, but it will be minimal)
    raw, events = raw.resample( preprocess_options['resample'], events=events )
    
    # Save resample events and raw
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_raw.fif.gz'
    raw.save(raw_fif_file, overwrite=True)
    event_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    mne.write_events(event_file, events)
    
    ### Step 2: Estimate ICA
    # Apply HPF to all channels and a 60Hz Notch filter to eogs
    raw.filter(preprocess_options['ica_lowcutoff'], None, 
               skip_by_annotation=['boundary'])
    raw.notch_filter([60,120], picks=['eog'])
    
    # Make ICA Epochs
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=preprocess_options['ica_tmin'], 
                        tmax=preprocess_options['ica_tmax'], 
                        baseline=(None,None), reject=None, preload=True)
    
    # Detect eog at stim onsets
    print('Finding blinks at onsets..')
    veog_data = epochs.copy().crop(tmin=-.15, tmax=.15).pick_channels(['VEOG']).get_data()
    veog_diff = np.abs(veog_data.max(axis=2) - veog_data.min(axis=2))
    blink_inds = np.where(veog_diff.squeeze()>preprocess_options['blink_thresh'])[0]
    print('Epochs to drop:', blink_inds)
    
    # Drop peak-to-peak only on EEG channels
    ar = AutoReject(n_jobs=4, verbose='tqdm')
    _, drop_log = ar.fit(epochs).transform(epochs, return_log=True)
    
    # Make color index
    epoch_colors = [None for x in range(events.shape[0])]
    for blink in blink_inds:
        epoch_colors[blink] = 'blue'
    for i, ep in enumerate(drop_log.bad_epochs):
        if ep:
            epoch_colors[i] = 'blue'
    colors = []
    for col in epoch_colors:
        colors.append([col]*len(epochs.info['ch_names']))
    
    # Visual inspect
    epochs.plot(n_channels=65, n_epochs=5, block=True,
                    scalings=dict(eeg=150e-6, eog=300e-6), 
                    epoch_colors=colors, picks='all')
    
    # Save ICA epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-ica_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)
    
    # Estimate ICA
    #ica = ICA(method='fastica', max_iter=1000, random_state=97)
    ica = ICA(method='picard', max_iter=1000, random_state=97, 
              fit_params=dict(ortho=True, extended=True), 
              verbose=True)
    ica.fit(epochs)
    
    # Save ICA
    ica_file = deriv_path / f'{sub_string}_task-{task}_ica.fif.gz'
    ica.save(ica_file)
    
    # Find EOG artifacts
    eog_inds, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_inds
    eog_ica_plot = ica.plot_scores(eog_scores, labels=['VEOG', 'HEOG'])
    eog_ica_file = fig_path / f'{sub_string}_task-{task}_ic_eog_scores.png'
    eog_ica_plot.savefig(eog_ica_file, dpi=600)
    plt.close(eog_ica_plot)
    
    # Plot all component properties
    ica.plot_components(inst=epochs, reject=None,
                        psd_args=dict(fmax=70))
    ica.save(ica_file)
    
    # Manually inspect 
    for ic in ica.exclude:
        ica.plot_properties(epochs, picks=ic, show=False,
                           psd_args=dict(fmax=70))
        ic_file = fig_path / f'{sub_string}_task-{task}_ic{ic:02}_properties.png'
        plt.savefig(ic_file, dpi=600)
        plt.close('all')
    