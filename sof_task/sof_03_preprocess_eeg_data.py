"""
Script: 02_sof_preprocess_eeg.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the SOF (scene, object, face) task. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from mne.io import read_raw_fif
from mne.preprocessing import read_ica
from mne.time_frequency import tfr_morlet, psd_welch
import mne

from autoreject import (AutoReject, get_rejection_threshold)


from sof_config import (bids_dir, deriv_dir, task, preprocess_options, bv_montage)
from sof_config import event_dict as event_id

# Ask for subject IDs to analyze
print('What IDs are being preprocessed?')
print('(Enter multiple values separated by a comma; e.g., 101,102)')
sub_list = input('Enter IDs: ')
sub_list = sub_list.split(',')
print(sub_list)

for sub in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_string = f'sub-{sub}'
    bids_path = bids_dir / sub_string
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  BIDS Folder: {bids_path}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: LOAD RESAMPLED DATA AND EVENTS
    # Load Raw EEG data from derivatives folder
    resamp_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_raw.fif.gz'
    raw = read_raw_fif(resamp_fif_file, preload=True)

    # Load events
    events_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    events = mne.read_events(events_file)
    
    ### Step 2: Load Metadata for event epochs creation
    ## Update metadata file (events.tsv)
    events_bids_file = bids_path / 'eeg' / f'{sub_string}_task-{task}_events.tsv'
    metadata = pd.read_csv(events_bids_file, sep='\t')
    metadata = metadata[metadata['trial_type'] != 'boundary']
    metadata.sample = events[:,0]
    metadata.onset = metadata.sample / raw.info['sfreq']
    metadata_file = deriv_path / f'{sub_string}_task-{task}_metadata.tsv'
    metadata.to_csv(metadata_file, sep='\t')
    
    ### Step 3: Filter Data and apply IC estimates and rereference/re-baseline
    # Load ICA
    ica_file = deriv_path / f'{sub_string}_task-{task}_desc-ica_ica.fif.gz'
    ica = read_ica(ica_file)

    # High Pass Filter raw and make epochs
    raw.filter(preprocess_options['lowcutoff'], None, 
               skip_by_annotation=['boundary'])
    raw.notch_filter([60,120], picks=['eog'])
    
    # Make Epochs from raw
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=preprocess_options['tmin'], 
                        tmax=preprocess_options['tmax'], 
                        metadata=metadata, baseline=(None,None), 
                        reject=None, preload=True)
    epochs.set_montage(bv_montage)
    
    # Apply ICA (in place)
    ica.apply(epochs)
    
    # Interpolate channels if needed, and set new montage to standard_1005
    epochs.interpolate_bads()
    mne.add_reference_channels(epochs, 'FCz', copy=False)
    epochs.set_montage(bv_montage)
    
    # Add FCz and reference to average and set montage to standard_1005
    epochs.set_eeg_reference(ref_channels='average')
   
    # Baseline to -200 to 0 
    epochs.apply_baseline((-.2,0))

    ### Step 4: Artifact Rejection
    #Find blinks at onset
    veog_data = epochs.copy().crop(tmin=-.15, tmax=.15).pick_channels(['VEOG']).get_data()
    veog_diff = np.abs(veog_data.max(axis=2) - veog_data.min(axis=2))
    blink_inds = np.where(veog_diff.squeeze()>preprocess_options['blink_thresh'])[0]
    print('Epochs with blink at stim onset:', blink_inds)

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
                scalings=dict(eeg=100e-6, eog=250e-6), 
                epoch_colors=colors, picks='all')

    # Save cleaned epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)
    events_save_file = deriv_path / f'{sub_string}_task-{task}_desc-cleaned_metadata.tsv'
    epochs.metadata.to_csv(events_save_file, sep='\t')