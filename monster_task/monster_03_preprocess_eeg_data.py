"""
Script: 03_sof_preprocess_eeg.py
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
import mne

from autoreject import get_rejection_threshold

from monster_config import (bids_dir, deriv_dir, task, preprocess_options, 
                        bv_montage, get_sub_list)
from monster_config import event_dict as event_id

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    bids_path = bids_dir / sub_string
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  BIDS Folder: {bids_path}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: LOAD RESAMPLED DATA AND EVENTS
    # Load Raw EEG data from derivatives folder
    resamp_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
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
    metadata.to_csv(metadata_file, sep='\t', index=False)
    
    ### Step 3: Filter Data and apply IC estimates and rereference/re-baseline
    # Load ICA
    ica_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_ica.fif.gz'
    ica = read_ica(ica_file)

    # High Pass Filter raw and make epochs
    raw.filter(preprocess_options['highpass'], None, 
               skip_by_annotation=['boundary'])
    raw.filter(None,40, picks=['eog'])
    raw.notch_filter([60,120], picks=['eog'])
    
    # Make Epochs from raw
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=preprocess_options['tmin'], 
                        tmax=preprocess_options['tmax'], 
                        metadata=metadata, baseline=(None,None), 
                        reject=None, preload=True)
    
    # Apply ICA (in place)
    ica.apply(epochs)
    
    # Interpolate channels if needed, and set new montage to bv_montage
    if len(epochs.info['bads'])>0:
        epochs.interpolate_bads()
    else:
        print('No channels were interpolated')
    mne.add_reference_channels(epochs, 'FCz', copy=False)
    epochs.set_channel_types(mapping=dict(FCz='eeg'))
    epochs.set_montage(bv_montage, on_missing='ignore')
    
    # Add FCz and reference to average and set montage to standard_1005
    epochs.set_eeg_reference(ref_channels='average')
   
    # Baseline to -200 to 0 
    epochs.apply_baseline((-.2,0))

    # Save epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-removedICs_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)

    # Make JSON
    json_info = {
        'Description': 'Epochs with Artifactual ICs removed',
        'sfreq': epochs.info['sfreq'],
        'reference': 'average',
        'filter': {
            'eeg': {
                'highpass': epochs.info['highpass'],
                'lowpass': epochs.info['lowpass'],
                'notch': 'n/a'
            },
            'eog': {
                'highpass': epochs.info['highpass'],
                'lowpass': 40.0,
                'notch': [60.0, 120.0]
            }
        },
        'tmin': epochs.times.min(),
        'tmax': epochs.times.max(),
        'bad_epochs': 'n/a',
        'bad_channels': epochs.info['bads'],
        'metadata': metadata_file.name
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-removedICs_epo.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file
    
    # Extract epoch data for ease of computation
    epoch_data = epochs.get_data(picks=['eeg'])
    
    # Get the number of good channels
    n_good_channels = len(mne.pick_types(epochs.info, eeg=True, eog=False, exclude='bads'))
    print(f'# of good channels: {n_good_channels}')
    n_thresh_channels = (n_good_channels*preprocess_options['perc_good_chans'])
    
    # Exclude voltages > +/100 microvolts
    max_voltage = np.abs(epoch_data).max(axis=2)
    ext_val_nchan = (np.sum(max_voltage > preprocess_options['ext_val_thresh'], axis=1))
    ext_val_bad = ext_val_nchan >= n_thresh_channels
    print(f'Epochs with extreme voltage on more than {n_thresh_channels} channels:', ext_val_bad.nonzero()[0])
    
    # Exclude epochs based on Global Rejection Threshold with 8 epochs
    reject = get_rejection_threshold(epochs, ch_types='eeg')
    p2p_vals = np.abs(epoch_data.max(axis=2) - epoch_data.min(axis=2))
    p2p_nchan = np.sum(p2p_vals >= reject['eeg'], axis=1)
    p2p_bad = p2p_nchan > n_thresh_channels
    print(f'Epochs exceeding global P2P on more than {n_thresh_channels} channels:', p2p_bad.nonzero()[0])
    
    # Detect eog at stim onsets
    veog_data = epochs.copy().apply_baseline((None,None)).crop(tmin=-.1, tmax=.1).pick_channels(['VEOG']).get_data()
    veog_diff = np.abs(veog_data.max(axis=2) - veog_data.min(axis=2))
    blink_inds = np.where(veog_diff.squeeze()>preprocess_options['blink_thresh'])[0]
    print('Epochs with blink at stim onset:', blink_inds)
    
    # Make color index
    n_channels = len(epochs.info.ch_names)    
    epoch_colors = list()
    for i in np.arange(epochs.events.shape[0]):
        epoch_colors.append([None]*(n_channels-1) + ['k'])
        if i in blink_inds:
            epoch_colors[i] = ['b'] * n_channels
        if p2p_bad[i]:
            epoch_colors[i] = ['m'] * n_channels
        if ext_val_bad[i]:
            epoch_colors[i] = ['c'] * n_channels
            
    # Visual inspect
    epochs.plot(n_channels=66, n_epochs=5, block=True,
                scalings=dict(eeg=100e-6, eog=300e-6), 
                epoch_colors=epoch_colors, picks='all')
    
    # Find bad epochs
    bad_epochs = []
    for i, epo in enumerate(epochs.drop_log):
        if len(epo) > 0:
            bad_epochs.append(i)

    # Interpolate channels if needed, and set new montage to bv_montage 
    if len(epochs.info['bads'])>0:
        epochs.interpolate_bads()
    else:
        print('No channels were interpolated')

    # Save cleaned epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)
    events_save_file = deriv_path / f'{sub_string}_task-{task}_desc-cleaned_metadata.tsv'
    epochs.metadata.to_csv(events_save_file, sep='\t', index=False)
    
    # Make JSON
    json_info = {
        'Description': 'Manually cleaned epochs',
        'sfreq': epochs.info['sfreq'],
        'reference': 'average',
        'filter': {
            'eeg': {
                'highpass': epochs.info['highpass'],
                'lowpass': epochs.info['lowpass'],
                'notch': 'n/a'
            },
            'eog': {
                'highpass': epochs.info['highpass'],
                'lowpass': 40.0,
                'notch': [60.0, 120.0]
            }
        },
        'tmin': epochs.times.min(),
        'tmax': epochs.times.max(),
        'bad_epochs': bad_epochs,
        'proportion_rejected_epochs': len(bad_epochs)/len(epochs),
        'bad_channels': epochs.info['bads'],
        'metadata': events_save_file.name, 
        'artifact_detection': {
            'extreme_value': [int(i) for i in ext_val_bad.nonzero()[0]],
            'global_p2p': [int(i) for i in p2p_bad.nonzero()[0]],
            'blink_at_onset': [int(i) for i in blink_inds]
        }
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    
    # Make epochs with mastoid reference and save
    epochs.set_eeg_reference(ref_channels=['TP9','TP10'])
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-mastoids_desc-cleaned_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)
    
    # Save json with mastoid reference
    json_info['reference'] = 'mastoids'
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-mastoids_desc-cleaned_epo.json'
    with open(json_file, 'w') as outfile: 
        json.dump(json_info, outfile, indent=4)

    