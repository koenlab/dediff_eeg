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
import mne

from autoreject import (Ransac, get_rejection_threshold)

from monster_config import (bids_dir, deriv_dir, event_dict, 
                        task, preprocess_options, bv_montage,
                        n_interpolates, consensus, get_sub_list)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### STEP 0: SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    bids_path = bids_dir / sub_string
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  BIDS Folder: {bids_path}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: LOAD DATA AND UPDATE EVENTS
    # Load Raw EEG data from derivatives folder
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-import_raw.fif.gz'
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
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw.save(raw_fif_file, overwrite=True)

    # Save events resampled
    event_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    mne.write_events(event_file, events)
    
    # Make a JSON
    json_info = {
        'Description': 'Resampled Events Information',
        'columns': ['onset', 'duration', 'code'],
        'onset_units': 'samples',
        'sfreq': raw.info['sfreq'],
        'codes': event_id,
        'num_events_adjusted': events_adjusted
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file

    ### Step 2: Estimate ICA
    # Apply HPF to all channels and a 60Hz Notch filter to eogs
    raw.filter(preprocess_options['ica_lowcutoff'], None, 
               skip_by_annotation=['boundary'])
    raw.filter(None,30, picks=['eog'])
    raw.notch_filter([60,120], picks=['eog'])
    
    # Make ICA Epochs
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=preprocess_options['ica_tmin'], 
                        tmax=preprocess_options['ica_tmax'], 
                        baseline=(None,None), reject=None, preload=True)
    epochs.set_montage(bv_montage)
    
    # Autodetect bad channels
    picks = mne.pick_channels(epochs.info['ch_names'], [], 
                              exclude=['FT9','FT10','TP9','TP10', 'VEOG', 'HEOG'])
    ransac = Ransac(verbose=False, n_jobs=4, min_corr=.70, picks=picks)
    ransac.fit(epochs)
    if len(ransac.bad_chs_):
        for chan in ransac.bad_chs_:
            epochs.info['bads'].append(chan)
    print(f'RANSAC Bad Channels: {ransac.bad_chs_}')
    
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
                    scalings=dict(eeg=150e-6, eog=300e-6), 
                    epoch_colors=epoch_colors, picks=['eeg','eog'])
    
    # Save ICA epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)

    # Find bad epochs
    bad_epochs = []
    for i, epo in enumerate(epochs.drop_log):
        if len(epo) > 0:
            bad_epochs.append(i)

    # Make a JSON
    json_info = {
        'Description': 'Epochs for ICA',
        'sfreq': epochs.info['sfreq'],
        'reference': 'FCz',
        'filter': {
            'lowcutoff': epochs.info['highpass'],
            'highcutoff': epochs.info['lowpass'],
            'notch': 60.0,
            'Description': 'Notch only applied to EOG channels'
                  },
        'tmin': epochs.times.min(),
        'tmax': epochs.times.max(),
        'bad_epochs': bad_epochs,
        'bad_channels': epochs.info['bads'],
        'proportion_rejected_epochs': len(bad_epochs)/len(epochs),
        'metadata': 'n/a',
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_epo.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file
    
    # Estimate ICA
    ica = ICA(method='picard', max_iter=1000, random_state=97, 
              fit_params=dict(ortho=True, extended=True), 
              verbose=True)
    ica.fit(epochs)
    
    # Save ICA
    ica_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_ica.fif.gz'
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

    # Make a JSON
    json_info = {
        'Description': 'ICA components',
        'sfreq': ica.info['sfreq'],
        'reference': 'FCz',
        'ica_method': {
            'method': 'picard',
            'fit_params': dict(ortho=True, extended=True)
                      },
        'filter': {
            'lowcutoff': ica.info['highpass'],
            'highcutoff': ica.info['lowpass'],
                  },
        'n_components': len(ica.info['chs']),
        'proportion_components_flagged': len(ica.exclude)/len(ica.info['ch_names']),
        'flagged_components': [int(x) for x in ica.exclude],
        'eog_scores': {
            'description': 'Correlation with VEOG and HEOG bipolar recordings',
            'veog_scores': {f'comp{i:02d}': float(x) for i, x in enumerate(eog_scores[0])},
            'heog_scores': {f'comp{i:02d}': float(x) for i, x in enumerate(eog_scores[1])},
                      }
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_ica.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file

    # Save raw with bads attached
    del raw # just in case
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw = read_raw_fif(raw_fif_file, preload=True)
    raw.info['bads'] = epochs.info['bads']
    raw.save(raw_fif_file, overwrite=True)

    # Make JSON file
    json_info = {
        'Description': 'Resampled continuous data',
        'sfreq': raw.info['sfreq'],
        'filter': {
            'lowcutoff': raw.info['highpass'],
            'highcutoff': raw.info['lowpass'],
                  },
        'reference': 'FCz',
        'bad_channels': raw.info['bads'],
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file
    

    