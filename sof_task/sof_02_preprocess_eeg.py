"""
Script: 02_sof_preprocess_eeg.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the SOF (scene, object, face) task. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import json

from mne.io import read_raw_fif
from mne import events_from_annotations
from mne.preprocessing import ICA

from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

from autoreject import get_rejection_threshold

import mne

from sof_config import bids_dir, deriv_dir, event_dict, task, preprocess_options

# Ask for subject IDs to analyze
print('What IDs are being preprocessed?')
print('(Enter multiple values separated by a comma; e.g., 101,102)')
sub_list = input('Enter IDs: ')
sub_list = sub_list.split(',')
print(sub_list)

for sub in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID
    sub_string = f'sub-{sub}'
    print(f'Preprocessing task-{task} data for {sub_string}')
    
    # Define some sobject directories
    bids_path = bids_dir / sub_string
    deriv_path = deriv_dir / sub_string
    
    ### STEP 1: LOAD DATA AND UPDATE EVENTS
    # Load Raw EEG data from derivatives folder
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-import_raw.fif.gz'
    raw = read_raw_fif(raw_fif_file, preload=True)

    # Read events from annotations
    events, event_id = mne.events_from_annotations(raw, event_id=event_dict)

    ## Adjust events with photosensor
    # Loop through and group markers and photosensor data
    # Changes happen inplace
    threshold = .85
    min_dif = .005 # 5 ms difference
    for event in events:
        eeg, times = raw.copy().pick_channels(['Photosensor']).get_data(
            start=event[0]-20,stop=event[0]+51,return_times=True)
        latencies = (times * raw.info['sfreq']).astype(int)
        cutoff = threshold * eeg.max()
        mask = np.where(eeg > cutoff)[1][0]
        psensor_onset = latencies[mask]
        if (np.abs(event[0] - psensor_onset)/raw.info['sfreq']) > min_dif:
            print('updating onset')
            event[0] = psensor_onset
            
    ## Remove Photosensor from channels
    raw.drop_channels('Photosensor')

    ## Resample events and raw to 250Hz (yes this causes jitter, but it will be minimal)
    raw, events = raw.resample( preprocess_options['resample'], events=events )
    
    # Save resample events and raw
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_raw.fif.gz'
    raw.save(raw_fif_file, overwrite=True)
    event_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    mne.write_events(event_file, events)
    
    ## Update metadata file (events.tsv)
    events_bids_file = sub_bids_dir / 'eeg' / f'{sub_string}_task-{task}_events.tsv'
    metadata = pd.read_csv(events_bids_file, sep='\t')
    metadata = metadata[metadata['trial_type'] != 'boundary']
    metadata.sample = events[:,0]
    metadata.onset = metadata.sample / raw.info['sfreq']
    events_save_file = deriv_path / f'{sub_string}_task-{task}_metadata.tsv'
    metadata.to_csv(events_save_file, sep='\t')
    
    ### ICA Path
    # Make Epochs
    ica_raw = raw.copy().filter( 1, None, skip_by_annotation=['boundary'])
    ica_epochs = mne.Epochs(ica_raw, events, event_id=event_id, 
                            tmin=-1.0, tmax=1.0, baseline=(None,None),
                            preload=True)
    
    # Detect eog at stim onsets
    veog_data = ica_epochs.copy().crop(tmin=-1.5, tmax=1.5).pick_channels(['VEOG']).get_data()
    veog_diff = np.abs(veog_data.max(axis=2) - veog_data.min(axis=2))
    print('Finding blinks at onsets..')
    blink_inds = np.where(veog_diff.squeeze()>preprocess_options['blink_thresh'])
    print('Epochs to drop:', blink_inds)
    arr_blink_inds = np.array(blink_inds)
    chdim_blink_inds = np.squeeze(arr_blink_inds)

    #Drop onset blinks from epochs
    epochs.drop(chdim_blink_inds, 'Blink at Stim Onset')
    veog_events = find_eog_events(raw, ch_name='VEOG')
    blink_at_onsets = []
    blink_win = .1
    event_counter = 0
    for event in events:
        time_diff = np.abs((veog_events[:,0] - event[0]) / raw.info['sfreq'])
        if np.any(time_diff <= blink_win):
            blink_at_onsets.append(event_counter)
        event_counter += 1
    ica_epochs.drop(blink_at_onsets, reason='OnsetBlink')
    
    # Drop peak-to-peak only on EEG channels
    reject = get_rejection_threshold(ica_epochs, ch_types='eeg')
    print(reject)
    peak_to_peak = np.max(ica_epochs.get_data(), axis=2) - np.min(ica_epochs.get_data(), axis=2)
    peak_to_peak = np.any( peak_to_peak > reject['eeg'], axis=1)
    if np.sum(peak_to_peak) < 50:
        ica_epochs.drop(peak_to_peak, reason='GlobalP2P')
    else:
        print('Too many epochs rejected on Global P2P...not applying')
    
    # Visual inspect
    ica_epochs.plot(n_channels=65,n_epochs=5,block=True,
                    scalings=dict(eeg=150e-6, eog=250e-6), picks='all')
    
    # Save ICA epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-ica_epo.fif.gz'
    ica_epochs.save(epochs_fif_file, overwrite=True)
    
    
    # ESTIMATE ICA
    ica = ICA(method='fastica', max_iter=1000, 
              random_state=97)
    ica.fit(ica_epochs)
    
    # Save ICA
    ica_file = deriv_path / f'{sub_string}_task-{task}_ica.fif.gz'
    ica.save(ica_file)
    
    eog_epochs = create_eog_epochs(raw)
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
    ica.exclude = eog_inds
    ica.plot_components(inst=ica_epochs)
    ica.plot_sources(inst=ica_epochs,block=True)
    ica.save(ica_file)
    
    ### Main Preprocessing path
    raw.filter( 0.05, None)
    
    # Add an empty channel
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                            tmin=-2.0, tmax=2.0, baseline=(None,None),
                            metadata=metadata, preload=True)
    ica.apply(epochs)
    epochs.interpolate_bads()
    mne.add_reference_channels(epochs, 'FCz', copy=False)
    epochs.set_eeg_reference()
    epochs.apply_baseline((-.2,0))
    epochs.set_montage('standard_1005')
    
    ar = AutoReject(n_interpolate=np.array([1,4,32]), 
                    consensus=np.linspace(0,1.0,num=11),
                    thresh_method='bayesian_optimization', 
                    n_jobs=4, random_state=42, verbose=False)
    ar.fit(epochs)
    
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    ar.save( deriv_path / f'{sub_string}_task-{task}_autoreject.hd5')
    
    epochs.drop(reject_log.bad_epochs, reason='Autoreject')
    epochs.drop(blink_at_onsets, reason='OnsetBlink')
    epochs.plot(n_channels=65, n_epochs=5, block=True, 
                scalings=dict(eeg=100e-6, eog=250e-6), picks='all')
    
    # Save cleaned epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_desc-cleaned_epo.fif.gz'
    epochs.save(epochs_fif_file, overwrite=True)
    events_save_file = deriv_path / f'{sub_string}_task-{task}_desc-cleaned_metadata.tsv'
    epochs.metadata.to_csv(events_save_file, sep='\t')
    
    # Make faces evoked 
    face_query = "category == 'faces' and repeat==1 and n_responses==0"
    faces = epochs[face_query].crop(-.2,.5).average()
    faces_file = deriv_path / f'{sub_string}_task-{task}_cond-faces_filt-none_ave.fif.gz'
    faces.save(faces_file)
    
    # Make scenes evoked
    scene_query = "category == 'scenes' and repeat==1 and n_responses==0"
    scenes = epochs[scene_query].crop(-.2,.5).average()
    scenes_file = deriv_path / f'{sub_string}_task-{task}_cond-scenes_filt-none_ave.fif.gz'
    scenes.save(scenes_file)
    
    # Make objects evoked
    object_query = "category == 'objects' and repeat==1 and n_responses==0"
    objects = epochs[scene_query].crop(-.2,.5).average()
    objects_file = deriv_path / f'{sub_string}_task-{task}_cond-objects_filt-none_ave.fif.gz'
    objects.save(objects_file)
    
    # # power 
    # freqs = np.arange(start=3, stop=50)
    # cycles = 7
    # power = tfr_morlet(epochs, freqs, 5, average=False, return_itc=False)
    
    # power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1, fmin=8, fmax=12,
    #                baseline=(-0.3, -.15), mode='logratio',
    #                title='Beta', show=True)
    # power.apply_baseline(baseline=(-.3,-.15), mode='logratio')
    # face_tfr = power['face/novel'].average()
    # object_tfr = power['object/novel'].average()
    # scene_tfr = power['scene/novel'].average()
    
    # face_diff_tfr = mne.combine_evoked([face_tfr, object_tfr], weights=[1,-1])
    # face_diff_tfr.plot_topomap(tmin=0.5, tmax=1, fmin=8, fmax=12,
    #                title='Beta', show=True)
    # scene_diff_tfr = mne.combine_evoked([scene_tfr, object_tfr], weights=[1,-1])
    # scene_diff_tfr.plot_topomap(tmin=0.5, tmax=1, fmin=8, fmax=12,
    #                title='Beta', show=True)