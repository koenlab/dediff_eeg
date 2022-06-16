"""
Script: 00_study_data_import.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for
the memory task.
"""

# Import Libraries
import sys
import os

os.chdir(os.path.split(__file__)[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import numpy as np
import pandas as pd
import json
from random import (random, randrange)

from mne.io import read_raw_brainvision
from mne import events_from_annotations
from mne_bids import BIDSPath, write_raw_bids

import mne

from study_config import (bids_dir, source_dir, deriv_dir,
                          event_dict, task, bad_chans)
from functions import get_sub_list

# Overwrite BIDS
overwrite = True

# Event Dictionaries for renaming and output
# Rename dictionary
rename_dict = {
    'New Segment/': 'boundary',
    'Marker/M 11': 'scene',
    'Marker/M 21': 'object',
}

# Add boundary to event_dict
event_dict['boundary'] = -99

# Data columns to keep and add
# List of data columns to drop in study and test files
cols_to_drop = ['date', 'expName', 'eeg_record', 'ran', 'order', 'image_file']

# Dictionary of columns to rename in study data file
study_cols_to_rename = {
    'frameRate': 'frame_rate',
    'psychopyVersion': 'psychopy_version',
    'TrialNumber': 'trial_number',
    'n_resps': 'study_n_responses'
}

# Dictionary of columns to rename in study data file
test_cols_to_rename = {
    'frameRate': 'frame_rate',
    'psychopyVersion': 'psychopy_version',
    'TrialNumber': 'trial_number',
    'n_resps': 'test_n_responses'
}

# Columns to transfer from Study to Test data
# Image is for matching keys
study_to_test = ['image', 'study_n_responses', 'study_resp', 'study_rt']

# Columns to transfer from Test data to Study
# Image is for matching keys
test_to_study = ['image', 'test_resp', 'test_rt']

# List of columns to add to *events.tsv from study_data
cols_to_add = ['trial_number', 'image', 'category', 'subcategory',
               'study_resp', 'study_rt', 'study_n_responses',
               'study_correct', 'test_resp', 'test_rt']

# Get Subject List
sub_list = get_sub_list(source_dir, is_source=True, allow_all=True)
for sub in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    sub_id = sub.replace('sub-', '')
    bids_id = sub_id[-3:]
    source_path = source_dir / sub
    bids_path = BIDSPath(subject=bids_id, task=task,
                         datatype='eeg', root=bids_dir)
    deriv_path = deriv_dir / f'sub-{bids_id}'
    deriv_path.mkdir(parents=True, exist_ok=True)
    print(f'Making BIDS data for sub-{bids_id} ({sub_id}) on task-{task}')
    print(f'  Source Path: {source_path}')
    print(f'  BIDS Path: {bids_path.directory}')
    print(f'  Derivative Path: {deriv_path}')

    # If EEG file alread writen skip this person
    exist_check = bids_path.copy().update(suffix='eeg', extension='vhdr')
    if exist_check.fpath.is_file() and not overwrite:
        print('SUBJECT BIDS DATA EXISTS: SKIPPING')
        continue

    # WRITE EEG TO BIDS FORMAT
    # Define the source data file
    source_vhdr = source_path / f'{sub}_task-{task}_run-01_eeg.vhdr'

    # Read in raw bv from source
    raw = read_raw_brainvision(source_vhdr, misc=['Photosensor'],
                               eog=['VEOG', 'HEOG'])

    # Fix channel order (swap VEOG and HEOG)
    if bids_id in ['121', '230', '237']:
        ch_names = raw.copy().info['ch_names']
        ch_names[-1], ch_names[-2] = ch_names[-2], ch_names[-1]
        raw = raw.reorder_channels(ch_names)
        raw.rename_channels(dict(VEOG='HEOG', HEOG='VEOG'))

    # Update line frequency to 60 Hz
    raw.info['line_freq'] = 60.0

    # Update event descriptions
    description = raw.annotations.description
    for i, (old_name, new_name) in enumerate(rename_dict.items()):
        description[description == old_name] = new_name

    # For sub-239...remove first 6 events
    if bids_id == '239':
        description[1:6] = 'bad'

    # Extract Events
    events, event_id = events_from_annotations(raw, event_id=event_dict)

    # Get bad channels and update
    sub_bad_chans = bad_chans.get(bids_id)
    if sub_bad_chans is not None:
        raw.info['bads'] = sub_bad_chans['channels']

    # Write BIDS Output
    write_raw_bids(raw, bids_path=bids_path, event_id=event_id,
                   events_data=events, overwrite=True, verbose=False)

    # UPDATE CHANNELS.TSV
    # Load *channels.tsv file
    bids_path.update(suffix='channels', extension='.tsv')
    chans_data = pd.read_csv(bids_path.fpath, sep='\t')

    # Add status_description
    chans_data['status_description'] = 'n/a'
    if sub_bad_chans is not None:
        for chan, reason in sub_bad_chans.items():
            chans_data.loc[chans_data['name'] == chan,
                           ['status_description']] = reason

    # Add EEGReference
    chans_data['reference'] = 'FCz'
    for chan in ['VEOG', 'HEOG', 'Photosensor']:
        chans_data.loc[chans_data['name'] == chan,
                       ['reference']] = 'n/a'

    # Overwrite file
    chans_data.to_csv(bids_path.fpath, sep='\t', index=False)

    # PROCESS STUDY DATA FILE
    # Read in the study*.tsv behavioral file
    study_source_file = source_path / f'{sub}_task-{task}_run-01_beh.tsv'
    study_data = pd.read_csv(study_source_file, sep='\t')

    # Rename and drop data columns
    study_data.rename(columns=study_cols_to_rename, inplace=True)
    study_data.drop(columns=cols_to_drop, inplace=True)

    # Replace NaN and -99 with 'n/a' for resp and rt, respectively
    study_data['study_resp'].replace(-99.0, 'n/a', inplace=True)
    study_data['study_resp_key'].replace('na', 'n/a', inplace=True)
    study_data['study_rt'].replace(-99.0, 'n/a', inplace=True)
    study_data['category'].replace('objects', 'object', inplace=True)
    study_data['category'].replace('scenes', 'scene', inplace=True)
    study_data.replace(['None', '', '--'], 'n/a', inplace=True)

    # Replace subject id and select needed data columns
    study_data['id'] = bids_id

    # Add study correct
    study_data['study_correct'] = 0  # initialize to wrong
    for index, row in study_data.iterrows():
        if row.subcategory == 'natural':

            if row.study_resp == 2:
                study_data.at[index, 'study_correct'] = 1

        elif row.subcategory == 'manmade':

            if row.study_resp == 1:
                study_data.at[index, 'study_correct'] = 1

    # PROCESS TEST DATA FILE
    # Read in the study*.tsv behavioral file
    test_source_file = source_path / f'{sub}_task-test_run-01_beh.tsv'
    test_data = pd.read_csv(test_source_file, sep='\t')

    # Rename and drop data columns
    test_data.rename(columns=test_cols_to_rename, inplace=True)
    test_data.drop(columns=cols_to_drop, inplace=True)
    test_data.drop(columns='resp_acc', inplace=True)

    # Update to 'n/a' values
    test_data['test_resp'].replace(-99.0, 'n/a', inplace=True)
    test_data['test_resp_key'].replace('na', 'n/a', inplace=True)
    test_data['test_rt'].replace(-99.0, 'n/a', inplace=True)
    test_data['category'].replace('objects', 'object', inplace=True)
    test_data['category'].replace('scenes', 'scene', inplace=True)
    test_data.replace(['None', '', '--'], 'n/a', inplace=True)

    # Replace subject id and select needed data columns
    test_data['id'] = bids_id

    # MERGE STUDY AND TEST DATA
    # Add test to study
    study_data = study_data.merge(test_data[test_to_study],
                                  how='inner', on='image')

    # Add study to test
    test_data = test_data.merge(study_data[study_to_test],
                                how='left', on='image')
    test_data.fillna(value='n/a', inplace=True)

    # HANDLE SPECIAL SUBJECTS ON DATA FILE
    if bids_id == '120':
        bad_study_trials = np.arange(28)
        bad_trials = study_data.iloc[bad_study_trials]['image'].to_list()
        study_data.drop(index=bad_study_trials, inplace=True)
        bad_test_trials = []
        for i, image in enumerate(test_data['image']):
            if image in bad_trials:
                bad_test_trials.append(i)
        test_data.drop(index=bad_test_trials, inplace=True)
        study_data.reset_index(inplace=True)
        test_data.reset_index(inplace=True)
    if bids_id == '239':
        bad_study_trials = np.arange(6)
        bad_trials = study_data.iloc[bad_study_trials]['image'].to_list()
        study_data.drop(index=bad_study_trials, inplace=True)
        bad_test_trials = []
        for i, image in enumerate(test_data['image']):
            if image in bad_trials:
                bad_test_trials.append(i)
        test_data.drop(index=bad_test_trials, inplace=True)
        study_data.reset_index(inplace=True)
        test_data.reset_index(inplace=True)

    # SAVE STUDY AND TEST DATA
    # Update BIDSPath
    bids_path.update(datatype='beh')
    bids_path.directory.mkdir(parents=True, exist_ok=True)

    # Save study data
    study_save_file = bids_path.directory / \
        f'sub-{bids_id}_task-{task}_beh.tsv'
    study_data.to_csv(study_save_file, sep='\t', index=False)

    # Save test data
    test_save_file = bids_path.directory / f'sub-{bids_id}_task-test_beh.tsv'
    test_data.to_csv(test_save_file, sep='\t', index=False)

    # UPDATE *_EVENTS.TSV
    # Load *events.tsv
    bids_path.update(datatype='eeg', suffix='events')
    events_data = pd.read_csv(bids_path.fpath, sep='\t')

    # Remove duplicat rows (mne-bids .5 bug) if needed
    if events.shape[0]*2 == events_data.shape[0]:
        events_data.drop(index=np.arange(1, events_data.shape[0]+1, step=2),
                         inplace=True)
    events_data.reset_index(inplace=True)

    # Add new columnas as "n/a" values
    events_data[cols_to_add] = 'n/a'

    # Update with values
    counter = 0  # Keep track of current row in study_data
    for index, row in events_data.iterrows():
        if row['trial_type'] != 'boundary':
            this_trial = study_data.iloc[counter]
            for col in cols_to_add:
                events_data.at[index, col] = this_trial[col]
            counter += 1

        # Overwrite *events.tsv
    events_data.to_csv(bids_path.fpath, sep='\t', index=False)

    # UPDATE *eeg_json
    # Load JSON
    bids_path.update(suffix='eeg', extension='json')
    with open(bids_path.fpath, 'r') as file:
        eeg_json = json.load(file)

    # Update keys
    eeg_json['EEGReference'] = 'FCz'
    eeg_json['EEGGround'] = 'Fpz'

    # Save EEG JSON
    with open(bids_path.fpath, 'w') as file:
        json.dump(eeg_json, file)

    # Write Raw and Events to .fif.gz file
    # Write Raw instance
    raw_out_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_ref-FCz_desc-import_raw.fif.gz'
    raw.save(raw_out_file, overwrite=overwrite)

    # Make a JSON
    json_info = {
        'Description': 'Import from BrainVision Recorder',
        'sfreq': raw.info['sfreq'],
        'reference': 'FCz'
    }
    json_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_ref-FCz_desc-import_raw.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file

    # Write events
    events_out_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_desc-import_eve.txt'
    mne.write_events(events_out_file, events)

    # Write events
    events_out_file = deriv_path / \
        f'sub-{bids_id}_task-{task}_desc-import_eve.txt'
    mne.write_events(events_out_file, events)

    # Make a JSON
    json_info = {
        'Description': 'Events from Brain Vision Import',
        'columns': ['onset', 'duration', 'code'],
        'onset_units': 'samples',
        'sfreq': raw.info['sfreq'],
        'codes': event_id
    }
    json_file = deriv_path / f'{sub}_task-{task}_desc-import_eve.json'
    try:
        json_file.unlink()
    except:
        pass
    json_file = deriv_path / f'sub-{bids_id}_task-{task}_desc-import_eve.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file

