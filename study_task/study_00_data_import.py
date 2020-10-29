"""
Script: 00_study_data_import.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the memory task. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import json

from mne.io import read_raw_brainvision
from mne import events_from_annotations
from mne_bids import BIDSPath, write_raw_bids
from mne_bids.copyfiles import copyfile_brainvision

import mne

from study_config import bids_dir, source_dir, deriv_dir, event_dict, task, bad_chans

#####---Overwrite BIDS---#####
overwrite = True

#####---Anonymize Dictionary---#####
# Update to make random days back +/- 120 days
anonymize = {
    'daysback': (365*randrange(100,120)) + (randrange(-120,120) + random())
}

#####---Event Dictionaries for renaming and output---#####
# Rename dictionary
rename_dict = {
    'New Segment/': 'boundary',
    'Marker/M 11': 'scene',
    'Marker/M 21': 'object',
}

# Add boundary to event_dict
event_dict['boundary'] = -99

#####---Data columns to keep and add---#####
# List of data columns to drop in study and test files
cols_to_drop = ['date','expName','eeg_record','ran','order','image_file']

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
cols_to_add = ['trial_number', 'image', 'category', 'subcategory', 'study_resp', 
               'study_rt', 'study_n_responses', 'study_correct', 'test_resp', 'test_rt']

#####---Get Subject List---#####
for sub in source_dir.glob('sub-*'):
    
    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_string = sub.name
    sub_id = sub_string.replace('sub-','')
    bids_id = sub_id.replace('p3e2s','')
    source_path = source_dir / sub_string
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
    
    ### WRITE EEG TO BIDS FORMAT ###
    # Define the source data file 
    source_vhdr = source_path / f'{sub_string}_task-{task}_run-01_eeg.vhdr'

    # Read in raw bv from source
    raw = read_raw_brainvision(source_vhdr, misc=['Photosensor'],
                               eog=['VEOG','HEOG'])

    # Update line frequency to 60 Hz
    raw.info['line_freq'] = 60.0
    
    # Update event descriptions
    description = raw.annotations.description
    for old_name, new_name in rename_dict.items():
        description[description == old_name] = new_name
    
    # Extract Events
    events, event_id = events_from_annotations(raw, event_id=event_dict)
    
    # Get bad channels and update
    sub_bad_chans = bad_chans.get(bids_id)
    if sub_bad_chans is not None:
        raw.info['bads'] = sub_bad_chans['channels']
        
    # Write BIDS Output
    write_raw_bids(raw, bids_path=bids_path, event_id=event_id, 
                   events_data=events, anonymize=anonymize,
                   overwrite=True, verbose=False)
    
    ### UPDATE CHANNELS.TSV ###
    # Load *channels.tsv file
    bids_path.update(suffix='channels', extension='.tsv')
    chans_data = pd.read_csv(bids_path.fpath, sep='\t')
    
    # Add status_description
    chans_data['status_description'] = 'n/a'
    if sub_bad_chans is not None:
        for chan, reason in sub_bad_chans.items():
            chans_data.loc[chans_data['name'] == chan, ['status_description']] = reason
    
    # Add EEGReference
    chans_data['reference'] = 'FCz'
    for chan in ['VEOG', 'HEOG', 'Photosensor']:
        chans_data.loc[chans_data['name']==chan, ['reference']] = 'n/a'
    
    # Overwrite file
    chans_data.to_csv(bids_path.fpath, sep='\t',index=False)
        
    ### PROCESS STUDY DATA FILE ###
    # Read in the study*.tsv behavioral file
    study_source_file = source_path / f'{sub_string}_task-{task}_run-01_beh.tsv'
    study_data = pd.read_csv(study_source_file,sep='\t')
    
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
    study_data['study_correct'] = 0 # initialize to wrong
    for index, row in study_data.iterrows():
        if row.subcategory == 'natural':
            
            if row.study_resp == 2:
                study_data.at[index,'study_correct'] = 1

        elif row.subcategory == 'manmade':

            if row.study_resp == 1:
                study_data.at[index,'study_correct'] = 1

    ### PROCESS TEST DATA FILE ###
    # Read in the study*.tsv behavioral file
    test_source_file = source_path / f'{sub_string}_task-test_run-01_beh.tsv'
    test_data = pd.read_csv(test_source_file,sep='\t')
    
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

    ### MERGE STUDY AND TEST DATA ###
    # Add test to study
    study_data = study_data.merge(test_data[test_to_study], how='inner', on='image')

    # Add study to test
    test_data = test_data.merge(study_data[study_to_test], how='left', on='image')
    test_data.fillna(value='n/a', inplace=True)

    ### SAVE STUDY AND TEST DATA ###
    # Update BIDSPath
    bids_path.update(datatype='beh')
    bids_path.directory.mkdir(parents=True, exist_ok=True)
    
    # Save study data
    study_save_file = bids_path.directory / f'sub-{bids_id}_task-{task}_beh.tsv'
    study_data.to_csv(study_save_file, sep='\t', index=False)
    
    # Save test data
    test_save_file = bids_path.directory / f'sub-{bids_id}_task-test_beh.tsv'
    test_data.to_csv(test_save_file, sep='\t', index=False)
    
    ### UPDATE *_EVENTS.TSV ###
    # Load *events.tsv
    bids_path.update(datatype='eeg', suffix='events')
    events_data = pd.read_csv(bids_path.fpath, sep='\t')

    # Remove duplicat rows (mne-bids .5 bug) if needed
    if events.shape[0]*2 == events_data.shape[0]:
        events_data.drop(index=np.arange(1,events_data.shape[0]+1, step=2), 
                        inplace=True)
    events_data.reset_index()

    # Add new columnas as "n/a" values
    events_data[cols_to_add] = 'n/a'

    # Update with values
    counter = 0 # Keep track of current row in study_data
    for index, row in events_data.iterrows():
        if row['trial_type'] != 'boundary':
            this_trial = study_data.iloc[counter]
            for col in cols_to_add:
                events_data.at[index, col] = this_trial[col]
            counter += 1

     # Overwrite *events.tsv
    events_data.to_csv(bids_path.fpath, sep='\t', index=False)

    ### UPDATE *eeg_json
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

    ### Write Raw and Events to .fif.gz file
    # Write Raw instance
    raw_out_file = deriv_path / f'sub-{bids_id}_task-{task}_desc-import_raw.fif.gz'
    raw.save(raw_out_file, overwrite=overwrite)

    # Write events
    events_out_file = deriv_path / f'sub-{bids_id}_task-{task}_desc-import_eve.txt'
    mne.write_events(events_out_file, events)
   
