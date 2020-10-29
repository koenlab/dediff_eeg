"""
Script: 00_monster_data_import.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the Monster task. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import json
from random import (random, randrange)

from mne.io import read_raw_brainvision
from mne import events_from_annotations
from mne_bids import BIDSPath, write_raw_bids
from mne_bids.copyfiles import copyfile_brainvision

import mne

from monster_config import bids_dir, source_dir, deriv_dir, event_dict, task, bad_chans

#####---Overwrite BIDS---#####
overwrite = True

#####---Event Dictionary to Keep---#####
rename_dict = {
    'New Segment/': 'boundary',
    'Marker/M 10': 'bot/stan/a1',
    'Marker/M 11': 'bot/stan/a2',
    'Marker/M 12': 'bot/stan/a3',
    'Marker/M 13': 'bot/stan/a4',
    'Marker/M 14': 'bot/stan/a5',
    'Marker/M 15': 'bot/stan/a6',
    'Marker/M 16': 'bot/stan/a7',
    'Marker/M 17': 'bot/stan/a8',
    'Marker/M 20': 'bot/odd/a1',
    'Marker/M 21': 'bot/odd/a2',
    'Marker/M 22': 'bot/odd/a3',
    'Marker/M 23': 'bot/odd/a4',
    'Marker/M 24': 'bot/odd/a5',
    'Marker/M 25': 'bot/odd/a6',
    'Marker/M 26': 'bot/odd/a7',
    'Marker/M 27': 'bot/odd/a8',
    'Marker/M110': 'top/stan/a1',
    'Marker/M111': 'top/stan/a2',
    'Marker/M112': 'top/stan/a3',
    'Marker/M113': 'top/stan/a4',
    'Marker/M114': 'top/stan/a5',
    'Marker/M115': 'top/stan/a6',
    'Marker/M116': 'top/stan/a7',
    'Marker/M117': 'top/stan/a8',
    'Marker/M120': 'top/odd/a1',
    'Marker/M121': 'top/odd/a2',
    'Marker/M122': 'top/odd/a3',
    'Marker/M123': 'top/odd/a4',
    'Marker/M124': 'top/odd/a5',
    'Marker/M125': 'top/odd/a6',
    'Marker/M126': 'top/odd/a7',
    'Marker/M127': 'top/odd/a8',
    }

# Add boundary to event_dict
event_dict['boundary'] = -99

#####---Data columns to keep and add---#####
# List of data columns to drop behavioral data file
cols_to_drop = ['date','expName','eeg_record','Unnamed: 50', 'resp_hand']

# Dictionary of columns to rename in behavioral data file
cols_to_rename = {
    'frameRate': 'frame_rate',
    'psychopyVersion': 'psychopy_version',
    'TrialNumber': 'trial_number'
}

# List of columns to add to *events.tsv from behavioral data
cols_to_add = ['angle_bin', 'correct', 'correct_resp', 'gabor_loc', 
               'phase', 'resp', 'rt', 'this_angle']

#####---Get Subject List---#####
for sub in source_dir.glob('sub-*'):
    
    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID
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

    ## Anonymize Dictionary
    anonymize = {
        'daysback': (365*randrange(100,110)) + (randrange(-120,120) + random())
    }

    # Read in raw bv from source
    raw = read_raw_brainvision(source_vhdr, misc=['Photosensor'],
                               eog=['VEOG','HEOG'])
    raw.anonymize(daysback=anonymize['daysback'])

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
                   events_data=events, overwrite=True, verbose=False)

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
        
    ### PROCESS BEHAVIORAL DATA FILE ###
    # Read in the sof*.tsv behavioral file
    beh_source_file = source_path / f'{sub_string}_task-{task}_run-01_beh.tsv'
    beh_data = pd.read_csv(beh_source_file, sep='\t')
    beh_data.rename(columns=cols_to_rename, inplace=True)
    beh_data.drop(columns=cols_to_drop, inplace=True)

    # Replace subject id and select needed data columns
    beh_data['id'] = bids_id
    
    # Replace NaN and -99 with 'n/a' for resp and rt, respectively
    beh_data['resp'].fillna('n/a', inplace=True)
    beh_data['rt'].replace(-99.0, 'n/a', inplace=True)
    beh_data['correct'].fillna(0, inplace=True)
    
    # Convert accuracy to integer
    beh_data['correct'] = beh_data['correct'].astype(int)
    
    # Fil in some more values
    beh_data.replace(['None', '', '--'], 'n/a', inplace=True)

    # Save behavioral data
    bids_path.update(datatype='beh')
    bids_path.directory.mkdir(parents=True, exist_ok=True)
    beh_save_file = bids_path.directory / f'sub-{bids_id}_task-{task}_beh.tsv'
    beh_data.to_csv(beh_save_file, sep='\t', index=False)
        
    ### UPDATE *_EVENTS.TSV ###
    # Load *events.tsv
    bids_path.update(datatype='eeg', suffix='events')
    events_data = pd.read_csv(bids_path.fpath, sep='\t')

    # Remove duplicat rows (mne-bids .5 bug) if needed
    if events.shape[0]*2 == events_data.shape[0]:
        events_data.drop(index=np.arange(1,events_data.shape[0]+1, step=2), 
                        inplace=True)
    events_data.reset_index()

    # Add new columns from beh_data
    events_data[cols_to_add] = beh_data[cols_to_add]
    
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
        