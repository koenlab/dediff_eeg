"""
Script: 00_monster_data_import.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the Monster task. 
"""

#####---Import Libraries---#####
import os
import sys

import numpy as np
import pandas as pd
import json

from mne.io import read_raw_brainvision
from mne import events_from_annotations
from mne_bids import write_raw_bids, make_bids_basename

from monster_config import bids_dir, source_dir, event_dict, task, bad_chans

#####---Change Directory to Script Directory---#####
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

#####---Overwrite BIDS---#####
overwrite = True

#####---Anonymize Dictionary---#####
anonymize = {
    'daysback': 365.1275*100
}

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
    bids_basename = make_bids_basename(subject=bids_id,
                                       task=task)
    print(f'Making BIDS data for sub-{bids_id} ({sub_id}) on task-{task}')
    
    # Define some sobject directories
    sub_source_dir = source_dir / sub_string
    sub_bids_dir = bids_dir / f'sub-{bids_id}'
    sub_eeg_dir = sub_bids_dir / 'eeg'
    sub_beh_dir = sub_bids_dir / 'beh'
    
    # Make behavioral directory
    sub_beh_dir.mkdir(parents=True, exist_ok=True)
      
    # If EEG file alread writen skip this person
    if  sub_bids_dir.joinpath(f'{bids_basename}_eeg.vhdr').is_file() and not overwrite:
        continue

    ### WRITE EEG TO BIDS FORMAT ###
    # Read in the raw bv file
    bv_file = source_dir / sub_string / f'{sub_string}_task-{task}_run-01_eeg.vhdr'
    raw = read_raw_brainvision(bv_file,
                               misc=['Photosensor'],
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
    write_raw_bids(raw, bids_basename, bids_dir,
                   event_id=event_id, 
                   events_data=events, 
                   anonymize=anonymize,
                   overwrite=overwrite,
                   verbose=False)
    
    ### UPDATE CHANNELS.TSV ###
    # Load *channels.tsv file
    chans_file = sub_eeg_dir / f'sub-{bids_id}_task-{task}_channels.tsv'
    chans_data = pd.read_csv(chans_file, sep='\t')
    
    # Add status_description
    chans_data['status_description'] = 'n/a'
    if sub_bad_chans is not None:
        for chan, reason in sub_bad_chans.items():
            chans_data['status_description'][chans_data['name']==chan] = reason
    
    # Add EEGReference
    chans_data['reference'] = 'FCz'
    for chan in ['VEOG' 'HEOG' 'Photosensor']:
        chans_data['reference'][chans_data['name']==chan] = 'n/a'
    
    # Overwrite file
    chans_data.to_csv(chans_file, sep='\t')
        
    ### PROCESS BEHAVIORAL DATA FILE ###
    # Read in the sof*.tsv behavioral file
    beh_source_file = sub_source_dir / f'{sub_string}_task-{task}_run-01_beh.tsv'
    beh_data = pd.read_csv(beh_source_file,sep='\t')
    
    # Replace subject id and select needed data columns
    beh_data['id'] = bids_id
    
    # Rename and drop data columns
    beh_data.rename(columns=cols_to_rename, inplace=True)
    beh_data.drop(columns=cols_to_drop, inplace=True)
    
    # Replace NaN and -99 with 'n/a' for resp and rt, respectively
    beh_data['resp'].fillna('n/a', inplace=True)
    beh_data['rt'].replace(-99.0, 'n/a', inplace=True)
    beh_data['correct'].fillna(0, inplace=True)
    
    # Convert accuracy to integer
    beh_data['correct'] = beh_data['correct'].astype(int)
    
    # Save behavioral data
    beh_out_file = sub_beh_dir / f'sub-{bids_id}_task-{task}_beh.tsv'
    beh_data.to_csv(beh_out_file, sep='\t')
        
    ### UPDATE *_EVENTS.TSV ###
    # Load *events.tsv
    events_file = sub_eeg_dir / f'sub-{bids_id}_task-{task}_events.tsv'
    events_data = pd.read_csv(events_file,sep='\t')
    events_data[cols_to_add] = beh_data[cols_to_add]
    
    # Overwrite *events.tsv
    events_data.to_csv(events_file, sep='\t')

    ### UPDATE *eeg_json
    # Load JSON
    eeg_json_file = sub_eeg_dir / f'sub-{bids_id}_task-{task}_eeg.json'
    with open(eeg_json_file,'r') as file:
        eeg_json = json.load(file)
    
    # Update keys
    eeg_json['EEGReference'] = 'EEG channels onlined referenced to FCz. EOG channels are bipolar'
    eeg_json['EEGGround'] = 'Fpz'
    #eeg_json['PowerLineFrequency'] = 60
    
    # Save EEG JSON
    with open(eeg_json_file,'w') as file:
        json.dump(eeg_json, file)
        