"""
Script: 00_sof_data_import.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the SOF (scene, object, face) task. 
"""

#####---Import Libraries---#####
import os
import sys

import pandas as pd
import json

from mne.io import read_raw_brainvision
from mne import events_from_annotations
from mne_bids import write_raw_bids, make_bids_basename

from sof_config import bids_dir, source_dir, event_dict, task, bad_chans

#####---Change Directory to Script Directory---#####
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

#####---Overwrite BIDS---#####
overwrite = True

#####---Anonymize Dictionary---#####
anonymize = {
    'daysback': 365.1275*100
}

#####---Event Dictionaries for renaming and output---#####
# Rename dictionary
rename_dict = {
    'New Segment/': 'boundary',
    'Marker/M 11': 'scene/novel',
    'Marker/M 12': 'scene/1back',
    'Marker/M 21': 'object/novel',
    'Marker/M 22': 'object/1back',
    'Marker/M 31': 'face/novel',
    'Marker/M 32': 'face/1back'
}

# Add boundary to event_dict
event_dict['boundary'] = -99

#####---Data columns to keep and add---#####
# List of data columns to drop behavioral data file
cols_to_drop = ['date','expName','eeg_record','ran','order','image_file']

# Dictionary of columns to rename in behavioral data file
cols_to_rename = {
    'frameRate': 'frame_rate',
    'psychopyVersion': 'psychopy_version',
    'TrialNumber': 'trial_number'
}

# List of columns to add to *events.tsv from behavioral data
cols_to_add = ['trial_number','category','subcategory','repeat','resp',
               'rt','correct','n_responses']

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
            chans_data.loc[chans_data['name'] == chan, ['status_description']] = reason
    
    # Add EEGReference
    chans_data['reference'] = 'FCz'
    for chan in ['VEOG', 'HEOG', 'Photosensor']:
        chans_data.loc[chans_data['name']==chan, ['reference']] = 'n/a'
    
    # Overwrite file
    chans_data.to_csv(chans_file, sep='\t',index=False)
        
    ### PROCESS BEHAVIORAL DATA FILE ###
    # Read in the sof*.tsv behavioral file
    beh_source_file = sub_source_dir / f'{sub_string}_task-{task}_run-01_beh.tsv'
    beh_data = pd.read_csv(beh_source_file,sep='\t')
    
    # Replace NaN and -99 with 'n/a' for resp and rt, respectively
    beh_data['resp'] = beh_data['resp'].fillna('n/a')
    beh_data['rt'] = beh_data['rt'].replace(-99.0,'n/a')
    
    # Convert accuracy to integer
    beh_data['correct'] = (beh_data['repeat'].isin([1]) & beh_data['resp'].isin(['n/a'])) | (beh_data['repeat'].isin([2]) & beh_data['resp'].isin(['j']))
    beh_data['correct'] = beh_data['correct'].astype(int)
    
    # Code # of resposnes
    beh_data['n_responses'] = beh_data['response.keys'].replace('None','').str.len()
    
    # Replace subject id and select needed data columns
    beh_data['id'] = bids_id
    
    # Rename and drop data columns
    beh_data.rename(columns=cols_to_rename, inplace=True)
    beh_data.drop(columns=cols_to_drop, inplace=True)
    
    # Save behavioral data
    beh_save_file = sub_beh_dir / f'sub-{bids_id}_task-{task}_beh.tsv'
    beh_data.to_csv(beh_save_file, sep='\t', index=False)
    

    ### UPDATE *_EVENTS.TSV ###
    # Load *events.tsv
    events_file = sub_eeg_dir / f'sub-{bids_id}_task-{task}_events.tsv'
    events_data = pd.read_csv(events_file,sep='\t')

    # Add new columnas as "n/a" values
    events_data[cols_to_add] = 'n/a'

    # Update with values
    counter = 0 # Keep track of current row in beh_data
    for index, row in events_data.iterrows():
        if row['trial_type'] != 'boundary':
            this_trial = beh_data.iloc[counter]
            for col in cols_to_add:
                events_data.at[index, col] = this_trial[col]
            counter += 1

    # Overwrite *events.tsv
    events_data.to_csv(events_file, sep='\t', index=False)

    ### UPDATE *eeg_json
    # Load JSON
    eeg_json_file = sub_eeg_dir / f'sub-{bids_id}_task-{task}_eeg.json'
    with open(eeg_json_file,'r') as file:
        eeg_json = json.load(file)
    
    # Update keys
    eeg_json['EEGReference'] = 'FCz'
    eeg_json['EEGGround'] = 'Fpz'
    #eeg_json['PowerLineFrequency'] = 60
    
    # Save EEG JSON
    with open(eeg_json_file,'w') as file:
        json.dump(eeg_json, file)
   
