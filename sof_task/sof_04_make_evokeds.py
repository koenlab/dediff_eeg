"""
Script: sof_04_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


from mne import read_epochs
import mne

from sof_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Creating evoked for task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: Load manually cleaned epochs
    # Read in Cleaned Epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    if not epochs_fif_file.is_file():
        continue
    epochs = read_epochs(epochs_fif_file)

    # Make evokeds
    evokeds = []
    for cond in ['scene','object','face']:
        query = f"category == '{cond}' and repeat==1 and n_responses==0"
        evoked = epochs[query].average()
        evoked.comment = cond
        evokeds.append(evoked)
        
    # Make a copy of evokeds for ease of combining
    cond_evokeds = evokeds.copy()
    
    # Make contrast list
    contrasts = {
        'all': [1/3]*3,
        'scene-object': [1,-1,0],
        'face-object': [0,-1,1],
        'scene-other': [2,-1,-1],
        'face-other': [-1,-1,2]
    }
    
    # Add in difference waves
    for contrast, weights in contrasts.items():
        evoked = mne.combine_evoked(cond_evokeds, weights=weights)
        evoked.comment = contrast
        evokeds.append(evoked)
    
    # Make original evokeds
    evokeds_copy = [x.copy() for x in evokeds]
    
    # Crop evokeds
    evokeds = [x.crop(preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']) for x in evokeds]
    
    # Write evoked file
    evoked_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    mne.write_evokeds(evoked_fif_file, evokeds)
    
    # Make JSON
    json_info = {
        'Description': 'Evoked data with no additional filtering',
        'sfreq': evokeds[0].info['sfreq'],
        'reference': 'average',
        'filter': {
            'eeg': {
                'highpass': evokeds[0].info['highpass'],
                'lowpass': evokeds[0].info['lowpass'],
                'notch': 'n/a'
            },
            'eog': {
                'highpass': evokeds[0].info['highpass'],
                'lowpass': 40.0,
                'notch': [60.0, 120.0]
            }
        },
        'tmin': evokeds[0].times.min(),
        'tmax': evokeds[0].times.max(),
        'evoked_objects': {x.comment:i for i, x in enumerate(evokeds)},
        'n_avg': {x.comment:x.nave for x in evokeds}
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file
    
    # Apply 20Hz LPF then crop
    evokeds = evokeds_copy.copy()
    evokeds = [x.filter(None,20, picks=['eeg']) for x in evokeds]
    evokeds = [x.crop(preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']) for x in evokeds]
   
    # Write evoked file
    evoked_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_lpf-20_ave.fif.gz'
    mne.write_evokeds(evoked_fif_file, evokeds)
    
    # Make JSON
    json_info = {
        'Description': 'Evoked data with Low-Pass Filter',
        'sfreq': evokeds[0].info['sfreq'],
        'reference': 'average',
        'filter': {
            'eeg': {
                'highpass': evoked.info['highpass'],
                'lowpass': 'n/a',
                'notch': 'n/a'
            },
            'eog': {
                'highpass': evoked.info['highpass'],
                'lowpass': evoked.info['lowpass'],
                'notch': [60.0, 120.0]
            }
        },
        'tmin': evokeds[0].times.min(),
        'tmax': evokeds[0].times.max(),
        'evoked_objects': {x.comment:i for i, x in enumerate(evokeds)},
        'n_avg': {x.comment:x.nave for x in evokeds}
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_lpf-20_ave.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
    del json_info, json_file
    
    