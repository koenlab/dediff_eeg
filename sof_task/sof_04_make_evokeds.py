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
from collections import OrderedDict
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
    
    # Loop through references
    refs = ['avg', 'mastoids']
    for ref in refs:
        
        # Set ref value
        if ref == 'avg':
            ref_json = 'average'
        else:
            ref_json = ref
    
        ### STEP 1: Load manually cleaned epochs
        # Read in Cleaned Epochs
        epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
        if not epochs_fif_file.is_file():
            continue
        epochs = read_epochs(epochs_fif_file)
                
        ### Step 2: Make evokeds for relevant conditions
        evokeds = []
        evokeds_key = OrderedDict()
        queries = {
            'novel': "repeat==1 and n_responses==0",
            'repeat': "repeat==2 and n_responses==1",
            'scene': "category=='scene' and repeat==1 and n_responses==0",
            'object': "category=='object' and repeat==1 and n_responses==0",
            'face': "category=='face' and repeat==1 and n_responses==0"
            }
        for i, (comment, query) in enumerate(queries.items()):
            evoked = epochs[query].average()
            evoked.comment = comment
            evokeds.append(evoked)
            evokeds_key[comment] = i
        end_i = i
        
        ### Step 3: Make difference waves    
        # Make contrast list
        contrasts = {
            'scene-object': dict(conds=['scene','object'],weights=[1,-1]),
            'face-object': dict(conds=['face','object'], weights=[1,-1]),
            'scene-other': dict(conds=['scene','object','face'], weights=[1,-.5,-.5]),
            'face-other': dict(conds=['face','object','scene'], weights=[1,-.5,-.5])
        }
        
        # Add in difference waves
        for contrast, v in contrasts.items():
            cond_evokeds = [evokeds[evokeds_key[x]] for x in v['conds']]
            evoked = mne.combine_evoked(cond_evokeds, weights=v['weights'])
            evoked.comment = contrast
            evokeds.append(evoked)
            end_i += 1
            evokeds_key[contrast] = end_i
        
        ### Step 4: filter and crop evokeds 
        # Filter evokeds
        evokeds_filt = [x.copy().filter(None,20, picks=['eeg']) for x in evokeds]

        # Crop evokeds
        tmin, tmax = preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']
        evokeds = [x.crop(tmin=tmin,tmax=tmax) for x in evokeds]
        evokeds_filt = [x.crop(tmin=tmin,tmax=tmax) for x in evokeds_filt]
        
         ### Step 5: write evokeds
        # Write evoked file
        evoked_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_lpf-none_ave.fif.gz'
        mne.write_evokeds(evoked_fif_file, evokeds)
        
        # Make JSON
        json_info = {
            'Description': 'Evoked data with no additional filtering',
            'sfreq': evokeds[0].info['sfreq'],
            'reference': ref_json,
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
            'evoked_objects': evokeds_key,
            'n_avg': {x.comment:x.nave for x in evokeds}
        }
        json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_lpf-none_ave.json'
        with open(json_file, 'w') as outfile:
            json.dump(json_info, outfile, indent=4)
        del json_info, json_file
        
        # Write evoked file with filtered data
        evoked_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_lpf-20_ave.fif.gz'
        mne.write_evokeds(evoked_fif_file, evokeds_filt)
        
        # Make JSON
        json_info = {
            'Description': 'Evoked data with Low-Pass Filter',
            'sfreq': evokeds_filt[0].info['sfreq'],
            'reference': ref_json,
            'filter': {
                'eeg': {
                    'highpass': evokeds_filt[0].info['highpass'],
                    'lowpass': evokeds_filt[0].info['lowpass'],
                    'notch': 'n/a'
                },
                'eog': {
                    'highpass': evokeds_filt[0].info['highpass'],
                    'lowpass': evokeds_filt[0].info['lowpass'],
                    'notch': [60.0, 120.0]
                }
            },
            'tmin': evokeds_filt[0].times.min(),
            'tmax': evokeds_filt[0].times.max(),
            'evoked_objects': evokeds_key,
            'n_avg': {x.comment:x.nave for x in evokeds_filt}
        }
        json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_lpf-20_ave.json'
        with open(json_file, 'w') as outfile:
            json.dump(json_info, outfile, indent=4)
        del json_info, json_file
    
    