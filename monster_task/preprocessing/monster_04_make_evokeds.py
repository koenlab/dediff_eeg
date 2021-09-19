"""
Script: monster_04_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest.
"""

# Import Libraries
import sys
import os

os.chdir(sys.path[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import numpy as np
from collections import OrderedDict
import json

from mne import read_epochs
import mne

from monster_config import (deriv_dir, task, preprocess_options)
from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    print(f'Creating evoked for task-{task} data for {sub}')
    print(f'  Derivatives Folder: {deriv_path}')

    # Loop through references
    refs = ['avg', 'mastoids']
    for ref in refs:

        # Set ref value
        if ref == 'avg':
            ref_json = 'average'
        else:
            ref_json = ref

        # STEP 1: Load manually cleaned epochs and re-reference
        # Read in Cleaned Epochs
        epochs_fif_file = deriv_path / \
            f'{sub}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
        if not epochs_fif_file.is_file():
            continue
        epochs = read_epochs(epochs_fif_file)

        # Step 2: Make evokeds for relevant conditions
        # Initialize variables
        evokeds = []
        evokeds_key = OrderedDict()

        # Make Queries
        queries = {
            'standard': "letter_type=='standard' and correct==1",
            'oddball': "letter_type=='oddball' and correct==1",
            'top': "gabor_loc=='top' and letter_type=='standard' and" +
                   "correct==1",
            'bottom': "gabor_loc=='bottom' and letter_type=='standard' and" +
                      "correct==1"
            }
        for a in np.arange(1, 9):
            queries[f'abin{a}'] = f"abin_label=='bin{a}' and" + \
                "letter_type=='standard' and correct==1"

        # Extract querie evokeds
        for i, (comment, query) in enumerate(queries.items()):
            evoked = epochs[query].average()
            evoked.comment = comment
            evokeds.append(evoked)
            evokeds_key[comment] = i
        end_i = i

        # Step 3: Make difference waves
        # Make contrast list
        contrasts = {
            'oddball-standard':
                dict(conds=['oddball', 'standard'], weights=[1, -1]),
            'top-bottom': dict(conds=['top', 'bottom'], weights=[1, -1])
        }

        # Add in difference waves
        for contrast, v in contrasts.items():
            cond_evokeds = [evokeds[evokeds_key[x]] for x in v['conds']]
            evoked = mne.combine_evoked(cond_evokeds, weights=v['weights'])
            evoked.comment = contrast
            evokeds.append(evoked)
            end_i += 1
            evokeds_key[contrast] = end_i

        # Crop evokeds
        tmin = preprocess_options['evoked_tmin'],
        tmax = preprocess_options['evoked_tmax']
        evokeds = [x.crop(tmin=tmin, tmax=tmax) for x in evokeds]

        # Step 4: write evokeds
        # Write evoked file
        evoked_fif_file = deriv_path / \
            f'{sub}_task-{task}_ref-{ref}_lpf-none_ave.fif.gz'
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
            'n_avg': {x.comment: x.nave for x in evokeds}
        }
        json_file = deriv_path / \
            f'{sub}_task-{task}_ref-{ref}_lpf-none_ave.json'
        with open(json_file, 'w') as outfile:
            json.dump(json_info, outfile, indent=4)
        del json_info, json_file
