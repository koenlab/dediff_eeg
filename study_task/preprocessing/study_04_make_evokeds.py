"""
Script: study_04_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest.
"""

# Import Libraries
import sys
import os

os.chdir(os.path.split(__file__)[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

from collections import OrderedDict
import json

from mne import read_epochs
import mne

from study_config import (deriv_dir, task, preprocess_options)
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

        # STEP 1: Load manually cleaned epochs
        # Read in Cleaned Epochs
        epochs_fif_file = deriv_path / \
            f'{sub}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
        if not epochs_fif_file.is_file():
            continue
        epochs = read_epochs(epochs_fif_file)

        # Step 2: Make evokeds for relevant conditions
        evokeds = []
        evokeds_key = OrderedDict()
        queries = {
            'all': "study_n_responses==1 and test_resp in [1,2,3,4,5,6]",
            'scene': ("category=='scene' and study_n_responses==1 and" +
                      "test_resp in [1,2,3,4,5,6]"),
            'object': ("category=='object' and study_n_responses==1 and" +
                       "test_resp in [1,2,3,4,5,6]"),
            'hit65': "study_n_responses==1 and test_resp in [5,6]",
            'miss65': "study_n_responses==1 and test_resp in [1,2,3,4]",
            'scene-hit65': ("category=='scene' and study_n_responses==1 and" +
                            "test_resp in [5,6]"),
            'scene-miss65': ("category=='scene' and study_n_responses==1 and" +
                             "test_resp in [1,2,3,4]"),
            'object-hit65': ("category=='object' and study_n_responses==1 " +
                             "and test_resp in [5,6]"),
            'object-miss65': ("category=='object' and study_n_responses==1 " +
                              "and test_resp in [1,2,3,4]"),
            'hit6': "study_n_responses==1 and test_resp in [6]",
            'miss6': "study_n_responses==1 and test_resp in [1,2,3,4,5]",
            'scene-hit6': ("category=='scene' and study_n_responses==1 " +
                           "and test_resp in [6]"),
            'scene-miss6': ("category=='scene' and study_n_responses==1 " +
                            "and test_resp in [1,2,3,4,5]"),
            'object-hit6': ("category=='object' and study_n_responses==1 " +
                            "and test_resp in [6]"),
            'object-miss6': ("category=='object' and study_n_responses==1 " +
                             "and test_resp in [1,2,3,4,5]")
        }
        for i, (comment, query) in enumerate(queries.items()):
            evoked = epochs[query].average()
            evoked.comment = comment
            evokeds.append(evoked)
            evokeds_key[comment] = i
        end_i = i

        # Step 3: Make difference waves
        # Make contrast list
        contrasts = {
            'scene-object': dict(conds=['scene', 'object'], weights=[1, -1]),
            'hit-miss65': dict(conds=['hit65', 'miss65'], weights=[1, -1]),
            'scene-hit-miss65':
                dict(conds=['scene-hit65', 'scene-miss65'], weights=[1, -1]),
            'object-hit-miss65':
                dict(conds=['object-hit65', 'object-miss65'], weights=[1, -1]),
            'hit-miss6': dict(conds=['hit6', 'miss6'], weights=[1, -1]),
            'scene-hit-miss6':
                dict(conds=['scene-hit6', 'scene-miss6'], weights=[1, -1]),
            'object-hit-miss6':
                dict(conds=['object-hit6', 'object-miss6'], weights=[1, -1])
        }

        # Add in difference waves
        for contrast, v in contrasts.items():
            cond_evokeds = [evokeds[evokeds_key[x]] for x in v['conds']]
            evoked = mne.combine_evoked(cond_evokeds, weights=v['weights'])
            evoked.comment = contrast
            evokeds.append(evoked)
            end_i += 1
            evokeds_key[contrast] = end_i

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
