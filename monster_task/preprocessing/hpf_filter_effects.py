"""
Script: monster_03_preprocess_eeg.py
Creator: Joshua D. Koen
Description: This script removes artifactual ICs from
the EEG data and then detects remaining artifacts.
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

from mne.io import read_raw_fif
from mne.preprocessing import read_ica
import mne

from monster_config import (deriv_dir, task,
                            preprocess_options,
                            bv_montage)
from monster_config import event_dict as event_id
from functions import get_sub_list

# Define filters to use
hpfs = [None, .01, .1, .3, .5]
method = 'fir'

dataframes = []

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub in sub_list:

    # SUBJECT INFORMATION DEFINITION
    print(sub)
    age_group = 'young' if int(sub[-3:]) < 200 else 'older'

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub

    # STEP 1: LOAD STUFF
    # Load Raw EEG data from derivatives folder
    resamp_fif_file = deriv_path / \
        f'{sub}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw_orig = read_raw_fif(resamp_fif_file, preload=True)

    # Load events
    events_file = deriv_path / f'{sub}_task-{task}_desc-resamp_eve.txt'
    events = mne.read_events(events_file)

    # Get Metadata
    metadata_file = deriv_path / f'{sub}_task-{task}_metadata.tsv'
    metadata = pd.read_csv(metadata_file, sep='\t')

    # Load ICA
    ica_file = deriv_path / f'{sub}_task-{task}_ref-FCz_desc-ica_ica.fif.gz'
    ica = read_ica(ica_file)

    # Load bad epochs
    json_file = deriv_path / f'{sub}_task-{task}_ref-avg_desc-cleaned_epo.json'
    with open(json_file, 'r') as f:
        json_info = json.load(f)
    bad_epochs = json_info['bad_epochs']
    bad_channels = json_info['bad_channels']

    for hpf in hpfs:
        print(f'\t{hpf}')
        # High Pass Filter raw and make epochs
        raw = raw_orig.copy()
        raw.filter(hpf, None, skip_by_annotation=['boundary'])
        raw.filter(None, 40, picks=['eog'])
        raw.notch_filter([60, 120], picks=['eog'])

        # Make Epochs from raw
        epochs = mne.Epochs(raw, events, event_id=event_id,
                            tmin=preprocess_options['tmin'],
                            tmax=preprocess_options['tmax'],
                            metadata=metadata, baseline=(None, None),
                            reject=None, preload=True)

        # Apply ICA (in place)
        ica.apply(epochs)

        # Interpolate channels if needed, and set new montage to bv_montage
        epochs.info['bads'] = bad_channels
        if len(epochs.info['bads']) > 0:
            epochs.interpolate_bads()
        else:
            print('No channels were interpolated')
        mne.add_reference_channels(epochs, 'FCz', copy=False)
        epochs.set_channel_types(mapping=dict(FCz='eeg'))
        epochs.set_montage(bv_montage, on_missing='ignore')

        # Add FCz and reference to average and set montage to standard_1005
        epochs.set_eeg_reference(ref_channels=['TP9', 'TP10'])

        # Baseline to -200 to 0
        epochs.apply_baseline((-.2, 0))

        # Drop epochs
        epochs.drop(bad_epochs)

        # Make evokeds
        channels = ['Fz', 'Cz', 'Pz']
        standard = (epochs["letter_type=='standard' and correct==1"]
                    .average()
                    .filter(None, 40))
        standard.comment = 'standard'
        standard.pick_channels(channels).reorder_channels(channels)
        oddball = (epochs["letter_type=='oddball' and correct==1"]
                   .average()
                   .filter(None, 40))
        oddball.comment = 'oddball'
        oddball.pick_channels(channels).reorder_channels(channels)

        # Extract mean amplitudes
        n2_standard = (standard
                       .copy()
                       .crop(tmin=.2, tmax=.3)
                       .get_data()).mean(axis=-1) * 1e6
        n2_oddball = (oddball
                      .copy()
                      .crop(tmin=.2, tmax=.3)
                      .get_data()).mean(axis=-1) * 1e6
        p3_standard = (standard
                       .copy()
                       .crop(tmin=.3, tmax=.6)
                       .get_data()).mean(axis=-1) * 1e6
        p3_oddball = (oddball
                      .copy()
                      .crop(tmin=.3, tmax=.6)
                      .get_data()).mean(axis=-1) * 1e6

        # Make a data dictionary
        if hpf is None:
            filter = 'dc'
        else:
            filter = hpf
        data_dict = {
            'id': [sub] * 6,
            'age': [age_group] * 6,
            'filter': [filter] * 6,
            'channel': channels * 2,
            'condition': ['standard'] * 3 + ['oddball'] * 3,
            'n2_amp': np.concatenate((n2_standard, n2_oddball)),
            'p3_amp': np.concatenate((p3_standard, p3_oddball))
        }
        this_df = pd.DataFrame(data_dict)

        # add to data frames
        dataframes.append(this_df)

all_df = pd.concat(dataframes)
all_df.to_csv('hpf_setting_test.csv', index=False)