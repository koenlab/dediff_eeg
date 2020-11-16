"""
Script: sof_22_redo_ICA_flagging.py
Creator: Joshua D. Koen
Description: Redo flagging of IC components
"""

#####---Import Libraries---#####
import pandas as pd
import json 

from mne import read_epochs
from mne.preprocessing import read_ica
import mne

from sof_config import (bids_dir, deriv_dir, task, get_sub_list)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')

    # # Load ICA Epochs
    # epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_epo.fif.gz'
    # epochs = read_epochs(epochs_fif_file)

    # # Load ICA
    # ica_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_ica.fif.gz'
    # ica = read_ica(ica_file)

    # # Plot ICA
    # ica.plot_components(inst=epochs, reject=None,
    #                     psd_args=dict(fmax=70))
    # ica.save(ica_file)

    # # Load json file for updating
    # json_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-ica_ica.json'
    # with open(json_file, 'r') as f:
    #     json_info = json.load(f)

    # # Only update description, proportion of flagged components, and glaffed components
    # json_info['Description'] = 'ICA components'
    # json_info['flagged_components'] = [int(x) for x in ica.exclude]
    # json_info['proportion_components_flagged'] = len(ica.exclude)/len(ica.info['ch_names'])
    
    # # Resave json file
    # with open(json_file, 'w') as f:
    #     json.dump(json_info, f, indent=4)
    
    files = ['ica', 'removedICs', 'cleaned']
    for x in files:
        if x == 'ica':
            tmin, tmax = -1.0, 1.0
            ref = 'FCz'
        else:
            tmin, tmax = -1.7, 1.7
            ref = 'avg'
        json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_desc-{x}_epo.json'
        if not json_file.is_file():
            continue
        with open(json_file, 'r') as f:
            json_info = json.load(f)
        
        json_info['tmin'] = tmin
        json_info['tmax'] = tmax
        if x == 'cleaned':
            json_info['metadata'] = (deriv_path / f'{sub_string}_task-_desc-cleaned_metadata.tsv').name
        with open(json_file, 'w') as f:
            json.dump(json_info, f, indent=4)
