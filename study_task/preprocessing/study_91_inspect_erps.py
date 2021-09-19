"""
Script: study_91_inspect_sof_erps.py
Creator: Joshua D. Koen
Description: Visuall inspect ERPs with interactive graphs
"""

# Import Libraries
import sys
import os

os.chdir(sys.path[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import json

from mne.viz import plot_compare_evokeds
import mne

from study_config import (deriv_dir, task)
from functions import get_sub_list

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Ask to filter ERPS
filter_data = input('Do you want to apply a low-pass '
                    'filter to the data (y/n)? ').lower()
if filter_data == 'y':
    lcutoff = float(input('What is the cutoff?: '))

for sub_string in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'

    # EVOKEDS Section (no lp filter only) This is AVG reference only
    # Load evokeds
    evoked_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    evokeds = mne.read_evokeds(evoked_fif_file, verbose=False)

    # Load evokeds json
    evoked_json_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-mastoids_lpf-none_ave.json'
    with open(evoked_json_file, 'r') as f:
        evoked_json = json.load(f)
    evokeds_key = evoked_json['evoked_objects']

    # Filter in place if requested
    if filter_data == 'y':
        print('Filtering ERPs...')
        for evoked in evokeds:
            evoked.filter(0, lcutoff, verbose=False)

    # Scenes vs. Objects
    conds = ['scene', 'object']
    these_evokeds = [evokeds[evokeds_key[x]] for x in evokeds_key.keys()
                     if x in conds]
    plot_compare_evokeds(these_evokeds, axes='topo', show=True,
                         title='Scenes vs. Objects')

    # Scenes: Hit vs. Misses (65 are hits)
    conds = ['scene-hit65', 'scene-miss65']
    these_evokeds = [evokeds[evokeds_key[x]] for x in evokeds_key.keys()
                     if x in conds]
    plot_compare_evokeds(these_evokeds, axes='topo', show=True,
                         title='Scene: Hit vs. Miss')

    # Objcts: Hit vs. Misses (65 are hits)
    conds = ['object-hit65', 'object-miss65']
    these_evokeds = [evokeds[evokeds_key[x]] for x in evokeds_key.keys()
                     if x in conds]
    plot_compare_evokeds(these_evokeds, axes='topo', show=True,
                         title='Scene: Hit vs. Miss')
