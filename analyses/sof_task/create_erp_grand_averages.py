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


from mne import read_evokeds
import mne

from sof_config import (analysis_dir, deriv_dir, task, get_sub_list,
                        bad_subs)

# Make an empty list
all_dfs = []

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    sub_list.remove(bad_sub) 
    
# Make empty lists to store conditions
novel = []
repeat = []
scene = []
object = []
face = []
    
# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    
    # Load evokeds
    evoked_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    evokeds = read_evokeds(evoked_fif_file, verbose=False)
    
    # Load evokeds json
    evoked_json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.json'
    with open(evoked_json_file, 'r') as f:
        evoked_json = json.load(f)
    evokeds_key = evoked_json['evoked_objects']
    
    # Store in the group lists
    for cond in ['novel', 'repeat', 'scene', 'object', 'face']:
        this_evoked = evokeds[evokeds_key[cond]]
    