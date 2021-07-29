"""
Script: sof_04_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

# Import libraries
import json

from mne import read_evokeds, write_evokeds
from sof_config import (analysis_dir, deriv_dir, task,
                        get_sub_list, bad_subs)

# Make output folders
out_sof = analysis_dir / 'erp' / 'sof_erps'
out_sof.mkdir(exist_ok=True, parents=True)
out_rep = analysis_dir / 'erp' / 'repetition_erps'
out_rep.mkdir(exist_ok=True, parents=True)

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    sub_list.remove(bad_sub)

# Make empty lists to store conditions
young = {
    'novel': [],
    'repeat': [],
    'scene': [],
    'object': [],
    'face': []
}
older = {
    'novel': [],
    'repeat': [],
    'scene': [],
    'object': [],
    'face': []
}

# Ask for subject IDs to analyze
for sub in sub_list:

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    if int(sub[-3:]) < 200:
        age = 'young'
    else:
        age = 'older'
    print(sub, age)

    # Load evokeds
    evoked_fif_file = deriv_path / \
        f'{sub}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    evokeds = read_evokeds(evoked_fif_file, verbose=False)

    # Load evokeds json
    evoked_json_file = deriv_path / \
        f'{sub}_task-{task}_ref-avg_lpf-none_ave.json'
    with open(evoked_json_file, 'r') as f:
        evoked_json = json.load(f)
    evokeds_key = evoked_json['evoked_objects']

    # Store in the group lists
    for cond in ['novel', 'repeat', 'scene', 'object', 'face']:

        # Load the evoked
        this_evoked = evokeds[evokeds_key[cond]]

        # Append to correct dictionary
        if age == 'young':
            young[cond].append(this_evoked)
        else:
            older[cond].append(this_evoked)

        # Save individual erp
        out_file = f'{sub}_task-sof_cond-{cond}_ave.fif.gz'
        if cond in ['novel', 'repeat']:
            out_file = out_rep / out_file
        else:
            out_file = out_sof / out_file
        this_evoked.save(out_file)

# Write the evokeds
for cond in ['novel', 'repeat', 'scene', 'object', 'face']:
    young_file = f'young_task-sof_cond-{cond}_ave.fif.gz'
    older_file = f'young_task-sof_cond-{cond}_ave.fif.gz'
    if cond in ['novel', 'repeat']:
        write_evokeds(out_rep / young_file, young[cond])
        write_evokeds(out_rep / older_file, older[cond])
    else:
        write_evokeds(out_sof / young_file, young[cond])
        write_evokeds(out_sof / older_file, older[cond])
