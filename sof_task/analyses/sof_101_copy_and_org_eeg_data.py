"""
Script: sof_04_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

# Import libraries
import sys
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

from shutil import copyfile

from mne import (read_evokeds, write_evokeds, read_epochs)
from sof_config import (analysis_dir, deriv_dir, task,
                        bad_subs)
from functions import get_sub_list

# Make output folders
out_erps = analysis_dir / 'erps'
out_erps.mkdir(exist_ok=True, parents=True)
out_epochs = analysis_dir / 'epochs'
out_epochs.mkdir(exist_ok=True, parents=True)
out_fooof = analysis_dir / 'fooof'
out_fooof.mkdir(exist_ok=True, parents=True)

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    sub_list.remove(bad_sub)

# Make empty dictionaries to store conditions
young_erps = {}
older_erps = {}
all_erps = {}

# Ask for subject IDs to analyze
for sub in sub_list:

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    if int(sub[-3:]) < 200:
        age = 'young'
    else:
        age = 'older'
    print(sub, age)

    # Move and copy epochs (reference average)
    epochs_file = f'{sub}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(deriv_path / epochs_file, verbose=False)
    epochs.save(out_epochs / epochs_file, overwrite=True)

    # Move and copy epochs (reference average, no low-pass filter)
    evoked_file = f'{sub}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    evokeds = read_evokeds(deriv_path / evoked_file,
                           baseline=(None, 0), verbose=False)
    write_evokeds(out_erps / evoked_file, evokeds)

    # Add individual evokeds into dictionary
    conds = [e.comment for e in evokeds]
    for i, c in enumerate(conds):
        if c not in young_erps:
            young_erps[c] = []
            older_erps[c] = []
            all_erps[c] = []

        all_erps[c].append(evokeds[i])
        if age == 'young':
            young_erps[c].append(evokeds[i])
        else:
            older_erps[c].append(evokeds[i])

    # Move and copy fooof
    fooof_file = f'{sub}_task-sof_ref-avg_desc-firstcorrect_fooof.json'
    copyfile(deriv_path / fooof_file, out_fooof / fooof_file)

for cond in young_erps.keys():
    young_file = f'young_task-sof_cond-{cond}_ave.fif.gz'
    write_evokeds(out_erps / young_file, young_erps[cond])
    older_file = f'older_task-sof_cond-{cond}_ave.fif.gz'
    write_evokeds(out_erps / older_file, older_erps[cond])
    all_file = f'all_task-sof_cond-{cond}_ave.fif.gz'
    write_evokeds(out_erps / all_file, all_erps[cond])
