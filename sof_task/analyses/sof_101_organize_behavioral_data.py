"""
Script: sof_102_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

# Import libraries
import sys
import os

os.chdir(os.path.split(__file__)[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import pandas as pd

from sof_config import (analysis_dir, deriv_dir, task,
                        bad_subs)
from functions import get_sub_list

# Make the output directory
out_path = analysis_dir / 'behavior'
out_path.mkdir(parents=True, exist_ok=True)

# Make an empty list
all_dfs = []
all_trials = []

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    if bad_sub in sub_list:
        sub_list.remove(bad_sub)

# Loop over subjects)
for sub in sub_list:

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    print(f'Loading task-{task} data for {sub}')

    # Write summary data file
    summary_file = deriv_path / \
        f'{sub}_task-{task}_desc-summarystats_beh.tsv'
    all_dfs.append(pd.read_csv(summary_file, sep='\t'))

    # Copy trial_data
    trial_file = deriv_path / \
        f'{sub}_task-{task}_desc-triallevel_beh.tsv'
    all_trials.append(pd.read_csv(trial_file, sep='\t'))

# Make the output directory
out_path = analysis_dir / 'behavior'
out_path.mkdir(parents=True, exist_ok=True)

# Merge summary data into into group df
summary_file = out_path / f'task-{task}_group_summary_data.tsv'
pd.concat(all_dfs).to_csv(summary_file, sep='\t', index=False)

# Merge and save trial level data
trial_file = out_path / f'task-{task}_group_alltrials_data.tsv'
trial_df = pd.concat(all_trials)
cols_2_drop = ['frame_rate', 'psychopy_version',
               'index', 'response.rt', 'response.keys']
trial_df.drop(columns=cols_2_drop, inplace=True)
for c in trial_df.columns:
    if 'started' in c or 'stopped' in c:
        trial_df.drop(columns=c, inplace=True)
trial_df['age'] = 'young'
trial_df.loc[trial_df['id'] >= 200, 'age'] = 'older'
id = trial_df['id']
age = trial_df['age']
trial_df.drop(columns=['id', 'age'], inplace=True)
trial_df.insert(0, 'id', id)
trial_df.insert(1, 'age', age)
trial_df.to_csv(trial_file, sep='\t', index=False)
