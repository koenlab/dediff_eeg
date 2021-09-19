"""
Script: monster_01_compute_behavioral_data.py
Creator: Joshua D. Koen
Description: This script analyzes the behavioral data for the
MONSTER task.
"""

# Import Libraries
import sys
import os

os.chdir(sys.path[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from monster_config import (bids_dir, deriv_dir, task,
                            get_sub_list)

# Overwrite
overwrite = True

# Get Subject List
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub in sub_list:

    # Get subject information
    sub_id = sub.replace('sub-', '')
    out_path = deriv_dir / sub
    out_path.mkdir(parents=True, exist_ok=True)

    # Make figure
    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches([11, 7])
    fig.suptitle(sub, fontsize=18)

    # Load the behavioral data file
    beh_file = bids_dir / sub / \
        'beh' / f'{sub}_task-{task}_beh.tsv'
    beh_data = pd.read_csv(beh_file, sep='\t')

    # Check if participant has been done and, unless I want to overwrite, skip
    trial_file = out_path / f'{sub}_task-{task}_desc-triallevel_beh.tsv'
    if trial_file.is_file() and not overwrite:
        continue

    # Trial Counts
    # Calculate trial counts
    group_by = ['gabor_loc', 'letter_type', 'abin_label']
    counts = beh_data.groupby(group_by)['correct'].value_counts()
    counts = counts.unstack('abin_label')
    counts.fillna('0', inplace=True)

    # Make the table for trial counts
    colLabels = [f'{x}' for x in counts.columns.to_flat_index()]
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('Trial Counts', fontsize=18, fontweight='bold')
    rowLabels = [
        f'{x[0][0:3]}-{x[1][0:3]}-{x[2]}' for x in counts.index.to_flat_index()
        ]
    ax1.table(cellText=counts.values, colLabels=colLabels, rowLabels=rowLabels,
              loc='upper center')
    ax1.axis('off')

    # Compute proportions (accuracy) and plot it
    proportions = beh_data.query('n_resp==1').groupby(
        group_by[0:2])['correct'].mean().unstack('gabor_loc')

    # Plot accuracy
    ax2 = plt.subplot(2, 2, 3)
    proportions.plot(kind='bar', ax=ax2)
    ax2.legend(loc='upper center', mode='expand', ncol=3,
               bbox_to_anchor=(.7, 1.1, .7, .12),
               frameon=False, title='Gabor Location')
    ax2.tick_params(axis='x', rotation=0)
    ax2.set_title('Accuracy', y=1.2, fontsize=18, fontweight='bold')
    ax2.set_ylabel('p(Correct)', fontsize=16)
    ax2.set_xlabel('Trial Type', fontsize=16)

    # MEDIAN RT MEASURES
    # Compute RT measures
    median_rts = beh_data.query('correct==1 and n_resp==1').groupby(
        group_by[0:2])['rt'].median().to_frame()
    mean_rts = beh_data.query('correct==1 and n_resp==1').groupby(
        group_by[0:2])['rt'].mean().to_frame()
    sd_rts = beh_data.query('correct==1 and n_resp==1').groupby(
        group_by[0:2])['rt'].std().to_frame()

    # Plot median RT
    ax3 = plt.subplot(2, 2, 4)
    median_rts.unstack('gabor_loc').plot(kind='bar', ax=ax3, legend=False)
    ax3.tick_params(axis='x', rotation=0)
    ax3.set_title('Median RT', y=1.20, fontsize=18, fontweight='bold')
    ax3.set_ylabel('RT (sec)', fontsize=16)
    ax3.set_xlabel('Trial Type', fontsize=16)

    # Save plot to figures
    fig_path = deriv_dir / sub / 'figures'
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_file = fig_path / f'{sub}_task-{task}_beh_performance.png'
    fig.savefig(fig_file, dpi=600)

    # Make Data frame to store output
    # Make dictionary and basic values
    out_dict = OrderedDict()
    out_dict = {
        'id': [sub_id],
        'age': ['young' if int(sub_id) < 200 else 'older'],
        'set': [beh_data.iloc[0]['stim_set']]
    }

    # Make combination of names
    gabors = ['top', 'bottom']
    trial_types = ['standard', 'oddball']
    conditions = list()
    for trial_type in trial_types:
        for gabor in gabors:
            conditions.append([trial_type, gabor])

    # Add in no resposnes
    for cond in conditions:
        this_key = f'{cond[0]}_{cond[1]}_p_no_resp'
        this_query = f'letter_type=="{cond[0]}" and gabor_loc=="{cond[1]}"'
        out_dict[this_key] = (beh_data.query(this_query)['n_resp'] == 0).mean()

    # Add in accuracy values
    for cond in conditions:
        this_key = f'{cond[0]}_{cond[1]}_acc'
        out_dict[this_key] = [proportions.loc[cond[0]][cond[1]]]

    for val in ['median', 'mean', 'sd']:
        for cond in conditions:
            this_key = f'{cond[0]}_{cond[1]}_{val}_rt'
            out_dict[this_key] = eval(
                f'[{val}_rts.loc[("{cond[1]}", "{cond[0]}")]["rt"]]')
    # Write summary data file
    summary_file = out_path / \
        f'{sub}_task-{task}_desc-summarystats_beh.tsv'
    pd.DataFrame().from_dict(out_dict).to_csv(
        summary_file, index=False, sep='\t')

    # Write copy of behavior data to derivatives
    trial_file = out_path / f'{sub}_task-{task}_desc-triallevel_beh.tsv'
    beh_data.to_csv(trial_file, sep='\t', index=False)

