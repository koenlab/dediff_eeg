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

from mne import (read_evokeds, pick_channels)
from mne.channels import combine_channels

from sof_config import (analysis_dir, deriv_dir, task, bad_subs)
from functions import (get_sub_list, mean_amplitude)

# Handle directories
out_erps = analysis_dir / 'erps'

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    sub_list.remove(bad_sub)

# Ask for subject IDs to analyze
for i, sub in enumerate(sub_list):

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    if int(sub[-3:]) < 200:
        age = 'young'
    else:
        age = 'older'
    print(sub, age)

    # Load epochs (difference waves)
    evoked_file = f'{sub}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    scene_diff = read_evokeds(out_erps / evoked_file,
                              condition='scene-object', verbose=False)
    face_diff = read_evokeds(out_erps / evoked_file,
                             condition='face-object', verbose=False)

    # Make virtual electrodes
    new_chans = {
        'mean(P7/PO7)': pick_channels(scene_diff.info['ch_names'],
                                      ['PO7', 'P7']),
        'mean(P8/PO8)': pick_channels(scene_diff.info['ch_names'],
                                      ['PO8', 'P8'])
    }
    scene_diff = combine_channels(scene_diff, new_chans)
    face_diff = combine_channels(face_diff, new_chans)

    # Extract data
    scene_peaks = mean_amplitude(scene_diff, tmin=.15, tmax=.224)
    scene_peaks.insert(0, 'id', sub.replace('sub-', ''))
    scene_peaks.insert(1, 'age_group', age)
    scene_peaks.insert(2, 'condition', 'scene')
    scene_peaks.insert(3, 'hemisphere', ['left', 'right'])
    face_peaks = mean_amplitude(face_diff, tmin=.1, tmax=.17)
    face_peaks.insert(0, 'id', sub.replace('sub-', ''))
    face_peaks.insert(1, 'age_group', age)
    face_peaks.insert(2, 'condition', 'face')
    face_peaks.insert(3, 'hemisphere', ['left', 'right'])
    sub_df = pd.concat([scene_peaks, face_peaks])

    # Handle output data frame
    if i == 0:
        df_long = sub_df.copy()
    else:
        df_long = df_long.append(sub_df)

# Save the long data frame
out_file = f'task-{task}_roi_mean_amplitude_diff_wave_long.tsv'
df_long.to_csv(out_erps / out_file, sep='\t', index=False)

# Load it in to fix datatype issues
df_long = pd.read_csv(out_erps / out_file, sep='\t')
df_wide = df_long.pivot_table(values=['mean_amplitude'],
                              index=['id', 'age_group'],
                              columns=['condition', 'hemisphere'])
df_wide.columns = ['_'.join(col).strip() for col in df_wide.columns.values]
df_wide.reset_index(inplace=True)
out_file = f'task-{task}_roi_mean_amplitude_diff_wave_wide.tsv'
df_wide.to_csv(out_erps / out_file, sep='\t', index=False)
