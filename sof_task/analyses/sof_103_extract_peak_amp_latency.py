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
import matplotlib.pyplot as plt

from mne import (read_evokeds, pick_channels)
from mne.channels import combine_channels

from sof_config import (analysis_dir, deriv_dir, task, bad_subs)
from functions import (get_sub_list, peak_amp_lat)

# Handle directories
out_erps = analysis_dir / 'erps'

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    if bad_sub in sub_list:
        sub_list.remove(bad_sub)

show_fig = False

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

    # Filter them with a 'gentle' low-pass filter of 30Hz
    scene_diff_30lpf = scene_diff.copy().filter(None, 30,
                                                verbose=False)
    face_diff_30lpf = face_diff.copy().filter(None, 30,
                                              verbose=False)

    # Make virtual electrodes
    new_chans = {
        'mean(P7/PO7)': pick_channels(scene_diff.info['ch_names'],
                                      ['PO7', 'P7']),
        'mean(P8/PO8)': pick_channels(scene_diff.info['ch_names'],
                                      ['PO8', 'P8'])
    }
    scene_diff = combine_channels(scene_diff, new_chans)
    scene_diff_30lpf = combine_channels(scene_diff_30lpf, new_chans)
    face_diff = combine_channels(face_diff, new_chans)
    face_diff_30lpf = combine_channels(face_diff_30lpf, new_chans)

    # Extract and plot scene data
    tmin, tmax = .12, .25
    scene_peaks = peak_amp_lat(scene_diff_30lpf, mode='pos', width=0,
                               tmin=tmin, tmax=tmax)
    scene_peaks.insert(0, 'id', sub.replace('sub-', ''))
    scene_peaks.insert(1, 'age_group', age)
    scene_peaks.insert(2, 'condition', 'scene')
    scene_peaks.insert(3, 'hemisphere', ['left', 'right'])

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(8, 8)
    var_list = ['peak_amplitude', 'peak_latency', 'tmin', 'tmax']
    for ai, ch in enumerate(scene_diff_30lpf.ch_names):
        amp, lat, tmin, tmax = scene_peaks.iloc[ai][var_list]
        mv = scene_diff_30lpf.data[ai, :]
        times = scene_diff_30lpf.times
        ax[ai].plot(times, mv, 'k', linewidth=1)
        ax[ai].plot(lat, amp, 'r|')
        ax[ai].axhline(0, linestyle='--', color='black', linewidth=.5)
        ax[ai].axvline(0, linestyle='--', color='black', linewidth=.5)
        ax[ai].set_title(f'{sub}: {ch}')
        ax[ai].set_xlim((-.2, .35))
        ax[ai].axhline(amp, linestyle=':', color='red', linewidth=.5)
        ax[ai].axvspan(tmin, tmax, alpha=.4, color='grey')
    plt.tight_layout()
    if show_fig:
        plt.show()

    # Extract and plot face data
    tmin, tmax = .10, .18
    if sub == 'sub-130':
        tmax = .16
    if sub == 'sub-113':
        tmax = .15
    face_peaks = peak_amp_lat(face_diff_30lpf, mode='neg', width=0,
                              tmin=tmin, tmax=tmax)
    face_peaks.insert(0, 'id', sub.replace('sub-', ''))
    face_peaks.insert(1, 'age_group', age)
    face_peaks.insert(2, 'condition', 'face')
    face_peaks.insert(3, 'hemisphere', ['left', 'right'])
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(8, 8)
    for ai, ch in enumerate(scene_diff_30lpf.ch_names):
        amp, lat, tmin, tmax = face_peaks.iloc[ai][var_list]
        mv = face_diff_30lpf.data[ai, :]
        times = face_diff_30lpf.times
        ax[ai].plot(times, mv, 'k', linewidth=1)
        ax[ai].plot(lat, amp, 'r|')
        ax[ai].axhline(0, linestyle='--', color='black', linewidth=.5)
        ax[ai].axvline(0, linestyle='--', color='black', linewidth=.5)
        ax[ai].set_title(f'{sub}: {ch}')
        ax[ai].set_xlim((-.2, .35))
        ax[ai].axhline(amp, linestyle=':', color='red', linewidth=.5)
        ax[ai].axvspan(tmin, tmax, alpha=.4, color='grey')
    plt.tight_layout()
    if show_fig:
        plt.show()

    # Handle output data frame
    sub_df = pd.concat([scene_peaks, face_peaks])
    if i == 0:
        df_long = sub_df.copy()
    else:
        df_long = df_long.append(sub_df)

# Save the long data frame
out_file = f'task-{task}_roi_peaks_diff_wave_long.tsv'
df_long.to_csv(out_erps / out_file, sep='\t', index=False)

# Load it in to fix datatype issues
df_long = pd.read_csv(out_erps / out_file, sep='\t')
df_wide = df_long.pivot_table(values=['peak_amplitude', 'peak_latency'],
                              index=['id', 'age_group'],
                              columns=['condition', 'hemisphere'])
df_wide.columns = ['_'.join(col).strip() for col in df_wide.columns.values]
df_wide.reset_index(inplace=True)
out_file = f'task-{task}_roi_peaks_diff_wave_wide.tsv'
df_wide.to_csv(out_erps / out_file, sep='\t', index=False)
