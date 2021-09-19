"""
Script: study_99_make_report.py
Creator: Joshua D. Koen
Description: Makes an html report of the data
"""

# Import Libraries
import sys
import os

os.chdir(sys.path[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import numpy as np
import json
import matplotlib.pyplot as plt
from math import ceil

import mne
from mne import Report
from mne.preprocessing import read_ica
from mne.viz import plot_compare_evokeds

from study_config import (deriv_dir, report_dir, task,
                          bv_montage)
from study_config import event_dict as event_id
from functions import get_sub_list

# Get rid of warning
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Generating report for task-{task} data from {sub_string}')

    # Initialize the report
    # Make the report object
    report = Report(subject=sub_string,
                    title=f'{sub_string}: task-{task} report',
                    image_format='png', verbose=True, projs=False,
                    subjects_dir=None)

    # Behavioral Section
    # Plot behavioral data
    behav_fig_file = fig_path / f'{sub_string}_task-{task}_beh_performance.png'
    report.add_images_to_section(behav_fig_file,
                                 captions='Behavior: Performance Summary',
                                 section='Behavior')

    # EEG Section
    # Load the Raw data
    raw_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw = mne.io.read_raw_fif(raw_fif_file, preload=True)

    # Load the Epochs
    epoch_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-mastoids_desc-cleaned_epo.fif.gz'
    epochs = mne.read_epochs(epoch_fif_file, preload=True)

    # Load Epochs json
    json_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-mastoids_desc-cleaned_epo.json'
    with open(json_file, 'r') as f:
        json_info = json.load(f)

    # Plot Sensors
    fig = bv_montage.plot(show=False)
    report.add_figs_to_section(fig, captions='EEG: Montage Layout',
                               section='EEG')

    # Get events
    events_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    events = mne.read_events(events_file)
    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                              event_id=event_id, show=False)
    report.add_figs_to_section(fig, captions='EEG: Events', section='EEG',
                               scale=1.5)

    # Add PSD plot from raw and epochsepochs
    fig = plt.figure()
    fig.set_size_inches(12, 7)
    ax1 = plt.subplot(221)
    raw.plot_psd(picks=['eeg'], xscale='linear', ax=ax1, show=False)
    ax1.set_title('Raw: Linear Scale')
    ax2 = plt.subplot(222)
    raw.plot_psd(picks=['eeg'], xscale='log', ax=ax2, show=False)
    ax2.set_title('Raw: Log Scale')
    ax3 = plt.subplot(223)
    epochs.plot_psd(picks=['eeg'], xscale='linear', ax=ax3, show=False)
    ax3.set_title('Epochs (Cleaned): Linear Scale')
    ax4 = plt.subplot(224)
    epochs.plot_psd(picks=['eeg'], xscale='log', ax=ax4, show=False)
    ax4.set_title('Epochs (Cleaned): Log Scale')
    ax4.set_xlabel('log(Frequency)')
    report.add_figs_to_section(fig, captions='EEG: Channel PSD', section='EEG')
    plt.close()

    # Plot epochs drop log
    fig = epochs.plot_drop_log(subject=sub_string, show=False)
    report.add_figs_to_section(fig, captions='EEG: Drop Log', section='EEG')

    # Load IC Clean epochs
    ic_rm_epoch_fif_file = epoch_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-avg_desc-removedICs_epo.fif.gz'
    ic_rm_epochs = mne.read_epochs(ic_rm_epoch_fif_file)

    # Make color index
    n_channels = len(ic_rm_epochs.info.ch_names)
    epoch_colors = list()
    for i in np.arange(ic_rm_epochs.events.shape[0]):
        epoch_colors.append([None]*(n_channels-1) + ['k'])
        if i in json_info['bad_epochs']:
            epoch_colors[i] = ['r'] * n_channels

    # Plot the epoched data with a scroll bar
    print('Plotting epochs...this can take a while')
    plt.close()
    n_epochs = len(ic_rm_epochs)
    epoch_nums = np.arange(0, n_epochs)
    epochs_per_plot = 8
    n_plots = ceil(n_epochs / epochs_per_plot)
    figs = []
    captions = []
    count = 0
    for s in np.arange(0, n_epochs, epochs_per_plot):

        # Get the epochs to plot
        these_epochs = []
        for e in range(count, count+epochs_per_plot):
            if e in epoch_nums:
                these_epochs.append(e)
        count += epochs_per_plot

        # Update caption
        captions.append(f'Epochs {these_epochs[0]}-{these_epochs[-1]}')

        # Plot
        tmp_epochs = ic_rm_epochs.copy()[these_epochs]
        tmp_colors = [epoch_colors[x] for x in these_epochs]
        figs.append(
            tmp_epochs.plot(n_channels=66, n_epochs=epochs_per_plot,
                            scalings=dict(eeg=100e-6, eog=200e-6),
                            show=False, epoch_colors=tmp_colors,
                            picks=['eeg', 'eog'])
        )
        print(f'Figs completed: {len(figs)}')

    # Add slidebar figs
    report.add_slider_to_section(figs, captions=captions, section='EEG',
                                 title='EEG: Cleaned Epochs', scale=2)
    plt.close()

    # ICA Section
    # Load ICA epochs
    ica_epoch_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-FCz_desc-ica_epo.fif.gz'
    ica_epochs = mne.read_epochs(ica_epoch_fif_file, preload=True)

    # Load ICA
    ica_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-FCz_desc-ica_ica.fif.gz'
    ica = read_ica(ica_file)

    # Plot ICA Component maps
    figs = ica.plot_components(reject=None, psd_args=dict(fmax=70), show=False)
    captions = ['*Greyed out components were excluded'] * len(figs)
    report.add_slider_to_section(figs, section='ICA', captions=captions,
                                 title='ICA: Component Maps', scale=1)
    plt.close()

    # Plot excluded ICs
    figs = ica.plot_properties(ica_epochs, picks=ica.exclude, show=False,
                               psd_args=dict(fmax=70))
    captions = [f'Component {x:03}' for x in ica.exclude]
    report.add_slider_to_section(figs, captions=captions, section='ICA',
                                 title='ICA: Excluded Component Properties',
                                 scale=1)
    plt.close()

    # Plot included ICs
    good_ics = np.setdiff1d(np.arange(ica.info['nchan']), ica.exclude)
    figs = ica.plot_properties(ica_epochs, picks=good_ics, show=False,
                               psd_args=dict(fmax=70))
    captions = [f'Component {x:03}' for x in good_ics]
    report.add_slider_to_section(figs, captions=captions, section='ICA',
                                 title='ICA: Kept Component Properties',
                                 scale=1)
    plt.close()

    # EVOKEDS Section (no lp filter only) This is AVG reference only
    # Load evokeds
    evoked_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.fif.gz'
    evokeds = mne.read_evokeds(evoked_fif_file, verbose=False)

    # Load evokeds json
    evoked_json_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-avg_lpf-none_ave.json'
    with open(evoked_json_file, 'r') as f:
        evoked_json = json.load(f)
    evokeds_key = evoked_json['evoked_objects']

    # Plots for each condition
    for cond, i in evokeds_key.items():

        # Skip if not needed
        if cond not in ['scene', 'object', 'scene-object']:
            continue

        # Initialize and gather info
        figs = []
        evoked = evokeds[i].copy().crop(
            tmin=-.2, tmax=.6).apply_baseline((-.2, 0))
        times = [-0.1, .1, .13, .18, .3, .5]

        # Make Figures
        figs.append(evoked.plot_image(show=False, picks=['eeg'],
                    titles=dict(eeg=cond)))
        figs.append(evoked.plot_joint(picks=['eeg'], times=times,
                                      topomap_args=dict(average=.05),
                                      title=cond,
                                      show=False))
        figs.append(evoked.plot_topomap(
            times=np.arange(-.1, evoked.tmax, .025),
            show=False, average=.05, nrows=4,
            title=f'{cond} topomaps (.5s avg)'))

        # Add slider section
        report.add_slider_to_section(
            figs, section='ERP', title=f'ERP (Avg. ref.): {cond}', scale=1)
        plt.close()

    # Add Select topos to ERP
    # Scene vs. Objects
    conds = ['scene', 'object']
    these_evokeds = [evokeds[evokeds_key[x]].copy().crop(
        tmin=-.2, tmax=.6).apply_baseline(
            (-.2, 0)) for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(these_evokeds, title='Scene and Object Trials',
                                axes='topo', show=False, show_sensors=True)
    fig2 = plot_compare_evokeds(
        evokeds[evokeds_key['scene-object']].copy().crop(
            tmin=-.2, tmax=.6).apply_baseline((-.2, 0)),
        title='Scene and Object Trials - Left Hemisphere',
        show=False, show_sensors=True,
        picks=['TP7', 'P7', 'PO7'], combine='mean')
    fig3 = plot_compare_evokeds(
        evokeds[evokeds_key['scene-object']].copy().crop(
            tmin=-.2, tmax=.6).apply_baseline((-.2, 0)),
        title='Scene and Object Trials - Right Hemisphere',
        show=False, show_sensors=True,
        picks=['TP8', 'P8', 'PO8'], combine='mean')
    captions = [
        'Scene and Object Trials',
        'Scene and Object Trials - Left Hemisphere',
        'Scene and Object Trials - Right Hemisphere'
        ]
    figs = [fig1[0], fig2[0], fig3[0]]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Avg. ref.): Topo - Scene vs. Object Trials', scale=1.5)
    plt.close()

    # EVOKEDS Section (no lp filter only) This is Mastoid reference only
    # Load evokeds
    evoked_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-mastoids_lpf-none_ave.fif.gz'
    evokeds = mne.read_evokeds(evoked_fif_file, verbose=False)

    # Load evokeds json
    evoked_json_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-mastoids_lpf-none_ave.json'
    with open(evoked_json_file, 'r') as f:
        evoked_json = json.load(f)
    evokeds_key = evoked_json['evoked_objects']

    # Plots for each condition
    for cond, i in evokeds_key.items():

        # Skip if not needed
        if cond in ['scene', 'object', 'scene-object']:
            continue

        # Initialize and gather info
        figs = []
        evoked = evokeds[i]
        times = 'peaks'

        # Make Figures
        figs.append(evoked.plot_image(
            show=False, picks=['eeg'], titles=dict(eeg=cond)))
        figs.append(evoked.plot_joint(
            picks=['eeg'], times=times, topomap_args=dict(average=.05),
            title=cond, show=False))
        figs.append(evoked.plot_topomap(
            times=np.arange(-.15, evoked.tmax, .1), show=False,
            average=.05, nrows=4, title=f'{cond} topomaps (.5s avg)'))

        # Add slider section
        report.add_slider_to_section(
            figs, section='ERP', title=f'ERP (Mastoid Ref.): {cond}', scale=1)
        plt.close()

    # Hits vs. Misses (65)
    conds = ['hit65', 'miss65']
    these_evokeds = [
        evokeds[evokeds_key[x]] for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(
        these_evokeds, title='Hits vs. Misses (65 as old)',
        axes='topo', show=False, show_sensors=True)

    # Plot 100ms time window topomaps of difference wave
    # (min-max scaled) (centered on 0,.5, etc.)
    times = np.arange(.05, evokeds[0].tmax, .1)
    evoked = evokeds[evokeds_key['hit-miss65']]
    fig2 = evoked.plot_topomap(times=times, show=False,
                               average=.1, nrows=4, ncols=5)

    # Make the slider
    figs = [fig1[0], fig2]
    captions = [
        'Hits vs. Misses (65 as old)',
        'Hits - Misses (65 as old) Topos (100ms bins)'
        ]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Mastoid ref.): Hits - Misses (65 as old)', scale=1.5)
    plt.close()

    # Hits vs. Misses (6)
    conds = ['hit6', 'miss6']
    these_evokeds = [
        evokeds[evokeds_key[x]] for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(
        these_evokeds, title='Hits vs. Misses (6 as old)',
        axes='topo', show=False, show_sensors=True)

    # Plot 100ms time window topomaps of difference wave
    # (min-max scaled) (centered on 0,.5, etc.)
    times = np.arange(.05, evokeds[0].tmax, .1)
    evoked = evokeds[evokeds_key['hit-miss6']]
    fig2 = evoked.plot_topomap(times=times, show=False,
                               average=.1, nrows=4, ncols=5)

    # Make the slider
    figs = [fig1[0], fig2]
    captions = [
        'Hits vs. Misses (6 as old)',
        'Hits - Misses (6 as old) Topos (100ms bins)'
        ]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Mastoid ref.): Hits - Misses (6 as old)', scale=1.5)
    plt.close()

    # Scene Hits vs. Misses (65)
    conds = ['scene-hit65', 'scene-miss65']
    these_evokeds = [
        evokeds[evokeds_key[x]] for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(
        these_evokeds, title='Scenes: Hits vs. Misses (65 as old)',
        axes='topo', show=False, show_sensors=True)

    # Plot 100ms time window topomaps of difference wave
    # (min-max scaled) (centered on 0,.5, etc.)
    times = np.arange(.05, evokeds[0].tmax, .1)
    evoked = evokeds[evokeds_key['scene-hit-miss65']]
    fig2 = evoked.plot_topomap(times=times, show=False,
                               average=.1, nrows=4, ncols=5)

    # Make the slider
    figs = [fig1[0], fig2]
    captions = [
        'Scenes: Hits vs. Misses (65 as old)',
        'Scenes: Hits - Misses (65 as old) Topos (100ms bins)'
        ]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Mastoid ref.): Scene Hits - Misses (65 as old)', scale=1.5)
    plt.close()

    # Scene Hits vs. Misses (6)
    conds = ['scene-hit6', 'scene-miss6']
    these_evokeds = [
        evokeds[evokeds_key[x]] for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(
        these_evokeds, title='Scenes: Hits vs. Misses (6 as old)',
        axes='topo', show=False, show_sensors=True)

    # Plot 100ms time window topomaps of difference wave
    # (min-max scaled) (centered on 0,.5, etc.)
    times = np.arange(.05, evokeds[0].tmax, .1)
    evoked = evokeds[evokeds_key['scene-hit-miss6']]
    fig2 = evoked.plot_topomap(times=times, show=False,
                               average=.1, nrows=4, ncols=5)

    # Make the slider
    figs = [fig1[0], fig2]
    captions = [
        'Scenes: Hits vs. Misses (6 as old)',
        'Scenes: Hits - Misses (6 as old) Topos (100ms bins)'
        ]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Mastoid ref.): Scene Hits - Misses (6 as old)', scale=1.5)
    plt.close()

    # Objects Hits vs. Misses (65)
    conds = ['object-hit65', 'object-miss65']
    these_evokeds = [
        evokeds[evokeds_key[x]] for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(
        these_evokeds, title='Objects: Hits vs. Misses (65 as old)',
        axes='topo', show=False, show_sensors=True)

    # Plot 100ms time window topomaps of difference wave
    # (min-max scaled) (centered on 0,.5, etc.)
    times = np.arange(.05, evokeds[0].tmax, .1)
    evoked = evokeds[evokeds_key['object-hit-miss65']]
    fig2 = evoked.plot_topomap(times=times, show=False,
                               average=.1, nrows=4, ncols=5)

    # Make the slider
    figs = [fig1[0], fig2]
    captions = [
        'Objects: Hits vs. Misses (65 as old)',
        'Objects: Hits - Misses (65 as old) Topos (100ms bins)'
        ]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Mastoid ref.): Object Hits - Misses (65 as old)', scale=1.5)
    plt.close()

    # Objects Hits vs. Misses (6)
    conds = ['object-hit6', 'object-miss6']
    these_evokeds = [
        evokeds[evokeds_key[x]] for x in evokeds_key.keys() if x in conds]
    fig1 = plot_compare_evokeds(
        these_evokeds, title='Objects: Hits vs. Misses (6 as old)',
        axes='topo', show=False, show_sensors=True)

    # Plot 100ms time window topomaps of difference wave
    # (min-max scaled) (centered on 0,.5, etc.)
    times = np.arange(.05, evokeds[0].tmax, .1)
    evoked = evokeds[evokeds_key['object-hit-miss6']]
    fig2 = evoked.plot_topomap(times=times, show=False,
                               average=.1, nrows=4, ncols=5)

    # Make the slider
    figs = [fig1[0], fig2]
    captions = [
        'Objects: Hits vs. Misses (6 as old)',
        'Objects: Hits - Misses (6 as old) Topos (100ms bins)'
        ]
    report.add_slider_to_section(
        figs, section='ERP', captions=captions,
        title='ERP (Mastoid ref.): Object Hits - Misses (6 as old)', scale=1.5)
    plt.close()

    # Save report
    report_file = report_dir / f'{sub_string}_task-{task}_report.html'
    report.save(report_file, overwrite=True, open_browser=False)
