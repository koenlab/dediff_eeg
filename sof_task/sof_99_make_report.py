"""
Script: 03_sof_preprocess_eeg.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the SOF (scene, object, face) task. 
"""

#####---Import Libraries---#####
import numpy as np
import pandas as pd
import json
from itertools import islice
import matplotlib.pyplot as plt
from math import ceil

import mne
from mne import Report
from mne import pick_types
from mne.viz import (plot_compare_evokeds, plot_epochs,
                     plot_evoked_joint, plot_evoked_topo, 
                     plot_evoked_topomap, plot_raw, 
                     plot_raw_psd, plot_raw_psd_topo, 
                     plot_ica_properties, plot_ica_sources)

from sof_config import (bids_dir, deriv_dir, report_dir, task, 
                        get_sub_list, make_epochs_html,
                        bv_montage)
from sof_config import event_dict as event_id

# Get rid of warning
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Generating report for task-{task} data from {sub_string}')
    
    ### Initialize the report ###
    # Make the report object
    report = Report(subject=sub_string, title=f'{sub_string}: task-{task} report',
                    image_format='png', verbose=True, projs=False,
                    subjects_dir=None)
    
    # Plot behavioral data
    behav_fig_file = fig_path / f'{sub_string}_task-{task}_beh_performance.png'
    report.add_images_to_section(behav_fig_file, captions='Behavior: Performance Summary',
                                 section='Behavior')
    
    # Load the Raw data
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw = mne.io.read_raw_fif(raw_fif_file, preload=True)
    
    # Load the Epochs
    epoch_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs = mne.read_epochs(epoch_fif_file, preload=True)
    
    # Load Epochs json
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.json'
    with open(json_file, 'r') as f:
        json_info = json.load(f)
    
    # Plot Sensors
    fig = bv_montage.plot(show=False)
    report.add_figs_to_section(fig, 
                               captions='Electrodes: Montage Layout', section='Electrodes')
        
    # Get events
    events_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    events = mne.read_events(events_file)
    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], 
                              event_id=event_id, show=False)
    report.add_figs_to_section(fig, captions='Events: Event Types', section='Events',
                               scale=1.5)
    
    # Add PSD plot from raw and epochsepochs
    fig = plt.figure()
    fig.set_size_inches(12,7)
    ax1 = plt.subplot(221)
    raw.plot_psd(picks=['eeg'], xscale='linear', ax=ax1, show=False)
    ax1.set_title('Raw: Linear Scale')
    ax2 = plt.subplot(222)
    raw.plot_psd(picks=['eeg'], xscale='log', ax=ax2, show=False)
    ax2.set_title('Raw: Log Scale')
    ax3 = plt.subplot(223)
    epochs.plot_psd(picks=['eeg'], xscale='linear', ax=ax3, show=False)
    ax3.set_title('Epochs: Linear Scale')
    ax4 = plt.subplot(224)
    epochs.plot_psd(picks=['eeg'], xscale='log', ax=ax4, show=False)
    ax4.set_title('Epochs: Log Scale')
    ax4.set_xlabel('log(Frequency)')
    report.add_figs_to_section(fig, captions='Channel PSD: Plots', section='Channel PSD')
    
    
    # Make color index
    n_channels = len(epochs.info.ch_names)    
    epoch_colors = list()
    for i in np.arange(epochs.events.shape[0]):
        epoch_colors.append([None]*(n_channels-1) + ['k'])
        if i in json_info['bad_epochs']:
            epoch_colors[i] = ['r'] * n_channels
        
    # Plot the epoched data with a scroll bar
    print('Plotting epochs...this can take a while')
    plt.close()
    n_epochs = len(epochs)
    epoch_nums = np.arange(0,n_epochs)
    epochs_per_plot = 5
    n_plots = ceil(n_epochs / epochs_per_plot)
    figs = []
    captions = []
    count = 0
    for s in np.arange(0,n_epochs,epochs_per_plot):
        
        # Get the epochs to plot
        these_epochs = []
        for e in range(count, count+epochs_per_plot):
            if e in epoch_nums:
                these_epochs.append(e)
        count += epochs_per_plot
        
        # Update caption
        captions.append(f'Epochs {these_epochs[0]}-{these_epochs[-1]}')
        
        # Plot
        tmp_epochs = epochs.copy()[these_epochs]
        tmp_colors = [epoch_colors[x] for x in these_epochs]
        figs.append(
            tmp_epochs.plot(n_channels=66, n_epochs=epochs_per_plot, 
                scalings=dict(eeg=100e-6, eog=200e-6), 
                show=False, epoch_colors=tmp_colors, picks=['eeg','eog'])  
        )
        print(f'Figs completed: {len(figs)}')
   
    # Add slidebar figs
    report.add_slider_to_section(figs, captions=captions, section='Epochs', 
                                 title='Epochs: Epoch Level Data', scale=1.5)    
    plt.close()
    
    
    # # Plot the Raw data
    # tmax = raw._last_time
    # plot_duration = 20
    # n_step =int((plot_duration)*raw.info['sfreq'])
    # scalings = dict(eeg=40e-6, eog=200e-6)
    # plt.close()
    # figs = []
    # captions = []
    # for s in islice(raw.times, None, len(raw.times), n_step):
    #     captions.append('{:0.2f} to {:0.2f} secs'.format(s, s+plot_duration))
    #     figs.append(raw.plot(
    #         n_channels=len(raw.info['chs']),
    #         show=False, remove_dc=True,
    #         scalings=scalings, 
    #         start=s, duration=plot_duration
    #         ))
        
    # Load JSON file and add table
    epochs_info = make_epochs_html(sub_string, json_file, epochs)
    report.add_htmls_to_section(epochs_info.render(), 
                                captions='Epochs: Info',
                                section='Epochs')
   
   
    # Save report
    report_file = report_dir / f'{sub_string}_task-{task}_report.html'
    report.save(report_file, overwrite=True, open_browser=False)   