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

import mne
from mne import Report
from mne import pick_types
from mne.viz import (plot_compare_evokeds, plot_epochs,
                     plot_evoked_joint, plot_evoked_topo, 
                     plot_evoked_topomap, plot_raw, 
                     plot_raw_psd, plot_raw_psd_topo, 
                     plot_ica_properties, plot_ica_sources)

from sof_config import (bids_dir, deriv_dir, report_dir, task, 
                        get_sub_list, make_raw_html, bv_montage)
from sof_config import event_dict as event_id

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Generating report for task-{task} data from {sub_string}')
    
    ### Start the report ###
    # Make the report object
    report = Report(subject=sub_string, title=f'{sub_string}: task-{task} report',
                    image_format='png', verbose=True, projs=False,
                    subjects_dir=None)
        
    # Load the raw information
    raw_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-FCz_desc-resamp_raw.fif.gz'
    raw = mne.io.read_raw_fif(raw_fif_file, preload=True)
        
    # Plot Sensors
    report.add_figs_to_section(bv_montage.plot(show=False), captions='Channel Locations', section='Channel Locations')
    
    # Get events
    events_file = deriv_path / f'{sub_string}_task-{task}_desc-resamp_eve.txt'
    events = mne.read_events(events_file)
    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], 
                              event_id=event_id, show=False)
    report.add_figs_to_section(fig, captions='Event Types', section='Events')
    
    ## PROCESS RESAMPLED RAW DATA
    # Add info to raw section
    raw_info = make_raw_html(sub_string, raw)
    report.add_htmls_to_section(raw_info.render(), 
                                captions='Raw Info',
                                section='Raw')
    
    # Add PSD plot per channel
    bads = raw.info['bads']
    raw.info['bads'] = []
    fig = plt.figure()
    fig.set_size_inches(7,7)
    ax1 = plt.subplot(211)
    raw.plot_psd(picks=['eeg'], xscale='linear', ax=ax1, show=False)
    ax1.set_title('Linear Scale')
    ax2 = plt.subplot(212)
    raw.plot_psd(picks=['eeg'], xscale='log', ax=ax2, show=False)
    ax2.set_title('Log Scale')
    ax2.set_xlabel('log(Frequency)')
    report.add_figs_to_section(fig, captions='Channel PSD', section='Raw')
    
    
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
        
    # # Add slidebar figs
    # report.add_slider_to_section(figs, captions=captions, section='Raw', 
    #                              title='Resampled Continuous Data', scale=.70)
    
    # plt.close()
    # Save report
    report_file = report_dir / f'{sub_string}_task-{task}_report.html'
    report.save(report_file, overwrite=True, open_browser=False)   