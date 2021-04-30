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
from matplotlib import cm, colors, colorbar
import json


from mne import read_epochs
from mne.time_frequency import psd_welch
from mne.viz import plot_topomap
import mne

from fooof import (FOOOF, FOOOFGroup)
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

from sof_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    # print(f'Creating evoked for task-{task} data for {sub_string}')
    # print(f'  Derivatives Folder: {deriv_path}')
    
    # Load epochs file
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    if not epochs_fif_file.is_file():
        continue
    epochs = read_epochs(epochs_fif_file)["repeat==1 and n_responses==0"]
    epochs.drop_channels(['VEOG','HEOG','FT9','FT10','TP9','TP10'])
    
    # Estimate the PSD in the pre-stimulus window
    spectrum, freqs = psd_welch(epochs, tmin=-1.0, tmax=0.0, 
                               fmin=2, fmax=100,
                               n_per_seg=200, n_overlap=150)
    spectrum = np.mean(spectrum, axis=(0))
    
    # Run FOOF
    fm = FOOOF(peak_width_limits=(2.0, 8.0), aperiodic_mode='fixed')

    # Set the frequency range to fit the model
    freq_range = [3, 40]

    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fm.fit(freqs, spectrum, freq_range)
    fm.report()
    
    fg = FOOOFGroup(peak_width_limits=(2.0, 8.0), aperiodic_mode='fixed',
                    min_peak_height=0.05, max_n_peaks=6)
    fg.fit(freqs, spectrum, [3, 30])
    fg.print_results()
    fg.plot()
    
    alphas = get_band_peak_fg(fg, [15,30])
    alpha_pw = alphas[:,1]
    
    exps = fg.get_params('aperiodic_params', 'exponent')
    f = plot_topomap(alpha_pw, epochs.info, cmap=cm.viridis, contours=0)
    f.colorbar()