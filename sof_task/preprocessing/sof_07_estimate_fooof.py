"""
Script: sof_07_estimate_fooof.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data runs the
FOOOF algorithm. The script also writes an HTML report.

FOOOF is run only on data from the 1 second prestimulus period
for trials that are first presentation only (not repeats) that
did not recieve a response (i.e., a commission error)
"""

#####---Import Libraries---#####
import numpy as np
import matplotlib.pyplot as plt

from mne import read_epochs
from mne.time_frequency import psd_welch

from fooof import FOOOFGroup

from sof_config import (deriv_dir, task, get_sub_list)

# Make report folder
report_dir = deriv_dir / 'fooof_reports'
report_dir.mkdir(exist_ok=True, parents=True)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string

    # Load epochs file
    epochs_fif_file = deriv_path / \
        f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    if not epochs_fif_file.is_file():
        continue
    epochs = read_epochs(epochs_fif_file)["repeat==1 and n_responses==0"]
    epochs.filter(None, 40)
    epochs.drop_channels(['VEOG', 'HEOG', 'FT9', 'FT10', 'TP9', 'TP10'])
    epochs.apply_baseline(baseline=(None, 0))

    # Estimate the PSD in the pre-stimulus window
    spectrum, freqs = psd_welch(epochs, tmin=-1.0, tmax=0.0,
                                fmin=2, fmax=30, n_fft=250,
                                average='median',
                                n_per_seg=200, n_overlap=150)
    spectrum = np.mean(spectrum, axis=0)

    # Run FOOF
    fm = FOOOFGroup(peak_width_limits=(2.0, 12.0),
                    aperiodic_mode='fixed',
                    peak_threshold=1)

    # Set the frequency range to fit the model
    freq_range = [2, 30]

    # Fit the FOOOF model
    fm.fit(freqs, spectrum, freq_range)

    # Save the FOOOF model in derivatives
    fooof_out_file = f'{sub_string}_task-sof_ref-avg_desc-firstcorrect_fooof'
    fm.save(fooof_out_file, file_path=deriv_path,
            save_results=True, save_data=True)

    # Save Results
    fooof_results_file = f'{sub_string}_task-sof_fooofreport.pdf'
    fm.save_report(fooof_results_file, file_path=report_dir)

    # # Make a FOOOF object for individual data
    # for i in np.arange(spectrum.shape[0]):
    #     fi = FOOOF(peak_width_limits=(2.0, 12.0),
    #                aperiodic_mode='fixed',
    #                peak_threshold=1)
    #     fi.fit(freqs, spectrum[i], freq_range)
    #     fi.plot()
    #     fig = plt.gcf()