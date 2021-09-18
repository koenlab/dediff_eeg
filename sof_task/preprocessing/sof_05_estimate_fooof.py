"""
Script: sof_07_estimate_fooof.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data runs the
FOOOF algorithm. The script also writes an HTML report.

FOOOF is run only on data from the 1 second prestimulus period
for trials that are first presentation only (not repeats) that
did not recieve a response (i.e., a commission error)
"""

# Import libraries
import sys
import os

os.chdir(sys.path[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import numpy as np
import pandas as pd

from mne import read_epochs
from mne.time_frequency import psd_welch

from fooof import FOOOFGroup

from sof_config import (deriv_dir, task)
from functions import get_sub_list

# Make report folder
report_dir = deriv_dir / 'fooof_reports'
report_dir.mkdir(exist_ok=True, parents=True)


# FOOF fit
def fit_fooof(epochs):
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
    return fm.copy()


# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub in sub_list:

    # SUBJECT INFORMATION DEFINITION
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub

    # Load epochs file
    epochs_fif_file = deriv_path / \
        f'{sub}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    if not epochs_fif_file.is_file():
        continue
    epochs = read_epochs(epochs_fif_file)["repeat==1 and n_responses==0"]
    epochs.drop_channels(['VEOG', 'HEOG', 'FT9', 'FT10', 'TP9', 'TP10'])
    epochs.apply_baseline(baseline=(None, 0))

    # Run first pass
    fm_orig = fit_fooof(epochs)

    # Save the FOOOF model in derivatives
    fooof_out_file = f'{sub}_task-sof_ref-avg_desc-orig_fooof'
    fm_orig.save(fooof_out_file, file_path=deriv_path,
                 save_results=True, save_data=True)

    # Save Results
    fooof_results_file = f'{sub}_task-sof_desc-orig_fooofreport.pdf'
    fm_orig.save_report(fooof_results_file, file_path=report_dir)

    # Find bad r2 channel
    epochs.info['bads'] = []
    for i, r in enumerate(fm_orig.get_params('r_squared')):
        if r < .75:
            epochs.info['bads'].append(epochs.info['ch_names'][i])
    epochs.interpolate_bads()

    # Store fm_orig data in a data frame
    fooof_df = pd.DataFrame({
        'ch_names': epochs.info['ch_names'],
        'exp_orig': fm_orig.get_params('aperiodic', col='exponent'),
        'offset_orig': fm_orig.get_params('aperiodic', col='offset'),
        'r2_orig': fm_orig.get_params('r_squared'),
        'error_orig': fm_orig.get_params('error')
    })
    fooof_df.insert(0, 'id', sub)

    # Run second pass
    fm_interp = fit_fooof(epochs)

    # Save the FOOOF model in derivatives
    fooof_out_file = f'{sub}_task-sof_ref-avg_desc-interp_fooof'
    fm_interp.save(fooof_out_file, file_path=deriv_path,
                   save_results=True, save_data=True)

    # Save Results
    fooof_results_file = f'{sub}_task-sof_desc-interp_fooofreport.pdf'
    fm_interp.save_report(fooof_results_file, file_path=report_dir)

    # Add to data frame
    fooof_df['exp_interp'] = fm_interp.get_params('aperiodic', col='exponent')
    fooof_df['offset_interp'] = fm_interp.get_params('aperiodic', col='offset')
    fooof_df['r2_interp'] = fm_interp.get_params('r_squared')
    fooof_df['error_interp'] = fm_interp.get_params('error')

    # Save dataframe
    df_file = deriv_path / f'{sub}_task-sof_fooof.tsv'
    fooof_df.to_csv(df_file, sep='\t', index=False)
