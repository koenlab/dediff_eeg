"""
Script: 03_sof_preprocess_eeg.py
Creator: Joshua D. Koen
Description: This script imports data from sourcedata to bids format for 
the SOF (scene, object, face) task. 
"""

#####---Import Libraries---#####
import mne


from sof_config import (bids_dir, deriv_dir, task,
                        file_patterns, get_sub_list)

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    fig_path = deriv_path / 'figures'
    print(f'Generating report for task-{task} data from {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')

    ### Start the report ###
    # Make the report object
    report = mne.Report(subject=sub_string, title=f'{sub_string}: task-{task} report',
                        image_format='svg', raw_psd=True, verbose=True, projs=False,
                        subjects_dir=None)
    report._sort_sections=True

    # Parse the EEG files in the subjects derivative directory
    report.parse_folder(deriv_path, pattern='*resamp*raw.fif.gz', render_bem=False, 
                        sort_sections=True)

    # Add information about epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs = mne.read_epochs(epochs_fif_file, preload=True)
    
    # Save report
    report_file = deriv_dir / f'{sub_string}_task-{task}_report.html'
    report.save(report_file, overwrite=True, open_browser=False)   