"""
Script: study_06_compute_tf_morlet.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

#####---Import Libraries---#####
from sys import modules
import numpy as np
import json

from mne import read_epochs
from mne.time_frequency import tfr_morlet
import mne

from study_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

scenes = []
objects = []
smos = []

# Define which reference to use
ref = 'avg'    

# Frequecies to estimate
freqs = np.arange(3,51,1) # Estimate 4-50Hz in linear intervals
n_cycles = 5 # of cycles

# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Creating TF data (Morlet) for task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    # Load epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file)
    
    # Estiamte TFR
    tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                     return_itc=False, picks=['eeg'], average=False,
                     output='power', n_jobs=1, verbose=True)
    
    # Save the power
    tfr_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_method-morlet_value-power_tfr.h5'
    tfr.save(tfr_file, overwrite=True)
    
    # Make JSON
    json_info = {
        'Description': 'TFR Power from Morlet Wavelets',
        'baseline': dict(twin='n/a', mode='n/a', sort_keys=True), 
        'sfreq': tfr.info['sfreq'],
        'reference': 'average',
        'tmin': tfr.times.min(),
        'tmax': tfr.times.max(),
        'freqs': freqs.tolist(),
        'n_cycles': n_cycles
    }
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-{ref}_method-morlet_value-power_tfr.json'
    with open(json_file, 'w') as outfile:
        json.dump(json_info, outfile, indent=4)
        
# #     # # 
#     scene = tfr["category=='scene' and study_n_responses==1 and test_resp in [5,6]"].average()
#     scenes.append(scene.apply_baseline((-.4, -.1), mode='logratio'))
    
#     # scene.plot_topo(picks=['eeg'], baseline=(-.5,.2), mode='logratio')
#     obj = tfr["category=='scene' and study_n_responses==1 and test_resp in [1,2,3,4]"].average()
#     objects.append(obj.apply_baseline((-.4, -.1), mode='logratio'))
#     # object.plot_topo(picks=['eeg'], baseline=(-.5,.2), mode='logratio')
#     evokeds = [scene,obj]
#     smo = mne.combine_evoked(evokeds, weights=[1, -1])
#     smos.append(smo)
#     smo.plot_topo(picks=['eeg'], tmin=-.5, tmax=1.5, fmax=25)
    
# # scene_grand = mne.grand_average(scenes)
# # scene_grand.plot_topo(picks=['eeg'])
# # obj_grand = mne.grand_average(objects)
# # obj_grand.plot_topo(picks=['eeg'])
# # smo_grand = mne.grand_average(smos)
# # smo_grand.plot_topo(picks=['eeg'])