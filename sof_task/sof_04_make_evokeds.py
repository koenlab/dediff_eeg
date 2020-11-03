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
import json

from mne import read_epochs
import mne

from sof_config import (bids_dir, deriv_dir, task, preprocess_options)

# Ask for subject IDs to analyze
print('What IDs are being preprocessed?')
print('(Enter multiple values separated by a comma; e.g., 101,102)')
sub_list = input('Enter IDs: ')
sub_list = sub_list.split(',')
print(sub_list)

face_list = []
scene_list = []
object_list = []

for sub in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    sub_string = f'sub-{sub}'
    deriv_path = deriv_dir / sub_string
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: Load manually cleaned epochs
    # Read in Cleaned Epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    epochs = read_epochs(epochs_fif_file)

    # Make face/novel evoked
    faces_query = "category == 'face' and repeat==1 and n_responses==0"
    faces = epochs[faces_query].crop(preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']).average()
    # faces_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_cond-face_filt-none_desc-cleaned_ave.fif.gz'
    # faces.save(faces_file)
    # faces_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_cond-face_filt-20lpf_desc-cleaned_ave.fif.gz'
    # faces_lpf = faces.copy().filter(None, preprocess_options['evoked_highcutoff'])
    # faces_lpf.save(faces_file)
    
    # Make scene/novel evoked
    scenes_query = "category == 'scene' and repeat==1 and n_responses==0"
    scenes = epochs[scenes_query].crop(preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']).average()
    # scenes_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_cond-scene_filt-none_desc-cleaned_ave.fif.gz'
    # scenes.save(scenes_file)
    # scenes_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_cond-scene_filt-20lpf_desc-cleaned_ave.fif.gz'
    # scenes_lpf = scenes.copy().filter(None, preprocess_options['evoked_highcutoff'])
    # scenes_lpf.save(scenes_file)
    
    # Make object evoked
    objects_query = "category == 'object' and repeat==1 and n_responses==0"
    objects = epochs[objects_query].crop(preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']).average()
    # objects_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_cond-object_filt-none_desc-cleaned_ave.fif.gz'
    # objects.save(objects_file)
    # objects_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_cond-object_filt-20lpf_desc-cleaned_ave.fif.gz'
    # objects_lpf = objects.copy().filter(None, preprocess_options['evoked_highcutoff'])
    # objects_lpf.save(objects_file)

    face_list.append(faces)
    scene_list.append(scenes)
    object_list.append(objects)
    
    
    # evokeds = [faces, scenes, objects]
    # mne.viz.plot_compare_evokeds(evokeds, picks='PO8')
    # evokeds_lpf = [faces_lpf, scenes_lpf, objects_lpf]
    mne.viz.plot_compare_evokeds(grands, picks='PO8')
    
    # diff_v_objects = [
    #     mne.combine_evoked(evokeds, weights=[1, 0, -1]),
    #     mne.combine_evoked(evokeds, weights=[0, 1, -1])
    # ]
    # mne.viz.plot_compare_evokeds(diff_v_objects, picks='PO8')
    # diff_v_objects_lpf = [
    #     mne.combine_evoked(evokeds_lpf, weights=[1, 0, -1]),
    #     mne.combine_evoked(evokeds_lpf, weights=[0, 1, -1])
    # ]
    # mne.viz.plot_compare_evokeds(diff_v_objects_lpf, picks='PO8')

    # diff_v_other = [
    #     mne.combine_evoked(evokeds, weights=[2, -1, -1]),
    #     mne.combine_evoked(evokeds, weights=[-1, 2, -1])
    # ]
    # mne.viz.plot_compare_evokeds(diff_v_other, picks='PO8')
    # diff_v_other_lpf = [
    #     mne.combine_evoked(evokeds_lpf, weights=[2, -1, -1]),
    #     mne.combine_evoked(evokeds_lpf, weights=[-1, 2, -1])
    # ]
    # mne.viz.plot_compare_evokeds(diff_v_other_lpf, picks='PO8')
    # diff_v_objects[0].get_pe    

    # face1 = diff_v_objects_lpf[0]
    # ch1, lat1, amp1 = face1.pick('PO8').get_peak(tmin=.10, tmax=.20, mode='neg', return_amplitude=True)
    # face2 = diff_v_other_lpf[0]
    # ch2, lat2, amp2 = face2.pick('PO8').get_peak(tmin=.10, tmax=.20, mode='neg', return_amplitude=True)

    # scene1 = diff_v_objects_lpf[1]
    # ch1, lat1, amp1 = scene1.pick('PO8').get_peak(tmin=.15, tmax=.25, mode='pos', return_amplitude=True)
    # scene2 = diff_v_other_lpf[1]
    # ch2, lat2, amp2 = scene2.pick('PO8').get_peak(tmin=.15, tmax=.25, mode='pos', return_amplitude=True)