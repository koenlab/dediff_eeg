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

from sof_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

evokeds = dict(scene=[],object=[],face=[])
data = dict(sub=[],age=[],
        r_face_peak_voltage=[],r_face_peak_latency=[], 
        l_face_peak_voltage=[],l_face_peak_latency=[], 
        r_scene_peak_voltage=[],r_scene_peak_latency=[],
        l_scene_peak_voltage=[],l_scene_peak_latency=[])


# Ask for subject IDs to analyze
sub_list = get_sub_list(deriv_dir, allow_all=True)
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Preprocessing task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')
    
    ### STEP 1: Load manually cleaned epochs
    # Read in Cleaned Epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.fif.gz'
    if not epochs_fif_file.is_file():
        continue
    epochs = read_epochs(epochs_fif_file)

    # Add to data
    data['sub'].append(sub_string)
    data['age'].append('young' if int(sub_string.replace('sub-','')) < 200 else 'older')
    
    # Make evokeds
    these_evokeds = []
    for cond in ['scene','object','face']:
        query = f"category == '{cond}' and repeat==1 and n_responses==0"
        evoked = epochs[query].crop(preprocess_options['evoked_tmin'], preprocess_options['evoked_tmax']).average()
        evoked.comment = cond
        save_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_cond-{cond}_ave.fif.gz'
        evoked.save(save_file)
        evokeds[cond].append(evoked)
        these_evokeds.append(evoked)

    scene_diff = mne.combine_evoked(these_evokeds,weights=[1,-1,0])
    face_diff = mne.combine_evoked(these_evokeds,weights=[0,-1,1])
    rois = {
        'right': mne.pick_channels(face_diff.info['ch_names'], ['PO8','P8'] ),
        'left': mne.pick_channels(face_diff.info['ch_names'], ['PO7','P7'] )
    }
    
    for roi, chs in rois.items():
        
        # Process scene 
        scene_roi = mne.channels.combine_channels(scene_diff, dict(roi=chs))
        _, t, v = scene_roi.get_peak(mode='pos', return_amplitude=True, 
                                      tmin=.15, tmax=.25)
        data[f'{roi[0]}_scene_peak_latency'].append(t*1000)
        data[f'{roi[0]}_scene_peak_voltage'].append(v*1e6)

        # Process face
        face_roi = mne.channels.combine_channels(face_diff, dict(roi=chs))
        _, t, v = face_roi.get_peak(mode='neg', return_amplitude=True, 
                                      tmin=.1, tmax=.18)
        data[f'{roi[0]}_face_peak_latency'].append(t*1000)
        data[f'{roi[0]}_face_peak_voltage'].append(v*1e6)
        
# import pandas as pd
# df = pd.DataFrame(data)
# from scipy.stats import ttest_ind
# dvs = df.columns[2:].tolist()

# stats = {}
# for dv in dvs:
#     y = df[df['age']=='young'][dv]
#     o = df[df['age']=='older'][dv]
#     stats[dv] = ttest_ind(y,o)

# # grands = [mne.grand_average(evokeds[x]) for x in evokeds.keys()]
# grands = []
# for cond in evokeds.keys():
#     this_grand = mne.grand_average(evokeds[cond])
#     this_grand.comment=cond
#     grands.append(this_grand)

# diff = [
#     mne.combine_evoked(grands, weights=[1,-1,0]),
#     mne.combine_evoked(grands, weights=[0,-1,1])
# ]
# diff[0].comment = 'face-object'
# diff[1].comment = 'scene-object'
#     # evokeds = [faces, scenes, objects]
#     # mne.viz.plot_compare_evokeds(evokeds, picks='PO8')
#     # evokeds_lpf = [faces_lpf, scenes_lpf, objects_lpf]
#     mne.viz.plot_compare_evokeds(grands, picks=['PO8','P8','P6'], combine='mean')
#     mne.viz.plot_compare_evokeds(diff, picks=['PO8','P8','P6'], combine='mean')
# diff[1].plot_joint(times=[.13,.175])
# _, old_t1, old_v1 = diff[0].pick('PO8').get_peak(mode='neg', return_amplitude=True, tmin=.1, tmax=.17)
# _, old_t2, old_v2 = diff[1].pick('PO8').get_peak(mode='pos', return_amplitude=True, tmin=.15, tmax=.25)

# print('Face')
# print(t1,old_t1)
# print(v1*1e6,old_v1*1e6)    

# print('Scene')
# print(t2,old_t2)
# print(v2*1e6,old_v2*1e6)    


#     # diff_v_objects = [
#     #     mne.combine_evoked(evokeds, weights=[1, 0, -1]),
#     #     mne.combine_evoked(evokeds, weights=[0, 1, -1])
#     # ]
#     # mne.viz.plot_compare_evokeds(diff_v_objects, picks='PO8')
#     # diff_v_objects_lpf = [
#     #     mne.combine_evoked(evokeds_lpf, weights=[1, 0, -1]),
#     #     mne.combine_evoked(evokeds_lpf, weights=[0, 1, -1])
#     # ]
#     # mne.viz.plot_compare_evokeds(diff_v_objects_lpf, picks='PO8')

#     # diff_v_other = [
#     #     mne.combine_evoked(evokeds, weights=[2, -1, -1]),
#     #     mne.combine_evoked(evokeds, weights=[-1, 2, -1])
#     # ]
#     # mne.viz.plot_compare_evokeds(diff_v_other, picks='PO8')
#     # diff_v_other_lpf = [
#     #     mne.combine_evoked(evokeds_lpf, weights=[2, -1, -1]),
#     #     mne.combine_evoked(evokeds_lpf, weights=[-1, 2, -1])
#     # ]
#     # mne.viz.plot_compare_evokeds(diff_v_other_lpf, picks='PO8')
#     # diff_v_objects[0].get_pe    

#     # face1 = diff_v_objects_lpf[0]
#     # ch1, lat1, amp1 = face1.pick('PO8').get_peak(tmin=.10, tmax=.20, mode='neg', return_amplitude=True)
#     # face2 = diff_v_other_lpf[0]
#     # ch2, lat2, amp2 = face2.pick('PO8').get_peak(tmin=.10, tmax=.20, mode='neg', return_amplitude=True)

#     # scene1 = diff_v_objects_lpf[1]
#     # ch1, lat1, amp1 = scene1.pick('PO8').get_peak(tmin=.15, tmax=.25, mode='pos', return_amplitude=True)
#     # scene2 = diff_v_other_lpf[1]
#     # ch2, lat2, amp2 = scene2.pick('PO8').get_peak(tmin=.15, tmax=.25, mode='pos', return_amplitude=True)