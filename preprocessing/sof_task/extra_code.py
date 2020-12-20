def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data    


        
    # Load prior marks if they exist (from JSON file)
    marked_bad = [False] * len(epochs)
    json_file = deriv_path / f'{sub_string}_task-{task}_ref-avg_desc-cleaned_epo.json'
    if json_file.is_file():
        with open(json_file, 'r') as f:
            marked = json.load(f)['bad_epochs']
        for x in marked:
            marked_bad[x]=True
    

    # Make color index
    n_channels = len(epochs.info.ch_names)    
    epoch_colors = list()
    for i in np.arange(epochs.events.shape[0]):
        epoch_colors.append([None]*(n_channels-1) + ['k'])
        if i in blink_inds:
            epoch_colors[i] = ['b'] * n_channels
        if reject_log.bad_epochs[i]:
            epoch_colors[i] = ['m'] * n_channels
        if marked_bad[i]:
            epoch_colors[i] = ['g'] * n_channels

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from mne.time_frequency import psd_welch
from mne.viz import plot_topomap

new_epochs = epochs.copy().crop(tmin=-1.1,tmax=0).pick_types(eeg=True, exclude=['TP9','TP10','FT9','FT10'])
psd, freqs = psd_welch(new_epochs, 
                       fmin=1, fmax=40, 
                       average='median')
psd = psd.mean(axis=0)
psd = np.asarray(psd)
fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                peak_threshold=2., max_n_peaks=6, verbose=False)
freq_range = [2, 25]
fg.fit(freqs, psd, freq_range)
fg.plot()
bands = Bands({'theta': [3, 7],
               'alpha': [7, 14],
               'beta': [15, 30]})
alphas = get_band_peak_fg(fg, bands.alpha)
alpha_pw = alphas[:,1]
fig = plt.figure()
plot_topomap(alpha_pw, new_epochs.info, cmap=cm.viridis, contours=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Create a topomap for the current oscillation band
    mne.viz.plot_topomap(band_power, new_epochs.info, cmap=cm.viridis, contours=0,
                         axes=axes[ind], show=False);

    # Set the plot title
    axes[ind].set_title(label + ' power', {'fontsize' : 20})
    
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Extracted and plot the power spectrum model with the most band power
    fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

    # Set some plot aesthetics & plot title
    axes[ind].yaxis.set_ticklabels([])
    axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})

fig = plt.figure()
exps = fg.get_params('aperiodic_params', 'exponent')
plot_topomap(exps, new_epochs.info, cmap=cm.viridis, contours=0)
    # # Manually inspect 
    # for ic in ica.exclude:
    #     ica.plot_properties(epochs, picks=ic, show=False,
    #                        psd_args=dict(fmax=70))
    #     ic_file = fig_path / f'{sub_string}_task-{task}_ic{ic:02}_properties.png'
    #     plt.savefig(ic_file, dpi=600)
    #     plt.close('all')

# Make faces evoked 
    face_query = "category == 'face' and repeat==1 and n_responses==0"
    faces = epochs[face_query].crop(-.2,.5).average()
    #faces_file = deriv_path / f'{sub_string}_task-{task}_cond-faces_filt-none_ave.fif.gz'
    #faces.save(faces_file)
    
    # Make scenes evoked
    scene_query = "category == 'scene' and repeat==1 and n_responses==0"
    scenes = epochs[scene_query].crop(-.2,.5).average()
    #scenes_file = deriv_path / f'{sub_string}_task-{task}_cond-scenes_filt-none_ave.fif.gz'
    #scenes.save(scenes_file)
    
    # Make objects evoked
    object_query = "category == 'object' and repeat==1 and n_responses==0"
    objects = epochs[object_query].crop(-.2,.5).average()
    #objects_file = deriv_path / f'{sub_string}_task-{task}_cond-objects_filt-none_ave.fif.gz'
    #objects.save(objects_file)

    evokeds = [faces.filter(None,20), scenes.filter(None,20), objects.filter(None,20)]
    mne.viz.plot_compare_evokeds(evokeds, axes='topo')
    diff_waves = [mne.combine_evoked(evokeds, weights=[1, -.5, -.5]),
                  mne.combine_evoked(evokeds, weights=[-.5, 1, -.5])
    ]
    mne.viz.plot_compare_evokeds(diff_waves, axes='topo')
    diff_waves2 = [mne.combine_evoked(evokeds, weights=[1, 0, -1]),
                  mne.combine_evoked(evokeds, weights=[0, 1, -1])
    ]
    mne.viz.plot_compare_evokeds(diff_waves2, axes='topo')

    diff_waves[0].pick_types(eeg=True,exclude=['FT9', 'FT10', 'TP9', 'TP10']).set_montage(bv_montage).plot_topomap(times=np.arange(.08,.24,.01), average=0.02, outlines='head')
    diff_waves[1].pick_types(eeg=True,exclude=['FT9', 'FT10', 'TP9', 'TP10']).set_montage(bv_montage).plot_topomap(times=np.arange(.08,.24,.01), average=0.02, outlines='head')

    # # power 
    # freqs = np.arange(start=3, stop=50`)
    # cycles = 7
    # power = tfr_morlet(epochs, freqs, 5, average=False, return_itc=False)
    
    # power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1, fmin=8, fmax=12,
    #                baseline=(-0.3, -.15), mode='logratio',
    #                title='Beta', show=True)
    # power.apply_baseline(baseline=(-.3,-.15), mode='logratio')
    # face_tfr = power['face/novel'].average()
    # object_tfr = power['object/novel'].average()
    # scene_tfr = power['scene/novel'].average()
    
    # face_diff_tfr = mne.combine_evoked([face_tfr, object_tfr], weights=[1,-1])
    # face_diff_tfr.plot_topomap(tmin=0.5, tmax=1, fmin=8, fmax=12,
    #                title='Beta', show=True)
    # scene_diff_tfr = mne.combine_evoked([scene_tfr, object_tfr], weights=[1,-1])
    # scene_diff_tfr.plot_topomap(tmin=0.5, tmax=1, fmin=8, fmax=12,
    #                title='Beta', show=True)
    
    
    
    
    
    evokeds.append(mne.combine_evoked(cond_evokeds,))
    
    # Make scene - object difference wave
    evoked = mne.combine_evoked(cond_evokeds, weights=[1,-1,0])
    evoked.comment = 'scene-object'
    evokeds.append(evoked)
    
    # Make face - object difference wave
    evoked_list = [
        evokeds['face'],
        evokeds['object']
    ]
    weights = [1,-1]
    evokeds['face-object'] = mne.combine_evoked(evoked_list, weights=weights)
    
    # Make scene- and face-other
    evoked_list = [x for x in evokeds.values()]
    contrasts = {
        'scene-other': [1,-.5,-.5],
        'face-other': [-.5,-.5,1]
    }
    
    
        # MAKE JSON INFORMATION

    scene_diff = mne.combine_evoked(these_evokeds,weights=[1,-1,0]).filter(None,20)
    face_diff = mne.combine_evoked(these_evokeds,weights=[0,-1,1]).filter(None,20)
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
        
import pandas as pd
df = pd.DataFrame(data)
from scipy.stats import ttest_ind
dvs = df.columns[2:].tolist()

stats = {}
for dv in dvs:
    y = df[df['age']=='young'][dv]
    o = df[df['age']=='older'][dv]
    stats[dv] = ttest_ind(y,o)



young = [i for i,val in enumerate((df['age'] == 'young').values.tolist()) if val]
old = [i for i,val in enumerate((df['age'] == 'older').values.tolist()) if val]

grand_young = [mne.grand_average(list(np.array(evokeds[x])[young])).filter(None,20) for x in evokeds.keys()]
grand_old = [mne.grand_average(list(np.array(evokeds[x])[old])).filter(None,20) for x in evokeds.keys()]

for i, v in enumerate(['scene','object','face']):
    grand_young[i].comment = v + 'young'
    grand_old[i].comment = v + 'older'

ylim = dict(eeg=[-5,12])
f = mne.viz.plot_compare_evokeds(grand_young, picks=['PO7','P7'], combine='mean', ylim=ylim)
f[0].savefig(bids_dir / '..' / 'young_grand_left.jpg')
f = mne.viz.plot_compare_evokeds(grand_old, picks=['PO7','P7'], combine='mean', ylim=ylim)
f[0].savefig(bids_dir / '..' / 'old_grand_left.jpg')

diff_scene = [
    mne.combine_evoked(grand_young, weights=[1,-1,0]),
    mne.combine_evoked(grand_old, weights=[1,-1,0])
]
diff_scene[0].comment = 'young'
diff_scene[1].comment = 'old'
ylim = dict(eeg=[-1,5])
f = mne.viz.plot_compare_evokeds(diff_scene, picks=['PO7','P7'], 
                                combine='mean', ylim=ylim, colors=['m','g'])
f[0].savefig(bids_dir / '..' / 'diff_grand_scene_left.jpg')

diff_face = [
    mne.combine_evoked(grand_young, weights=[0,-1,1]),
    mne.combine_evoked(grand_old, weights=[0,-1,1])
]
diff_face[0].comment = 'young'
diff_face[1].comment = 'old'
ylim = dict(eeg=[-6,2])
f = mne.viz.plot_compare_evokeds(diff_face, picks=['PO7','P7'], 
                                combine='mean', ylim=ylim, colors=['m','g'])
f[0].savefig(bids_dir / '..' / 'diff_grand_face_left.jpg')

# # grands = [mne.grand_average(evokeds[x]) for x in evokeds.keys()]
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