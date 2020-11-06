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