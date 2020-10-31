    
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