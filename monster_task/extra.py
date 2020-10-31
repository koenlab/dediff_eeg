epochs_ar = epochs_ar['stan']
top = epochs_ar['gabor_loc=="top" and correct==1'].copy().crop(tmin=-.2, tmax=.6).average()
bot = epochs_ar['bot'].copy().crop(tmin=-.2, tmax=.6).average()

evokeds = [top.filter(None,20), bot.filter(None,20)]
diff = mne.combine_evoked(evokeds, weights=[1, -1])
mne.viz.plot_compare_evokeds(evokeds, axes='topo')