"""
Script: Angle-Bin Decoding
Creator: Joshua D. Koen
Description: This script use the Bae and Luck procedure to
    perform decoding of 8 angle bins. Briefly, the script:
    1. Loads Epoched data
    2. Splits into 16 bins (crossed with top/bottom and angle bin)
        This must be divisible 3 (maximum of 16 epochs per bin)
        Only on standard trials)
    3. ERPs made for each bin, then averaged over top/bottom (assuming equal weights)
    4. Decoding done on a time-by-time basis with all scalp electrodes
        Classifier is a SVM using 1-vs-all
        Chance is 1/8 (or 12.5%)
"""

#####---Import Libraries---#####
import itertools

import numpy as np
from numpy.random import shuffle
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OutputCodeClassifier
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

from mne import read_epochs
import mne

from monster_config import (bids_dir, deriv_dir, task, preprocess_options, get_sub_list)

# Determine number of simulations
n_sims = 10
n_folds = 3

def trim_epochs(inst, n_trials, n_folds):
    epoch_id = np.arange(len(inst))
    shuffle(epoch_id)
    n_drop = len(inst) - (n_trials*n_folds)
    if n_drop > 0:
        drop = epoch_id[-n_drop:]
        inst.drop(drop)
    return inst

def epochs_2_evokeds(top, bot, n_trials, n_folds):
    """
    This function takes top and bottom epoch bins, cut down to
    n_trials per epoch. top and bot are then averaged with equal
    weighting
    """
    
    # Make output list
    evokeds = []
    
    # Handle top and bottom trimming
    top = trim_epochs(top.copy(), n_trials, n_folds)
    bot = trim_epochs(bot.copy(), n_trials, n_folds)
    
    # Make the n_folds top epochs
    top_id = np.arange(len(top))
    shuffle(top_id)
    top_id = np.split(top_id, n_folds)
    
    # Make the n_folds top epochs
    bot_id = np.arange(len(bot))
    shuffle(bot_id)
    bot_id = np.split(bot_id, n_folds)
    
    # Get all the evokes
    for fold in np.arange(n_folds):
        top_evk = top[top_id[fold]].average()
        bot_evk = bot[bot_id[fold]].average()
        evks = [top_evk, bot_evk]
        evk = mne.combine_evoked(evks, weights=[.5,.5])
        evk.comment = f'g{fold}'
        evokeds.append(evk)
        
    return evokeds    
        

# Ask for subject IDs to analyze
# sub_list = get_sub_list(deriv_dir, allow_all=True)
sub_list = [x.name for x in deriv_dir.glob('sub-*') if x.is_dir()]
sub_list = [x for x in sub_list if int(x[-3:])>=200 and x not in ['sub-228','sub-218']]
sub_list.sort()
group_scores = []
for sub_string in sub_list:

    ### SUBJECT INFORMATION DEFINITION ###
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Creating evoked for task-{task} data for {sub_string}')
    print(f'  Derivatives Folder: {deriv_path}')


    ### STEP 1: Load manually cleaned epochs and re-reference 
    # Read in Cleaned Epochs
    epochs_fif_file = deriv_path / f'{sub_string}_task-{task}_ref-mastoids_desc-cleaned_epo.fif.gz'
    if not epochs_fif_file.is_file():
        continue
    epochs = read_epochs(epochs_fif_file, preload=True)
    epochs.drop_channels(['FT9','FT10','TP9','TP10'])
    epochs.filter(None, 6, h_trans_bandwidth=2.5, pad='symmetric')
    # epochs.filter(8, 12, l_trans_bandwidth=1.0, h_trans_bandwidth=1.0, pad='symmetric')
    # epochs.apply_hilbert(envelope=True)
    # epochs._data = epochs._data**2
    epochs.resample(50)
    # epochs.crop(tmin=-.1, tmax=.6)
    times = epochs.times

    ### STEP 2: Get separate epochs objects for each conditoin
    # Get for each condition
    a1_top = epochs['top/stan/a1']['correct==1']
    a1_bot = epochs['bot/stan/a1']['correct==1']
    a2_top = epochs['top/stan/a2']['correct==1']
    a2_bot = epochs['bot/stan/a2']['correct==1']
    a3_top = epochs['top/stan/a3']['correct==1']
    a3_bot = epochs['bot/stan/a3']['correct==1']
    a4_top = epochs['top/stan/a4']['correct==1']
    a4_bot = epochs['bot/stan/a4']['correct==1']
    a5_top = epochs['top/stan/a5']['correct==1']
    a5_bot = epochs['bot/stan/a5']['correct==1']
    a6_top = epochs['top/stan/a6']['correct==1']
    a6_bot = epochs['bot/stan/a6']['correct==1']
    a7_top = epochs['top/stan/a7']['correct==1']
    a7_bot = epochs['bot/stan/a7']['correct==1']
    a8_top = epochs['top/stan/a8']['correct==1']
    a8_bot = epochs['bot/stan/a8']['correct==1']

    # Find minimum length of trials needed per bin
    bins = [f'a{x+1}' for x in np.arange(8)]
    locs = ['top', 'bot']
    conds = itertools.product(bins, locs)
    n_total_epochs = [eval(f'len({x}_{y})') for x, y in conds]
    n_epochs_per_bin = int(np.floor(np.min(n_total_epochs) / n_folds))

    ### Step 3: Loop over n_sims
    all_scores = []
    for sim in np.arange(n_sims):
        
        # Get new bins
        a1 = epochs_2_evokeds(a1_top.copy(), a1_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a2 = epochs_2_evokeds(a2_top.copy(), a2_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a3 = epochs_2_evokeds(a3_top.copy(), a3_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a4 = epochs_2_evokeds(a4_top.copy(), a4_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a5 = epochs_2_evokeds(a5_top.copy(), a5_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a6 = epochs_2_evokeds(a6_top.copy(), a6_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a7 = epochs_2_evokeds(a7_top.copy(), a7_bot.copy(), 
                                n_epochs_per_bin, n_folds)
        a8 = epochs_2_evokeds(a8_top.copy(), a8_bot.copy(), 
                                n_epochs_per_bin, n_folds)

        # Make X and y
        all = [x.data for x in a1] + \
            [x.data for x in a2] + \
            [x.data for x in a3] + \
            [x.data for x in a4] + \
            [x.data for x in a5] + \
            [x.data for x in a6] + \
            [x.data for x in a7] + \
            [x.data for x in a8]
        X = np.array(all)
        y = [1 for x in a1] + \
            [2 for x in a2] + \
            [3 for x in a3] + \
            [4 for x in a4] + \
            [5 for x in a5] + \
            [6 for x in a6] + \
            [7 for x in a7] + \
            [8 for x in a8]

        # Make pipeline
        svc = LinearSVC(penalty='l2', C=1.0, max_iter=10000, random_state=1)
        # ecoc = OutputCodeClassifier(svc, code_size=.5, random_state=1)
        # # ecoc = OutputCodeClassifier(svc, code_size=2.0, randome_state=1)
        clf = make_pipeline(StandardScaler(), 
                            OutputCodeClassifier(svc, code_size=2.0, random_state=1))
        # clf = make_pipeline(StandardScaler(),
        #                     LinearSVC(penalty='l2', C=1.0, multi_class='ovr',
        #                         max_iter=10000))
        # # clf = make_pipeline(StandardScaler(with_std=False),
        #                     LinearSVC(penalty='l2', C=1.0, multi_class='ovr',
        #                         max_iter=10000))
        
        
        # Make X and y vectors
        time_decod = SlidingEstimator(clf, n_jobs=1, verbose=False)
        scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=1)
        all_scores.append(scores)
    
    # Subject scores 
    group_scores.append(np.mean(all_scores, axis=(0,1)))  
    del epochs      
    
    
# # Plot
scores = np.mean(group_scores, axis=0)
fig, ax = plt.subplots()
ax.plot(times, scores, label='score')
ax.axhline(.125, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('Accuracy')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.ylim(bottom=0,top=.20)
plt.savefig('older_adults.png', format='png')


# a1 = mne.combine_evoked([a1_top.average(), a1_bot.average()], weights=[.5,.5])
# a2 = mne.combine_evoked([a2_top.average(), a2_bot.average()], weights=[.5,.5])
# a3 = mne.combine_evoked([a3_top.average(), a3_bot.average()], weights=[.5,.5])
# a4 = mne.combine_evoked([a4_top.average(), a4_bot.average()], weights=[.5,.5])
# a5 = mne.combine_evoked([a5_top.average(), a5_bot.average()], weights=[.5,.5])
# a6 = mne.combine_evoked([a6_top.average(), a6_bot.average()], weights=[.5,.5])
# a7 = mne.combine_evoked([a7_top.average(), a7_bot.average()], weights=[.5,.5])
# a8 = mne.combine_evoked([a8_top.average(), a8_bot.average()], weights=[.5,.5])


# topo_times=np.arange(.05, .65, .05)
# topo_avg = .05
# min, max = -12, 12
# fig, ax = plt.subplots(8,12)
# a1.plot_topomap(axes=ax[0], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a2.plot_topomap(axes=ax[1], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a3.plot_topomap(axes=ax[2], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a4.plot_topomap(axes=ax[3], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a5.plot_topomap(axes=ax[4], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a6.plot_topomap(axes=ax[5], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a7.plot_topomap(axes=ax[6], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)
# a8.plot_topomap(axes=ax[7], times=topo_times, average=topo_avg, colorbar=False, vmin=min, vmax=max)

# fig, ax = plt.subplots(1,8)
# a1.plot_topomap(axes=ax[0], times=[.08], average=.04, colorbar=False)
# a2.plot_topomap(axes=ax[1], times=[.08], average=.04, colorbar=False)
# a3.plot_topomap(axes=ax[2], times=[.08], average=.04, colorbar=False)
# a4.plot_topomap(axes=ax[3], times=[.08], average=.04, colorbar=False)
# a5.plot_topomap(axes=ax[4], times=[.08], average=.04, colorbar=False)
# a6.plot_topomap(axes=ax[5], times=[.08], average=.04, colorbar=False)
# a7.plot_topomap(axes=ax[6], times=[.08], average=.04, colorbar=False)
# a8.plot_topomap(axes=ax[7], times=[.08], average=.04, colorbar=False)