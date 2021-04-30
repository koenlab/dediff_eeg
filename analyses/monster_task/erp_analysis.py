#!/usr/bin/env python
# coding: utf-8

# ### MONSTER Task Analysis
# 
# This notebook conducts some preliminary analyses on the P3 and C1 wave analyses across subjects. 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'qt')

# Import libraries
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# from mne import (read_evokeds, grand_average)
import mne

# Setup analysis options
task = 'monster'

# Setup directory paths
my_os = platform.system()
if my_os == 'Darwin':
    server_dir = Path('/Volumes/koendata')
elif my_os == 'Linux':
    server_dir = Path('/koenlab/koendata')
else:
    server_dir = Path('X:')
data_dir = server_dir / 'EXPT' / 'nd003' / 'exp2' / 'data'
deriv_dir = data_dir / 'derivatives' / f'task-{task}'
analysis_dir = data_dir / 'analyses' / f'task-{task}' / 'erps'

# Gather subject ids
sub_list = [sub.name for sub in deriv_dir.glob('sub-*')]
bad_ids = ['sub-218', 'sub-228', 'sub-230', 'sub-231', 'sub-233', 'sub-235', 'sub-236']
sub_list = [sub for sub in sub_list if sub not in bad_ids]
sub_list.sort()


# In[1]:



# Reference and Channels to Drop
drop_chans = ['TP9', 'TP10', 'FT9', 'FT10']

# Conditions
conds = ['standard', 'oddball', 'top', 'bottom']

# Print Info
print('Subjects in the analysis:', sub_list)
    
# Load filtered EVOKED data and extract for each condition
young_evokeds = dict(
    standard=[],
    oddball=[],
    top=[],
    bottom=[]
)
older_evokeds = dict(
    standard=[],
    oddball=[],
    top=[],
    bottom=[]
)
for sub in sub_list:
    
    # Load in evoked object (this is a list)
    evoked_file = deriv_dir / sub / f'{sub}_task-{task}_ref-mastoids_lpf-none_ave.fif.gz'
    evokeds = mne.read_evokeds(evoked_file, verbose=False)
    evokeds = [evoked.drop_channels(drop_chans) for evoked in evokeds]
    
    # Load json to get object list
    json_file = deriv_dir / sub / f'{sub}_task-{task}_ref-mastoids_lpf-20_ave.json'
    with open(json_file, 'r') as f:
        info = json.load(f)
    
    # Add to all_evokeds
    for cond in conds:
        if int(sub[-3:]) < 200:
            young_evokeds[cond].append(evokeds[info['evoked_objects'][cond]])
        else:
            older_evokeds[cond].append(evokeds[info['evoked_objects'][cond]])


# #### P3 Wave
# 
# Here I focus on the P3 Wave. First, I make the grand average ERPs

# In[5]:



# Gather the standard and oddball conditions for young adults and plot grand average
young_standard  = young_evokeds['standard']
young_oddball = young_evokeds['oddball']
young_diff   = [mne.combine_evoked([s, o], weights=[-1,1]) for s, o in zip(young_standard, young_oddball)]

# Make grand averages`
young_grand_standard  = mne.grand_average(young_standard)
young_grand_oddball = mne.grand_average(young_oddball)
young_grands = {
    'Young-Standard': young_grand_standard,
    'Young-Oddball': young_grand_oddball
}
young_grand_diff   = mne.grand_average(young_diff)

# Plot topo of ERP grand averages
mne.viz.plot_compare_evokeds(young_grands, axes='topo', combine=None,
                            title='Young Adults: standard vs. oddballed Grand Average')
fig = young_grand_diff.plot_topomap(times=np.arange(.1,.6,.05), average=.05, 
                              nrows=2, ncols=5, colorbar=True)


# In[ ]:


young_evokeds['standard'][0].info


# In[7]:



# Gather the standard and oddball conditions for older adults and plot grand average
older_standard  = older_evokeds['standard']
older_oddball = older_evokeds['oddball']
older_diff   = [mne.combine_evoked([s, o], weights=[-1,1]) for s, o in zip(older_standard, older_oddball)]

# Make grand averages
older_grand_standard  = mne.grand_average(older_standard)
older_grand_oddball = mne.grand_average(older_oddball)
older_grands = {
    'older-Standard': older_grand_standard,
    'older-Oddball': older_grand_oddball
}
older_grand_diff   = mne.grand_average(older_diff)

# Plot topo of ERP grand averages
mne.viz.plot_compare_evokeds(older_grands, axes='topo', combine=None,
                            title='older Adults: standard vs. oddballed Grand Average')
fig = older_grand_diff.plot_topomap(times=np.arange(.1,.6,.05), average=.05, 
                              nrows=2, ncols=5, colorbar=True)


# In[10]:


grands = dict(Young=young_grand_diff, Older=older_grand_diff)
mne.viz.plot_compare_evokeds(grands, axes='topo', combine=None, 
                            title='Young and Older Adult P3 Difference Waves')
plt.savefig('p3_young_vs_older_topo.png',dpi=600)


# #### C1 Wave
# 
# Below, I analyze the C1 wave

# In[11]:



# Gather the top and bottom conditions for young adults and plot grand average
young_top  = young_evokeds['top']
young_bottom = young_evokeds['bottom']
young_c1_diff   = [mne.combine_evoked([t, b], weights=[1,-1]) for t, b in zip(young_top, young_bottom)]

# Make grand averages`
young_grand_top  = mne.grand_average(young_top)
young_grand_bottom = mne.grand_average(young_bottom)
young_c1_grands = {
    'Top': young_grand_top,
    'Bottom': young_grand_bottom
}
young_c1_grand_diff   = mne.grand_average(young_c1_diff)

# Plot topo of ERP grand averages
fig = mne.viz.plot_compare_evokeds(young_c1_grands, picks=['Pz','Oz','POz','PO3','PO4'], combine='mean',
                            title='Young Adults: top vs. bottomed Grand Average')
# fig = young_c1_grand_diff.plot_topomap(times=np.arange(.04,.2,.02), average=.02, 
#                               nrows=2, ncols=4, colorbar=True)


# In[12]:



# Gather the top and bottom conditions for older adults and plot grand average
older_top  = older_evokeds['top']
older_bottom = older_evokeds['bottom']
older_c1_diff   = [mne.combine_evoked([t, b], weights=[1,-1]) for t, b in zip(older_top, older_bottom)]

# Make grand averages`
older_grand_top  = mne.grand_average(older_top)
older_grand_bottom = mne.grand_average(older_bottom)
older_c1_grands = {
    'Top': older_grand_top,
    'Bottom': older_grand_bottom
}
older_c1_grand_diff   = mne.grand_average(older_c1_diff)

# Plot topo of ERP grand averages
mne.viz.plot_compare_evokeds(older_c1_grands, axes='topo', combine=None,
                            title='older Adults: top vs. bottomed Grand Average')
fig = older_c1_grand_diff.plot_topomap(times=np.arange(.04,.2,.02), average=.02, 
                              nrows=2, ncols=4, colorbar=True)


# In[13]:


tb_colors = dict(Top='#2CA02C', Bottom='#9467BD')
fig = plt.figure()
ax = plt.subplot(2,4,(1,2) )
mne.viz.plot_compare_evokeds(young_c1_grands, picks=['Pz','Oz','POz','PO3','PO4'], combine='mean',
                            title='Young Adults', axes=ax, colors=tb_colors)
plt.title('Young Adults')
plt.legend(loc='lower right')
ax = plt.subplot(2,4,(3,4) )
mne.viz.plot_compare_evokeds(older_c1_grands, picks=['Pz','Oz','POz','PO3','PO4'], combine='mean',
                            title='Older Adults', axes=ax, colors=tb_colors)
plt.title('Older Adults')
plt.legend(loc='lower right')


ax = plt.subplot(2,4,(6,7))
c1_grands = dict(Young=young_c1_grand_diff, Older=older_c1_grand_diff)
mne.viz.plot_compare_evokeds(c1_grands, picks=['Pz','Oz','POz','PO3','PO4'], combine='mean', 
                             title='Young and Older Adult C1 Difference Waves', axes=ax)
plt.title('Top-Bottom Difference Waves')
plt.legend(loc='lower right')

fig.tight_layout(pad=.75)
save_file = analysis_dir / 'c1_cluster_fig.png'
plt.savefig(save_file, dpi=600)


# young_grand_top.copy().pick_channels(['Pz','Oz','POz','PO3','PO4']).plot_sensors(show_names=True)
# plt.title('Sensors')
                            
                             


# In[40]:


young_grand_top.copy().pick_channels(['Pz','Oz','POz','PO3','PO4']).plot_sensors(show_names=True)


# In[ ]:


evokeds[0].plot_sensors


# In[ ]:


c1_grands[0].plot_sensors()

