"""
Script: study_01_compute_behavioral_data.py
Creator: Joshua D. Koen
Description: This script analyzes the behavioral data for the
Memory (study) task. 
"""

### Import needed libraries
import matlab
from matlab.engine import start_matlab
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from study_config import (bids_dir, deriv_dir, task)

# Start matlab engine
eng = start_matlab()
eng.addpath(eng.genpath('/opt/matlab_software/roc_toolbox'), nargout=0)

# Make matlab commands
gen_pars = '[x0,lb,ub]=gen_pars(model,nBins,nConds,parNames);'
fit_roc  = "roc_data=roc_solver(targf,luref,model,fitStat,x0,lb,ub,'figure',false)"
save_roc = "save(save_file,'roc_data')"

### Overwrite 
overwrite = True

#####---Get Subject List---#####
for sub in bids_dir.glob('sub-*'):

    ### Get subject information
    sub_string = sub.name
    sub_id = sub_string.replace('sub-', '')
    out_path = deriv_dir / sub_string 
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Make figure
    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches([11,8.5])
    fig.suptitle(sub_string, fontsize=18)
    
    ## Load the behavioral data file
    beh_file = bids_dir / sub_string / 'beh' / f'{sub_string}_task-test_beh.tsv'
    beh_data = pd.read_csv(beh_file, sep='\t')

    # Check if participant has been done and, unless I want to overwrite, skip
    trial_file = out_path / f'{sub_string}_task-{task}_desc-triallevel_beh.tsv'
    if trial_file.is_file() and not overwrite:
        continue
    
    ### Trial Counts - Test
    # Split old and new items
    group_by = ['category','item_type']
    query = '((item_type=="old" and study_n_responses==1) or (item_type=="new")) and test_n_responses==1'
    
    # Calculate trial counts 
    counts = beh_data.query(query).groupby(group_by)['test_resp'].value_counts(sort=False)
    counts = counts.unstack('test_resp')
    counts = counts.iloc[:, ::-1] # Reverse column order
    counts.columns = [str(x) for x in np.arange(6,0,-1)]
    counts.fillna(0.0, inplace=True)
    counts.sort_index(level='category',ascending=False, inplace=True)
    assert(counts.shape[1]==6)
    
    # Make the table for trial counts
    colLabels = [f'{x}' for x in counts.columns.to_flat_index()]
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('Trial Counts',fontsize=18, fontweight='bold')
    rowLabels = [f'{x[0]}-{x[1]}' for x in counts.index.to_flat_index()]
    plt.table(cellText=counts.values, colLabels=colLabels, rowLabels=rowLabels,
             loc='upper center')
    ax1.axis('off')
    
    # Make targf and luref matrices
    idx = pd.IndexSlice
    targf = counts.loc[idx[:,'old'], :]
    luref = counts.loc[idx[:,'new'], :]

    # Make and update matlab workspace
    matlab_vars = {
        'model': 'dpsd',
        'nBins': matlab.double([targf.shape[1]]),
        'nConds': matlab.double([targf.shape[0]]), 
        'parNames': ['Ro','F'],
        'fitStat': '-LL',
        'targf': matlab.double(targf.values.astype(int).tolist()),
        'luref': matlab.double(luref.values.astype(int).tolist()),
    }
    for k,v in matlab_vars.items():
        eng.workspace[k]=v

    # Run the ROC Solver
    eng.eval(gen_pars, nargout=0)
    eng.eval(fit_roc, nargout=0)
    roc_file = str(out_path / f'{sub_string}_task-test_roc_data.mat')
    eng.save(roc_file, nargout=0)
    
    # Get a dictionary of the roc_data structure
    roc_data = eng.workspace['roc_data']
    pickle_file = out_path / f'{sub_string}_task-test_roc_data.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(roc_data, f)
    
    # Extract needed data
    hits = np.asarray(roc_data['observed_data']['accuracy_measures']['HR'])
    fas = np.asarray(roc_data['observed_data']['accuracy_measures']['FAR'])
    pr = hits-fas
    Ro = np.asarray(roc_data['dpsd_model']['parameters']['Ro'])
    F = np.asarray(roc_data['dpsd_model']['parameters']['F'])
    c = np.asarray(roc_data['dpsd_model']['parameters']['criterion'])
    
    # Compute overal pR
    overall_hit = (targf.values.sum(axis=0) / targf.values.sum()).cumsum()[2]
    overall_fa  = (luref.values.sum(axis=0) / luref.values.sum()).cumsum()[2]
    overall_pr  = overall_hit-overall_fa
    
    ## MEDIAN RT MEASURES FOR STUDY ITEMS
    
    query = 'item_type=="old" and study_n_responses==1 and test_n_responses==1'
    
    # Compute RT measures
    median_rts = beh_data.query('correct==1 and n_resp==1').groupby(group_by[0:2])['rt'].median().to_frame()
    mean_rts = beh_data.query('correct==1 and n_resp==1').groupby(group_by[0:2])['rt'].mean().to_frame()
    sd_rts = beh_data.query('correct==1 and n_resp==1').groupby(group_by[0:2])['rt'].std().to_frame()
    
    # Plot median RT
    ax3 = plt.subplot(2,2,4)
    median_rts.unstack('gabor_loc').plot(kind='bar', ax=ax3, legend=False)
    ax3.tick_params(axis='x', rotation=0)
    ax3.set_title('Median RT', y=1.20, fontsize=18, fontweight='bold')
    ax3.set_ylabel('RT (sec)', fontsize=16)
    ax3.set_xlabel('Trial Type', fontsize=16)
    
    # Save plot to figures
    fig_path = deriv_dir / sub_string / 'figures'
    fig_path.mkdir(parents=True, exist_ok=True)    
    fig_file = fig_path / f'{sub_string}_task-{task}_beh_performance.png'
    fig.savefig(fig_file, dpi=600)
        
    # Make Data frame to store output
    # Make dictionary and basic values
    out_dict = OrderedDict()
    out_dict = {
        'id': [sub_id], 
        'age': ['young' if int(sub_id) < 200 else 'older'],
        'set': [beh_data.iloc[0]['stim_set']]
    }
    
    # Make combination of names 
    gabors = ['top', 'bottom']
    trial_types = ['standard', 'oddball']
    conditions = list()
    for trial_type in trial_types:
        for gabor in gabors:
            conditions.append([trial_type, gabor])
            
    # Add in no resposnes
    for cond in conditions:
        this_key = f'{cond[0]}_{cond[1]}_p_no_resp'
        this_query = f'letter_type=="{cond[0]}" and gabor_loc=="{cond[1]}"'
        out_dict[this_key] = (beh_data.query(this_query)['n_resp']==0).mean()
        
    # Add in accuracy values
    for cond in conditions:
        this_key = f'{cond[0]}_{cond[1]}_acc'
        out_dict[this_key] = [proportions.loc[cond[0]][cond[1]]]
    
    for val in ['median', 'mean', 'sd']:
        for cond in conditions:
            this_key = f'{cond[0]}_{cond[1]}_{val}_rt'
            out_dict[this_key] = eval(f'[{val}_rts.loc[("{cond[1]}", "{cond[0]}")]["rt"]]')
    # Write summary data file
    summary_file = out_path / f'{sub_string}_task-{task}_desc-summarystats_beh.tsv'
    pd.DataFrame().from_dict(out_dict).to_csv(summary_file, index=False, sep='\t')    
    
    # Write copy of behavior data to derivatives
    trial_file = out_path / f'{sub_string}_task-{task}_desc-triallevel_beh.tsv'
    beh_data.to_csv(trial_file, sep='\t', index=False)
    
    
    
