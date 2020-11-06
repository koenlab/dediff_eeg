"""
Script: study_01_compute_behavioral_data.py
Creator: Joshua D. Koen
Description: This script analyzes the behavioral data for the
Memory (study) task. 
"""

### Import needed libraries
import matlab
from matlab.engine import start_matlab

# Start matlab engine
eng = start_matlab()
eng.addpath(eng.genpath('/opt/matlab_software/roc_toolbox'), nargout=0)

# Make matlab commands
gen_pars = '[x0,lb,ub]=gen_pars(model,nBins,nConds,parNames);'
fit_roc  = "roc_data=roc_solver(targf,luref,model,fitStat,x0,lb,ub,'subID',sub,'condNames',conditions,'figure',false)"
save_roc = "save(save_file,'roc_data')"

### NORMAL LIBRARIES
import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict

from study_config import (bids_dir, deriv_dir, task)

### Overwrite 
overwrite = True

# Functions to read in .mat file
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

sub_list = [x.name for x in bids_dir.glob('sub-*')]
sub_list.sort()

#####---Get Subject List---#####
for sub_string in sub_list:

    ### Get subject information
    # sub_string = sub.name
    sub_id = sub_string.replace('sub-', '')
    out_path = deriv_dir / sub_string 
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Make figure
    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches([9,9])
    fig.suptitle(sub_string, fontsize=18)
    
    ## Load the behavioral data file
    beh_file = bids_dir / sub_string / 'beh' / f'{sub_string}_task-test_beh.tsv'
    beh_data = pd.read_csv(beh_file, sep='\t')
    memory_bin = []
    study_acc = []
    for row in beh_data.itertuples():
        # Code memory bin
        if row.item_type=='old' and row.study_n_responses==1 and row.test_n_responses==1:
            if row.test_resp in [5,6]:
                memory_bin.append('hit')
            else:
                memory_bin.append('miss')
        elif row.item_type=='new' and row.test_n_responses==1:
            if row.test_resp in [5,6]:
                memory_bin.append('fa')
            else:
                memory_bin.append('cr')
        else:
            memory_bin.append('n/a')
        
        # Code study accuracy
        if row.item_type=='old' and row.study_n_responses==1 and row.test_n_responses==1:
            if row.subcategory=='natural' and row.study_resp==2:
                study_acc.append(1)
            elif row.subcategory=='manmade' and row.study_resp==1:
                study_acc.append(1)
            else:
                study_acc.append(0)
        else:
            study_acc.append(np.nan)
    
    beh_data['memory_bin'] = memory_bin
    beh_data['study_acc'] = study_acc
    del memory_bin, study_acc
    
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
    ax1 = plt.subplot(421)
    ax1.set_title('Trial Counts',fontsize=18, fontweight='bold')
    rowLabels = [f'{x[0]}-{x[1]}' for x in counts.index.to_flat_index()]
    ax1.table(cellText=counts.values, colLabels=colLabels, rowLabels=rowLabels,
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
        'sub_id': sub_id,
        'conditions': ['scene', 'object']
    }
    for k,v in matlab_vars.items():
        eng.workspace[k]=v

    # Run the ROC Solver, save it, then load it
    eng.eval(gen_pars, nargout=0)
    eng.eval(fit_roc, nargout=0)
    roc_file = str(out_path / f'{sub_string}_task-test_roc_data.mat')
    eng.save(roc_file, nargout=0)
    roc_data = loadmat(roc_file)['roc_data']
    
    # Extract needed data
    hits = roc_data['observed_data']['accuracy_measures']['HR']
    fas = roc_data['observed_data']['accuracy_measures']['FAR']
    pr = hits-fas
    Ro = roc_data['dpsd_model']['parameters']['Ro']
    F = roc_data['dpsd_model']['parameters']['F']
    c = roc_data['dpsd_model']['parameters']['criterion']
    
    # Compute overal pR
    overall_hit = (targf.values.sum(axis=0) / targf.values.sum()).cumsum()[2]
    overall_fa  = (luref.values.sum(axis=0) / luref.values.sum()).cumsum()[2]
    overall_pr  = overall_hit-overall_fa
    
    # Get observed and predicted ROC values
    targ_obs = roc_data['observed_data']['target']['cumulative']
    lure_obs = roc_data['observed_data']['lure']['cumulative']
    targ_pred = roc_data['dpsd_model']['predicted_rocs']['roc']['target']
    lure_pred = roc_data['dpsd_model']['predicted_rocs']['roc']['lure']
    
    # Make ROC plot
    ax2 = plt.subplot(223, aspect='equal')
    ax2.set_xlim((0.0, 1.0))
    ax2.set_xmargin(0)
    ax2.set_xlabel('False Alarm Rate', fontsize=14)
    ax2.set_ylim((0.0, 1.0))    
    ax2.set_ymargin(0)
    ax2.set_ylabel('Hit Rate', fontsize=14)
    colors = ['tab:blue', 'tab:orange']
    for cond, color in enumerate(colors):
        ax2.scatter(lure_obs[cond], targ_obs[cond], c=color).set_clip_on(False)
    ax2.legend(['scene','object'], fontsize=14, loc='upper center', frameon=False,
               mode='expand', ncol=2, bbox_to_anchor=(0,1.05,1 ,.12))
    for cond, color in enumerate(colors):
        ax2.plot(lure_pred[cond], targ_pred[cond], color=color)
    ax2.plot((0,1),(0,1), color='k', linestyle='--', linewidth=1)
    
    # Make the memory parameter plot for Ro
    ax3 = plt.subplot(445)
    Ro_plot = pd.DataFrame(dict(scene=Ro[0], object=Ro[1]), index=['estimate'])
    Ro_plot.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Estimate')
    ax3.set_ylim((0, None))
    ax3.set_title('Ro', fontsize=14)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax3.legend().remove()
    ax3.set_box_aspect(1.25)
    
    # Make the memory parameter plot for Ro
    ax4 = plt.subplot(446)
    F_plot = pd.DataFrame(dict(scene=F[0], object=F[1]), index=['estimate'])
    F_plot.plot(kind='bar', ax=ax4)
    ax4.set_ylim((0, None))
    ax4.set_title("Familiarity (d')", fontsize=14)
    ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax4.legend().remove()
    ax4.set_box_aspect(1.25)
    
    # Plot trial counts in memory bins (
    query = 'item_type=="old" and study_n_responses==1 and test_n_responses==1'
    mem_bin_count = beh_data.query(query).groupby('category')['memory_bin'].value_counts().unstack('category')
    mem_bin_count = mem_bin_count[['scene','object']]
    ax5 = plt.subplot(443)
    mem_bin_count.plot(kind='bar', ax=ax5, legend=False)
    ax5.axhline(20, color='k', linestyle='--')
    ax5.set_xlabel('Memory Bin')
    ax5.set_box_aspect(1.25)
    ax5.tick_params(axis='x', rotation=0)
    ax5.set_title('Memory Bin Counts')
    
    # Plot Recognition Accuracy (Hit-FA)
    ax6 = plt.subplot(444)
    recog_data = pd.DataFrame(np.append(pr,overall_pr), columns=['pR'], index=['scene', 'object', 'overall'] )
    barlist = recog_data.plot(kind='bar', ax=ax6)
    barlist.get_children()[1].set_color(colors[1])
    barlist.get_children()[2].set_color('grey')
    ax6.axhline(.10, color='k', linestyle='--')
    ax6.set_title('Recognition pR')
    ax6.legend().remove()
    ax6.set_box_aspect(1.25)
    ax6.tick_params(axis='x', rotation=45)
    ax6.set_ylim((0,1))
    for i, v in enumerate(recog_data.values):
        x_pos = barlist.get_children()[i]._x0
        ax6.text(x_pos - .125, v+.015, f'{v[0]:.2f}', color='k')
    ax6.legend(barlist.get_children(),['scene', 'object', 'overall'], 
               loc='upper center', mode='expand', ncol=3, 
               bbox_to_anchor=(-4,1.4,4.5,.12), frameon=False,
               fontsize=18)
    
    ## Study Performance Data
    group_by = ['category', 'memory_bin']
    query = 'item_type=="old" and study_n_responses==1 and test_n_responses==1'
    
    # Compute RT measures by subsequent memory
    study_median_rts = beh_data.query(query).groupby(group_by)['study_rt'].median().to_frame()
    study_mean_rts = beh_data.query(query).groupby(group_by)['study_rt'].mean().to_frame()
    study_sd_rts = beh_data.query(query).groupby(group_by)['study_rt'].std().to_frame()
    
    # Plot median RT
    ax7 = plt.subplot(8,8,(29,38))
    study_rt_plot = study_median_rts.unstack('category')
    study_rt_plot = study_rt_plot[[('study_rt','scene'),('study_rt','object')]]
    study_rt_plot.plot(kind='bar', ax=ax7, legend=False)
    ax7.set_xlabel(None)
    ax7.tick_params(axis='x', rotation=0)
    ax7.set_title('Study Median RT')
    ax7.set_ylabel('RT (sec)')
    ax7.set_box_aspect(1.6)
    ax7.set_ylim((study_median_rts.values.min()*.95,study_median_rts.values.max()*1.05))
    ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
   
    # Compute study acc measures
    study_acc = beh_data.query(query).groupby(group_by)['study_acc'].mean().to_frame()
    
    # Plot study ACC measures
    ax8 = plt.subplot(8,8,(31,40))
    study_acc_plot = study_acc.unstack('category')
    study_acc_plot = study_acc_plot[[('study_acc','scene'),('study_acc','object')]]
    study_acc_plot.plot(kind='bar', ax=ax8, legend=False)
    ax8.set_xlabel(None)
    ax8.tick_params(axis='x', rotation=0)
    ax8.set_title('Study Accuracy')
    ax8.set_ylabel('p(Correct)')
    ax8.set_box_aspect(1.6)
    ax8.set_ylim((.5,1.0))
    
    ## Get Test RT
    # Setup query
    group_by = ['category', 'memory_bin']
    query = '((item_type=="old" and study_n_responses==1) or (item_type=="new")) and test_n_responses==1'
    
    # Test RTs 
    test_median_rts = beh_data.query(query).groupby(group_by)['test_rt'].median().to_frame()
    test_mean_rts = beh_data.query(query).groupby(group_by)['test_rt'].mean().to_frame()
    test_sd_rts = beh_data.query(query).groupby(group_by)['test_rt'].std().to_frame()
    
    # Plot test RTs
    ax9 = plt.subplot(7,2,(12,14))
    test_rt_plot = test_median_rts.unstack('category')
    test_rt_plot = test_rt_plot[[('test_rt','scene'),('test_rt','object')]]
    test_rt_plot.plot(kind='bar', ax=ax9, legend=False)
    ax9.set_xlabel('Memory Bin')
    ax9.tick_params(axis='x', rotation=0)
    ax9.set_title('Test Median RT')
    ax9.set_ylabel('RT (sec)')
    ax9.set_ylim(test_rt_plot.values.min()*.6,test_rt_plot.values.max()*1.10)
    ax9.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax9.set_box_aspect(.625)
    
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
    
    # Add Memory performance measures
    out_dict['overall_hit_rate'] = overall_hit
    out_dict['overall_fa_rate'] = overall_fa
    out_dict['overall_pr'] = overall_pr
    out_dict['scene_hit_rate'] = hits[0]
    out_dict['object_hit_rate'] = hits[1]
    out_dict['scene_fa_rate'] = fas[0]
    out_dict['object_fa_rate'] = fas[1]
    out_dict['scene_item_pr'] = pr[0]
    out_dict['object_item_pr'] = pr[1]
    out_dict['scene_recollection'] = Ro[0]
    out_dict['object_recollection'] = Ro[1]
    out_dict['scene_familiarity'] = F[0]
    out_dict['object_familiarity'] = F[1]
    for i1, cond in enumerate(['scene', 'object']):
        for i2, v in enumerate(np.arange(5,0,-1)):
            out_dict[f'{cond}_c{v}'] = c[i1,i2]
    for cond in ['scene','object']:
        for itype in ['old','new']:
            for col in targf.columns:
                out_dict[f'{cond}_{itype}_conf_{col}'] = counts.loc[(cond,itype), col]
    
    # Add Study RT Measurse
    for val in ['median', 'mean', 'sd']:
        this_rt = eval(f'study_{val}_rts')
        for cond in ['scene','object']:
            for mem_bin in ['hit','miss']:
                out_dict[f'{cond}_{mem_bin}_{val}_study_rt'] = this_rt.loc[(cond,mem_bin)]['study_rt']
                
    # Add Test RT Measurse
    for val in ['median', 'mean', 'sd']:
        this_rt = eval(f'test_{val}_rts')
        for cond in ['scene','object']:
            for mem_bin in ['hit','miss','cr','fa']:
                out_dict[f'{cond}_{mem_bin}_{val}_test_rt'] = this_rt.loc[(cond,mem_bin)]['test_rt']
    
    # Write summary data file
    summary_file = out_path / f'{sub_string}_task-{task}_desc-summarystats_beh.tsv'
    pd.DataFrame().from_dict(out_dict).to_csv(summary_file, index=False, sep='\t')    
    
    # Write copy of behavior data to derivatives
    trial_file = out_path / f'{sub_string}_task-{task}_desc-triallevel_beh.tsv'
    beh_data.to_csv(trial_file, sep='\t', index=False)
    
    
    
    
    
