from pathlib import Path
import platform
import numpy as np
from mne.channels import read_custom_montage

### THESE ARE THINGS I CAN CHANGE AND UPDATE ###
### Define task name ###
task = 'sof'

### List of Known Bad Channels ###
bad_chans = {
    '115': {
        'channels': ['TP10'], 
        'reason': ['excessive line noise']
        }
}

### Autoreject parameters
n_interpolates = np.array([6, 8, 10, 12])
consensus = np.linspace(0.2, 1.0, 9)

### Dictionary of preprocessing options
preprocess_options = {
    'blink_thresh': 150e-6,
    'ext_val_thresh': 100e-6,
    'perc_good_chans': .10,
    'resample': 250, 
    'highpass': .1, 
    'tmin': -1.0,
    'tmax': 1.0,
    'baseline': (-.2, 0),
    'evoked_tmin': -.2,
    'evoked_tmax': .6, 
    'evoked_lowpass': 20.0, 
    'ica_highpass': 1,
    'ica_baseline': (None, None)
}

### BELOW IS RATHER FIXED ###
#####---Determine Top Directory---#####
# This is platform dependent and retrutns a Path class object
# Get the server directory
my_os = platform.system()
if my_os == 'Darwin':
    server_dir = Path('/Volumes/koendata')
elif my_os == 'Linux':
    server_dir = Path('/koenlab/koendata')
else:
    server_dir = Path('X:')

### data_dir ###
# This is the top-level data directory folder                   
data_dir = server_dir / 'EXPT' / 'nd003' / 'exp2' / 'data'

### source_dir ###
# This is the source_data directory
source_dir = data_dir / 'sourcedata'

### bids_dir ###
# This is the bids formatted output directory
bids_dir = data_dir / 'bids'
bids_dir.mkdir(parents=True, exist_ok=True)

### Derivatives directory ###
deriv_dir = bids_dir / 'derivatives' / f'task-{task}'
deriv_dir.mkdir(parents=True, exist_ok=True)

### Report directory ###
report_dir = deriv_dir / 'reports'
report_dir.mkdir(parents=True, exist_ok=True)

### BVEF File
bvef_file = data_dir / 'scripts' / 'brainvision_64.bvef'
bv_montage = read_custom_montage(bvef_file, head_size=.08)

### Define event dictionary ###
event_dict = {
    'scene/novel': 11,
    'scene/1back': 12,
    'object/novel': 21,
    'object/1back': 22,
    'face/novel': 31,
    'face/1back': 32,
}

# Define subject list function
def get_sub_list(data_dir, allow_all=False, is_source=False):
    # Ask for subject IDs to analyze
    print('What IDs are being preprocessed?')
    print('(Enter multiple values separated by a comma; e.g., 101,102)')
    if allow_all:
        print('To process all subjects, type all')
    
    sub_list = input('Enter IDs: ')
    
    if sub_list == 'all' and allow_all:
        if is_source:
            sub_list = [x.name for x in data_dir.glob('sub-p3e2s*')]
        else:
            sub_list = [x.name for x in data_dir.glob('sub-*')]
    else:
        sub_list = sub_list.split(',')
        if is_source:
            sub_list = [f'sub-p3e2s{x}' for x in sub_list]
        else:
            sub_list = [f'sub-{x}' for x in sub_list]

    sub_list.sort()
    return sub_list

### HTML GENERATORS

from dominate.tags import *
from mne import pick_types
from collections import OrderedDict
import json

def make_raw_html(sub, raw):
    
    # Initialize table as a list object
    html = li(_class='raw', id=sub)
    html.add(h4(''))
    with html.add(table(_class='table table-hover')):
        
        with tbody():
        
            # Sampling frequency
            with tr():
                th(u'Sampling Freuency')
                td(u'{:0.2f} Hz'.format(raw.info['sfreq']))
            
            # Get lowpass filter
            with tr():
                th(u'Highpass')
                td(u'{:0.2f} Hz'.format(raw.info['highpass']))
            
            # Get lowpass filter
            with tr():
                th(u'Lowpass')
                td(u'{:0.2f} Hz'.format(raw.info['lowpass']))
            
            # Line Noise Frequency
            with tr():
                th(u'Line Frequency')
                td(u'60.00 Hz')
            
            # Add in measurement time
            with tr():
                th(u'Measurement Duration')
                td(u'{:0.2f} seconds'.format(raw._last_time))
            
            # Online Reference
            with tr():
                th(u'Reference Channel')
                td(u'FCz')  
            
            # Number of EEG Channels
            with tr():
                th(u'# of Good EEG Channels')
                td(u'{:d}'.format(len(pick_types(raw.info, eeg=True))))
            
            # Get bad channels
            if raw.info['bads'] is not None:
                bads = ', '.join(raw.info['bads'])
            else:
                bads = 'None'
            with tr():
                th(u'Interpolated Channels')
                td(u'{}'.format(bads))
        
            # EOG Channels
            pick_eog = pick_types(raw.info, meg=False, eog=True)
            if len(pick_eog) > 0:
                eog = ', '.join(np.array(raw.info['ch_names'])[pick_types(raw.info, eog=True)])
            else:
                eog = 'None'
            with tr():
                th(u'EOG Channels')
                td(u'{}'.format(eog))
            
    return html

def make_epochs_html(sub, json_file, epochs):
    
    # Load json data
    with open(json_file, 'r') as f:
        info = json.load(f)
    
    # Initialize table as a list object
    html = div(style='margin-left: 16.666666666666664%')
    with html.add(li(_class=u'raw', id='epochstable')):
        with table(_class=u'table table-hover'):
            with tbody():
            
                # Sampling frequency
                with tr():
                    th(u'Sampling Freuency')
                    td(u'{:0.2f} Hz'.format(info['sfreq']))
                
                # Get lowpass filter
                with tr():
                    th(u'Highpass')
                    td(u'{:0.2f} Hz'.format(epochs.info['highpass']))
                
                # Get lowpass filter
                with tr():
                    th(u'Lowpass')
                    td(u'{:0.2f} Hz'.format(epochs.info['lowpass']))
                
                # Line Noise Frequency
                with tr():
                    th(u'Line Frequency')
                    td(u'60.00 Hz')
                
                # Online Reference
                with tr():
                    th(u'Reference Channel')
                    td(u'{}'.format(info['reference']))
                
                # Get bad channels
                if info['bad_channels'] is not None:
                    bads = ', '.join(info['bad_channels'])
                else:
                    bads = 'None'
                with tr():
                    th(u'Interpolated Channels')
                    td(u'{}'.format(bads))

                # Get number of rejected epochs
                with tr():
                    th(u'# of Epochs Rejected')
                    td(u'{}'.format(len(info['bad_epochs'])))
                
                # Get proportion of bad epochs
                with tr():
                    th(u'Percent Epochs Rejection')
                    td(u'{:0.2f}%'.format(info['proportion_rejected_epochs']*100))
            
    return html