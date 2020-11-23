from pathlib import Path
import platform
import numpy as np
from mne.channels import read_custom_montage

### THESE ARE THINGS I CAN CHANGE AND UPDATE ###
### Define task name ###
task = 'monster'

### List of Known Bad Channels ###
bad_chans = {
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
    'tmin': -.5,
    'tmax': 1.0,
    'baseline': (-.2, 0),
    'evoked_tmin': -.2,
    'evoked_tmax': .8, 
    'evoked_highcutoff': 20.0, 
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

### Analysis Directory
analysis_dir = data_dir / 'analyses' / f'task-{task}'
analysis_dir.mkdir(parents=True, exist_ok=True)

### Report directory ###
report_dir = deriv_dir / 'reports'
report_dir.mkdir(parents=True, exist_ok=True)

### BVEF File
bvef_file = data_dir / 'scripts' / 'brainvision_64.bvef'
bv_montage = read_custom_montage(bvef_file, head_size=.08)

### Define event dictionary ###
event_dict = {
    'bot/stan/a1': 10,
    'bot/stan/a2': 11,
    'bot/stan/a3': 12,
    'bot/stan/a4': 13,
    'bot/stan/a5': 14,
    'bot/stan/a6': 15,
    'bot/stan/a7': 16,
    'bot/stan/a8': 17,
    'bot/odd/a1': 20,
    'bot/odd/a2': 21,
    'bot/odd/a3': 22,
    'bot/odd/a4': 23,
    'bot/odd/a5': 24,
    'bot/odd/a6': 25,
    'bot/odd/a7': 26,
    'bot/odd/a8': 27,
    'top/stan/a1': 110,
    'top/stan/a2': 111,
    'top/stan/a3': 112,
    'top/stan/a4': 113,
    'top/stan/a5': 114,
    'top/stan/a6': 115,
    'top/stan/a7': 116,
    'top/stan/a8': 117,
    'top/odd/a1': 120,
    'top/odd/a2': 121,
    'top/odd/a3': 122,
    'top/odd/a4': 123,
    'top/odd/a5': 124,
    'top/odd/a6': 125,
    'top/odd/a7': 126,
    'top/odd/a8': 127
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
