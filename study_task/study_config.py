from pathlib import Path
import platform
import numpy as np
from mne.channels import read_custom_montage
import scipy.io as spio

### THESE ARE THINGS I CAN CHANGE AND UPDATE ###
### Define task name ###
task = 'study'

### List of Known Bad Channels ###
bad_chans = {
}

### Autoreject parameters
n_interpolates = np.array([1, 2, 4, 8, 12])
consensus = np.linspace(0.4, 1.0, 7)

### Dictionary of preprocessing options
preprocess_options = {
    'blink_thresh': 150e-6,
    'resample': 250, 
    'lowcutoff': .1, 
    'tmin': -1.7,
    'tmax': 2.7,
    'baseline': (-.2, 0),
    'evoked_tmin': -.2,
    'evoked_tmax': .6, 
    'evoked_highcutoff': 20.0, 
    'ica_lowcutoff': 1,
    'ica_tmin': -1.0, 
    'ica_tmax': 2.0,
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

### BVEF File
bvef_file = data_dir / 'scripts' / 'brainvision_64.bvef'
bv_montage = read_custom_montage(bvef_file, head_size=.08)

### Define event dictionary ###
event_dict = {
    'scene': 11,
    'object': 21,
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
