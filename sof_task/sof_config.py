from pathlib import Path
import platform
import numpy as np
from mne.channels import read_custom_montage

import os
import sys
os.chdir(sys.path[0])

# Bad Subjects not included in group analysis
bad_subs = ['sub-106', 'sub-202', 'sub-218']

# Define task name
task = 'sof'

# List of Known Bad Channels
bad_chans = {
    '115': {
        'channels': ['TP10'],
        'reason': ['excessive line noise']
        }
}

# Autoreject parameters
n_interpolates = np.array([6, 8, 10, 12])
consensus = np.linspace(0.2, 1.0, 9)

# Dictionary of preprocessing options
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

# THESE ARE FIXED VARIABLES TO NOT UPDATE
# This is platform dependent and returns a Path class object
# Get the server directory
my_os = platform.system()
if my_os == 'Darwin':
    server_dir = Path('/Volumes/koendata')
elif my_os == 'Linux':
    server_dir = Path('/koenlab/koendata')
else:
    server_dir = Path('X:')

# This is the top-level data directory folder
data_dir = server_dir / 'EXPT' / 'nd003' / 'exp2' / 'data'

# This is the source_data directory
source_dir = data_dir / 'sourcedata'

# This is the bids formatted output directory
bids_dir = data_dir / 'bids'
bids_dir.mkdir(parents=True, exist_ok=True)

# Derivatives directory
deriv_dir = data_dir / 'derivatives' / f'task-{task}'
deriv_dir.mkdir(parents=True, exist_ok=True)

# Report directory
report_dir = deriv_dir / 'reports'
report_dir.mkdir(parents=True, exist_ok=True)

# Analysis Directory
analysis_dir = data_dir / 'analyses' / f'task-{task}'
analysis_dir.mkdir(parents=True, exist_ok=True)

# BVEF File and Montage
bvef_file = Path('..') / 'brainvision_64.bvef'
bv_montage = read_custom_montage(bvef_file, head_size=.08)

# Define event dictionary
event_dict = {
    'scene/novel': 11,
    'scene/1back': 12,
    'object/novel': 21,
    'object/1back': 22,
    'face/novel': 31,
    'face/1back': 32,
}
