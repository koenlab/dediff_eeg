from pathlib import Path
import platform
import numpy as np
from mne.channels import read_custom_montage

import os
import sys
os.chdir(os.path.split(__file__)[0])

# Bad Subjects not included in group analysis
bad_subs = ['sub-216', 'sub-218']

# Define task name
task = 'study'

# List of Known Bad Channels
bad_chans = {
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
    'tmax': 2.0,
    'baseline': (-.4, 0),
    'evoked_tmin': -.4,
    'evoked_tmax': 2.0,
    'evoked_highcutoff': 20.0,
    'ica_highpass': 1,
    'ica_baseline': (None, None)
}

# BELOW IS RATHER FIXED
# Determine Top Directory
# This is platform dependent and retrutns a Path class object
# Get the server directory
my_os = platform.system()
if my_os == 'Darwin':
    server_dir = Path('/Volumes/koendata')
elif my_os == 'Linux':
    server_dir = Path('/koenlab/koendata')
else:
    server_dir = Path('X:')

# data_dir
# This is the top-level data directory folder
data_dir = server_dir / 'EXPT' / 'nd003' / 'exp2' / 'data'

# source_dir
# This is the source_data directory
source_dir = data_dir / 'sourcedata'

# bids_dir
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

# BVEF File
bvef_file = Path('..') / 'brainvision_64.bvef'
bv_montage = read_custom_montage(bvef_file, head_size=.08)

# Define event dictionary
event_dict = {
    'scene': 11,
    'object': 21,
    }
