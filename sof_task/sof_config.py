from pathlib import Path
import platform

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

### Dictionary of preprocessing options
preprocess_options = {
    'blink_thresh': 150,
    'resample': 250, 
    'lowcutoff': .05, 
    'tmin': -2.0,
    'tmax': 2.0,
    'baseline': (-.2, 0),
    'ica_lowcutoff': 1,
    'ica_tmin': -1.0, 
    'ica_tmax': 1.0,
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

### Define event dictionary ###
event_dict = {
    'scene/novel': 11,
    'scene/1back': 12,
    'object/novel': 21,
    'object/1back': 22,
    'face/novel': 31,
    'face/1back': 32,
}