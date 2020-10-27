from pathlib import Path
import platform

### THESE ARE THINGS I CAN CHANGE AND UPDATE ###
### Define task name ###
task = 'study'

### List of Known Bad Channels ###
bad_chans = {
}

### Dictionary of preprocessing options
preprocess_options = {
    'blink_thresh': 150,
    'resample': 250, 
    'lowcutoff': .5, 
    'epoch_tmin': -2.0,
    'epoch_tmax': 2.0
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
    'scene': 11,
    'object': 21,
    }