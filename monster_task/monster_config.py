from pathlib import Path
import platform

### List of Known Bad Channels ###
bad_chans = {
}

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

### Define task name ###
task = 'monster'

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
    'top/odd/a8': 127,
    }



