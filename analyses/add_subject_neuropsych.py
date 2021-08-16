from pathlib import Path
import platform

import pandas as pd
 
# Determine path to data base on OS
my_os = platform.system()
if my_os == 'Darwin':
    server_dir = Path('/Volumes/koendata')
elif my_os == 'Linux':
    server_dir = Path('/koenlab/koendata')
else:
    server_dir = Path('X:')
nd3_dir = server_dir / 'EXPT' / 'nd003' / 'exp2' / 'data'
    
bids_dir = nd3_dir / 'bids'

# Load neuropsych file
neuro_file = nd3_dir / 'analyses' / 'neuropsych' / \
    'neuropsych_aggregate_data.tsv'
neuro_df = pd.read_csv(neuro_file, sep='\t')

for i, row in neuro_df.iterrows():
    
    sub = row['id']
    print(sub)
    this_file = bids_dir / sub / 'beh' / f'{sub}_task-neuropsych_beh.tsv'
    row.to_frame().transpose().to_csv(this_file, sep='\t', index=False)



