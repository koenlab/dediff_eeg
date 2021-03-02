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
    
# Load in the neuropsych database
npsych_file = server_dir / 'Neuropsychs' / 'Neuropsych_Database' / 'neuropsych_scores.xlsx'
npsych_df = pd.read_excel(npsych_file, sheet_name='data', 
                          skiprows=2, usecols='B:BK')
npsych_df.set_index('NEUROPSYCH ID', inplace=True)

# Load linking file
linking_file = nd3_dir / 'linking_neuropsycID_to_nd3ID.xlsx'
linking_df = pd.read_excel(linking_file, sheet_name='Sheet1', 
                            usecols='A:G')
linking_df.set_index('Neuro ID', inplace=True)

# Make new dataframe with neuro information
nd3_neuro_df = linking_df.join(npsych_df)
nd3_neuro_df['nd3 ID'] = nd3_neuro_df['nd3 ID'].str.replace('p3e2s','sub-')
nd3_neuro_file = nd3_dir / 'neuro_data_files.tsv'
nd3_neuro_df.to_csv(nd3_neuro_file, sep='\t', index=False)

from scipy.stats import ttest_ind

ttests = {}
variables = nd3_neuro_df.columns[15:]
for col in variables:
    
    y = nd3_neuro_df.where(nd3_neuro_df['AGE'] < 50).dropna()[col]
    o = nd3_neuro_df.where(nd3_neuro_df['AGE'] > 50).dropna()[col]
    ttests[col] = ttest_ind(y, o)
results_df = pd.DataFrame.from_dict(ttests,orient='Index')
results_df.columns = ['statistic','pvalue']

# Load participants.tsv
# par_file = nd3_dir / 'bids' / 'participants.tsv' 
# par_df = pd.read_csv(par_file, sep='\t')

