"""
Script: sof_102_make_evokeds.py
Creator: Joshua D. Koen
Description: This script loads cleaned epoch data and makes
evoked objects for conditions of interest. 
"""

# Import libraries
import sys
import os

os.chdir(os.path.split(__file__)[0])
sys.path.append('../../')  # For functions file
sys.path.append('..')  # For config file

import pandas as pd

from sof_config import (analysis_dir, deriv_dir, task, bad_subs)
from functions import (get_sub_list)

# Handle directories
out_dir = analysis_dir / 'fooof'

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    sub_list.remove(bad_sub)

# Ask for subject IDs to analyze
for i, sub in enumerate(sub_list):

    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub
    if int(sub[-3:]) < 200:
        age = 'young'
    else:
        age = 'older'
    print(sub, age)

    # Load FOOOF data
    fooof_file = deriv_path / f'{sub}_task-{task}_fooof.tsv'
    fooof_df = pd.read_csv(fooof_file, sep='\t')
    fooof_df.insert(1, 'age', age)

    # Handle output data frame
    if i == 0:
        df_long = fooof_df.copy()
    else:
        df_long = df_long.append(fooof_df)

# Save the long data frame
out_file = f'task-{task}_aperiodic_by_channel.tsv'
df_long.to_csv(out_dir / out_file, sep='\t', index=False)
