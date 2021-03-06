import numpy as np
import pandas as pd

from sof_config import (analysis_dir, deriv_dir, task, get_sub_list,
                        bad_subs)

# Make an empty list
all_dfs = []

# Define the subjects
sub_list = get_sub_list(deriv_dir, allow_all=True)

# Remove subjects from list
for bad_sub in bad_subs:
    sub_list.remove(bad_sub) 

# Loop over subjects)
for sub_string in sub_list:
    
    # Define the Subject ID and paths
    deriv_path = deriv_dir / sub_string
    print(f'Loading task-{task} data for {sub_string}')
    
    # Write summary data file
    summary_file = deriv_path / f'{sub_string}_task-{task}_desc-summarystats_beh.tsv'
    all_dfs.append(pd.read_csv(summary_file, sep='\t'))

# Merge into group df
group_df = pd.concat(all_dfs)

# Save the data from
out_path = analysis_dir / 'behavior'
out_path.mkdir(parents=True, exist_ok=True)
out_file = out_path / f'task-{task}_group_behavioral_data.tsv'
group_df.to_csv(out_file, sep='\t', index=False)