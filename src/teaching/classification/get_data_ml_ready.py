# Author: Xiuxia Du
# 2025-03-27

import os
import pandas as pd

data_dir = '/Users/xdu4/Documents/Duxiuxia/data_and_results_for_repo/teaching_machine_learning/data'
result_dir = data_dir

data_file_name = 'sample_data.csv'
sample_factor_file_name = 'sample_factor.csv'

# make data ML-ready
in_file_full_name = os.path.join(data_dir, data_file_name)
data_in_df = pd.read_csv(in_file_full_name)

in_file_full_name = os.path.join(data_dir, sample_factor_file_name)
sample_factor_df = pd.read_csv(in_file_full_name)

data_for_analysis_df = data_in_df.copy()
data_for_analysis_df.drop(columns=['name', 'mz', 'rt'], inplace=True)

for cur_sample in data_for_analysis_df.columns:
    tf = sample_factor_df['sample_name'] == cur_sample
    cur_sample_index = sample_factor_df.at[tf[tf].index[0], 'sample_index']

    data_for_analysis_df.rename(columns={cur_sample: cur_sample_index}, inplace=True)

out_file_name = 'sample_data_ready_for_machine_learning.csv'
out_file_full_name = os.path.join(result_dir, out_file_name)
data_for_analysis_df.to_csv(out_file_full_name)

# get the sample metadata file ready
sample_factor_for_analysis_df = sample_factor_df[['sample_index', 'sample_factor']]

out_file_name = 'sample_factor_ready_for_machine_learning.csv'
out_file_name = os.path.join(result_dir, out_file_name)
sample_factor_for_analysis_df.to_csv(out_file_name, index=False)