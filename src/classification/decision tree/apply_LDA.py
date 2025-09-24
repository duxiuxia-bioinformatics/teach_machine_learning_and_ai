# Author: Xiuxia Du
# 2025-03-27

import os
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

data_dir = '/Users/xdu4/Documents/Duxiuxia/data_and_results_for_repo/teaching_machine_learning/data'
result_dir = '/Users/xdu4/Documents/Duxiuxia/data_and_results_for_repo/teaching_machine_learning/result/LDA'

in_file_name = 'sample_data_ready_for_machine_learning.csv'
in_file_full_name = os.path.join(data_dir, in_file_name)
sample_data_df = pd.read_csv(in_file_full_name, index_col=0)

in_file_name = 'sample_factor_ready_for_machine_learning.csv'
in_file_full_name = os.path.join(data_dir, in_file_name)
sample_factor_df = pd.read_csv(in_file_full_name)

# apply_LDA
# remove samples that do not have a sample factor
tf = sample_factor_df['sample_factor'].isnull()

sample_to_remove_list = list(sample_factor_df.loc[tf, 'sample_index'])
sample_to_remove_list_str = [str(x) for x in sample_to_remove_list]

sample_factor_df.drop(index=tf[tf].index, inplace=True)
sample_data_df.drop(columns=sample_to_remove_list_str, inplace=True)

out_file_name = 'sample_data_for_LDA.csv'
out_file_full_name = os.path.join(result_dir, out_file_name)
sample_data_df.to_csv(out_file_full_name, index=False)

out_file_name = 'sample_factor_for_LDA.csv'
out_file_full_name = os.path.join(result_dir, out_file_name)
sample_factor_df.to_csv(out_file_full_name, index=False)

# do LDA
X = sample_data_df.transpose()
Y = sample_factor_df['sample_factor']

sklearn_LDA = LDA(n_components=1)
sklearn_LDA_projection = sklearn_LDA.fit_transform(X, Y)
sklearn_LDA_projection = -sklearn_LDA_projection
sklearn_LDA_projection_df = pd.DataFrame(sklearn_LDA_projection, \
                                         index=sample_factor_df['sample_index'], \
                                         columns=['projection'])
out_file_name = 'projection.csv'
out_file_full_name = os.path.join(result_dir, out_file_name)
sklearn_LDA_projection_df.to_csv(out_file_full_name, index=False)

fig, ax = plt.subplots()
tf = (sample_factor_df['sample_factor'] == 0)
sample_list = list(sample_factor_df.loc[tf, 'sample_index'])
ax.scatter(sklearn_LDA_projection_df.loc[sample_list, 'projection'], \
           np.zeros((len(sample_list), 1)), \
           color='g', s=10)

tf = (sample_factor_df['sample_factor'] == 1)
sample_list = list(sample_factor_df.loc[tf, 'sample_index'])
ax.scatter(sklearn_LDA_projection_df.loc[sample_list, 'projection'], \
           np.ones((len(sample_list), 1)), \
           color='r', s=10)
fig.show()

xx = 1