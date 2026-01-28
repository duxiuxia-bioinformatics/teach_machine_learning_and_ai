# 2025 Fall

import pandas as pd
import os, sys
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# =================================
# plotting parameters
# =================================
fig_width = 5
fig_height = 3
marker_size = 9
plt.rcParams['figure.figsize'] = (fig_width, fig_height)

# =================================
# Add the parent directory to the Python path to import ML_toolbox
# =================================
def get_parent_dir():
    try:
        # Works when running a script
        current_dir = os.path.dirname(__file__)
    except NameError:
        # Works in notebooks / REPL
        current_dir = os.getcwd()
    return os.path.abspath(os.path.join(current_dir, '..'))

parent_dir = get_parent_dir()
sys.path.append(parent_dir)

from ML_toolbox.d_mlr_gradient_descent_class import MLR
# =================================

# =================================
# Parameters
corr_threshold = 0.997
# =================================

def main():
    # =================================================
    # import data and sample metadata
    # =================================================
    # import data
    in_file_name = '../../data/sample_data_for_project_final.csv'
    data_in_df = pd.read_csv(in_file_name, index_col=0, header=0)

    # =================================================
    # linear regression by sklearn
    # =================================================
    corr_df = data_in_df.T.corr()

    tf = corr_df > corr_threshold
    xx = tf.sum().sum()

    tf_upper_no_diagonal = pd.DataFrame(np.triu(tf, k=1), index=tf.index, columns=tf.columns)

    positions = (tf_upper_no_diagonal.stack()[tf_upper_no_diagonal.stack()]).index

    out_file_name = 'positions_needed.pickle'
    file_handle = open(out_file_name, 'wb')
    pickle.dump(positions, file_handle)
    file_handle.close()

    fig, ax = plt.subplots()
    ax.scatter(data_in_df.loc[positions[0][0]], data_in_df.loc[positions[0][1]], color='blue', s=marker_size)
    ax.set_xlabel(positions[0][0])
    ax.set_ylabel(positions[0][1])

    return

if __name__ == "__main__":
    main()