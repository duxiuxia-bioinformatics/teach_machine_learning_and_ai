# 2025 Fall

import pandas as pd
import os, sys
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

from ML_toolbox.d_PCA import MyPCA
# =================================


def main():
    # =================================================
    # import data and sample metadata
    # =================================================
    # import data
    in_file_name = '../../data/sample_data_for_project_final.csv'
    data_in_df = pd.read_csv(in_file_name, index_col=0, header=0)
    data_in_df = data_in_df.T

    # import sample metadata
    in_file_name = '../../data/sample_metadata_for_project_final.csv'
    sample_metadata_df = pd.read_csv(in_file_name, index_col=0, header=0)

    # =================================================
    # Standardize the data
    # =================================================
    standard_scaler = StandardScaler()
    data_standardized = standard_scaler.fit_transform(data_in_df)

    # =================================================
    # Do PCA analysis using my class
    # =================================================
    my_pca_obj = MyPCA(n_components=2)
    my_pca_results = my_pca_obj.fit_transform(data_standardized)

    out_file_name = 'my_pca_results.pickle'
    with open(out_file_name, 'wb') as file_handle:
        pickle.dump(my_pca_results, file_handle)
    file_handle.close()

    my_pca_obj.plot_results()
    print('My PCA percent variance explained:', my_pca_results.percent_variance_explained)

    my_pca_scores = my_pca_results['scores'][:, :2]
    my_pca_scores_df = pd.DataFrame(data=my_pca_scores, index=sample_metadata_df.index, columns=['PC1', 'PC2'])

    # get indices for type 0 samples and type 1 samples
    tf_0 = (sample_metadata_df['sample_label'] == 0)
    tf_1 = (sample_metadata_df['sample_label'] == 1)

    # make the scores plot
    fig, ax = plt.subplots()
    ax.scatter(my_pca_scores_df.loc[tf_0, 'PC1'], my_pca_scores_df.loc[tf_0, 'PC2'], s=marker_size, color='blue', label='0')
    ax.scatter(my_pca_scores_df.loc[tf_1, 'PC1'], my_pca_scores_df.loc[tf_1, 'PC2'], s=marker_size, color='red', label='1')
    ax.legend(loc='upper right')
    fig.show()


    # =================================================
    # Do PCA analysis using sklearn
    # =================================================
    sklearn_pca_obj = PCA(n_components=2)
    sklearn_pca_obj.fit(data_standardized)
    sklearn_scores = sklearn_pca_obj.fit_transform(data_standardized)
    sklearn_scores_df = pd.DataFrame(sklearn_scores, columns=['PC1', 'PC2'], index=data_in_df.index)

    fig, ax = plt.subplots()
    ax.scatter(sklearn_scores_df.loc[tf_0, 'PC1'], sklearn_scores_df.loc[tf_0, 'PC2'], s=marker_size, color='blue', label='0')
    ax.scatter(sklearn_scores_df.loc[tf_1, 'PC1'], sklearn_scores_df.loc[tf_1, 'PC2'], s=marker_size, color='red', label='1')
    ax.legend(loc='upper right')
    fig.show()

    print('sklearn PCA explained variance ratio:', sklearn_pca_obj.explained_variance_ratio_)

    return

if __name__ == '__main__':
    main()