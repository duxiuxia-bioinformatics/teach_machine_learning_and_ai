import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn import datasets

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# =================================
# Add the parent directory to the Python path
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
# =================================

from ML_toolbox.d_PCA import MyPCA

def main():
    data_in = datasets.load_iris()
    X = data_in.data

    # standardize the data
    X_standardized = StandardScaler().fit_transform(X)

    num_of_samples = X_standardized.shape[0]
    num_of_variables = X_standardized.shape[1]

    # do PCA using custom-made PCA
    my_pca_obj = MyPCA(n_components=2)
    pca_results = my_pca_obj.fit_transform(X_standardized)
    my_pca_obj.plot_results()

    # use sklearn PCA
    sklearn_pca_obj = PCA(n_components=min(num_of_samples, num_of_variables))
    sklearn_pca_obj.fit(X_standardized)
    sklearn_pca_scores = sklearn_pca_obj.fit_transform(X_standardized)

    fig, ax = plt.subplots()
    ax.scatter(sklearn_pca_scores[:, 0], sklearn_pca_scores[:, 1], c='blue')
    pc1_percentage = round(sklearn_pca_obj.explained_variance_ratio_[0] * 100, 0)
    pc2_percentage = round(sklearn_pca_obj.explained_variance_ratio_[1] * 100, 0)
    ax.set_xlabel(f'PC 1 ({pc1_percentage}%)')
    ax.set_ylabel(f'PC 2 ({pc2_percentage}%)')
    fig.show()
    return

if __name__ == "__main__":
    main()


