import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

fig_width = 5
fig_height = 3

class MyPCA:

    def __init__(self, n_components):
        self.num_of_components = n_components

    def fit_transform(self, X):

        # get covariance matrix of the data
        covariance_matrix = np.cov(X, rowvar=False)

        # eigendecomposition of the covariance matrix
        w, v = LA.eig(covariance_matrix)

        w = w.real
        v = v.real

        # sort eigenvalues in descending order
        II = w.argsort()[::-1]
        all_eigenvalues = w[II]
        all_eigenvectors = v[:, II]

        # get percent variance
        percent_variance_explained = all_eigenvalues / sum(all_eigenvalues) * 100

        # get scores
        pca_scores = np.matmul(X, all_eigenvectors)

        # collect PCA results
        self.pca_results = {'data_for_pca': X,\
                            'eigenvalues': all_eigenvalues, \
                            'percent_variance_explained': percent_variance_explained,\
                            'loadings': all_eigenvectors,\
                            'scores': pca_scores}

        return self.pca_results

    def plot_results(self):
        # scree plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.scatter(range(len(self.pca_results['percent_variance_explained'])), \
                   self.pca_results['percent_variance_explained'],
                   color='blue')
        ax.set_title('scree plot')
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        ax.set_ylim((-10.0, 110.0))
        # fig.show()


        # scores plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.scatter(self.pca_results['scores'][:, 0], self.pca_results['scores'][:, 1], color='blue')
        ax.set_title('scores plot')
        ax.set_xlabel('PC1 (' + str(round(self.pca_results['percent_variance_explained'][0])) + '%)')
        ax.set_ylabel('PC2 (' + str(round(self.pca_results['percent_variance_explained'][1])) + '%)')
        # fig.show()

        # loadings plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.scatter(self.pca_results['loadings'][:, 0], self.pca_results['loadings'][:, 1], color='blue')
        ax.set_title('loadings plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        for i in range(self.pca_results['loadings'].shape[0]):
            ax.text(self.pca_results['loadings'][i, 0], self.pca_results['loadings'][i, 1], 'x' + str(i + 1))
        # fig.show()
