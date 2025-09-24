import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

class d_PCA:

    def __init__(self, num_of_components):
        self.num_of_components = num_of_components

    def fit_transform(self, x, corr_logic=False):
        # column_mean = x.mean(axis=0)
        # column_mean_stacked = np.tile(column_mean, reps=(x.shape[0], 1))
        # x_mean_centered = x - column_mean_stacked

        # use mean_centered data or standardized mean_centered data
        if not corr_logic:
            # data_for_pca = x_mean_centered
            data_for_pca = x
        else:
            # mean centering
            column_mean = x.mean(axis=0)
            column_mean_stacked = np.tile(column_mean, reps=(x.shape[0], 1))
            x_mean_centered = x - column_mean_stacked

            column_sd = np.std(x, axis=0)
            column_sd_stacked = np.tile(column_sd, reps=(x.shape[0], 1))
            data_for_pca = x / column_sd_stacked

        # get covariance matrix of the data
        covariance_matrix = np.cov(data_for_pca, rowvar=False)

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
        pca_scores = np.matmul(data_for_pca, all_eigenvectors)

        # collect PCA results
        self.pca_results = {'raw_data': x,\
                            'data_for_pca': data_for_pca,\
                            'eigenvalues': all_eigenvalues, \
                            'percent_variance_explained': percent_variance_explained,\
                            'loadings': all_eigenvectors,\
                            'scores': pca_scores,\
                            'data_after_pretreatment': data_for_pca}

        return self.pca_results

    def plot_results(self, plot_output_folder):
        # scree plot
        fig, ax = plt.subplots()
        ax.scatter(range(len(self.pca_results['percent_variance_explained'])), \
                   self.pca_results['percent_variance_explained'],
                   color='blue')
        ax.set_title('scree plot')
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        ax.set_ylim((-10.0, 110.0))
        fig.show()
        out_file_name = 'scree_plot.pdf'
        out_file_full_name = plot_output_folder + out_file_name
        fig.savefig(out_file_full_name)

        # scores plot
        fig, ax = plt.subplots()
        ax.scatter(self.pca_results['scores'][:, 0], self.pca_results['scores'][:, 1], color='blue')
        ax.set_title('scores plot')
        ax.set_xlabel('PC1 (' + str(round(self.pca_results['percent_variance_explained'][0])) + '%)')
        ax.set_ylabel('PC2 (' + str(round(self.pca_results['percent_variance_explained'][1])) + '%)')
        fig.show()
        out_file_name = 'scores_plot.pdf'
        out_file_full_name = plot_output_folder + out_file_name
        fig.savefig(out_file_full_name)

        # loadings plot
        fig, ax = plt.subplots()
        ax.scatter(self.pca_results['loadings'][:, 0], self.pca_results['loadings'][:, 1], color='blue')
        ax.set_title('loadings plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        for i in range(self.pca_results['loadings'].shape[0]):
            ax.text(self.pca_results['loadings'][i, 0], self.pca_results['loadings'][i, 1], 'x' + str(i + 1))
        fig.show()
        out_file_name = 'loadings_plot.pdf'
        out_file_full_name = plot_output_folder + out_file_name
        fig.savefig(out_file_full_name)

    # def fit_transform(self, x):
    #     columnMean = x.mean(axis=0)
    #     columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    #     xMeanCentered = x - columnMeanAll
    #
    #     # use mean_centered data or standardized mean_centered data
    #     if not self.corr_logic:
    #         dataForPca = xMeanCentered
    #     else:
    #         columnSD = np.std(x, axis=0)
    #         columnSDAll = np.tile(columnSD, reps=(x.shape[0], 1))
    #         dataForPca = x / columnSDAll
    #
    #     # get covariance matrix of the data
    #     covarianceMatrix = np.cov(dataForPca, rowvar=False)
    #
    #     # eigendecomposition of the covariance matrix
    #     eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    #
    #     eigenValues = eigenValues.real
    #     eigenVectors = eigenVectors.real
    #
    #     # sort eigenvalues in descending order
    #     II = eigenValues.argsort()[::-1]
    #     eigenValues = eigenValues[II]
    #     eigenVectors = eigenVectors[:, II]
    #
    #     # percentage of variance explained by each PC
    #     totalVariance = sum(eigenValues)
    #     percentVariance = np.zeros(len(eigenValues))
    #     for i in range(len(eigenValues)):
    #         percentVariance[i] = eigenValues[i] / totalVariance
    #
    #     # get scores
    #     pcaScores = np.matmul(dataForPca, eigenVectors)
    #
    #     # collect PCA results
    #     pcaResults = {'data': x,
    #                   'mean_centered_data': xMeanCentered,
    #                   'percent_variance': percentVariance,
    #                   'loadings': eigenVectors,
    #                   'scores': pcaScores,
    #                   'data_after_pretreatment': dataForPca}
    #
    #     return pcaResults