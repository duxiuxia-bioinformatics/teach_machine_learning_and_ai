import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn import datasets

from numpy import linalg as LA

src_dir = '/Users/xdu4/Documents/Duxiuxia/source_code_repository/github_duxiuxia-bioinformatics/teach_machine_learning'
os.chdir(src_dir)

from ML_toolbox.d_PCA import MyPCA
import config

def main():

    # -----------------------------------------------------------------------------------------
    # data file 1
    # -----------------------------------------------------------------------------------------
    np.random.seed(5)

    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names
    y = iris.target
    target_names = iris.target_names

    # examine the data
    # plot the data in 2D
    fig = plt.figure(figsize=(config.fig_width, config.fig_height))
    ax = fig.add_subplot(2, 1, 1)
    II = (y==0)
    ax.scatter(X[II,0], X[II, 1], color='blue')
    II = (y==1)
    ax.scatter(X[II,0], X[II, 1], color='red')
    II = (y==2)
    ax.scatter(X[II,0], X[II, 1], color='green')
    ax.set_title('iris data sepal information')
    ax.set_xlabel('length')
    ax.set_ylabel('width')

    ax = fig.add_subplot(2, 1, 2)
    II = (y==0)
    ax.scatter(X[II,2], X[II, 3], color='blue')
    II = (y==1)
    ax.scatter(X[II,2], X[II, 3], color='red')
    II = (y==2)
    ax.scatter(X[II,2], X[II, 3], color='green')
    ax.set_title('iris data petal information')
    ax.set_xlabel('length')
    ax.set_ylabel('width')
    fig.show()

    # plot the data in 3D
    fig = plt.figure(figsize=(config.fig_width, config.fig_height))
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax = fig.add_subplot(111, projection='3d')
    flower_name_and_label = [('Setosa', 0),
                             ('Versicolor', 1),
                             ('Virginiza', 2)]

    for name, label in flower_name_and_label:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean(),
                  X[y == label, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float64)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k', s=config.marker_size)

    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    # ax.dist = 12

    fig.show()

    # kmeans clustering
    estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
                  ('k_means_iris_3', KMeans(n_clusters=3)),
                  ('k_means_iris_random_init_1', KMeans(n_clusters=3, n_init=1, init='random')),
                  ('k_means_iris_random_init_2', KMeans(n_clusters=3, n_init=20, init='random')),
                  ('k_means_iris_random_k-means++', KMeans(n_clusters=3, n_init=20, init='k-means++'))]
    # n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.

    titles = ['8 clusters',
              '3 clusters',
              '3 clusters with random initialization 1',
              '3 clusters with random initialization 2',
              '3 clusters with k-means++ initialization']

    #for name, est in estimators:
    for i in range(len(estimators)):
        name = estimators[i][0]
        est = estimators[i][1]
        fig = plt.figure(figsize=(config.fig_width, config.fig_height))
        ax = fig.add_subplot(111, projection='3d')
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                   c=labels.astype(np.float64), edgecolor='k', s=config.marker_size)

        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[i])
        # ax.dist = 18

        fig.show()

    # kmeans and PCA
    KMeans_result_iris = KMeans(n_clusters=3, random_state=0).fit(X)
    KMeans_result_iris.labels_
    print(KMeans_result_iris.labels_)

    object_pca = MyPCA(num_of_components=2)
    pca_results = object_pca.fit_transform(x=X, corr_logic=True)

    fig = plt.figure(figsize=(config.fig_width, config.fig_height))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pca_results['scores'][0:49, 0], pca_results['scores'][0:49, 1], color='blue')
    ax.scatter(pca_results['scores'][50:99, 0], pca_results['scores'][50:99, 1], color='red')
    ax.scatter(pca_results['scores'][100:149, 0], pca_results['scores'][100:149, 1], color='green')
    ax.set_title("PCA scores plot")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.show()

    PCA_KMeans_result = KMeans(n_clusters=3, random_state=0).fit(pca_results['scores'][:, 0:2])
    print("\nKmeans cluster labels after PCA for Iris data:")
    print(PCA_KMeans_result.labels_)

    # -----------------------------------------------------------------------------------------
    # data file 2
    # -----------------------------------------------------------------------------------------
    inFileName = "./data/SCLC_study_output_filtered_2.csv"
    dataIn = pd.read_csv(inFileName, header=0, index_col=0)

    KMeans_result = KMeans(n_clusters=2, random_state=0).fit(dataIn)
    print("\nKmeans cluster labels for cell line data:")
    print(KMeans_result.labels_)
    print("\nKmeans cluster centers for cell line data:")
    print(KMeans_result.cluster_centers_)

    k_means_result = k_means(dataIn, n_clusters=2, init='k-means++')

if __name__ == '__main__':
    main()