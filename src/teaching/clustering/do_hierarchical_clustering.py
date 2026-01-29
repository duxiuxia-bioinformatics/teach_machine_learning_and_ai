#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:17:08 2017

@author: xdu4
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn import datasets

from sklearn.decomposition import PCA
import pandas as pd

# ====================================================================
# plotting parameters
# ====================================================================
fig_width = 10
fig_height = 8

marker_size = 14

# ====================================================================
# other parameters
# ====================================================================
np.random.seed(0)

def main():
    # ------------------------------------------
    # iris data
    # ------------------------------------------
    # 1. load the data
    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names
    y = iris.target
    target_names = iris.target_names

    # 2. direct HC
    # HC_model = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean')
    HC_model = AgglomerativeClustering(n_clusters=3, linkage='complete', metric='euclidean')
    HC_model.fit(X)
    print("\nCluster labels from HC for iris data:")
    print(HC_model.labels_)

    # 3. do PCA first and then HC
    PCA_result = PCA(n_components=4).fit_transform(X)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(PCA_result[0:49, 0], PCA_result[0:49, 1], color='blue', s=marker_size)
    ax.scatter(PCA_result[50:99, 0], PCA_result[50:99, 1], color='red', s=marker_size)
    ax.scatter(PCA_result[100:149, 0], PCA_result[100:149, 1], color='green', s=marker_size)
    fig.show()

    # then do hierarchical clustering
    # HC_model_2 = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='euclidean')
    HC_model_2 = AgglomerativeClustering(n_clusters=3, linkage='average', metric='euclidean')
    HC_model_2.fit(PCA_result[:, 0:2])
    print("\nCluster labels from PCA and then HC for iris data:")
    print(HC_model_2.labels_)

    # ------------------------------------------
    # cell line data
    # ------------------------------------------
    inFileName = "../data/SCLC_study_output_filtered_2.csv"
    dataIn = pd.read_csv(inFileName, header=0, index_col=0)

    HC_model_SCLC_1 = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='euclidean')
    HC_model_SCLC_1.fit(dataIn.values)
    print("\ncluster labels from HC with complete linkage and Euclidean distance for cell line data:")
    print(HC_model_SCLC_1.labels_)

    HC_model_SCLC_1 = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
    # ward minimizes the variance of the clusters being merged.
    HC_model_SCLC_1.fit(dataIn.values)
    print("\ncluster labels from HC with ward linkage and Euclidean distance for cell line data:")
    print(HC_model_SCLC_1.labels_)

    HC_model_SCLC_1 = AgglomerativeClustering(n_clusters=2, linkage='average', metric='euclidean')
    HC_model_SCLC_1.fit(dataIn.values)
    print("\ncluster labels from HC with average linkage and Euclidean distance for cell line data:")
    print(HC_model_SCLC_1.labels_)

    PCA_result = PCA(n_components=3).fit_transform(dataIn.values)
    HC_model_SCLC_1 = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
    HC_model_SCLC_1.fit(PCA_result)
    print("\ncluster labels from PCA (3 components) and then HC with ward linkage and Euclidean distance for cell line data:")
    print(HC_model_SCLC_1.labels_)

    PCA_result = PCA(n_components=4).fit_transform(dataIn.values)
    HC_model_SCLC_1 = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
    HC_model_SCLC_1.fit(PCA_result)
    print("\ncluster labels from PCA (4 components) and then HC with ward linkage and Euclidean distance for cell line data:")
    print(HC_model_SCLC_1.labels_)

    PCA_result = PCA(n_components=19).fit_transform(dataIn.values)
    HC_model_SCLC_1 = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
    HC_model_SCLC_1.fit(PCA_result)
    print("\ncluster labels from PCA (19 components) and then HC with ward linkage and Euclidean distance for cell line data:")
    print(HC_model_SCLC_1.labels_)

if __name__ == '__main__':
    main()