# Author: Xiuxia Du, 10/17/2019

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

index_example = 1
# index_example == 0: toy data
# index_example == 1: iris data
# index_example == 2: cell line data

def main():
    if index_example == 0:
        X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()

        obj_DBSCAN = DBSCAN(eps=6, min_samples=4).fit(X)
        print(obj_DBSCAN.labels_)

    if index_example == 1:
        data_in = datasets.load_iris()
        X = data_in.data

        num_of_samples = X.shape[0]

        distance_list = []
        for i in range(num_of_samples):
            for j in range(i+1, num_of_samples):
                cur_distance = sum((X[i, :] - X[j, :]) * (X[i, :] - X[j, :]))
                distance_list.append(cur_distance)

        plt.hist(distance_list)
        plt.show()

        object_DBSCAN = DBSCAN(eps=1, min_samples=6)
        object_DBSCAN.fit(data_in.data)
        print(object_DBSCAN.labels_)

    elif index_example == 2:
        in_file_name = '../data/SCLC_study_output_filtered_2.csv'
        data_in = pd.read_csv(in_file_name, index_col=0, header=0)
        X = data_in.to_numpy()

        num_of_samples = X.shape[0]

        distance_list = []
        for i in range(num_of_samples):
            for j in range(i+1, num_of_samples):
                cur_distance = sum((X[i, :] - X[j, :]) * (X[i, :] - X[j, :]))
                distance_list.append(cur_distance)

        plt.hist(distance_list)
        plt.show()

        object_DBSCAN = DBSCAN(eps=2e5, min_samples=8)
        object_DBSCAN.fit(data_in)
        print(object_DBSCAN.labels_)

        # use kmeans
        object_kmeans = KMeans(n_clusters=2, random_state=0)
        object_kmeans.fit(X)
        print(object_kmeans.labels_)
    else:
        exit('Unknown index_example!')

if __name__ == '__main__':
    main()