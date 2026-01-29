# Author: Xiuxia Du
# January 2021

import numpy as np
import d_similarity_measure

def main():
    v1 = np.array([1, 2, 3, 4])

    v2 = np.array([4, 3, 7, 5])

    object_get_similarity = d_similarity_measure.simimarity_measure(v1, v2)
    euclidean_distance = object_get_similarity.get_euclidean()
    cosine_distance = object_get_similarity.get_cosine()

if __name__ == "__main__":
    main()