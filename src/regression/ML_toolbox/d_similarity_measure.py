# Author: Xiuxia Du
# January 2021

import numpy as np
from numpy import linalg as LA

class simimarity_measure:
    # def __init__(self):
    #     xx = 1

    def get_euclidean(self, x1, x2):
        d = LA.norm(x1 - x2, ord=2)
        return d

    def get_cosine(self, x1, x2):
        d = 1.0 - np.dot(x1, x2) / (LA.norm(x1, ord=2) * LA.norm(x2, ord=2))
        return d

