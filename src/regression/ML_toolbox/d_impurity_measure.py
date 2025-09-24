# Author: Xiuxia Du
# February 2021

import numpy as np
from numpy import linalg as LA

class impurity_measure:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        unique_classes, counts = np.unique(self.y, return_counts=True)
        self.class_frequency = counts / (len(y))

    def get_gini(self):
        gini = 1.0 - sum(self.class_frequency**2)
        return gini

    def get_entropy(self):
        entropy = (-1.0) * sum(self.class_frequency * np.log2(self.class_frequency))
        return entropy

    def get_maxclassification_error(self):
        maxclassification_error = 1.0 - max(self.class_frequency)
        return maxclassification_error