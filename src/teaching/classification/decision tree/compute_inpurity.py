# Author: Xiuxia Du
# January 2021

import numpy as np
import d_impurity_measure
from sklearn import datasets

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    object_get_impurity = d_impurity_measure.impurity_measure(X, y)
    gini = object_get_impurity.get_gini()
    entropy = object_get_impurity.get_entropy()
    max_classification_error = object_get_impurity.get_maxclassification_error()



if __name__ == "__main__":
    main()