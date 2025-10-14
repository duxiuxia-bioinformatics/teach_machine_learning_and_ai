# Author: Xiuxia Du
# 2025-08-21

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
X, y = iris.data, iris.target
decision_tree_obj = tree.DecisionTreeClassifier()
decision_tree_obj.fit(X, y)
tree.plot_tree(decision_tree_obj)

xx = 1