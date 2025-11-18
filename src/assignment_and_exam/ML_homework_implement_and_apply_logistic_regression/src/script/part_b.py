import pandas as pd
import os, sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from ML_toolbox import d_logistic_regression

in_file_name = 'pfas_standardized.csv'
data_in_df = pd.read_csv(in_file_name)



xx = 1

# obj_logistic_regression = d_logistic_regression.logistic_regression(X=X_train,
#                                                                             y=Y_train,
#                                                                             delta_J_threshold=my_delta_J_threshold,
#                                                                             initial_theta=my_initial_theta,
#                                                                             learning_rate=my_learning_rate,
#                                                                             bool_regularization=my_bool_regularization,
#                                                                             regularization_lambda=my_regularization_lambda)
#
# optimal_theta, J = obj_logistic_regression.fit()
#
# # testing
# my_label_predict, my_prob_predict = obj_logistic_regression.predict(X_test)

