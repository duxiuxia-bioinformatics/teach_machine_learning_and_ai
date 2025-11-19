import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from ML_toolbox import d_logistic_regression

# ====================================
# training parameters
num_iterations = 50000
my_learning_rate = 1.5
my_delta_J_threshold = 1e-5
bool_augment_X = True

my_bool_regularization = False

if my_bool_regularization:
    my_regularization_lambda = 1.5
else:
    my_regularization_lambda = 0.0
# ====================================


# ====================================
# plotting parameters
fig_width = 8
fig_height = 6

marker_size = 10

plot_training_result = True

save_fig = True
# ====================================


in_file_name = 'pfas_standardized.csv'
data_in_df = pd.read_csv(in_file_name)

y = data_in_df[['disease']]
X = data_in_df.drop(columns=['disease'])
X.drop(columns=['PFOS'], inplace=True)

for i in range(data_in_df.shape[0]):
    # split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # standardize the training data
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    if bool_augment_X:
        X_train_scaled = sm.add_constant(X_train_scaled)
        X_test_scaled = sm.add_constant(X_test_scaled)

    y_train_array = y_train.to_numpy()

    my_num_of_variables = X_train.shape[1]
    my_initial_theta = np.zeros((my_num_of_variables, 1))

    obj_logistic_regression = d_logistic_regression.logistic_regression(X=X_train_scaled,
                                                                        y=y_train_array,
                                                                        delta_J_threshold=my_delta_J_threshold,
                                                                        initial_theta=my_initial_theta,
                                                                        learning_rate=my_learning_rate,
                                                                        bool_regularization=my_bool_regularization,
                                                                        regularization_lambda=my_regularization_lambda)

    optimal_theta, J = obj_logistic_regression.fit()

    # prediction
    my_label_predict, my_prob_predict = obj_logistic_regression.predict(X_test_scaled)

    # plot J and theta from training
    if my_bool_regularization:
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(range(len(J)), J, color='blue')
        ax.set_xlabel("iteration")
        ax.set_ylabel(r'$J$')
        fig_title = r'$\lambda$=' + str(my_regularization_lambda)
        ax.set_title(fig_title)
        fig.show()

        if save_fig:
            figure_file_name = "cost_vs_iteration.pdf"
            fig.savefig(figure_file_name)

    else:
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(range(len(J)), J, color='blue')
        ax.set_xlabel("iteration")
        ax.set_ylabel(r'$J$')
        fig.show()

    # compare with sklearn
    logistic_classifier = LogisticRegression(random_state=0,
                                             solver='liblinear',
                                             fit_intercept=True,
                                             tol=1e-6)
    logistic_classifier.fit(X_train, y_train_array)
    print("intercept:")
    print(logistic_classifier.intercept_)
    print("coefficients:")
    print(logistic_classifier.coef_)


#
# # testing
# my_label_predict, my_prob_predict = obj_logistic_regression.predict(X_test)

