import sys
import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

# Get the absolute path of the current file
try:
    # Works in .py scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for Jupyter
    current_dir = os.getcwd()

# Go up N levels (here N=2, but you can adjust)
project_root = os.path.abspath(os.path.join(current_dir, "..", '..'))

# Add the project root to sys.path if not already there so that the ML_toolbox can be imported
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from ML_toolbox import d_logistic_regression

import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

# ====================================================================
# plotting parameters
# ====================================================================
plt.close('all')

fig_width = 8
fig_height = 6

marker_size = 10

plot_training_result = True

save_fig = True

# ====================================================================
# other parameters
# ====================================================================
num_iterations = 50000
my_learning_rate = 1.5
my_delta_J_threshold = 1e-5

my_bool_regularization = False

if my_bool_regularization:
    my_regularization_lambda = 1.5
else:
    my_regularization_lambda = 0.0

def main():
    # ====================================================================
    # apply logistic regression to iris data
    # ====================================================================
    iris = load_iris()

    II = (iris.target != 0)

    my_X = iris.data[II, 2:4]
    my_Y = iris.target[II]
    my_Y = my_Y.reshape((len(my_Y), 1))

    II_1 = np.where(my_Y == 1)
    II_2 = np.where(my_Y == 2)

    my_Y[II_1] = 0.
    my_Y[II_2] = 1.0

    my_number_of_samples = my_X.shape[0]
    my_number_of_variables = my_X.shape[1]

    my_initial_theta = np.zeros((my_number_of_variables, 1))

    # pre-treatment of variables
    X_pretreated = np.zeros((my_number_of_samples, my_number_of_variables))
    for i in range(my_number_of_variables):
        cur_col = my_X[:, i]

        col_min = min(cur_col) * np.ones(my_number_of_samples)
        col_range = (max(cur_col) - min(cur_col)) * np.ones(my_number_of_samples)

        X_pretreated[:, i] = (cur_col - col_min) / col_range

    compare_with_scikit = []
    error_array = np.zeros(my_number_of_samples)
    # for index_testing in range(my_number_of_samples):
    for index_testing in [49]:
    # for index_testing in range(X_pretreated.shape[0]):
        print(index_testing)

        # training
        X_test = X_pretreated[index_testing, :]
        Y_test = my_Y[index_testing]

        X_train = np.delete(X_pretreated, index_testing, axis=0)
        Y_train = np.delete(my_Y, index_testing, axis=0)

        obj_logistic_regression = d_logistic_regression.logistic_regression(X=X_train,
                                                                            y=Y_train,
                                                                            delta_J_threshold=my_delta_J_threshold,
                                                                            initial_theta=my_initial_theta,
                                                                            learning_rate=my_learning_rate,
                                                                            bool_regularization=my_bool_regularization,
                                                                            regularization_lambda=my_regularization_lambda)

        optimal_theta, J = obj_logistic_regression.fit()

        # testing
        my_Y_predict = obj_logistic_regression.predict(X_test)

        # transform theta to use the non-pretreated X
        # optimal_theta_transformed = np.zeros((my_number_of_variables+1, 1))
        #
        # optimal_theta_transformed[0] = optimal_theta[0]
        # for i in range(my_number_of_variables):
        #     optimal_theta_transformed[0] = optimal_theta_transformed[0] - optimal_theta[i] * min(my_X[:, i]) / (max(my_X[:, i]) - min(my_X[:, i]))
        #
        # for i in range(my_number_of_variables):
        #     optimal_theta_transformed[i+1] = optimal_theta[i+1] / (max(my_X[:, i]) - min(my_X[:, i]))

        # logistic_classifier = LogisticRegression(random_state=0, solver=fit_intercept=True, C=1e15)

        # use scikit learn
        logistic_classifier = LogisticRegression(random_state=0,
                                                 solver='liblinear',
                                                 fit_intercept=True,
                                                 tol=1e-6)
        logistic_classifier.fit(X_train, Y_train.ravel())
        print("intercept:")
        print(logistic_classifier.intercept_)
        print("coefficients:")
        print(logistic_classifier.coef_)

        X_test = X_test.reshape((1, -1))
        scikit_Y_predict = logistic_classifier.predict(X_test)

        tf = (my_Y_predict == scikit_Y_predict)

        compare_with_scikit.append(tf[0][0])

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
            fig.show()

        plt.close('all')

    out_file_name = "../results/compare_with_scikit_learn_logistic_regression_LOO.csv"
    np.savetxt(out_file_name, compare_with_scikit, delimiter=",")

    # ====================================================================
    # apply logistic regression to banking data
    # ====================================================================
    # get banking data: The data is related to direct marketing campaigns
    # of a Portuguese banking institution. The classification goal is to
    # predict whether the client will subscribe to a term deposit (variable y).

    # The dataset provides customer's  information. It includes 41,188 records and 21 fields.

    # y: has the client subscribed to a term deposit? 1 - Yes, 0 = No

    # This material is adapted from
    # https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/

    in_file_name = "../data/banking.csv"
    data_in = pd.read_csv(in_file_name, header=0)
    data_in = data_in.dropna()

    print(data_in.shape)

    print(list(data_in.columns))
    data_in.head()

    # check the missing values
    data_in.isnull().sum()

    # randomly select 25% of the data for illustration
    samples_to_select = np.int_(data_in.shape[0] * np.random.rand(10000))
    data_in = data_in.iloc[samples_to_select, :]

    # barplot for the dependent variable
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(x='y', data=data_in, palette='hls')
    fig.show()

    # customer job distribution
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(y='job', data=data_in)
    fig.show()

    # customer marital status distribution
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(x='marital', data=data_in)
    fig.show()

    # customer credit in default
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(x='default', data=data_in)
    fig.show()

    # customer housing loan
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(x='housing', data=data_in)
    fig.show()

    # customer personal loan
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(x='loan', data=data_in)
    fig.show()

    # customer poutcome
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.countplot(x='poutcome', data=data_in)
    fig.show()

    # use a small number of features for building the logistic regression model
    data_for_analysis = data_in[['job', 'marital', 'default', 'housing', 'loan', 'poutcome', 'y']]

    # Create dummy variables
    data_for_analysis_dummy = pd.get_dummies(data_for_analysis,
                                             columns=['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
    data_for_analysis_dummy.head()

    # remove columns with unknown values
    all_columns = list(data_for_analysis_dummy)
    indices_for_unknown_column = [i for i, s in enumerate(all_columns) if 'unknown' in s]

    data_for_analysis_final = data_for_analysis_dummy.drop(data_for_analysis_dummy.columns[indices_for_unknown_column], axis=1)

    # check the independence between independent variables
    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(data_for_analysis_final.corr())
    fig.show()

    # get training and testing sets
    X = data_for_analysis_final.iloc[:, 1:]
    y = data_for_analysis_final.iloc[:, 0]

    num_of_samples = X.shape[0]
    num_of_test_samples = int(0.20 * num_of_samples)

    test_sample_index = np.int_(num_of_samples * np.random.rand(num_of_test_samples))
    train_sample_index = np.setdiff1d(np.arange(0, num_of_samples, 1), test_sample_index)
    X_train = X.iloc[train_sample_index, :]
    X_test = X.iloc[test_sample_index, :]
    Y_train = y.iloc[train_sample_index]
    Y_test = y.iloc[test_sample_index]

    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    # --------------------------------------------------------------------
    # logistic regression using sklearn
    # --------------------------------------------------------------------
    logistic_classifier = LogisticRegression(random_state=0,
                                             C=1.0,
                                             class_weight=None,
                                             dual=False,
                                             fit_intercept=True,
                                             intercept_scaling=1,
                                             max_iter=100,
                                             multi_class='ovr',
                                             n_jobs=1,
                                             penalty='l2',
                                             solver='liblinear',
                                             tol=0.0001,
                                             verbose=0,
                                             warm_start=False)
    logistic_classifier.fit(X_train, Y_train)
    print("intercept:")
    print(logistic_classifier.intercept_)
    print("coefficients:")
    print(logistic_classifier.coef_)

    # predict
    sklearn_y_prediction = logistic_classifier.predict(X_test)

    # classifier performance
    sklearn_confusion_matrix_for_logistic_classifier = confusion_matrix(Y_test, sklearn_y_prediction)

    print("sklearn confusion matrix:")
    print(sklearn_confusion_matrix_for_logistic_classifier)

    sklearn_TP = np.float(sklearn_confusion_matrix_for_logistic_classifier[0, 0])
    sklearn_TN = np.float(sklearn_confusion_matrix_for_logistic_classifier[1, 1])
    sklearn_FP = np.float(sklearn_confusion_matrix_for_logistic_classifier[0, 1])
    sklearn_FN = np.float(sklearn_confusion_matrix_for_logistic_classifier[1, 0])

    sklearn_sensitivity = sklearn_TP / (sklearn_TP + sklearn_FN) # also called recall
    sklearn_specificity = sklearn_TN / (sklearn_TN + sklearn_FP)
    sklearn_precision = sklearn_TP / (sklearn_TP + sklearn_FP)
    sklearn_accuracy = (sklearn_TP + sklearn_TN) / (sklearn_TP + sklearn_TN + sklearn_FP + sklearn_FN)

    print("classification report:")
    print(classification_report(Y_test, sklearn_y_prediction))

    # --------------------------------------------------------------------
    # logistic regression using my own gradient descent
    # --------------------------------------------------------------------
    my_number_of_samples = X_train.shape[0]
    my_number_of_variables = X_train.shape[1]

    my_initial_theta = np.zeros((my_number_of_variables, 1))

    obj_logistic_regression = d_logistic_regression.logistic_regression(X=X_train,
                                                                        y=Y_train,
                                                                        delta_J_threshold=my_delta_J_threshold,
                                                                        initial_theta=my_initial_theta,
                                                                        learning_rate=my_learning_rate)

    optimal_theta, J = obj_logistic_regression.do_gradient_descent()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(len(J)), J, color='blue')
    fig.show()

    # testing
    my_y_predict = obj_logistic_regression.predict(X_test)

    # classifier performance
    my_confusion_matrix_for_logistic_classifier = confusion_matrix(Y_test, my_y_predict)

    print("my confusion matrix:")
    print(my_confusion_matrix_for_logistic_classifier)

    my_TP = np.float(my_confusion_matrix_for_logistic_classifier[0, 0])
    my_TN = np.float(my_confusion_matrix_for_logistic_classifier[1, 1])
    my_FP = np.float(my_confusion_matrix_for_logistic_classifier[0, 1])
    my_FN = np.float(my_confusion_matrix_for_logistic_classifier[1, 0])

    my_sensitivity = my_TP / (my_TP + my_FN) # also called recall
    my_specificity = my_TN / (my_TN + my_FP)
    my_precision = my_TP / (my_TP + my_FP)
    my_accuracy = (my_TP + my_TN) / (my_TP + my_TN + my_FP + my_FN)

if __name__ == '__main__':
    main()








# # ====================================================================
# # logistic regression using gradient descent
# # ====================================================================
# m = X_train.shape[0]
#
# # add the x_0 column bo both X_train and X_test
# x_0 = pd.Series(np.ones(m), index=X_train.index)
# X_train = X_train.assign(x_0=x_0.values)
#
# all_columns = X_train.columns.tolist()
# all_columns = all_columns[-1:] + all_columns[:-1]
#
# X_train = X_train[all_columns]
#
# x_0 = pd.Series(np.ones(num_of_test_samples), index=X_test.index)
# X_test = X_test.assign(x_0=x_0.values)
#
# all_columns = X_test.columns.tolist()
# all_columns = all_columns[-1:] + all_columns[:-1]
#
# X_test = X_test[all_columns]
#
# n = X_train.shape[1]
#
# # convert from pandas DataFrame to ndarray
# X_train = X_train.as_matrix()
# X_test = X_test.as_matrix()
# y_train = y_train.as_matrix()
# y_test = y_test.as_matrix()
#
# # gradient descent
# iteration = 200
#
# theta = np.zeros((iteration, n))
# # theta[0, :] = 2.5 * np.ones(n)
# initial_theta_0 = logistic_classifier.intercept_
# initial_theta_1 = logistic_classifier.coef_[0]
# initial_theta = 1.0 + np.concatenate((initial_theta_0, initial_theta_1))
#
# theta[0, :] = initial_theta
#
# J = np.zeros(iteration)
#
# alpha = 5
#
# for index_iter in range(iteration-1):
#     if index_iter % 10 == 0:
#         print("iteration " + str(index_iter))
#
#     partial_derivative = np.zeros(n)
#
#     for index_sample in range(m):
#         cur_z = sum(X_train[index_sample, :] * theta[index_iter, :])
#         cur_y_hat = 1.0 / (1.0 + np.exp(-cur_z))
#         cur_residual = cur_y_hat - y_train[index_sample]
#
#         partial_derivative = partial_derivative + X_train[index_sample, :] * cur_residual
#
#         cur_cost = y_train[index_sample] * np.log10(cur_y_hat) + (1.0-y_train[index_sample]) * np.log10(1.0 - cur_y_hat)
#         J[index_iter] = J[index_iter] + cur_cost
#
#     J[index_iter] = -J[index_iter] / m
#
#     theta[index_iter+1, :] = theta[index_iter, :] - alpha * partial_derivative / m
#
# # calculate the last J(theta)
# for index_sample in range(m):
#     cur_z = sum(X_train[index_sample, :] * theta[iteration-1, :])
#     cur_y_hat = 1.0 / (1.0 + np.exp(-cur_z))
#     # cur_residual = cur_y_hat - y_train[index_sample]
#
#     cur_cost = y_train[index_sample] * np.log10(cur_y_hat) + (1.0-y_train[index_sample]) * np.log10(1.0 - cur_y_hat)
#     J[iteration-1] = J[iteration-1] + cur_cost
#
# J[iteration-1] = -J[iteration-1] / m
#
# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$J(\theta)$')
# ax.scatter(range(iteration), J, color='blue', s=marker_size)
# fig.show()
#
# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel(r'$\theta_1$')
# ax.set_ylabel(r'$J(\theta)$')
# ax.scatter(theta[:, 1], J, color='blue', s=marker_size)
# fig.show()
#
# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(2, 3, 1)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_0$')
# ax.scatter(range(iteration), theta[:, 0], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 2)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_1$')
# ax.scatter(range(iteration), theta[:, 1], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 3)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_2$')
# ax.scatter(range(iteration), theta[:, 2], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 4)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_3$')
# ax.scatter(range(iteration), theta[:, 3], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 5)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_4$')
# ax.scatter(range(iteration), theta[:, 4], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 6)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_5$')
# ax.scatter(range(iteration), theta[:, 5], color='blue', s=marker_size)
# fig.show()
#
# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(2, 3, 1)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_6$')
# ax.scatter(range(iteration), theta[:, 6], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 2)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_7$')
# ax.scatter(range(iteration), theta[:, 7], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 3)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_8$')
# ax.scatter(range(iteration), theta[:, 8], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 4)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_9$')
# ax.scatter(range(iteration), theta[:, 9], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 5)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{10}$')
# ax.scatter(range(iteration), theta[:, 10], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 6)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{11}$')
# ax.scatter(range(iteration), theta[:, 11], color='blue', s=marker_size)
# fig.show()
#
# # figure 3
# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(2, 3, 1)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{12}$')
# ax.scatter(range(iteration), theta[:, 12], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 2)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{13}$')
# ax.scatter(range(iteration), theta[:, 13], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 3)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{14}$')
# ax.scatter(range(iteration), theta[:, 14], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 4)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{15}$')
# ax.scatter(range(iteration), theta[:, 15], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 5)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{16}$')
# ax.scatter(range(iteration), theta[:, 16], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 6)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{17}$')
# ax.scatter(range(iteration), theta[:, 17], color='blue', s=marker_size)
# fig.show()
#
# # figure 4
# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = fig.add_subplot(2, 3, 1)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{18}$')
# ax.scatter(range(iteration), theta[:, 18], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 2)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{19}$')
# ax.scatter(range(iteration), theta[:, 19], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 3)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{20}$')
# ax.scatter(range(iteration), theta[:, 20], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 4)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{21}$')
# ax.scatter(range(iteration), theta[:, 21], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 5)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{22}$')
# ax.scatter(range(iteration), theta[:, 22], color='blue', s=marker_size)
#
# ax = fig.add_subplot(2, 3, 6)
# ax.set_xlabel(r'iteration')
# ax.set_ylabel(r'$\theta_{23}$')
# ax.scatter(range(iteration), theta[:, 23], color='blue', s=marker_size)
# fig.show()
#
# # prediction
# final_theta = theta[-1, :]
# final_theta = final_theta.reshape((len(final_theta), 1))
# z_predict = np.matmul(X_test, final_theta)
# y_predict_probability = 1.0 / (1.0 + np.exp(-z_predict))
#
# y_predict_binary = np.zeros(len(y_predict_probability))
# for i in range(len(y_predict_probability)):
#     if y_predict_probability[i] >= 0.5:
#         y_predict_binary[i] = 1.0
#     else:
#         y_predict_binary[i] = 0.0
#
# # get the confusion matrix
# TP = 0
# FP = 0
# TN = 0
# FN = 0
#
# for i in range(len(y_test)):
#     if y_test[i] == 1:
#         if y_predict_binary[i] == 1:
#             TP = TP + 1
#         else:
#             FN = FN + 1
#     else:
#         if y_predict_binary[i] == 1:
#             FP = FP + 1
#         else:
#             TN = TN + 1
#
# confusion_matrix_for_my_logistic_classifier = confusion_matrix(y_test, y_predict_binary)
# print('confusion matrix:')
# print(confusion_matrix_for_my_logistic_classifier)
#
# sklearn_theta_0 = logistic_classifier.intercept_
# sklearn_theta_1 = logistic_classifier.coef_[0]
# sklearn_theta = np.concatenate((sklearn_theta_0, sklearn_theta_1))
#
# z_predict = np.matmul(X_test, sklearn_theta)
# y_predict_probability_sklearn = 1.0 / (1.0 + np.exp(-z_predict))
#
# y_predict_binary_sklearn = np.zeros(len(y_predict_probability_sklearn))
# for i in range(len(y_predict_probability_sklearn)):
#     if y_predict_probability_sklearn[i] >= 0.5:
#         y_predict_binary_sklearn[i] = 1.0
#     else:
#         y_predict_binary_sklearn[i] = 0.0
#
# print(confusion_matrix(y_test, y_predict_binary_sklearn))