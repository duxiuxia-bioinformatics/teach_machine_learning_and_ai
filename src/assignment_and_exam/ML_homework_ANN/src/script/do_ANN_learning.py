# Author: Xiuxia Du
# edited: 2021-11-04


# =================================
# import packages
# =================================
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pickle

import matplotlib as mpl

# mpl.rcParams.update(mpl.rcParamsDefault)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =================================
# Add the parent directory to the Python path to import ML_toolbox
# =================================
def get_parent_dir():
    try:
        # Works when running a script
        current_dir = os.path.dirname(__file__)
    except NameError:
        # Works in notebooks / REPL
        current_dir = os.getcwd()
    return os.path.abspath(os.path.join(current_dir, '..'))

parent_dir = get_parent_dir()
sys.path.append(parent_dir)

from ML_toolbox.d_ANN import MyANN
# =================================

# import d_data_pretreatment

def save_obj(file_name, obj):
    with open(file_name, 'wb') as fid:
        pickle.dump(obj, fid)

def load_obj(file_name):
    with open(file_name, 'rb') as fid:
        return pickle.load(fid)

# ====================================================================
# plotting parameters
# ====================================================================
fig_width = 8
fig_height = 6
marker_size = 10

# ====================================================================
# model training parameters
# ====================================================================
my_learning_rate = 0.5
my_num_of_iterations = 500

# ====================================================================
# Other settings
# ====================================================================
np.random.seed(1)


def main():

    # load the iris data: use the setosa and versicolor for binary classification
    data_in = load_iris()
    X = data_in.data[0:100, 2:4]
    y = data_in.target[0:100]

    # split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    # standardize the training data
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)

    # use the mean and standard deviation computed from the training data to standardize the testing data
    X_test_scaled = standard_scaler.transform(X_test)

    # information
    my_num_of_variables = X_train_scaled.shape[1]
    my_num_of_outputs = 1
    my_num_of_hidden_units = 2

    my_initial_theta = {}
    my_initial_theta['layer_1'] = np.random.random((2, 3))
    my_initial_theta['layer_2'] = np.random.random((1, 3))

    # model training
    my_ann_obj = MyANN(num_of_variables=my_num_of_variables,
                       num_of_hidden_units=my_num_of_hidden_units,
                       num_of_outputs=my_num_of_outputs,
                       bool_is_classification=True,
                       initial_theta=my_initial_theta,
                       num_of_iterations=my_num_of_iterations,
                       learning_rate=my_learning_rate)
    theta_at_all_iterations, cost_list = my_ann_obj.fit(X_train, y_train)

    print('Optimal theta:')
    print(theta_at_all_iterations[-1])

    # arrange theta into an array for easy plotting
    theta_in_array = {}
    theta_in_array['layer_1'] = np.zeros((my_num_of_hidden_units, (my_num_of_variables + 1), len(theta_at_all_iterations)))
    theta_in_array['layer_2'] = np.zeros((my_num_of_outputs, (my_num_of_hidden_units + 1), len(theta_at_all_iterations)))

    for i in range(len(theta_at_all_iterations)):
        for j in range(my_num_of_hidden_units):
            for k in range(my_num_of_variables + 1):
                theta_in_array['layer_1'][j, k, i] = theta_at_all_iterations[i]['layer_1'][j, k]

    for i in range(len(theta_at_all_iterations)):
        for j in range(my_num_of_outputs):
            for k in range(my_num_of_hidden_units + 1):
                theta_in_array['layer_2'][j, k, i] = theta_at_all_iterations[i]['layer_2'][j, k]


    # plot cost vs iterations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('iterations')
    # ax.set_ylabel(r'$J(\theta)$')
    ax.set_ylabel('J')
    ax.scatter(range(len(cost_list)), cost_list, color='blue')
    fig.show()

    fig = plt.figure()
    # fig.suptitle(r'layer 1 $\theta$')
    fig.suptitle('layer 1 theta')
    count = 1
    for j in range(my_num_of_hidden_units):
        for k in range(my_num_of_variables + 1):
            ax = fig.add_subplot(my_num_of_hidden_units, my_num_of_variables + 1, count)
            ax.scatter(range(len(theta_at_all_iterations)), theta_in_array['layer_1'][j, k, :], color='blue')
            ax.set_xlabel('iterations')
            ax.set_ylabel('theta')
            count = count + 1
    fig.show()

    fig = plt.figure()
    # fig.suptitle(r'layer 2 $\theta$')
    fig.suptitle('layer 2 theta')
    count = 1
    for j in range(my_num_of_outputs):
        for k in range(my_num_of_hidden_units + 1):
            ax = fig.add_subplot(my_num_of_outputs, my_num_of_hidden_units + 1, count)
            ax.scatter(range(len(theta_at_all_iterations)), theta_in_array['layer_2'][j, k, :], color='blue')
            ax.set_xlabel('iterations')
            ax.set_ylabel('theta')
            count = count + 1
    fig.show()

    # fig = plt.figure()
    # fig.suptitle('predicted output')
    # ax = fig.add_subplot(2, 1, 1)
    # ax.scatter(range(len(h)), h_array[:, 0], color='blue')
    # count = count + 1
    # ax.set_xlabel('iterations')
    # ax.set_ylabel('h0')
    #
    # ax = fig.add_subplot(2, 1, 2)
    # ax.scatter(range(len(h)), h_array[:, 1], color='blue')
    # count = count + 1
    # ax.set_xlabel('iterations')
    # ax.set_ylabel('h1')
    # fig.show()

    plt.close('all')

    # test the model using X_test and y_test
    y_predicted = my_ann_obj.predict(X_test)

    my_accuracy_score = accuracy_score(y_test, y_predicted)
    print('Accuracy:', my_accuracy_score)



    return

if __name__ == '__main__':
    main()



# else:
#     # ====================================================================
#     # use ANN for classification: illustrated using the iris data set
#     # ====================================================================
#
#     # --------------------------------------------------------------------
#     # get data, data pretreatment, and prepare for neural network learning
#     # --------------------------------------------------------------------
#     # get iris data
#     iris = load_iris()
#
#     my_total_num_of_samples = 100
#
#     my_X = iris.data[0:my_total_num_of_samples, 2:4] # use petal data
#     my_Y = iris.target[0:my_total_num_of_samples]
#
#     treated_Y = np.reshape(my_Y, (len(my_Y), 1))
#
#     my_num_of_variables = my_X.shape[1]
#
#     # determine K
#     unique_output = np.unique(treated_Y)
#
#     if len(unique_output) == 2:
#         my_num_of_output_units = 1
#     else:
#         my_num_of_output_units = len(unique_output)
#
#     # data pretreatment
#     obj_pretreatment = d_data_pretreatment.data_pretreatment(X=my_X, treatment_method="range")
#     treated_X = obj_pretreatment.do_pretreatment()
#
#     # specify neural network architecture: one hidden layer
#     my_num_of_hidden_units = 2    # not including the bias unit
#
#     # --------------------------------------------------------------------
#     # training and leave-one-out cross validation
#     # --------------------------------------------------------------------
#     # training and testing
#     error_list = []
#     theta_all_training = []
#
#     error_list_sklearn = []
#     coefs_all_training_sklearn = []
#     intercepts_all_training_sklearn = []
#
#     for i in range(my_total_num_of_samples):
#
#         # get training and testing data using leave-one-out
#         x_test = treated_X[i, :]
#         y_test = treated_Y[i, :]
#
#         x_train = np.delete(treated_X, i, axis=0)
#         y_train = np.delete(treated_Y, i, axis=0)
#
#         my_num_of_samples = x_train.shape[0]
#
#         obj_ANN = d_ANN.ANN(num_of_variables=my_num_of_variables,
#                             num_of_hidden_units=my_num_of_hidden_units,
#                             num_of_outputs=my_num_of_output_units,
#                             num_of_iterations=my_num_of_iterations,
#                             learning_rate=my_learning_rate)
#
#         theta, cost_list, h = obj_ANN.fit(x_train, y_train)
#
#         # test
#         x_test = x_test.reshape((len(x_test), 1))
#         z, a = obj_ANN.do_forward_propagation(one_sample_x=x_test, w=theta[len(theta)-1])
#         h = a['layer_3']
#
#         if h[0][0] > 0.5:
#             y_predicted = 1.0
#         else:
#             y_predicted = 0.0
#
#         if y_predicted == y_test:
#             error_list.append(0.0)
#         else:
#             error_list.append(1.0)
#
#         theta_all_training.append(theta[len(theta)-1])
#
#         # =================================================================================
#         # compare with sklearn
#         # =================================================================================
#         MLP_classifier = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-15, random_state=1,
#                                        hidden_layer_sizes=(2))
#         MLP_classifier.fit(x_train, np.ravel(y_train))
#
#         coefs_all_training_sklearn.append(MLP_classifier.coefs_)
#         intercepts_all_training_sklearn.append(MLP_classifier.intercepts_)
#
#         y_predicted_sklearn = MLP_classifier.predict(x_test.reshape(1, -1))
#         if y_predicted_sklearn == y_test:
#             error_list_sklearn.append(0.0)
#         else:
#             error_list_sklearn.append(1.0)
#
#     error_array = np.array(error_list)
#     out_file_name = "error_array.csv"
#     np.savetxt(out_file_name, error_array, delimiter=",")
#
#     save_obj(file_name='theta_all_training.pkl', obj=theta_all_training)
#     import_theta = load_obj('theta_all_training.pkl')
#
#     save_obj(file_name='coefs_all_training_sklearn.pkl', obj=coefs_all_training_sklearn)
#
#     save_obj(file_name='intercepts_all_training_sklearn.pkl', obj=intercepts_all_training_sklearn)