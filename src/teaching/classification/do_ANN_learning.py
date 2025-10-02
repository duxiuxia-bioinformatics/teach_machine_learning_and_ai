# Author: Xiuxia Du
# edited: 2021-11-04

import numpy as np
import matplotlib.pyplot as plt
#from ../ML_toolbox import d_ANN
#from ../ML_toolbox import d_data_pretreatment

import d_ANN
import d_data_pretreatment

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

import pickle

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
# mpl.rcParams['text.usetex'] = True

# # for using latex
# from matplotlib import rc
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font', **{'family': 'serif', 'serif': ['Palatino']})
# rc('text', usetex=True)
#
# import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

def save_obj(file_name, obj):
    with open(file_name, 'wb') as fid:
        pickle.dump(obj, fid)

def load_obj(file_name):
    with open(file_name, 'rb') as fid:
        return pickle.load(fid)

# ====================================================================
# plotting parameters
# ====================================================================
plt.close('all')

fig_width = 8
fig_height = 6

marker_size = 10

bool_plot_training_result = True
save_fig = True

# ====================================================================
# other parameters
# ====================================================================
my_learning_rate = 0.5
my_num_of_iterations = 500

index_example = 2
# index_example=1: regression
# index_example=2: classification

# main
np.random.seed(1)

if index_example == 1:
    # ====================================================================
    # example of back propagation
    # ====================================================================
    my_initial_theta = {}

    my_X = np.array([0.05, 0.1])
    my_Y = np.array([0.01, 0.99])

    my_X = np.reshape(my_X, (1, len(my_X)))
    my_Y = np.reshape(my_Y, (1, len(my_Y)))

    my_num_of_samples = 1
    my_num_of_outputs = 2
    my_num_of_variables = 2

    # specify neural network architecture: one hidden layer
    my_num_of_hidden_units = 2    # not including the bias unit

    my_initial_theta['layer_1'] = np.array([[0.35, 0.15, 0.2], [0.35, 0.25, 0.3]])
    my_initial_theta['layer_2'] = np.array([[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]])

    obj_ANN = d_ANN.ANN(num_of_variables=my_num_of_variables,
                        num_of_hidden_units=my_num_of_hidden_units,
                        num_of_outputs=my_num_of_outputs,
                        bool_is_classification=False,
                        initial_theta=my_initial_theta,
                        num_of_iterations=my_num_of_iterations,
                        learning_rate=my_learning_rate)

    theta, cost_list, h = obj_ANN.fit(my_X, my_Y)

    # arrange theta into an array for easy plotting
    theta_in_array = {}
    theta_in_array['layer_1'] = np.zeros((my_num_of_hidden_units, (my_num_of_variables + 1), len(theta)))
    theta_in_array['layer_2'] = np.zeros((my_num_of_outputs, (my_num_of_hidden_units + 1), len(theta)))

    for i in range(len(theta)):
        for j in range(my_num_of_hidden_units):
            for k in range(my_num_of_variables + 1):
                theta_in_array['layer_1'][j, k, i] = theta[i]['layer_1'][j, k]

    for i in range(len(theta)):
        for j in range(my_num_of_outputs):
            for k in range(my_num_of_hidden_units + 1):
                theta_in_array['layer_2'][j, k, i] = theta[i]['layer_2'][j, k]

    h_array = np.zeros((len(h), 2))
    for i in range(len(h)):
        h_array[i, 0] = h[i][0]
        h_array[i, 1] = h[i][1]

    if bool_plot_training_result:
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
                ax.scatter(range(len(theta)), theta_in_array['layer_1'][j, k, :], color='blue')
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
                ax.scatter(range(len(theta)), theta_in_array['layer_2'][j, k, :], color='blue')
                ax.set_xlabel('iterations')
                ax.set_ylabel('theta')
                count = count + 1
        fig.show()

        fig = plt.figure()
        fig.suptitle('predicted output')
        ax = fig.add_subplot(2, 1, 1)
        ax.scatter(range(len(h)), h_array[:, 0], color='blue')
        count = count + 1
        ax.set_xlabel('iterations')
        ax.set_ylabel('h0')

        ax = fig.add_subplot(2, 1, 2)
        ax.scatter(range(len(h)), h_array[:, 1], color='blue')
        count = count + 1
        ax.set_xlabel('iterations')
        ax.set_ylabel('h1')
        fig.show()

        plt.close('all')

else:
    # ====================================================================
    # use ANN for classification: illustrated using the iris data set
    # ====================================================================

    # --------------------------------------------------------------------
    # get data, data pretreatment, and prepare for neural network learning
    # --------------------------------------------------------------------
    # get iris data
    iris = load_iris()

    my_total_num_of_samples = 100

    my_X = iris.data[0:my_total_num_of_samples, 2:4] # use petal data
    my_Y = iris.target[0:my_total_num_of_samples]

    treated_Y = np.reshape(my_Y, (len(my_Y), 1))

    my_num_of_variables = my_X.shape[1]

    # determine K
    unique_output = np.unique(treated_Y)

    if len(unique_output) == 2:
        my_num_of_output_units = 1
    else:
        my_num_of_output_units = len(unique_output)

    # data pretreatment
    obj_pretreatment = d_data_pretreatment.data_pretreatment(X=my_X, treatment_method="range")
    treated_X = obj_pretreatment.do_pretreatment()

    # specify neural network architecture: one hidden layer
    my_num_of_hidden_units = 2    # not including the bias unit

    # --------------------------------------------------------------------
    # training and leave-one-out cross validation
    # --------------------------------------------------------------------
    # training and testing
    error_list = []
    theta_all_training = []

    error_list_sklearn = []
    coefs_all_training_sklearn = []
    intercepts_all_training_sklearn = []

    for i in range(my_total_num_of_samples):

        # get training and testing data using leave-one-out
        x_test = treated_X[i, :]
        y_test = treated_Y[i, :]

        x_train = np.delete(treated_X, i, axis=0)
        y_train = np.delete(treated_Y, i, axis=0)

        my_num_of_samples = x_train.shape[0]

        obj_ANN = d_ANN.ANN(num_of_variables=my_num_of_variables,
                            num_of_hidden_units=my_num_of_hidden_units,
                            num_of_outputs=my_num_of_output_units,
                            num_of_iterations=my_num_of_iterations,
                            learning_rate=my_learning_rate)

        theta, cost_list, h = obj_ANN.fit(x_train, y_train)

        # test
        x_test = x_test.reshape((len(x_test), 1))
        z, a = obj_ANN.do_forward_propagation(one_sample_x=x_test, w=theta[len(theta)-1])
        h = a['layer_3']

        if h[0][0] > 0.5:
            y_predicted = 1.0
        else:
            y_predicted = 0.0

        if y_predicted == y_test:
            error_list.append(0.0)
        else:
            error_list.append(1.0)

        theta_all_training.append(theta[len(theta)-1])

        # =================================================================================
        # compare with sklearn
        # =================================================================================
        MLP_classifier = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-15, random_state=1,
                                       hidden_layer_sizes=(2))
        MLP_classifier.fit(x_train, np.ravel(y_train))

        coefs_all_training_sklearn.append(MLP_classifier.coefs_)
        intercepts_all_training_sklearn.append(MLP_classifier.intercepts_)

        y_predicted_sklearn = MLP_classifier.predict(x_test.reshape(1, -1))
        if y_predicted_sklearn == y_test:
            error_list_sklearn.append(0.0)
        else:
            error_list_sklearn.append(1.0)

    error_array = np.array(error_list)
    out_file_name = "error_array.csv"
    np.savetxt(out_file_name, error_array, delimiter=",")

    save_obj(file_name='theta_all_training.pkl', obj=theta_all_training)
    import_theta = load_obj('theta_all_training.pkl')

    save_obj(file_name='coefs_all_training_sklearn.pkl', obj=coefs_all_training_sklearn)

    save_obj(file_name='intercepts_all_training_sklearn.pkl', obj=intercepts_all_training_sklearn)