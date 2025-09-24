import sys
import os

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Get the absolute path of the current file
try:
    # Works in .py scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for Jupyter
    current_dir = os.getcwd()

# Go up N levels (here N=2, but you can adjust)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Add the project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ML_toolbox import d_mlr_gradient_descent_class
from ML_toolbox import d_lm_analytical_solution_class

# # --------------------------------------------------------------------------
# # set up paths
# # --------------------------------------------------------------------------
# # get the directory path of the running script
# # working_dir_absolute_path = os.path.dirname(os.path.realpath(__file__))
# #
# # toolbox_absolute_path = os.path.join(working_dir_absolute_path, "ML_toolbox")
# # data_absolute_path = os.path.join(working_dir_absolute_path, "data")
# #
# # sys.path.append(toolbox_absolute_path)
# # sys.path.append(data_absolute_path)
#


# --------------------------------------------------------------------------
# set up plotting parameters
# --------------------------------------------------------------------------
line_width_1 = 2
line_width_2 = 2
marker_1 = '.' # point
marker_2 = 'o' # circle
marker_size = 12
line_style_1 = ':' # dotted line
line_style_2 = '-' # solid line

boolean_using_existing_data = False

# parameters for numerical solutions
delta_J_threshold = 0.000001
learning_rate = 0.001

ind_example = 2
# ind_example == 1: plot the cost as a function of theta_0 and theta_1
# ind_example == 2: numerical solution

def main():
    if ind_example == 1:
        if boolean_using_existing_data:
            in_file_name = "../../data/linear_regression_test_data.csv"
            in_file_full_name = in_file_name
            # in_file_full_name = os.path.join(data_absolute_path, in_file_name)

            dataIn = pd.read_csv(in_file_full_name)
            x = np.array(dataIn['x'])
            y = np.array(dataIn['y'])
            y_theoretical = np.array(dataIn['y_theoretical'])
        else:
            n = 20
            # np.random.seed(0)

            x = -2 + 4 * np.random.rand(n)
            x = np.sort(x)

            beta_0 = 1.0
            beta_1 = 1.5
            sigma = 0.5

            epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)

            y_theoretical = beta_0 + beta_1 * x
            y = beta_0 + beta_1 * x + epsilon

            # --------------------------------------------------------------------------
            # linear regression using OLS
            # --------------------------------------------------------------------------
            n = len(x)

            x_bar = np.mean(x)
            y_bar = np.mean(y)

            # do linear regression using my own function
            lm_d_result = d_lm_analytical_solution_class.d_lm(x, y)

            # plot
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
            ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)
            ax.plot(x, lm_d_result['y_hat'], color='blue', label='predicted', linewidth=line_width_1)
            ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=line_width_1)
            ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title("Linear regression")
            ax.legend(loc='lower right', fontsize=9)
            fig.show()

            # --------------------------------------------------------------------------
            # cost function
            # --------------------------------------------------------------------------
            all_beta_1 = np.arange(start=beta_1 - 2.0, stop=beta_1 + 2.0, step=0.01)
            if beta_0 == 0:     # cost J is a function of beta_1 only
                J_vec = np.zeros(len(all_beta_1))

                for i in range(len(all_beta_1)):
                    current_beta_1 = all_beta_1[i]

                    for j in range(n):
                        current_y_hat = current_beta_1 * x[j]

                        J_vec[i] = J_vec[i] + (current_y_hat - y[j])**2

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(all_beta_1, J_vec)
                ax.set_xlabel(r'$\theta_{1}$')
                ax.set_ylabel(r'$J(\theta_1)$')
                fig.show()
                fig.savefig('cost function_1 variable.pdf', bbox_inches='tight')

            else:   # cost J is a function of beta_0 and beta_1
                all_beta_0 = np.arange(start=beta_0 - 2.0, stop=beta_0 + 2.0, step=0.1)

                beta_0_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))
                beta_1_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))

                J_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))

                for i in range(len(all_beta_1)):
                    current_beta_1 = all_beta_1[i]

                    for j in range(len(all_beta_0)):
                        current_beta_0 = all_beta_0[j]

                        beta_0_matrix[i, j] = current_beta_0
                        beta_1_matrix[i, j] = current_beta_1

                        for k in range(n):
                            current_y_hat = current_beta_0 + current_beta_1 * x[k]

                            J_matrix[i, j] = J_matrix[i, j] + (current_y_hat - y[k])**2

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                ax.plot_surface(beta_0_matrix, beta_1_matrix, J_matrix, cmap=cm.coolwarm)
                ax.set_xlabel(r'$\theta_0$')
                ax.set_ylabel(r'$\theta_1$')
                ax.set_zlabel(r'$J(\theta_0, \theta_1)$')
                fig.show()
                fig.savefig('cost function_2 variables.pdf', bbox_inches='tight')

    elif ind_example == 2: # gradient descent method
        # --------------------------------------------------------------------------
        # linear regression using gradient descent
        # --------------------------------------------------------------------------

        # import and plot raw data
        in_file_name = "../../../data/home_price.csv"
        # in_file_name = "../../data/linear_regression_test_data.csv"
        data_in_df = pd.read_csv(in_file_name)

        # x_name = ['size', 'number of bedrooms']
        x_name = ['size']
        y_name = ['price']

        variable_to_plot = ['size']

        fig, ax = plt.subplots()
        ax.scatter(data_in_df[variable_to_plot], data_in_df[y_name], marker='.', color='blue')
        ax.set_xlabel(variable_to_plot)
        ax.set_ylabel(y_name)
        fig.show()

        # normalize variables to make them have similar scale
        standard_scaler_obj = StandardScaler()
        standard_scaler_obj.fit(data_in_df[x_name + y_name])

        mean_needed_df = pd.Series(standard_scaler_obj.mean_, index=x_name + y_name)
        scale_needed_df = pd.Series(standard_scaler_obj.scale_, index=x_name + y_name)

        data_normalized_df = pd.DataFrame(standard_scaler_obj.transform(data_in_df[x_name+y_name]), \
                                          index=data_in_df.index, \
                                          columns=x_name + y_name)

        # get information on x (single variable) and y
        X = data_normalized_df[x_name]
        y = data_normalized_df[y_name]

        # augment X0
        X = sm.add_constant(X)

        number_of_variables = X.shape[1]

         # including X0
        initial_theta = np.zeros((number_of_variables, 1))

        # gradient descent
        obj_MLR = d_mlr_gradient_descent_class.MLR(delta_J_threshold=delta_J_threshold,
                                                       initial_theta=initial_theta,
                                                       learning_rate=learning_rate)

        obj_MLR.fit(X=X, y=y)
        optimal_theta = obj_MLR.optimal_theta
        J = obj_MLR.J

        y_hat = X @ optimal_theta

        # restore y_hat to the original data space
        y_hat_restored = y_hat * scale_needed_df[y_name][y_name].values + mean_needed_df[y_name].values


        fig, ax = plt.subplots()
        ax.scatter(data_in_df[variable_to_plot], data_in_df[y_name], marker='.', color='blue')
        ax.plot(data_in_df[variable_to_plot], y_hat_restored, color='red')
        ax.set_xlabel(variable_to_plot)
        ax.set_ylabel(y_name)
        fig.show()

        fig, ax = plt.subplots()
        ax.scatter(range(len(J)), J, marker='.', color='blue')
        ax.set_xlabel('iterations')
        ax.set_ylabel('J')
        fig.show()
    else:
        sys.exit('Unknown example index!')
    return

# for command line
if __name__ == '__main__':
    main()