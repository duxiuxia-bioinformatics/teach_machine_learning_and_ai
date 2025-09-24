# Author: Xiuxia Du
# Edited: Spring of 2023

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm


# --------------------------------------------------------------------------
# set up paths
# --------------------------------------------------------------------------
# get the directory path of the running script
# working_dir_absolute_path = os.path.dirname(os.path.realpath(__file__))
working_dir_absolute_path = '/Users/xdu4/Dropbox (UNC Charlotte)/Duxiuxia/Ocean/Teaching/ML/all/python/src'

toolbox_absolute_path = os.path.join(working_dir_absolute_path, "ML_toolbox")
data_absolute_path = os.path.join(working_dir_absolute_path, "data")

sys.path.append(toolbox_absolute_path)
sys.path.append(data_absolute_path)

from ML_toolbox import d_lm

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

# --------------------------------------------------------------------------
# other settings
# --------------------------------------------------------------------------
boolean_using_existing_data = False

def main():

    if boolean_using_existing_data:
        in_file_name = "linear_regression_test_data.csv"
        in_file_full_name = os.path.join(data_absolute_path, in_file_name)

        dataIn = pd.read_csv(in_file_full_name)
        x = np.array(dataIn['x'])
        y = np.array(dataIn['y'])
        y_theoretical = np.array(dataIn['y_theoretical'])
    else:
        n = 20
        # np.random.seed(0)

        x = -2 + 4 * np.random.rand(n)
        x = np.sort(x)

        beta_0 = 5.0
        beta_1 = 1.5
        sigma = 0.5

        epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)

        y_theoretical = beta_0 + beta_1 * x
        y = beta_0 + beta_1 * x + epsilon

    # plot the data
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
    fig.show()

    # --------------------------------------------------------------------------
    # linear regression
    # --------------------------------------------------------------------------
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    # do linear regression using my own function
    lm_d_result = d_lm.d_lm(x, y)

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
    # upper right, upper left, lower right, lower left, center left, center right, upper center, lower center

    # plot confidence intervals
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
    ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)
    ax.plot(x, lm_d_result['y_hat'], color='blue', label='predicted', linewidth=line_width_1)
    ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=line_width_1)
    ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Linear regression with CI of the regression line")
    ax.legend(loc='lower right', fontsize=9)

    for i in range(n):
        ax.plot([x[i], x[i]],
                [lm_d_result['y_hat_CI_lower_bound'][i], lm_d_result['y_hat_CI_upper_bound'][i]],
                color='magenta',
                linewidth=line_width_1)
    fig.show()

    # do linear regression using sklearn
    lm_sklearn= linear_model.LinearRegression()
    x_reshaped = x.reshape((len(x), 1))
    lm_sklearn.fit(x_reshaped, y)
    y_hat = lm_sklearn.predict(x_reshaped)

    lm_sklearn_result = {}
    lm_sklearn_result['beta_0_hat'] = lm_sklearn.intercept_
    lm_sklearn_result['beta_1_hat'] = lm_sklearn.coef_
    lm_sklearn_result['R2'] = r2_score(y, y_hat)
    lm_sklearn_result['mean_squared_error'] = mean_squared_error(y, y_hat)
    lm_sklearn_result['y_hat'] = y_hat

    # plot the data and the results
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
    ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)
    ax.plot(x, y_hat, color='blue', label='predicted', linewidth=line_width_1)
    ax.plot(x, np.ones(n) * y_bar, color='black', linestyle=':', linewidth=line_width_1)
    ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Linear regression with CI of the regression line")
    ax.legend(loc='lower right', fontsize=9)
    fig.show()

    # do linear regression using statsmodel
    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    Y2 = sm.add_constant(y)
    est = sm.OLS(x, Y2)
    est2 = est.fit()
    print(est2.summary())

    # --------------------------------------------------------------------------
    # more on the statistics of beta_1_hat and beta_0_hat
    # --------------------------------------------------------------------------
    if not boolean_using_existing_data:
        N = 1000 # number of sampling

        beta_0_hat_vec = np.zeros(N)    # initialize
        beta_1_hat_vec = np.zeros(N)    # initialize

        for i in range(N):
            epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)

            y_theoretical = beta_0 + beta_1 * x
            y = beta_0 + beta_1 * x + epsilon

            lm_d_result = d_lm.d_lm(x, y)

            beta_0_hat_vec[i] = lm_d_result['beta_0_hat']
            beta_1_hat_vec[i] = lm_d_result['beta_1_hat']

        # plot the histogram of beta_0_hat and beta_1_hat
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.hist(beta_0_hat_vec, bins=20, histtype='step')
        ax.set_xlim([4.0, 6.0])
        figure_title = "Histogram of beta_0_hat: n=" + str(n) + ", sigma=" + str(sigma)
        ax.set_title(figure_title)
        ax = fig.add_subplot(2, 1, 2)
        ax.hist(beta_1_hat_vec, bins=20, histtype='step')
        ax.set_xlim([1.0, 2.0])
        figure_title = "Histogram of beta_1_hat: n=" + str(n) + ", sigma=" + str(sigma)
        ax.set_title(figure_title)
        fig.show()

    # --------------------------------------------------------------------------
    # diagnostics
    # --------------------------------------------------------------------------
    # 1. are r and y_hat uncorrelated?
    r = y - lm_d_result['y_hat']
    np.corrcoef(r, lm_d_result['y_hat'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(lm_d_result['y_hat'], r, color='blue')
    ax.set_xlabel('y_hat')
    ax.set_ylabel('r')
    fig.show

    # --------------------------------------------------------------------------
    # Multiple linear regression
    # --------------------------------------------------------------------------
    # use diabetes data
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    print(diabetes.DESCR)

    diabetes_df = pd.DataFrame()

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

if __name__ == '__main__':
    main()