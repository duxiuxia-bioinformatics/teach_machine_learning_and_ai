import numpy as np
import sys
#from numpy.linalg import inv
from numpy import linalg as LA

class LDA:
    """"
    Class for Linear Discriminant Analysis.

    Attributes:
    """

    def __init__(self, num_of_classes):
        self.num_of_classes = num_of_classes

    def do_LDA_two_class(self, X, Y):
        targets = np.unique(Y)

        II_0 = (np.where(Y==targets[0]))[0]
        II_1 = (np.where(Y==targets[1]))[0]

        N_0 = len(II_0)
        N_1 = len(II_1)

        x0 = X[II_0, :]
        x1 = X[II_1, :]

        d = x1.shape[1]

        mu_0 = np.mean(x0, axis=0)
        mu_1 = np.mean(x1, axis=0)

        S_within_0 = np.zeros((d, d))
        S_within_1 = np.zeros((d, d))

        for i in range(N_0):
            cur_sample = x0[i, :]
            cur_sample_mean_centered = cur_sample - mu_0
            cur_sample_mean_centered = cur_sample_mean_centered.reshape((1, d))

            current_scatter = np.matmul(cur_sample_mean_centered.transpose(), cur_sample_mean_centered)

            S_within_0 = S_within_0 + current_scatter

        for i in range(N_1):
            cur_sample = x1[i, :]
            cur_sample_mean_centered = cur_sample - mu_1
            cur_sample_mean_centered = cur_sample_mean_centered.reshape((1, d))

            current_scatter = np.matmul(cur_sample_mean_centered.transpose(), cur_sample_mean_centered)

            S_within_1 = S_within_1 + current_scatter

        S_within = S_within_0 + S_within_1

        W = np.matmul(LA.inv(S_within), mu_0-mu_1)

        # normalize W
        W = W / np.sqrt(sum(W**2))

        # get threshold
        decision_boundary = sum(W * 0.5 * (mu_0 + mu_1))

        # projections
        projection_0 = np.matmul(W, x0.transpose())
        projection_1 = np.matmul(W, x1.transpose())

        # get mu in the projection space
        mu_0_in_projection_space = sum(mu_0 * W)
        mu_1_in_projection_space = sum(mu_1 * W)

        result = {}
        result['W'] = W
        result['projection_0'] = projection_0
        result['projection_1'] = projection_1
        result['mu_0_in_projection_space'] = mu_0_in_projection_space
        result['mu_1_in_projection_space'] = mu_1_in_projection_space
        result['decision_boundary'] = decision_boundary

        return result

    def do_LDA_multi_class(self, X, Y):
        targets = np.unique(Y)

        # number of targets
        num_of_targets = len(targets)

        # number of features
        d = X.shape[1]

        # overall mu
        mu = np.mean(X, axis=0)
        mu = mu.reshape((1, d))

        II_dictionary = {}
        x_dictionary = {}
        mu_dictionary = {}
        S_within_dictionary = {}

        # initialize and get the mu vector for each target
        for i in range(num_of_targets):
            II_dictionary[i] = np.where(Y==targets[i])
            II_dictionary[i] = II_dictionary[i][0]

            x_dictionary[i] = X[II_dictionary[i], :]

            mu_dictionary[i] = np.mean(x_dictionary[i], axis=0)
            mu_dictionary[i] = mu_dictionary[i].reshape((1, d))

            S_within_dictionary[i] = np.zeros((d, d))

        # compute S_within for each target
        for index_target in range(num_of_targets):
            for index_sample in range(len(II_dictionary[index_target])):
                cur_sample = x_dictionary[index_target][index_sample]
                cur_sample = cur_sample.reshape((1, d))

                cur_sample_mean_centered = cur_sample - mu_dictionary[index_target]

                cur_scatter = np.matmul(cur_sample_mean_centered.transpose(), cur_sample_mean_centered)

                S_within_dictionary[index_target] = S_within_dictionary[index_target] + cur_scatter

        # compute total S_within
        S_within = np.zeros((d, d))
        for i in range(num_of_targets):
            S_within = S_within + S_within_dictionary[i]

        # compute S_between
        S_between = np.zeros((d, d))
        for i in range(num_of_targets):
            distance = mu_dictionary[i] - mu

            cur_S_between = len(II_dictionary[i]) * np.matmul(distance.transpose(), distance)

            S_between = S_between + cur_S_between

        # solve for W
        eigenvalues, W = LA.eig(np.matmul(LA.inv(S_within), S_between))

        W = W[:, 0:num_of_targets-1]

        W = W.real

        # mu in the projection space
        projection = np.matmul(X, W)
        mu_tilde_dictionary = {}
        for i in range(num_of_targets):
            mu_tilde_dictionary[i] = np.mean(projection[II_dictionary[i], :], axis=0)

        result = {}
        result['W'] = W
        result['projection'] = projection
        result['eigenvalues'] = eigenvalues
        result['mu_tilde_dictionary'] = mu_tilde_dictionary

        return result

    def fit(self, X, Y):
        if self.num_of_classes == 2:
            result = self.do_LDA_two_class(X, Y)
        elif self.num_of_classes > 2:
            result = self.do_LDA_multi_class(X, Y)
        else:
            sys.exit("Unknown number of classes!")

        return result
