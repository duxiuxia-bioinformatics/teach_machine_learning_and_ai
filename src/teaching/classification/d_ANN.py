import numpy as np

class ANN:
    """
    Class for a neural network with one hidden layer.

    Attributes:
    """

    def __init__(self, num_of_variables,
                 num_of_hidden_units,
                 num_of_outputs,
                 num_of_iterations,
                 learning_rate,
                 bool_is_classification=True,
                 initial_theta={}):

        self.num_of_samples = 0
        self.num_of_variables = num_of_variables
        self.num_of_hidden_units = num_of_hidden_units
        self.num_of_outputs = num_of_outputs

        self.num_of_iterations = num_of_iterations
        self.learning_rate = learning_rate

        self.bool_is_classification = bool_is_classification

        if not initial_theta:
            self.initial_theta = self.get_initial_theta()
        else:
            self.initial_theta = initial_theta

    def get_initial_theta(self):
        initial_theta = {}

        initial_theta['layer_1'] = 2.0 * np.random.rand(self.num_of_hidden_units, self.num_of_variables+1) - 1.0
        initial_theta['layer_2'] = 2.0 * np.random.rand(self.num_of_outputs, self.num_of_hidden_units+1) - 1.0

        return initial_theta

    def get_z(self, ww, a):
        z = np.matmul(ww, a)
        return z

    def get_a(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def do_forward_propagation(self, one_sample_x, w):
        one_sample_x = one_sample_x.reshape((len(one_sample_x), 1))

        # hidden layer input and output
        one_sample_x = np.vstack((np.array([1.0]), one_sample_x))

        z_upper_2 = self.get_z(w['layer_1'], one_sample_x)
        a_upper_2 = self.get_a(z_upper_2)

        # output layer input and output
        a_upper_2 = np.vstack((np.array([1.0]), a_upper_2))
        z_upper_3 = self.get_z(w['layer_2'], a_upper_2)
        a_upper_3 = self.get_a(z_upper_3)

        z_upper_2 = np.reshape(z_upper_2, (len(z_upper_2), 1))
        a_upper_2 = np.reshape(a_upper_2, (len(a_upper_2), 1))
        z_upper_3 = np.reshape(z_upper_3, (len(z_upper_3), 1))
        a_upper_3 = np.reshape(a_upper_3, (len(a_upper_3), 1))

        z = {}
        z['layer_2'] = z_upper_2
        z['layer_3'] = z_upper_3

        a = {}
        a['layer_1'] = one_sample_x
        a['layer_2'] = a_upper_2
        a['layer_3'] = a_upper_3

        return z, a

    def get_mean_squared_cost_one_sample(self, actual, predicted):
        cost_one_sample = 0.5 * sum((actual-predicted)**2)
        return cost_one_sample

    def get_cross_entropy_cost_one_sample(self, actual, predicted):
        cost_one_sample = 0.0

        if self.num_of_outputs == 1:
            cost_one_sample = - actual * np.log(predicted) - (1.0 - actual) * (1.0 - np.log(predicted))
        else:
            for i in range(self.num_of_outputs):
                cost_one_output_unit = - actual[i] * np.log(predicted[i]) - (1.0 - actual[i]) * (1.0 - np.log(predicted[i]))

                cost_one_sample = cost_one_sample + cost_one_output_unit

        return cost_one_sample

    def get_derivative_of_total_cost_wrt_a_superscript_3(self, actual, predicted):
        if self.bool_is_classification:
            d = (predicted - actual) / (predicted * (1 - predicted))
        else:
            d = predicted - actual
        return d

    def get_derivative_of_a_wrt_z(self, a):
        d = a * (1.0 - a)
        return d

    def get_derivative_of_total_cost_wrt_z_superscript_3(self, actual, predicted):
        temp1 = self.get_derivative_of_total_cost_wrt_a_superscript_3(actual, predicted)
        temp2 = self.get_derivative_of_a_wrt_z(predicted)

        d = temp1 * temp2
        return d

    def do_back_propagation(self, actual_output, a, w):
        delta = {}
        delta['layer_3'] = self.get_derivative_of_total_cost_wrt_z_superscript_3(actual=actual_output,
                                                                                 predicted=a['layer_3'])

        d_total_cost_wrt_theta = {}
        d_total_cost_wrt_theta['layer_2'] = np.matmul(delta['layer_3'], a['layer_2'].transpose())

        d_total_cost_wrt_a_superscript_2 = np.matmul(delta['layer_3'].transpose(), w['layer_2'][0:2, 1:3])
        delta['layer_2'] = d_total_cost_wrt_a_superscript_2.transpose() * self.get_derivative_of_a_wrt_z(a['layer_2'])[1:3]

        d_total_cost_wrt_theta['layer_1'] = np.matmul(delta['layer_2'], a['layer_1'].transpose())

        return d_total_cost_wrt_theta

    def fit(self, X, Y):
        self.num_of_samples = X.shape[0]

        theta = []
        cost_list = []
        h = []

        theta.append(self.initial_theta)

        for index_iter in np.arange(start=1, stop=self.num_of_iterations, step=1):
            if index_iter % 100 == 0:
                print("iteration = " + str(index_iter))

            d_total_cost_wrt_theta_all_samples = {}
            d_total_cost_wrt_theta_all_samples['layer_1'] = np.zeros((self.num_of_hidden_units, self.num_of_variables+1))
            d_total_cost_wrt_theta_all_samples['layer_2'] = np.zeros((self.num_of_outputs, self.num_of_hidden_units+1))

            theta_this_iteration = theta[index_iter - 1]
            cost_all_samples_this_iteration = 0

            for index_sample in range(self.num_of_samples):
                x_one_sample = X[index_sample, :]
                x_one_sample = np.reshape(x_one_sample, (len(x_one_sample), 1))

                if self.num_of_outputs > 1:
                    y_one_sample = Y[index_sample, :]
                    y_one_sample = np.reshape(y_one_sample, (len(y_one_sample), 1))
                else:
                    y_one_sample = Y[index_sample]

                # forward propagation
                z, a = self.do_forward_propagation(one_sample_x=x_one_sample, w=theta_this_iteration)
                current_h = a['layer_3']

                if not self.bool_is_classification:
                    cur_cost = self.get_mean_squared_cost_one_sample(actual=y_one_sample, predicted=current_h)
                else:
                    cur_cost = self.get_cross_entropy_cost_one_sample(actual=y_one_sample, predicted=current_h)

                cost_all_samples_this_iteration = cost_all_samples_this_iteration + cur_cost

                # backward propagation
                d_total_cost_wrt_theta_one_sample = self.do_back_propagation(actual_output=y_one_sample,
                                                                             a=a,
                                                                             w=theta_this_iteration)

                d_total_cost_wrt_theta_all_samples['layer_1'] = d_total_cost_wrt_theta_all_samples['layer_1'] + d_total_cost_wrt_theta_one_sample['layer_1']
                d_total_cost_wrt_theta_all_samples['layer_2'] = d_total_cost_wrt_theta_all_samples['layer_2'] + d_total_cost_wrt_theta_one_sample['layer_2']

            # gradient descent: done with one iteration and update theta
            d_total_cost_wrt_theta_all_samples['layer_1'] = d_total_cost_wrt_theta_all_samples['layer_1'] / self.num_of_samples
            d_total_cost_wrt_theta_all_samples['layer_2'] = d_total_cost_wrt_theta_all_samples['layer_2'] / self.num_of_samples

            new_theta = {}
            new_theta['layer_2'] = theta_this_iteration['layer_2'] - self.learning_rate * d_total_cost_wrt_theta_all_samples['layer_2']
            new_theta['layer_1'] = theta_this_iteration['layer_1'] - self.learning_rate * d_total_cost_wrt_theta_all_samples['layer_1']

            theta.append(new_theta)
            cost_list.append(cost_all_samples_this_iteration)
            h.append(current_h)

        # done with all the iterations
        return theta, cost_list, h