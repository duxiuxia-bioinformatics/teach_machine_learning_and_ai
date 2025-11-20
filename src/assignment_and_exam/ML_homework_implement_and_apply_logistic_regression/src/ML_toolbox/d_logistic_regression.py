import numpy as np

class logistic_regression:
    def __init__(self, X, y, delta_J_threshold, initial_theta, learning_rate, bool_regularization=False, regularization_lambda=0):
        self.number_of_samples = X.shape[0]

        self.X = X
        self.y = y

        self.delta_J_threshold = delta_J_threshold

        self.initial_theta = initial_theta

        self.learning_rate = learning_rate

        self.number_of_variables = self.X.shape[1]

        self.optimal_theta = np.zeros((len(self.initial_theta), 1))

        self.bool_regularization = bool_regularization

        self.regularization_lambda = regularization_lambda

    def get_logistic(self, z):
        y_hat = 1.0 /(1.0 + np.exp(-z))

        return y_hat

    def get_cost(self, theta):
        J = 0.0

        if self.bool_regularization:
            for i in range(self.number_of_samples):
                current_x = self.X[i, :]
                current_y = float(self.y[i].item())

                current_z = np.matmul(current_x, theta).item()  # linear combination of variables
                current_y_hat = self.get_logistic(current_z)  # logistic function

                J = J + current_y * np.log(current_y_hat) + (1.0 - current_y) * np.log(1.0 - current_y_hat)

            J = -J

            temp = 0.0
            for j in np.arange(start=1, stop=self.number_of_variables, step=1):
                current_theta = theta[j].item()
                temp = temp + current_theta * current_theta

            J = J + temp * self.regularization_lambda / 2.0

        else:
            for i in range(self.number_of_samples):
                current_x = self.X[i, :]
                # current_y = float(np.asscalar(self.y[i]))
                current_y = float(self.y[i].item())

                current_z = np.matmul(current_x, theta).item()  # linear combination of variables
                current_y_hat = self.get_logistic(current_z)  # logistic function

                J = J + current_y * np.log(current_y_hat) + (1.0 - current_y) * np.log(1.0 - current_y_hat)

            J = -J

        J = J / (self.number_of_samples)

        return J

    def get_gradient(self, theta):
        gradient = np.zeros((self.number_of_variables, 1))

        # with regularization
        if self.bool_regularization:

            # gradient for theta_0
            for i in range(self.number_of_samples):
                current_x = self.X[i, :]
                current_y = self.y[i].item()

                current_z = np.matmul(current_x, theta).item()
                current_y_hat = self.get_logistic(current_z)

                current_error = current_y_hat - current_y

                gradient[0] = gradient[0] + current_error * current_x[0]

            # gradient for theta_1 ... theta_n
            for j in np.arange(start=1, stop=self.number_of_variables, step=1):

                for i in range(self.number_of_samples):
                    current_x = self.X[i, :]
                    current_y = self.y[i].item()

                    current_z = np.matmul(current_x, theta).item()
                    current_y_hat = self.get_logistic(current_z)

                    current_error = current_y_hat - current_y

                    gradient[j] = gradient[j] + current_error * current_x[j]

                gradient[j] = gradient[j] + self.regularization_lambda * theta[j]

        # without regularization
        else:
            for j in range(self.number_of_variables):

                for i in range(self.number_of_samples):
                    current_x = self.X[i, :]
                    current_y = self.y[i]

                    current_z = np.matmul(current_x, theta)
                    current_y_hat = self.get_logistic(current_z)

                    current_error = current_y_hat - current_y

                    gradient[j] = gradient[j] + current_error * current_x[j]

        gradient = gradient / self.number_of_samples

        return gradient

    def fit(self):
        J_all_iterations = []

        # get initial cost
        initial_cost = self.get_cost(self.initial_theta)

        J_all_iterations.append(initial_cost)

        # the first iteration
        # get gradient
        new_gradient = self.get_gradient(self.initial_theta)

        new_theta = self.initial_theta - self.learning_rate * new_gradient

        # get new cost
        new_cost = self.get_cost(new_theta)

        J_all_iterations.append(new_cost)

        cost_difference = abs(initial_cost - new_cost)

        while cost_difference > self.delta_J_threshold:
            previous_cost = new_cost
            previous_theta = new_theta

            # get gradient
            new_gradient = self.get_gradient(previous_theta)

            new_theta = previous_theta - self.learning_rate * new_gradient

            new_cost = self.get_cost(new_theta)

            J_all_iterations.append(new_cost)

            cost_difference = abs(previous_cost - new_cost)

        optimal_theta = new_theta

        self.optimal_theta = optimal_theta

        return optimal_theta, J_all_iterations

    def predict(self, X_test):
        # X_test = np.asmatrix(X_test)
        #
        # x0 = np.ones((X_test.shape[0], 1))
        # X_test_augmented = np.hstack((x0, X_test))

        z = np.matmul(X_test, self.optimal_theta)

        # get y_hat
        if X_test.shape[0] == 1:
            # if there is only one test sample
            prob_predict_array = self.get_logistic(z[0][0])

            if z[0][0] > 0.5:
                label_predict_array = np.array(int(1))
            else:
                label_predict_array = np.array(int(0))

        else:
            prob_predict_list = []
            for i in range(len(z)):
                cur_z = z[i]
                cur_prob_predict = self.get_logistic(cur_z)
                prob_predict_list.append(cur_prob_predict)

            prob_predict_array = np.array(prob_predict_list)

            label_predict_list = []
            for i in range(len(z)):
                cur_y_hat = prob_predict_list[i]
                if cur_y_hat > 0.5:
                    label_predict_list.append(int(1))
                else:
                    label_predict_list.append(int(0))
            label_predict_array = np.array(label_predict_list)

        return label_predict_array, prob_predict_array