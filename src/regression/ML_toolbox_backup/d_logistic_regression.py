import numpy as np

class logistic_regression:
    def __init__(self, X, y, delta_J_threshold, initial_theta, learning_rate, bool_regularization=False, regularization_lambda=0):
        self.number_of_samples = X.shape[0]

        x0 = np.ones((self.number_of_samples, 1))
        self.X = np.hstack((x0, X))

        self.y = y
        self.delta_J_threshold = delta_J_threshold

        initial_theta_for_x0 = np.array([0])
        self.initial_theta = np.vstack((initial_theta_for_x0, initial_theta))

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
                current_y = float(np.item(self.y[i]))

                current_z = np.asscalar(np.matmul(current_x, theta))  # linear combination of variables
                current_y_hat = self.get_logistic(current_z)  # logistic function

                J = J + current_y * np.log(current_y_hat) + (1.0 - current_y) * np.log(1.0 - current_y_hat)

            J = -J

            temp = 0.0
            for j in np.arange(start=1, stop=self.number_of_variables, step=1):
                current_theta = np.asscalar(theta[j])
                temp = temp + current_theta * current_theta

            J = J + temp * self.regularization_lambda / 2.0

        else:
            for i in range(self.number_of_samples):
                current_x = self.X[i, :]
                current_y = float(np.asscalar(self.y[i]))

                current_z = np.asscalar(np.matmul(current_x, theta))  # linear combination of variables
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
                current_y = np.asscalar(self.y[i])

                current_z = np.asscalar(np.matmul(current_x, theta))
                current_y_hat = self.get_logistic(current_z)

                current_error = current_y_hat - current_y

                gradient[0] = gradient[0] + current_error * current_x[0]

            # gradient for theta_1 ... theta_n
            for j in np.arange(start=1, stop=self.number_of_variables, step=1):

                for i in range(self.number_of_samples):
                    current_x = self.X[i, :]
                    current_y = np.asscalar(self.y[i])

                    current_z = np.asscalar(np.matmul(current_x, theta))
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
        X_test = np.asmatrix(X_test)

        x0 = np.ones((X_test.shape[0], 1))
        X_test_augmented = np.hstack((x0, X_test))

        z = np.matmul(X_test_augmented, self.optimal_theta)

        y_hat = self.get_logistic(z[0, 0])

        y_hat = np.array(y_hat)
        y_predict = np.zeros((X_test.shape[0], 1))
        II = np.where(y_hat > 0.5)
        if len(II[0]) > 0:
            y_predict[II] = 1.0

        return y_predict