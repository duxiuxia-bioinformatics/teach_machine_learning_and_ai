import numpy as np

class MLR:
    def __init__(self, delta_J_threshold, initial_theta, learning_rate):
        self.delta_J_threshold = delta_J_threshold
        self.initial_theta = initial_theta
        self.learning_rate = learning_rate

    def get_cost(self, theta):
        J = 0.0

        for i in range(self.number_samples):
            current_x = self.X[i, :].reshape((1, self.number_of_variables))
            current_y = self.y[i]

            # y_hat = np.matmul(theta.transpose(), current_x)
            y_hat = np.matmul(current_x, theta)
            y_hat = y_hat[0, 0]

            J = J + (y_hat - current_y)**2

        J = J / (2 * self.number_samples)

        return J

    def get_gradient(self, theta):
        gradient = np.zeros((self.number_of_variables, 1))

        for i in range(self.number_of_variables):
            for j in range(self.number_samples):
                current_x = self.X[j, :].reshape((1, self.number_of_variables))
                current_y = self.y[j]

                current_y_hat = np.matmul(current_x, theta)
                current_y_hat = current_y_hat[0, 0] # turn array to scalar
                current_error = current_y_hat - current_y

                gradient[i, 0] = gradient[i, 0] + current_error * current_x[0, i]

        gradient = gradient / self.number_samples

        return gradient

    def do_gradient_descent(self, ):
        J = []

        # get initial cost
        initial_cost = self.get_cost(self.initial_theta)

        J.append(initial_cost)

        # the first iteration
        # get gradient
        new_gradient = self.get_gradient(self.initial_theta)

        new_theta = self.initial_theta - self.learning_rate * new_gradient

        # get new cost
        new_cost = self.get_cost(new_theta)

        J.append(new_cost)

        cost_difference = abs(initial_cost - new_cost)

        while cost_difference > self.delta_J_threshold:
            previous_cost = new_cost
            previous_theta = new_theta

            # get gradient
            new_gradient = self.get_gradient(previous_theta)

            new_theta = previous_theta - self.learning_rate * new_gradient

            new_cost = self.get_cost(new_theta)

            J.append(new_cost)

            cost_difference = abs(previous_cost - new_cost)

        optimal_theta = new_theta

        self.optimal_theta = optimal_theta
        self.J = J

        return

    def fit(self, X, y):
        self.X = X.values
        self.y = y.values

        self.number_samples = self.X.shape[0]
        self.number_of_variables = self.X.shape[1]

        self.do_gradient_descent()
        return