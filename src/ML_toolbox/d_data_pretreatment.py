import numpy as np

class data_pretreatment():
    """
    Data pretreatment before applying machine learning methods.

    Attributes:
    """
    def __init__(self, X, treatment_method):
        self.X = X
        self.pretreatment_method = treatment_method

        self.num_of_samples = self.X.shape[0]
        self.num_of_variables = self.X.shape[1]

    def do_range_treatment(self):

        if self.num_of_samples > 1:
            treated_X = np.zeros((self.num_of_samples, self.num_of_variables))

            for i in range(self.num_of_variables):
                cur_column = self.X[:, i]

                col_min = np.min(cur_column) * np.ones(self.num_of_samples)
                col_range = (max(cur_column) - min(cur_column)) * np.ones(self.num_of_samples)

                treated_X[:, i] = (cur_column - col_min) / col_range
        else:
            print("No data pretreatment is needed!")
            treated_X = self.X

        return treated_X

    def do_pretreatment(self):
        if self.pretreatment_method == "range":
            treated_X = self.do_range_treatment()
        else:
            print("unknown pretreatment method")

        return treated_X
