import numpy as np

class data_model:
    """
    Class for creating data models.

    Attributes:
    """

    def __init__(self, data):
        self.data = np.array([])
        self.feature_names = []
        self.target = np.array([])
        self.target_names = []

        self.make_data_model(data)

    def make_data_model(self, data_in):
        num_of_variables = data_in.shape[1] - 1
        target = data_in.iloc[:, num_of_variables]
        data = data_in.iloc[:, 0:num_of_variables]

        target_names = np.unique(target)
        feature_names = (data_in.keys())[0:num_of_variables]

        self.data = data.values
        self.feature_names = feature_names.tolist()
        self.target = target.values
        self.target_names = target_names.tolist()


