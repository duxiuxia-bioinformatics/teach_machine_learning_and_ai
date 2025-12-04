# 2025 Fall

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # =================================================
    # import data and sample metadata
    # =================================================
    # import data
    in_file_name = '../../data/sample_data_for_project_final.csv'
    data_in_df = pd.read_csv(in_file_name, index_col=0, header=0)
    data_in_df = data_in_df.T

    # import sample metadata
    in_file_name = '../../data/sample_metadata_for_project_final.csv'
    sample_metadata_df = pd.read_csv(in_file_name, index_col=0, header=0)

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_in_df, sample_metadata_df, test_size=0.2, random_state=42)

    # standardize the data
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    # ===================================
    # run sklearn
    # ===================================
    MLP_classifier_obj = MLPClassifier(solver='lbfgs',
                                       activation='logistic',
                                       alpha=1e-15,
                                       random_state=1,
                                       hidden_layer_sizes=(2))
    MLP_classifier_obj.fit(X_train_scaled, y_train.to_numpy().ravel())

    probs_predicted_by_mlp = MLP_classifier_obj.predict_proba(X_test_scaled)
    y_predicted_by_mlp = MLP_classifier_obj.predict(X_test_scaled)
    MLP_accuracy = accuracy_score(y_test, y_predicted_by_mlp)
    print('Labels predicted by MLP:', y_predicted_by_mlp)
    print('True labels:', y_test)
    print('MLP accuracy:', MLP_accuracy)
    print('Probability predicted by MLP:', np.round(probs_predicted_by_mlp[:, 1], 2))

    return

if __name__ == '__main__':
    main()