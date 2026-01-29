# 2025 Fall

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

    # train the model
    sklearn_logit_obj = LogisticRegression(random_state=0,
                                                 solver='liblinear',
                                                 fit_intercept=False,
                                                 tol=1e-6)
    sklearn_logit_obj.fit(X_train_scaled, y_train.to_numpy().ravel())
    y_pred = sklearn_logit_obj.predict(X_test_scaled)
    sklearn_accuracy = accuracy_score(y_test, y_pred)
    print('sklearn logistic regression accuracy: {}'.format(sklearn_accuracy))
    return

if __name__ == '__main__':
    main()