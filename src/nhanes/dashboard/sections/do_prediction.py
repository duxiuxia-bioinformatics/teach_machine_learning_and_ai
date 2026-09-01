import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report
)

# from src.nhanes.dashboard.sections.do_inference import FONT_SIZE

RANDOM_STATE = 42
FONT_SIZE = 8
TICK_SIZE = 6

def do_prediction(merged_filtered_no_missing_df):
    st.dataframe(merged_filtered_no_missing_df)
    st.write(f'Data size: {merged_filtered_no_missing_df.shape}')
    st.divider()

    for cur_variable in ['URXDCB', 'URX14D']:
        for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
            data_for_prediction_df = merged_filtered_no_missing_df[[cur_variable, cur_outcome]]

            tf = data_for_prediction_df[cur_outcome] < 2.5
            cur_data_filtered_df = data_for_prediction_df.loc[tf, :]
            st.write(f'{cur_outcome}: {cur_variable} has {sum(tf)} rows')

            tf = cur_data_filtered_df[cur_outcome] == 2.0
            cur_data_filtered_df.loc[tf, cur_outcome] = 0.0
            X = np.log(cur_data_filtered_df[[cur_variable]])
            y = cur_data_filtered_df[cur_outcome]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.01,
                stratify=y,
                random_state=7
            )

            counts = y_train.value_counts()
            st.write(f'y_train counts:')
            st.write(counts)

            counts = y_test.value_counts()
            st.write(f'y_test counts:')
            st.write(counts)

            # fig, ax = plt.subplots(figsize=(6, 4))
            # counts.plot(kind='bar', ax=ax)
            # ax.set_xlabel(cur_outcome, fontsize=FONT_SIZE)
            # ax.set_ylabel('Count', fontsize=FONT_SIZE)
            # ax.set_title(f'{cycle_selected}: {cur_outcome} Distribution', fontsize=FONT_SIZE)
            # ax.tick_params(axis='both', labelsize=TICK_SIZE)
            # st.pyplot(fig)

            # tf = (y_train==0)


            lr_model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE))
            ])
            lr_model.fit(X_train, y_train)

            probs = lr_model.predict_proba(X_test)[:, 1]
            predicted_labels = (probs > 0.5).astype(int)

            cm = confusion_matrix(y_test, predicted_labels)
            fig, ax = plt.subplots(figsize=(4, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            ax.set_title('Logistic regression confusion matrix', fontsize=FONT_SIZE)
            st.pyplot(fig)
            st.divider()

            # build a SVM model
            my_C = 1  # 1.0
            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='linear', C=my_C))
            ])
            svm_pipeline.fit(X_train, y_train)
            predicted = svm_pipeline.predict(X_test)

            # build a neural network model
            mlp_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(20,),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ))
            ])

            mlp_pipeline.fit(X_train, y_train)

            y_pred_mlp = mlp_pipeline.predict(X_test)
            y_prob_mlp = mlp_pipeline.predict_proba(X_test)[:, 1]


            # precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
            # ap = average_precision_score(y_test, probs)
            #
            # plt.figure()
            # plt.plot(recall, precision, label=f"Avg Precision = {ap:.3f}")
            # plt.xlabel("Recall")
            # plt.ylabel("Precision")
            # plt.title("Precision–Recall Curve")
            # plt.legend()
            # plt.show()
    return
