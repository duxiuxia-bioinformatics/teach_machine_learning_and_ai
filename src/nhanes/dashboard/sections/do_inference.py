import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
FONT_SIZE = 8
TICK_SIZE = 6

def do_inference(cycle_selected, cur_outcome, merged_filtered_no_missing_df):

    disease_outcome = 'MCQ160C'

    for cur_quartile in [0, 1, 2, 3]:
        st.write(f'current quartile: {cur_quartile}')

        tf = merged_filtered_no_missing_df['URX14D_quartile'] == cur_quartile
        data_quartile_1_df = merged_filtered_no_missing_df.loc[tf, :]
        st.write('data')
        st.dataframe(data_quartile_1_df)

        # remove samples whose disease outcome is not 1 or 2
        tf = data_quartile_1_df[disease_outcome] < 2.5
        data_quartile_1_filtered_df = data_quartile_1_df.loc[tf, :]
        X = data_quartile_1_filtered_df[['URX14D']]
        y = data_quartile_1_filtered_df[disease_outcome]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.01,
            stratify=y,
            random_state=7
        )

        # barchart of the disease status
        counts = y_test.value_counts()
        fig, ax = plt.subplots(figsize=(4, 2))
        counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(cur_outcome, fontsize=FONT_SIZE)
        ax.set_ylabel('Count', fontsize=FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_title(f'{cycle_selected}, {cur_quartile}: y_test {disease_outcome} Distribution', fontsize=FONT_SIZE)
        st.pyplot(fig)

        counts = y_train.value_counts()
        fig, ax = plt.subplots(figsize=(4, 2))
        counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(cur_outcome, fontsize=FONT_SIZE)
        ax.set_ylabel('Count', fontsize=FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_title(f'{cycle_selected}, {cur_quartile}: y_train {disease_outcome} Distribution', fontsize=FONT_SIZE)
        st.pyplot(fig)


        st.write('X_train')
        st.dataframe(X_train)

        baseline_model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE))
        ])

        baseline_model.fit(X_train, y_train)

        # probs = baseline_model.predict_proba(X_test)[:, 1]
        # predicted_labels = (probs > 0.5).astype(int)
        # cm = confusion_matrix(y_test, predicted_labels)
        # fig, ax = plt.subplots(figsize=(4, 4))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot(ax=ax)
        # ax.set_title('Confusion Matrix')
        # st.pyplot(fig)

        st.write('coefficient')
        coefs = baseline_model.named_steps['lr'].coef_
        st.write(coefs)
        st.write(f'odds-ratio: {np.exp(coefs)}')

        intercept = baseline_model.named_steps['lr'].intercept_
        st.write('intercept')
        st.write(intercept)