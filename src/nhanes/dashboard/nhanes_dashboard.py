# Author: Xiuxia Du

import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report
)

RANDOM_STATE = 42
FONT_SIZE = 8
TICK_SIZE = 6

# ===================
# tabs
# ===================
cycle_selected = st.selectbox(
    'Select a cycle:',
    ['2005-2006', '2007-2008', '2009-2010'],
    index=0
)
st.write('You selected:', cycle_selected)

if cycle_selected == '2005-2006':
    lab_in_file_name = '../data/original/2005-2006/PP_D.xpt'
    questionnaire_in_file_name = '../data/original/2005-2006/MCQ_D.xpt'
elif cycle_selected == '2007-2008':
    lab_in_file_name = '../data/original/2007-2008/PP_E.xpt'
    questionnaire_in_file_name = '../data/original/2007-2008/MCQ_E.xpt'
elif cycle_selected == '2009-2010':
    lab_in_file_name = '../data/original/2009-2010/PP_F.xpt'
    questionnaire_in_file_name = '../data/original/2009-2010/MCQ_F.xpt'
else:
    cycle_selected = '2005-2006'

lab_data_df = pd.read_sas(lab_in_file_name, format='xport')
st.write('Lab measurements data:')
st.dataframe(lab_data_df)

questionnaire_data_df = pd.read_sas(questionnaire_in_file_name, format='xport')
st.write(f'Questionnaire data: {questionnaire_data_df.shape}')
st.dataframe(questionnaire_data_df)
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    tf = (questionnaire_data_df[cur_outcome] == 7.0)
    st.write(f'{cur_outcome} has {sum(tf)} 7s')

    tf = (questionnaire_data_df[cur_outcome].astype(str).str.contains('7'))
    st.write(f'{cur_outcome} has {sum(tf)} 7s')
    st.write(questionnaire_data_df.loc[tf, 'SEQN'])
    st.write('   ')

st.write(os.getcwd())
out_file_name = 'questionnaire_data.csv'
questionnaire_data_df.to_csv(out_file_name)

st.write('   ')
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    tf = (questionnaire_data_df[cur_outcome] == 9.0)
    st.write(f'{cur_outcome} has {sum(tf)} 9s')

tf = questionnaire_data_df['MCQ160M'] == 7
st.write(questionnaire_data_df.loc[tf, :])

merged_df = pd.merge(lab_data_df, questionnaire_data_df, on='SEQN')
st.write('After merging the lab measurements with the questionnaire:')
st.dataframe(merged_df)

st.write('merging')
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    tf = (merged_df[cur_outcome] == 7.0) | (merged_df[cur_outcome] == 9.0)
    st.write(f'{cur_outcome} has {sum(tf)} 7s or 9s')

# retrieve certain columns
needed_columns = ['URXDCB', 'URX14D', 'MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']
merged_filtered_df = merged_df[needed_columns]
st.write('Needed columns have been retrieved')
st.dataframe(merged_filtered_df)

# filter out rows where there are missing values
num_of_missing_values_per_row = merged_filtered_df.isnull().sum(axis=1)
tf = num_of_missing_values_per_row <= 0
merged_filtered_no_missing_df = merged_filtered_df.loc[tf, :]
st.write('After filtering missing values:')
st.dataframe(merged_filtered_no_missing_df)

st.write('after filtering missing values')
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    tf = (merged_filtered_no_missing_df[cur_outcome] == 7.0)
    st.write(f'{cur_outcome} has {sum(tf)} 7s')

st.write('after filtering missing values')
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    tf = (merged_filtered_no_missing_df[cur_outcome] == 9.0)
    st.write(f'{cur_outcome} has {sum(tf)} 9s')


st.write(f'size of the dataset for further analysis: {merged_filtered_no_missing_df.shape}')

# get the histogram of URXDCB and URX14D
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(np.log(merged_filtered_no_missing_df['URXDCB']), bins=20)
ax.set_xlabel('URXDCB')
ax.set_ylabel('Frequency')
ax.set_title(f'{cycle_selected}: Histogram of URXDCB')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(np.log(merged_filtered_no_missing_df['URX14D']), bins=20)
ax.set_xlabel('URX14D')
ax.set_ylabel('Frequency')
ax.set_title(f'{cycle_selected}: Histogram of URX14D')
st.pyplot(fig)

# get the bar chart of the questionnaire results
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    counts = merged_filtered_no_missing_df[cur_outcome].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax)
    ax.set_xlabel(cur_outcome)
    ax.set_ylabel('Count')
    ax.set_title(f'{cycle_selected}: {cur_outcome} Distribution')
    st.pyplot(fig)

    # plot the histogram of URXDCB for answer 1 and 2.
    fig, ax = plt.subplots(figsize=(6, 4))

    tf = merged_filtered_no_missing_df[cur_outcome] == 2.0
    st.write(sum(tf))
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URXDCB']), bins=20, color='green', label=f'{cur_outcome}=2')

    tf = merged_filtered_no_missing_df[cur_outcome] == 1.0
    st.write(sum(tf))
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URXDCB']), bins=20, color='red', label=f'{cur_outcome}=1')

    ax.set_xlabel('URXDCB')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{cycle_selected}: Histogram of URXDCB')
    ax.legend()
    st.pyplot(fig)

    # plot the histogram of URX14D for answer 1 and 2.
    fig, ax = plt.subplots(figsize=(6, 4))

    tf = merged_filtered_no_missing_df[cur_outcome] == 2.0
    st.write(sum(tf))
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URX14D']), bins=20, color='green', label=f'{cur_outcome}=2')

    tf = merged_filtered_no_missing_df[cur_outcome] == 1.0
    st.write(sum(tf))
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URX14D']), bins=20, color='red', label=f'{cur_outcome}=2')

    ax.set_xlabel('URX14D')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{cycle_selected}: Histogram of URX14D')
    ax.legend()
    st.pyplot(fig)

# Split the data into four quartiles and build a logistic regression model for each quartile
merged_filtered_no_missing_df['URX14D_quartile'] = pd.qcut(merged_filtered_no_missing_df['URX14D'], q=4, labels=False)
st.write('After quartiles:')
st.dataframe(merged_filtered_no_missing_df)

# ============= prediction
for cur_variable in ['URXDCB', 'URX14D']:
    for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
        data_for_prediction_df = merged_filtered_no_missing_df[[cur_variable, cur_outcome]]

        tf = data_for_prediction_df[cur_outcome] < 2.5
        cur_data_filtered_df = data_for_prediction_df.loc[tf, :]
        st.write(f'{cur_outcome}: {cur_variable} has {sum(tf)} rows')

        X = cur_data_filtered_df[[cur_variable]]
        y = cur_data_filtered_df[cur_outcome]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.01,
            stratify=y,
            random_state=7s
        )

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
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

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


# ============= statistical inference
# build a logistic regression model for first quartile URX14D
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


#
# baseline.fit(X_train, y_train)
#
# probs = baseline.predict_proba(X_test)[:, 1]  # probability of class 1
# preds_05 = (probs >= 0.5).astype(int)
#
# fpr, tpr, thresholds = roc_curve(y_test, probs)
# auc = roc_auc_score(y_test, probs)
#
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
# ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
# ax.set_xlabel("False Positive Rate")
# ax.set_ylabel("True Positive Rate (Recall)")
# ax.set_title("ROC Curve")
# ax.legend()
# st.pyplot(fig)

# print("ROC AUC:", auc)

# tf = merged_filtered_no_missing_df['URX14D_quartile'] == 1
# st.write(sum(tf))
#
# tf = merged_filtered_no_missing_df['URX14D_quartile'] == 2
# st.write(sum(tf))
#
# tf = merged_filtered_no_missing_df['URX14D_quartile'] == 3
# st.write(sum(tf))
