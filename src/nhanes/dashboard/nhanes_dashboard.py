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

# build a logistic regression model for first quartile URX14D
tf = merged_filtered_no_missing_df['URX14D_quartile'] == 0
data_quartile_1_df = merged_filtered_no_missing_df.loc[tf, :]
st.write('quartile 1:')
st.dataframe(data_quartile_1_df)
X = data_quartile_1_df['URX14D']
y = data_quartile_1_df['MCQ160C']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=7
)

baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE))
])
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
