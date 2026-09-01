# Author: Xiuxia Du

# =====================================
# import packages
# =====================================
import numpy as np
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



from sections.do_prediction import do_prediction
import sections.do_inference

# =====================================
# specify global parameters
# =====================================
RANDOM_STATE = 42
FONT_SIZE = 8
TICK_SIZE = 6

# =====================================
# =====================================
# =====================================
# select data release cycle
st.divider()
st.header('Select data release cycle')
cycle_selected = st.selectbox(
    'Select a cycle:',
    ['2005-2006', '2007-2008', '2009-2010'],
    index=0
)
st.write('Data release cycle that you have selected:', cycle_selected)

# load the data for the selected data release cycle
st.divider()
st.header('Load data')
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

# load lab measurement data
lab_data_df = pd.read_sas(lab_in_file_name, format='xport')


st.write('Lab measurements data:')
st.dataframe(lab_data_df)
st.write(f'Lab measurement data size: {lab_data_df.shape}')
st.divider()

# load questionnaire data
st.write('Load questionnaire data:')
questionnaire_data_df = pd.read_sas(questionnaire_in_file_name, format='xport')
st.dataframe(questionnaire_data_df)
st.write(f'Questionnaire data size: {questionnaire_data_df.shape}')
st.divider()

# merge the lab measurement with the questionnaire data
st.header('Merge lab measurements with questionnaire')
merged_df = pd.merge(lab_data_df, questionnaire_data_df, on='SEQN')
st.write('Merged lab measurements with the questionnaire:')
st.dataframe(merged_df)
st.write(f'Merged lab measurements with the questionnaire size: {merged_df.shape}')
st.divider()


# retrieve columns that are needed
needed_columns = ['URXDCB', 'URX14D', 'MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']
merged_filtered_df = merged_df[needed_columns]
st.write('After retrieving columns that are needed')
st.dataframe(merged_filtered_df)
st.write(f'Data size: {merged_filtered_df.shape}')
st.divider()

# filter out rows where there are missing values
st.header('Filter data')
num_of_missing_values_per_row = merged_filtered_df.isnull().sum(axis=1)
tf = num_of_missing_values_per_row <= 0
merged_filtered_no_missing_df = merged_filtered_df.loc[tf, :]
st.write('After filtering missing values:')
st.dataframe(merged_filtered_no_missing_df)
st.write(f'Data size: {merged_filtered_no_missing_df.shape}')
st.divider()

# ===========================================================
# get the histogram of URXDCB and URX14D
st.header('Examine the data')

st.subheader('Variables overall')
fig, ax = plt.subplots(figsize=(6, 4), dpi=50)
ax.hist(np.log(merged_filtered_no_missing_df['URXDCB']), bins=20)
ax.set_xlabel('URXDCB', fontsize=FONT_SIZE)
ax.set_ylabel('Number of missing values', fontsize=FONT_SIZE)
ax.set_ylabel('Frequency')
ax.set_title(f'{cycle_selected}: Histogram of URXDCB', fontsize=FONT_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
st.pyplot(fig, use_container_width=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(np.log(merged_filtered_no_missing_df['URX14D']), bins=20)
ax.set_xlabel('URX14D', fontsize=FONT_SIZE)
ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
ax.set_title(f'{cycle_selected}: Histogram of URX14D', fontsize=FONT_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
st.pyplot(fig)
st.divider()

# get the bar chart of the questionnaire results
st.subheader('Variables by health outcome')
fig, ax = plt.subplots(figsize=(6, 4), dpi=50)
for cur_outcome in ['MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']:
    st.write(f'{cur_outcome}:')
    counts = merged_filtered_no_missing_df[cur_outcome].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax)
    ax.set_xlabel(cur_outcome, fontsize=FONT_SIZE)
    ax.set_ylabel('Count', fontsize=FONT_SIZE)
    ax.set_title(f'{cycle_selected}: {cur_outcome} Distribution', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    st.pyplot(fig)

    # plot the histogram of URXDCB for answer 1 and 2.
    fig, ax = plt.subplots(figsize=(6, 4))
    tf = merged_filtered_no_missing_df[cur_outcome] == 2.0
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URXDCB']), bins=20, color='green', label=f'{cur_outcome}=2')
    tf = merged_filtered_no_missing_df[cur_outcome] == 1.0
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URXDCB']), bins=20, color='red', label=f'{cur_outcome}=1')
    ax.set_xlabel('URXDCB', fontsize=FONT_SIZE)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    ax.set_title(f'{cycle_selected}: Histogram of URXDCB', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    st.pyplot(fig)

    # plot the histogram of URX14D for answer 1 and 2.
    fig, ax = plt.subplots(figsize=(6, 4))
    tf = merged_filtered_no_missing_df[cur_outcome] == 2.0
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URX14D']), bins=20, color='green', label=f'{cur_outcome}=2')
    tf = merged_filtered_no_missing_df[cur_outcome] == 1.0
    ax.hist(np.log(merged_filtered_no_missing_df.loc[tf, 'URX14D']), bins=20, color='red', label=f'{cur_outcome}=2')

    ax.set_xlabel('URX14D', fontsize=FONT_SIZE)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    ax.set_title(f'{cycle_selected}: Histogram of URX14D', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    st.pyplot(fig)
    st.divider()

# do predictive modeling
st.header('Predictive modeling')
do_prediction(merged_filtered_no_missing_df)
st.divider()
# do_inference(cycle_selected, cur_outcome, merged_filtered_no_missing_df)
