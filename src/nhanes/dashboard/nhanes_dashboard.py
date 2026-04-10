# Author: Xiuxia Du

import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
st.write('Lab Measurements')
st.dataframe(lab_data_df)

questionnaire_data_df = pd.read_sas(questionnaire_in_file_name, format='xport')
st.write('Questionnaire')
st.dataframe(questionnaire_data_df)

merged_df = pd.merge(lab_data_df, questionnaire_data_df, on='SEQN')
st.write('merged_df')
st.dataframe(merged_df)

# retrieve certain columns
needed_columns = ['URXDCB', 'URX14D', 'MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ160B', 'MCQ160M', 'MCQ160K']
merged_filtered_df = merged_df[needed_columns]
st.write('merged_filtered_df')
st.dataframe(merged_filtered_df)

# filter out rows where there are missing values
num_of_missing_values_per_row = merged_filtered_df.isnull().sum(axis=1)
tf = num_of_missing_values_per_row <= 1
merged_filtered_no_missing_df = merged_filtered_df.loc[tf, :]
st.write('merged_filtered_no_missing_df')
st.dataframe(merged_filtered_no_missing_df)
st.write(merged_filtered_no_missing_df.shape)

# get a bar plot
merged_filtered_no_missing_df['MCQ160C'].value_counts().plot(kind='bar')
merged_filtered_no_missing_df['MCQ160C'].value_count



# data_in_df = pd.read_csv('PFAS_J.csv')
#
# st.write('This is the data analysis dashboard for the NHANES dataset.')
#
# # filter out the comment code columns
# columns_to_drop = ['SEQN', 'WTSB2YR', 'LBDPFDEL', 'LBDPFHSL', 'LBDMPAHL', 'LBDPFNAL', 'LBDPFUAL', 'LBDNFOAL', 'LBDBFOAL', 'LBDNFOSL', 'LBDMFOSL']
# data_in_df.drop(columns=columns_to_drop, inplace=True)
# st.dataframe(data_in_df)
# st.write(f'{data_in_df.shape[0]}')
#
# # remove empty rows
# num_of_missing_columns = data_in_df.isna().sum(axis=1)
# tf = num_of_missing_columns / len(data_in_df) > 0.9 * data_in_df.shape[1]
# data_in_df.drop(index=tf[tf].index, inplace=True)
# st.dataframe(data_in_df)
# st.write(f'{data_in_df.shape[0]}')
#
# fig, ax = plt.subplots()
# sns.heatmap(data_in_df, cmap='viridis', ax=ax)
# st.pyplot(fig)
#
# corr = data_in_df.corr()
# st.write(corr)
#
