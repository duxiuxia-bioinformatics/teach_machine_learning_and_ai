import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_in_df = pd.read_csv('PFAS_J.csv')

st.write('This is the data analysis dashboard for the NHANES dataset.')

# filter out the comment code columns
columns_to_drop = ['SEQN', 'WTSB2YR', 'LBDPFDEL', 'LBDPFHSL', 'LBDMPAHL', 'LBDPFNAL', 'LBDPFUAL', 'LBDNFOAL', 'LBDBFOAL', 'LBDNFOSL', 'LBDMFOSL']
data_in_df.drop(columns=columns_to_drop, inplace=True)
st.dataframe(data_in_df)
st.write(f'{data_in_df.shape[0]}')

# remove empty rows
num_of_missing_columns = data_in_df.isna().sum(axis=1)
tf = num_of_missing_columns / len(data_in_df) > 0.9 * data_in_df.shape[1]
data_in_df.drop(index=tf[tf].index, inplace=True)
st.dataframe(data_in_df)
st.write(f'{data_in_df.shape[0]}')

fig, ax = plt.subplots()
sns.heatmap(data_in_df, cmap='viridis', ax=ax)
st.pyplot(fig)

corr = data_in_df.corr()
st.write(corr)

