# Author: Xiuxia Du
# 2025-09-25

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

# ============================================
# apply logistic regresesino to the PFAS data
# ============================================
bool_use_sklearn = True

in_file_full_name = '../../../data/pfas.csv'
data_in_df = pd.read_csv(in_file_full_name)
print(data_in_df.head())

# log transform PFAS
data_in_df['log_PFOS'] = np.log(data_in_df['PFOS'])
print(data_in_df.head())

# log_pfas_col_name = ['log_PFOS']
# log_pfas_col_name = ['log_PFOS', 'age']
# log_pfas_col_name = ['log_PFOS', 'gender']
# log_pfas_col_name = ['log_PFOS', 'BMI']
log_pfas_col_name = ['log_PFOS', 'age', 'gender', 'BMI']

# standardize the data
X = data_in_df[log_pfas_col_name]

X_scaled = StandardScaler().fit_transform(X)

# fit model 1 with only the log_PFOS as the variable
if not bool_use_sklearn:
    X_scaled_with_constant = sm.add_constant(X_scaled)

    model_1 = sm.Logit(data_in_df['disease'], X_scaled_with_constant)
    result_1 = model_1.fit()
    print(result_1.summary())

    intercept = result_1.params['const']
    coef = result_1.params['x1']
    print(intercept)
    print(coef)
else:
    model_1 = LogisticRegression(solver='lbfgs', penalty=None)
    model_1.fit(X=X_scaled, y=data_in_df['disease'])
    print(log_pfas_col_name)
    print(model_1.intercept_)
    print(model_1.coef_)






xx = 1






