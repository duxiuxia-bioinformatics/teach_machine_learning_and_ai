import pandas as pd
import statsmodels.formula.api as smf

def fit_two_models(df, y_col="IL6", disease_col="disease", age_col="age", label="dataset"):
    print(f"\n=== {label} ===")
    print(df)

    # Model A: IL6 ~ disease
    mA = smf.ols(f"{y_col} ~ {disease_col}", data=df).fit()

    # Model B: IL6 ~ disease + age
    mB = smf.ols(f"{y_col} ~ {disease_col} + {age_col}", data=df).fit()

    print("\nModel A: IL6 ~ disease")
    print(f"  beta_disease = {mA.params[disease_col]:.4f}")
    print(f"  p-value      = {mA.pvalues[disease_col]:.4g}")
    print(f"  R^2          = {mA.rsquared:.4f}")

    print("\nModel B: IL6 ~ disease + age")
    print(f"  beta_disease = {mB.params[disease_col]:.4f}")
    print(f"  p-value      = {mB.pvalues[disease_col]:.4g}")
    print(f"  beta_age     = {mB.params[age_col]:.4f}")
    print(f"  p-value(age) = {mB.pvalues[age_col]:.4g}")
    print(f"  R^2          = {mB.rsquared:.4f}")

    # If you want full tables, uncomment:
    # print("\n--- Full summaries ---")
    # print(mA.summary())
    # print(mB.summary())

# ---------------------------
# Dataset 1: No age overlap
# ---------------------------
df1 = pd.DataFrame({
    "disease": [0,0,0,0, 1,1,1,1],
    "age":     [25,26,27,28, 55,56,57,58],
    "IL6":     [5.0,5.2,5.4,5.6, 7.8,8.0,8.2,8.4],
})

# -----------------------------------------
# Dataset 2: Slight overlap (age 40 and 45)
# -----------------------------------------
df2 = pd.DataFrame({
    "disease": [0,0,0,0, 1,1,1,1],
    "age":     [30,35,40,45, 40,45,50,55],
    "IL6":     [6.0,6.5,7.0,7.5, 7.0,7.5,8.0,8.5],
})

fit_two_models(df1, label="Toy dataset 1 (no age overlap)")
fit_two_models(df2, label="Toy dataset 2 (some age overlap)")
