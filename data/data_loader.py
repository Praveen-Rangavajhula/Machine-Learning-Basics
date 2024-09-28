from sklearn.datasets import load_diabetes
from scipy.stats import boxcox
import pandas as pd


def load_diabetes_data():
    diabetes_data = load_diabetes()

    df_diabetes = pd.DataFrame(
        data=diabetes_data.data,
        columns=diabetes_data.feature_names
    )
    df_diabetes['target'] = diabetes_data.target

    print(df_diabetes.head())

    df_diabetes['target'] = apply_box_cox_transformation(df_diabetes['target'])

    return df_diabetes

def apply_box_cox_transformation(target_series ):
    transformed_target , _, _ = boxcox(target_series , alpha=0.05)
    return transformed_target
