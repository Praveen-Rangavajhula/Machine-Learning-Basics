from sklearn.datasets import load_diabetes
from scipy.stats import boxcox
import pandas as pd

def load_diabetes_data():
    """
    Load the diabetes dataset, apply Box-Cox transformation to the target variable, and return it as a DataFrame.

    Returns:
    DataFrame: A pandas DataFrame containing the diabetes dataset with transformed target values.
    """
    diabetes_data = load_diabetes()

    df_diabetes = pd.DataFrame(
        data=diabetes_data.data,
        columns=diabetes_data.feature_names
    )

    df_diabetes['target'] = diabetes_data.target

    df_diabetes['target'] = apply_box_cox_transformation(df_diabetes['target'])

    return df_diabetes

def apply_box_cox_transformation(target_series):
    """
    Apply Box-Cox transformation to the target series.

    Parameters:
    target_series (Series): A pandas Series containing the target values to be transformed.

    Returns:
    Series: A pandas Series containing the transformed target values.
    """
    transformed_target, _, _ = boxcox(target_series, alpha=0.05)
    return transformed_target
