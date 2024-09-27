from sklearn.datasets import load_diabetes
import pandas as pd


def load_diabetes_data():
    diabetes_data = load_diabetes()

    df_diabetes = pd.DataFrame(
        data=diabetes_data.data,
        columns=diabetes_data.feature_names
    )
    df_diabetes['target'] = diabetes_data.target


    return df_diabetes
