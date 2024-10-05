import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_linearity(y_test, y_pred_eval):
    """Check for linearity by plotting actual vs predicted target values."""
    try:
        plt.figure(figsize=(14, 8))
        sns.scatterplot(x=y_test, y=y_pred_eval, color='r')
        plt.title("Checking for linearity:\nActual vs Predicted target values")
        plt.xlabel("Actual target values")
        plt.ylabel("Predicted target values")

        # Adding a reference line for better visualization
        max_value = max(y_test.max(), y_pred_eval.max())
        min_value = min(y_test.min(), y_pred_eval.min())
        plt.plot([min_value, max_value], [min_value, max_value], color='blue', linestyle='--', linewidth=2)

        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        plt.grid()
        plt.show()
    except Exception as e:
        print(f"Error in plot_linearity: {e}")


def plot_residual_normality(y_test, y_pred_eval):
    """Check for residual normality and mean by plotting residuals."""
    try:
        plt.figure(figsize=(14, 8))
        e = y_test - y_pred_eval
        sns.histplot(e, color='b', kde=True, stat='density', bins=30)
        plt.title("Checking for Residual Normality and Mean:\nResidual Error (e)")
        plt.xlabel("Residuals")
        plt.ylabel("Density")

        mean_residual = np.mean(e)
        plt.axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_residual:.2f}')

        plt.legend()
        plt.show()

        return e
    except Exception as e:
        print(f"Error in plot_residual_normality: {e}")
        return None


def plot_multivariate_normality(e):
    """Check for multivariate normality using a Q-Q plot and return the correlation coefficient."""
    try:
        plt.figure(figsize=(14, 8))
        _, (ax, _, r) = sp.stats.probplot(e, dist="norm", plot=plt)  # Specifying the normal distribution
        plt.title("Check for Multivariate Normality\nQ-Q Plot")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.show()

        print(f"Correlation coefficient (r): {r:.4f}")
        return r
    except Exception as e:
        print(f"Error in plot_multivariate_normality: {e}")
        return None


def plot_homoscedasticity_and_vif(y_pred_eval, e, X):
    """Check for homoscedasticity and calculate Variance Inflation Factor (VIF)."""
    try:
        # Check for Homoscedasticity
        plt.figure(figsize=(14, 8))
        sns.scatterplot(x=y_pred_eval, y=e, color='green')
        plt.title("Check for Homoscedasticity\nResidual Error vs Predicted target values")
        plt.xlabel("Predicted Target Values")
        plt.ylabel("Residuals")

        plt.axhline(0, color='red', linestyle='--', linewidth=2)  # Add a horizontal line at y=0
        plt.grid()
        plt.show()

        # Variance Inflation Factor
        # Create a DataFrame to hold the VIF values
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        print(vif_data)
        return vif_data
    except Exception as e:
        print(f"Error in plot_homoscedasticity_and_vif: {e}")
        return None
