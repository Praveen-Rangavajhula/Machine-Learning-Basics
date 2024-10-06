import numpy as np
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self):
        """
        Initialize the LinearRegression model.
        """
        self.theta = None

    def fit(self, X, y):
        """
        Fit the model using the Normal Equation method.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        """
        X_0 = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        try:
            self.theta = np.linalg.pinv(X_0.T @ X_0) @ (X_0.T @ y)  # Normal Equation
        except np.linalg.LinAlgError as e:
            print(f"Error in fitting model: {e}")
            self.theta = None

    def predict(self, X):
        """
        Predict the target values using the fitted model.

        Parameters:
        X (array-like): Feature matrix.

        Returns:
        array-like: Predicted target values.
        """
        if self.theta is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' before predicting.")
        X_0 = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        return X_0 @ self.theta

    def evaluate(self, X, y):
        """
        Evaluate the model performance using MSE and R² score.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.

        Returns:
        tuple: Mean Squared Error (MSE) and R² score.
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        return mse, r_squared

    def get_params(self):
        """
        Get model parameters: intercept and coefficients.

        Returns:
        list: Intercept and coefficients.

        Raises:
        ValueError: If the model is not fitted yet.
        """
        if self.theta is not None:
            intercept = self.theta[0]
            coefficients = self.theta[1:]
            return [intercept] + list(coefficients)
        else:
            raise ValueError("Model is not fitted yet. Please call 'fit' before getting parameters.")