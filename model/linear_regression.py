import numpy as np
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """ Fit the model using the Normal Equation method. """
        X_0 = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        self.theta = np.linalg.inv(X_0.T @ X_0) @ (X_0.T @ y)  # Normal Equation

    def predict(self, X):
        """ Predict the target values using the fitted model. """
        X_0 = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        return X_0 @ self.theta

    def evaluate(self, X, y):
        """ Evaluate the model performance using MSE and R² score. """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        return mse, r_squared

    def get_params(self):
        """ Get model parameters: intercept and coefficients. """
        if self.theta is not None:
            intercept = self.theta[0]
            coefficients = self.theta[1:]
            return [intercept] + list(coefficients)
        else:
            raise ValueError("Model is not fitted yet. Please call 'fit' before getting parameters.")
