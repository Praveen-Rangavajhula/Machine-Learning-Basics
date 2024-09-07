import numpy as np
import matplotlib.pyplot as plt


def compute_loss(y_pred, target):
    """
    Compute Mean Squared Error (MSE) loss.

    Parameters:
    - y_pred: Predicted target values.
    - target: Actual target values.

    Returns:
    - loss: The computed MSE loss.
    """
    return (1 / len(target)) * np.sum(np.power(y_pred - target, 2))


class LinearRegression:

    def __init__(self, learning_rate=0.25, epochs=10000):
        """
        Constructor to initialize parameters for the linear regression model.

        Parameters:
        - learning_rate: Step size for updating parameters during gradient descent.
        - epochs: Number of iterations to run gradient descent.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.parameter = np.random.rand()  # Initialize parameter (weight)
        self.bias = np.random.rand()  # Initialize bias (intercept)

    def predict(self, x):
        """
        Predict output using the learned parameter and bias.

        Parameters:
        - X: Input feature values.

        Returns:
        - y_pred: Predicted target values.
        """
        return self.parameter * x + self.bias

    def train(self, x, target):
        """
        Train the linear regression model using gradient descent.

        Parameters:
        - X: Input feature values.
        - target: Actual target values.
        """
        for i in range(self.epochs):
            y_pred = self.predict(x)

            # Gradient descent updates for parameter (weight) and bias
            self.bias -= self.learning_rate * (1 / len(target)) * np.sum(y_pred - target)
            self.parameter -= self.learning_rate * (1 / len(target)) * np.sum((y_pred - target) * x)

            # (Optional) Uncomment to print loss every 1000 epochs
            # if i % 1000 == 0:
            #     loss = self.compute_loss(y_pred, target)
            #     print(f'Epoch {i}, Loss: {loss}')

        final_loss = compute_loss(self.predict(x), target)
        print(f'Final Loss after training: {final_loss}')

    def plot(self, x, target):
        """
        Plot the original data points and the best-fit line.

        Parameters:
        - x: Input feature values.
        - target: Actual target values.
        """
        plt.scatter(x, target, label='Data points')
        plt.plot(x, self.predict(x), color='red', label='Best fit line')
        plt.title("Linear Regression - Best Fit Line")
        plt.xlabel("X")
        plt.ylabel("target")
        plt.legend()
        plt.show()
