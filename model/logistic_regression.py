import numpy as np

# TODO: FIX THE BUGS IN THE CODE BELOW
class LogisticRegression:

    def __init__(self, learning_rate=0.25, epochs=10000):
        """
        Constructor to initialize parameters for logistic regression.
        """
        self.epochs                 = epochs
        self.learning_rate          = learning_rate
        self.bias                   = np.random.randn()  # Initialize bias (intercept)
        self.parameter              = np.random.randn()  # Initialize weight (parameter)

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid function to map any real value to a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        """
        Predict probability using the learned parameter and bias.
        """
        z = self.bias + self.parameter * x
        return self.sigmoid(z)

    def train(self, x, y):
        """
        Train the logistic regression model using gradient descent.
        """
        m = len(y)  # Number of training examples

        for i in range(self.epochs):
            y_pred = self.predict(x)  # Get probabilities (predictions)

            # Compute the Negative Log Likelihood (NLL) loss
            epsilon = 1e-8
            nll = -(1 / m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

            # Gradient descent updates for parameter (weight) and bias
            self.bias -= self.learning_rate * (1/m) * np.sum(y_pred - y)
            self.parameter -= self.learning_rate * (1/m) * np.sum((y_pred - y) * x)

            # Optionally, print the loss every 1000 iterations
            if i % 1000 == 0 or i == self.epochs-1:
                print(f'Epoch {i}, Loss: {nll}')


    def classify(self, x, threshold=0.5):
        """
        Classify input data by applying a threshold on the predicted probabilities.
        """
        probabilities = self.predict(x)
        return np.array([1 if p > threshold else 0 for p in probabilities])

    @staticmethod
    def calculate_accuracy(y, y_pred):
        return np.mean(y == y_pred) * 100


