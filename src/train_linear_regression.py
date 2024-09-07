import numpy as np
import matplotlib.pyplot as plt
from model.linear_regression import LinearRegression

# Generate synthetic data for testing the model
np.random.seed(23)
X = 2 * np.random.rand(100, 1)  # Feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Target

# Instantiate the linear regression model
lr_model = LinearRegression(learning_rate=0.25, epochs=10000)

# Plot original data
plt.scatter(X, y)
plt.title("Original data")
plt.xlabel("X")
plt.ylabel("target")
plt.show()

# Train the model
lr_model.train(X, y)

# Plot the best fit line
lr_model.plot(X, y)
