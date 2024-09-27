import numpy as np
import matplotlib.pyplot as plt
from model.logistic_regression import LogisticRegression

# Generate some synthetic data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
true_parameters = [1.2, -3]

# Generate z and y_prob based on true parameters
z = true_parameters[0] * x + true_parameters[1]
y_prob = 1 / (1 + np.exp(-z))  # Sigmoid function

# Generate binary labels
y = (y_prob > np.random.rand(100, 1)).astype(int)

# Train the logistic regression model
log_reg_model = LogisticRegression(learning_rate=0.2, epochs=1000)
log_reg_model.train(x, y)

# Step 1: Plot the data points (feature x vs. label y)
plt.scatter(x, y, label='Data Points', c=y, cmap='bwr', edgecolor='k')

# Step 2: Generate a smooth range of x values for the sigmoid curve
x_values = np.linspace(0, 2, 100)

# Step 3: Compute the predicted probabilities (sigmoid curve) for the x_values
y_pred_prob = log_reg_model.predict(x_values)

# Step 4: Plot the sigmoid curve
plt.plot(x_values, y_pred_prob, label='Sigmoid Curve', color='blue')

# Step 5: Plot the decision boundary where probability = 0.5
decision_boundary = -log_reg_model.bias / log_reg_model.parameter  # Solving p(x) = 0.5 for x
plt.axvline(decision_boundary, color='red', linestyle='--', label=f'Decision Boundary (x={decision_boundary:.2f})')

# Step 6: Add labels, legend, and title
plt.xlabel('Feature (x)')
plt.ylabel('Probability / Label')
plt.title('Logistic Regression Decision Boundary')
plt.legend()

# Display the plot
plt.show()

# Calculate and print the accuracy
y_pred = log_reg_model.classify(x)
print("Accuracy:", log_reg_model.calculate_accuracy(y, y_pred))
