import pandas as pd
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data.data_loader import load_diabetes_data
from model.linear_regression import LinearRegression
import model.visualization as visualize

diabetes_data = load_diabetes_data()

print(diabetes_data.head())
X = diabetes_data.drop('target', axis=1)
y = diabetes_data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=49)

# Train using custom implementation (Uses Normal Equation)
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
theta = linear_regression.theta

# Create parameters for the linear regression model
parameters = ['theta_' + str(i) for i in range(len(theta))]
columns = ['intercept'] + list(X.columns.values)
parameter_df = pd.DataFrame({
    'Parameters': parameters,
    'Columns': columns,
    'Theta': theta
})

# Model Evaluation
J_mse, r_square = linear_regression.evaluate(X_train, y_train)

print('The Mean Square Error (MSE) or J (theta) is: ', J_mse)
print('R square obtained for normal equation method is: ', r_square)

# Sklearn regression module
sk_linear_regression = SKLinearRegression()
sk_linear_regression.fit(X_train, y_train)

# Get coefficients and intercept from the sklearn model
sk_theta = [sk_linear_regression.intercept_] + list(sk_linear_regression.coef_)
# Join the sklearn parameters to the existing DataFrame
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))

# Print the parameters DataFrame
print(parameter_df)

# Prediction using sklearn model
y_pred_sk = sk_linear_regression.predict(X_train)

# Evaluation for sklearn model
J_mse_sk = mean_squared_error(y_train, y_pred_sk)
r_square_sk = sk_linear_regression.score(X_train, y_train)

print('The Mean Square Error (MSE) or J (theta) for scikit learn library is: ', J_mse_sk)
print('R square obtained for scikit learn library is: ', r_square_sk)

visualize.plot_actual_vs_predicted(y_train, y_pred_sk)
