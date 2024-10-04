from statistics import linear_regression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data.data_loader import load_diabetes_data
from sklearn.model_selection import train_test_split

diabetes_data = load_diabetes_data()

print(diabetes_data.head())
X = diabetes_data.drop('target', axis=1)
y = diabetes_data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=49)

# Add a column of ones to X_train and X_test to represent the intercept term (x0 = 1)
X_train_0   = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_0    = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Normal Equation
theta = np.matmul(
    np.linalg.inv( np.matmul(X_train_0.T, X_train_0) ),
    np.matmul(X_train_0.T, y_train)
)

# Create parameters for the linear regression model
parameters = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
columns = ['intercept'] + list(X.columns.values)
parameter_df = pd.DataFrame({
    'Parameters': parameters,
    'Columns': columns,
    'Theta': theta
})

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

sk_theta = [linear_regression_model.intercept_] + list(linear_regression_model.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
print(parameter_df.head())

# Model Evaluation
y_pred_eval = np.matmul(X_test_0, theta)

# MSE calculation
J_mse = np.sum((y_pred_eval - y_test) ** 2) / (X_test_0.shape[0])

# R_square calculation
sse = J_mse * X_test_0.shape[0]
sst = np.sum((y_test - y_test.mean()) ** 2)
r_square = 1 - (sse/sst)
print('The Mean Square Error(MSE) or J(theta) is: ', J_mse)
print('R square obtained for normal equation method is: ', r_square)

# sklearn regression module
y_pred_sk = linear_regression_model.predict(X_test)

#Evaluvation: MSE

J_mse_sk = mean_squared_error(y_pred_sk, y_test)

# R_square
r_square_sk = linear_regression_model.score(X_test,y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse_sk)
print('R square obtained for scikit learn library is :',r_square_sk)

# TODO: Model validation and improving the performance further probably by using PCA to reduce the redundant features.
