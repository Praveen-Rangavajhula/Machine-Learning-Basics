import numpy as np

from data.data_loader import load_diabetes_data
from sklearn.model_selection import train_test_split

diabetes_data = load_diabetes_data()

print(diabetes_data.head())
X = diabetes_data.drop('target', axis=1)
y = diabetes_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=49)

X_train_0   = np.c_[np.ones((np.shape(X_train), 1)), X_train]
X_test_0    = np.c_[np.ones((np.shape(X_train), 1)), X_train]

# Normal Equation
theta = np.matmul(
    np.linalg.inv( np.matmul(X_train_0.T, X_train_0) ),
    np.matmul(X_train_0.T, y_train)
)

