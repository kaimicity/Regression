import numpy as np

# Define the models. Since models for predicting Y1 and Y2 are both multivariate linear regression models,
# they could be defined by the same function.
def modelFunc(x, theta):
    y_hat = []
    for obs in x:
        y_obs = 0
        for i in range(len(obs)):
            y_obs = y_obs + obs[i] * theta[i]
        y_hat.append(y_obs)
    return np.array(y_hat)

def predict(x, theta):
    y = 0
    for i in range(len(x)):
        y = y + x[i] * theta[i]
    return y

# Define loss function
def loss(x, y, theta):
    y_hat = modelFunc(x, theta)
    squared_error = (y - y_hat) ** 2
    return squared_error.mean() / 2

# Calculate R squared
def Rsquared(x, y, theta):
    y_hat = modelFunc(x, theta)
    y_mean = y.mean()
    squared_error = (y - y_hat) ** 2
    squared_total = (y - y_mean) ** 2
    return 1 - sum(squared_error) / sum(squared_total)

# Normal equation for model
def normalEquation(x, y, lamb):
    x = np.matrix(x)
    x_inner_product = np.dot(x.T, x)
    inv = np.linalg.inv(x_inner_product) + lamb * regularizationMatrix(x[0].size)
    theta = np.dot(inv, x.T)
    theta = np.dot(theta, y)
    theta = theta.tolist()
    theta = theta[0]
    return theta

# Matrix for regularization
def regularizationMatrix(n):
    eye = np.eye(n, dtype=float)
    eye[0][0] = 0
    return eye

