import numpy as np

# Produce theta
def predictLwlr(test_x, train_x, train_y, tau):
    train_x = np.matrix(train_x)
    w = produceWMatrix(test_x, train_x, tau)
    x_inner_product = np.dot(np.dot(train_x.T, w), train_x)
    theta = np.dot(x_inner_product.I, train_x.T)
    theta = np.dot(theta, w)
    theta = np.dot(theta, train_y)
    theta = theta.tolist()
    theta = theta[0]
    return theta

# def predict(test_x_set, train_x, train_y, tau):
#     y_hat = []
#     for obs in test_x_set:
#         print(obs)
#         y_obs = 0
#         theta = predictLwlr(obs, train_x, train_y, tau)
#         for i in range(len(obs)):
#             y_obs = y_hat + theta[i] * obs[i]
#         y_hat.append(y_obs)
#     return y_hat

# Predict the output for a group of input
def predictValue(text_x, train_x, train_y, tau):
    y_obs = 0
    theta = predictLwlr(text_x, train_x, train_y, tau)
    for i in range(len(text_x)):
        y_obs = y_obs + theta[i] * text_x[i]
    return y_obs

# Produce weight metrix
def produceWMatrix(test_x, train_x, tau):
    eye = np.eye(np.shape(train_x)[0], dtype=float)
    for i in range(np.shape(train_x)[0]):
        distance = test_x - train_x[i, :]
        eye[i][i] = np.exp(np.dot(distance, distance.T) / (-2.0 * tau ** 2))
    return eye

