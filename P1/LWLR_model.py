from preparation import data_train_y1, data_validation_y1, data_test_y1, data_train_y2, data_validation_y2, data_test_y2
import numpy as np
import LWLR_util
from model import y1_theta, y2_theta
from util import predict

data_train_y1 = data_train_y1.values
data_train_y2 = data_train_y2.values
data_validation_y1 = data_validation_y1.values
data_validation_y2 = data_validation_y2.values
data_test_y1 = data_test_y1.values
data_test_y2 = data_test_y2.values


# Extract training input and output sets for the models
y1_x = data_train_y1[:, : (len(data_train_y1[0]) - 1)]
y1_y = data_train_y1[:, (len(data_train_y1[0]) - 1)]
y2_x = data_train_y2[:, : (len(data_train_y2[0]) - 1)]
y2_y = data_train_y2[:, (len(data_train_y2[0]) - 1)]

# Extract validation input and output sets for the models
y1_vali_x = data_validation_y1[:, : (len(data_train_y1[0]) - 1)]
y1_vali_y = data_validation_y1[:, (len(data_train_y1[0]) - 1)]
y2_vali_x = data_validation_y2[:, : (len(data_train_y2[0]) - 1)]
y2_vali_y = data_validation_y2[:, (len(data_train_y2[0]) - 1)]

# Predict
tau = 0.1
for i in range(10):
    y1_vali_y_hat = LWLR_util.predictValue(y1_vali_x[i], y1_x, y1_y, tau)
    y1_vali_ne_hat = predict(y1_vali_x[i], y1_theta)
    print('Y1: LWLR Predict value: ' + str(y1_vali_y_hat) + ' Observe value: ' + str(y1_vali_y[i]) + ' Parametric model  predict value:' + str(y1_vali_ne_hat))

for i in range(10):
    y2_vali_y_hat = LWLR_util.predictValue(y2_vali_x[i], y2_x, y2_y, tau)
    y2_vali_ne_hat = predict(y2_vali_x[i], y2_theta)
    print('Y2: LWLR Predict value: ' + str(y2_vali_y_hat) + ' Observe value: ' + str(y2_vali_y[i]) + ' Parametric model  predict value:' + str(y2_vali_ne_hat))
