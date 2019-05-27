from preparation import data_train_y1, data_validation_y1, data_test_y1, data_train_y2, data_validation_y2, data_test_y2
import numpy as np
import util

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

# Add x0 to the input set of model s
y1_x = np.c_[np.ones_like(y1_x[:, 0]), y1_x]
y2_x = np.c_[np.ones_like(y2_x[:, 0]), y2_x]

# Train models and get sets of theta
regu_para = .00000000001
y1_theta = util.normalEquation(y1_x, y1_y, regu_para)
print(y1_theta)
y2_theta = util.normalEquation(y2_x, y2_y, regu_para)
print(y2_theta)

# Calculate the loss and validate the model
y1_loss = util.loss(y1_vali_x, y1_vali_y, y1_theta)
y1_r_squared = util.Rsquared(y1_vali_x, y1_vali_y, y1_theta)
print("Y1 Model: Loss: " + str(y1_loss) + ", Squared R:" + str(y1_r_squared))
y2_loss = util.loss(y2_vali_x, y2_vali_y, y2_theta)
y2_r_squared = util.Rsquared(y2_vali_x, y2_vali_y, y2_theta)
print("Y2 Model: Loss: " + str(y2_loss) + ", Squared R:" + str(y2_r_squared))

# Extract validation input and output sets for the models
y1_test_x = data_test_y1[:, : (len(data_train_y1[0]) - 1)]
y1_test_y = data_test_y1[:, (len(data_train_y1[0]) - 1)]
y2_test_x = data_test_y2[:, : (len(data_train_y2[0]) - 1)]
y2_test_y = data_test_y2[:, (len(data_train_y2[0]) - 1)]

# Fit the models with test set
y1_loss_test = util.loss(y1_test_x, y1_test_y, y1_theta)
y1_r_squared_test = util.Rsquared(y1_test_x, y1_test_y, y1_theta)
print("Y1 Model: Loss: " + str(y1_loss_test) + ", Squared R:" + str(y1_r_squared_test))
y2_loss_test = util.loss(y2_test_x, y2_test_y, y2_theta)
y2_r_squared_test = util.Rsquared(y2_test_x, y2_test_y, y2_theta)
print("Y2 Model: Loss: " + str(y2_loss_test) + ", Squared R:" + str(y2_r_squared_test))
