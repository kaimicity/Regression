import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv')

# Find and Drop Missing Value
data.info()



# Visualization for estimating density
data.X1.plot('hist', title='X1 Frequency')
plt.savefig('X1_Frequency.png')
plt.close()
data.X2.plot('hist', title='X2 Frequency')
plt.savefig('X2_Frequency.png')
plt.close()
data.X3.plot('hist', title='X3 Frequency')
plt.savefig('X3_Frequency.png')
plt.close()
data.X4.plot('hist', title='X4 Frequency')
plt.savefig('X4_Frequency.png')
plt.close()
data.X5.plot('hist', title='X5 Frequency')
plt.savefig('X5_Frequency.png')
plt.close()
data.X6.plot('hist', title='X6 Frequency')
plt.savefig('X6_Frequency.png')
plt.close()
data.X7.plot('hist', title='X7 Frequency')
plt.savefig('X7_Frequency.png')
plt.close()
data.X8.plot('hist', title='X8 Frequency')
plt.savefig('X8_Frequency.png')
plt.close()

# Normalize dataset (Feature scaling)
data_norm = (data - data.mean(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Split into train set, validation set and test set
data_train, data_test = train_test_split(data_norm, test_size=0.2, random_state=0)
data_train, data_validation = train_test_split(data_train, test_size=0.25, random_state=0)

# Statistical analyze data
print(data_train.describe())


# Visualization for exploring relationships between two feature
data_train.plot(kind='scatter', x='X1', y='Y1')
plt.savefig('X1_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X2', y='Y1')
plt.savefig('X2_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X3', y='Y1')
plt.savefig('X3_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X4', y='Y1')
plt.savefig('X4_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X5', y='Y1')
plt.savefig('X5_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X6', y='Y1')
plt.savefig('X6_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X7', y='Y1')
plt.savefig('X7_Y1_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X8', y='Y1')
plt.savefig('X8_Y1_Scatter.png')
plt.close()


data_train.plot(kind='scatter', x='X1', y='Y2')
plt.savefig('X1_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X2', y='Y2')
plt.savefig('X2_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X3', y='Y2')
plt.savefig('X3_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X4', y='Y2')
plt.savefig('X4_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X5', y='Y2')
plt.savefig('X5_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X6', y='Y2')
plt.savefig('X6_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X7', y='Y2')
plt.savefig('X7_Y2_Scatter.png')
plt.close()
data_train.plot(kind='scatter', x='X8', y='Y2')
plt.savefig('X8_Y2_Scatter.png')
plt.close()

# Feature selection
print(data_train.corr())
drop_list_y1 = ["X3", "X4", "X6", "X7", "X8", "Y2"]
drop_list_y2 = ["X3", "X6", "X7", "X8", "Y1"]
data_train_y1 = data_train.drop(drop_list_y1, axis=1)
data_train_y2 = data_train.drop(drop_list_y2, axis=1)
data_validation_y1 = data_validation.drop(drop_list_y1, axis=1)
data_validation_y2 = data_validation.drop(drop_list_y2, axis=1)
data_test_y1 = data_test.drop(drop_list_y1, axis=1)
data_test_y2 = data_test.drop(drop_list_y2, axis=1)

