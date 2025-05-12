# # Import necessary libraries.
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, validation_curve
# from sklearn import tree, neighbors, datasets
# from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
# from sklearn.linear_model import LinearRegression

# plt.close('all')

# # Define the MSE loss function
# def mse_loss(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# # Implementing gradient descent
# def gradient_descent(X, y, learning_rate=0.001, iterations=70):
#     print("Running gradient descent")
#     m = len(y)
#     # X_b = np.c_[np.ones((m, 1)), X]
#     X_b = np.c_[X]
#     # theta = np.random.randn(4, 1)
#     theta = np.array([[0.01], [0.01], [0.01], [0.01]])
#     loss_history = []

#     print("Running iterations")
#     for i in range(iterations):
#         # print("Iteration:", i)
#         gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
#         # print("Finished gradient calculation :", gradients)
#         theta = theta - learning_rate * gradients
#         # print("Finished calculating theta :", theta)
#         loss = mse_loss(y, X_b.dot(theta))
#         # print("Finished calculating loss :", loss)
#         loss_history.append(loss)
#         # print("Finished iteration:", i)        

#     return theta, loss_history

# ### if unable to load, run below to load the local dataset file
# dungeon_train_data = pd.read_csv('dungeon_sensorstats_train.csv')
# dungeon_test_data = pd.read_csv('dungeon_sensorstats_test.csv')

# # Extract data where the race class is "human"
# train_human_data = dungeon_train_data[dungeon_train_data['race'] == 'human']
# test_human_data = dungeon_test_data[dungeon_test_data['race'] == 'human']
# # print(train_human_data.head())

# # Trim the data to only include "intelligence", "stench", "sound", and "heat" features
# features = ["intelligence", "stench", "sound", "heat"]
# # features = ["intelligence"]
# train_trimmed_data = train_human_data[features]
# test_trimmed_data = test_human_data[features]
# # print(train_trimmed_data.head())

# # Extract the labels from the data
# train_labels = train_human_data['bribe']
# test_labels = test_human_data['bribe']

# # Set up the RNG for numpy.random
# RANDOM_SEED = 42
# P_CV = 0.2

# # Extract and split data into data and test
# data_train, data_test, labels_train, labels_test = train_test_split(train_trimmed_data, train_labels, test_size=P_CV, random_state=RANDOM_SEED)

# # Linear Regression model from sklearn
# model = LinearRegression()
# print("Start training model")
# model.fit(data_train, labels_train)

# # predict
# print("Predicting")
# predictions = model.predict(data_test)

# # MSE
# mse = mean_squared_error(labels_test, predictions)
# print("Mean Squared Error:", mse)

# # Running gradient descent
# theta, loss_history = gradient_descent(data_train, labels_train.values.reshape(-1, 1))

# print("The optimal weights:", theta)


# # Plotting the loss history
# plt.figure(figsize=(10, 6))
# plt.plot(loss_history, label='Loss History')
# plt.xlabel('Iterations')
# plt.ylabel('MSE Loss')
# plt.title('Loss History over Iterations')
# plt.legend()
# plt.grid(True)
# plt.show()


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn import tree, neighbors, datasets
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression

plt.close('all')

# Define the MSE loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Implementing gradient descent
def gradient_descent(X, y, learning_rate=0.001, iterations=70):
    print("Running gradient descent")
    m = len(y)
    # X_b = np.c_[np.ones((m, 1)), X]
    X_b = np.c_[X]
    # theta = np.random.randn(4, 1)
    theta = np.array([[0.01], [0.01], [0.01], [0.01]])
    loss_history = []

    print("Running iterations")
    for i in range(iterations):
        # print("Iteration:", i)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        # print("Finished gradient calculation :", gradients)
        theta = theta - learning_rate * gradients
        # print("Finished calculating theta :", theta)
        loss = mse_loss(y, X_b.dot(theta))
        # print("Finished calculating loss :", loss)
        loss_history.append(loss)
        # print("Finished iteration:", i)

    return theta, loss_history

# Load dataset
dungeon_train_data = pd.read_csv('dungeon_sensorstats_train.csv')
dungeon_test_data = pd.read_csv('dungeon_sensorstats_test.csv')

# Extract data where the race class is "human"
train_human_data = dungeon_train_data[dungeon_train_data['race'] == 'human']
test_human_data = dungeon_test_data[dungeon_test_data['race'] == 'human']

# Trim the data to only include "intelligence", "stench", "sound", and "heat" features
features = ["intelligence", "stench", "sound", "heat"]
# features = ["intelligence"]
train_trimmed_data = train_human_data[features]
test_trimmed_data = test_human_data[features]

# Extract the labels from the data
train_labels = train_human_data['bribe']
test_labels = test_human_data['bribe']

# Set up the RNG for numpy.random
RANDOM_SEED = 42
P_CV = 0.2

# Extract and split data into data and test
data_train, data_test, labels_train, labels_test = train_test_split(train_trimmed_data, train_labels, test_size=P_CV, random_state=RANDOM_SEED)

# Linear Regression model from sklearn
model = LinearRegression()
print("Start training model")
# model.fit(data_train, labels_train)
model.fit(train_trimmed_data, train_labels)

# predict
print("Predicting")
predictions = model.predict(test_trimmed_data)

# MSE
# mse = mean_squared_error(labels_test, predictions) # for validation
mse = mean_squared_error(test_labels, predictions) # for real test
print("Mean Squared Error:", mse)

# Running gradient descent
theta, loss_history = gradient_descent(data_train, labels_train.values.reshape(-1, 1))

print("The optimal weights:", theta)

# Plotting the loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Loss History')
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('Loss History over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# # Visualize the relationship between "intelligence" and predictions
# plt.figure(figsize=(10, 6))
# plt.scatter(test_human_data[features], test_labels, color='blue', label='Actual Intelligence')
# plt.scatter(test_human_data[features], predictions, color='red', label='Predicted Intelligence')
# plt.xlabel('Intelligence')
# plt.ylabel('Bribe')
# plt.title('Visualization of intelligence feature and its prediction')
# plt.legend()
# plt.grid(True)
# plt.show()
