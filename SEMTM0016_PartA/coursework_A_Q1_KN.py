import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set up the RNG for numpy.random
RANDOM_SEED = 42

data = np.load("sprites_greyscale10.npz")
train_X = data["train_X"]
train_Y = data["train_Y"]
test_X = data["test_X"]
test_Y = data["test_Y"]

# choose 2 classes
selected_classes = ["human", "lizard"]
train_mask = np.isin(train_Y, selected_classes)
test_mask = np.isin(test_Y, selected_classes)
train_X_subset = train_X[train_mask]
train_Y_subset = train_Y[train_mask]
test_X_subset = test_X[test_mask]
test_Y_subset = test_Y[test_mask]

# test for fixed k = 5
print("Start for fixed k hyperparameter")
k = 55
knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(train_X_subset, train_Y_subset)
train_predictions = knn.predict(train_X_subset)
test_predictions = knn.predict(test_X_subset)
train_accuracies = accuracy_score(train_Y_subset, train_predictions)
test_accuracies = accuracy_score(test_Y_subset, test_predictions)

print(test_predictions)
print(test_accuracies)

# Hyperparameter tuning for k in k-NN
P_CV = 0.2
RUNS = 5
MAX_K = 100
print("Start finding optimal k hyperparameter from 1 to 100 in 10 runs")

k_range = range(1, MAX_K + 1)
train_accuracies_run = np.zeros((RUNS, len(k_range)))
val_accuracies_run = np.zeros((RUNS, len(k_range)))

for run in range(RUNS):
    print("Run : ", run + 1)
    data_train_CV, data_val_CV, labels_train_CV, labels_val_CV = train_test_split(train_X_subset, train_Y_subset, test_size=P_CV, random_state=RANDOM_SEED)
    for i, k in enumerate(k_range):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(data_train_CV, labels_train_CV)
        train_predictions = knn.predict(data_train_CV)
        val_predictions = knn.predict(data_val_CV)
        train_accuracies_run[run, i] = accuracy_score(labels_train_CV, train_predictions)
        val_accuracies_run[run, i] = accuracy_score(labels_val_CV, val_predictions)

train_accuracies_mean = np.mean(train_accuracies_run, axis=0)
val_accuracies_mean = np.mean(val_accuracies_run, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(k_range, train_accuracies_mean, label='Train Accuracy')
plt.plot(k_range, val_accuracies_mean, label='Validation Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Validation Curve for k-NN (Averaged over Runs)')
plt.legend()
plt.grid(True)
plt.show()