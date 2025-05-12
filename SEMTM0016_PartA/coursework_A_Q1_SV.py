# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

# Set up the RNG for numpy.random
RANDOM_SEED = 42

data = np.load("sprites_greyscale10.npz")
train_X = data["train_X"]
train_Y = data["train_Y"]
test_X = data["test_X"]
test_Y = data["test_Y"]

# print test_Y to decide which classes will be chosen
# print(test_Y)

# choose 2 classes
selected_classes = ["human", "lizard"]

train_mask = np.isin(train_Y, selected_classes)
test_mask = np.isin(test_Y, selected_classes)

train_X_subset = train_X[train_mask]
train_Y_subset = train_Y[train_mask]
test_X_subset = test_X[test_mask]
test_Y_subset = test_Y[test_mask]

# test for fixed C = 1.0 (regularization strength)
print("Start for fixed C hyperparameter")
clf = LogisticRegression(C=0.075, random_state=RANDOM_SEED)
clf.fit(train_X_subset, train_Y_subset)
train_predictions = clf.predict(train_X_subset)
test_predictions = clf.predict(test_X_subset)
train_accuracies = accuracy_score(train_Y_subset, train_predictions)
test_accuracies = accuracy_score(test_Y_subset, test_predictions)

print(test_predictions)
print(test_accuracies)

# Logistic regression
P_CV = 0.2
RUNS = 2
# C_range = np.logspace(0, 2, 10)
C_range = np.logspace(-2, 2, 10)
# C_range = np.arange(1, 2000, 10)
print("Start finding C hyperparameter in 5 runs")

train_accuracies_run = np.zeros((RUNS, len(C_range)))
val_accuracies_run = np.zeros((RUNS, len(C_range)))

# Held-out validation
for run in range(RUNS):
    data_train_CV, data_val_CV, labels_train_CV, labels_val_CV = train_test_split(train_X_subset, train_Y_subset, test_size=P_CV, random_state=RANDOM_SEED)
    for i, C in enumerate(C_range):
        clf = LogisticRegression(C=C, random_state=RANDOM_SEED)
        clf.fit(data_train_CV, labels_train_CV)
        train_predictions = clf.predict(data_train_CV)
        val_predictions = clf.predict(data_val_CV)
        train_accuracies_run[run, i] = accuracy_score(labels_train_CV, train_predictions)
        val_accuracies_run[run, i] = accuracy_score(labels_val_CV, val_predictions)
        
train_accuracies_mean = np.mean(train_accuracies_run, axis=0)
val_accuracies_mean = np.mean(val_accuracies_run, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(C_range, train_accuracies_mean, label='Train Accuracy')
plt.plot(C_range, val_accuracies_mean, label='Validation Accuracy')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Logistic Regression (Averaged over 2 Runs)')
plt.legend()
plt.grid(True)
plt.show()
