# Import necessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve
from sklearn import tree, neighbors
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

# Set up the RNG for numpy.random
RANDOM_SEED = 42

data = np.load("sprites_greyscale10.npz")
train_X = data ["train_X"]
train_Y = data ["train_Y"]
test_X = data ["test_X"]
test_Y = data ["test_Y"]

print(train_X)

# print test_Y to decide which classes will be chosen
# print(test_Y)

# choose 2 classes
selected_classes = ["human", "lizard"]

train_mask = np.isin(train_Y, selected_classes)
test_mask = np.isin(test_Y, selected_classes)
# print(train_Y)
# print(train_mask)
# print(test_Y)
# print(test_mask)
# print("Number of training samples: ", np.sum(train_mask))
# print("Number of test samples: ", np.sum(test_mask))

train_X_subset = train_X[train_mask]
train_Y_subset = train_Y[train_mask]
test_X_subset = test_X[test_mask]
test_Y_subset = test_Y[test_mask]

# test for fixed max_leaf_nodes = 10
print("Start for fixed max_leaf_nodes hyperparameter")
clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)
clf.fit(train_X_subset, train_Y_subset)
train_predictions = clf.predict(train_X_subset)
test_predictions = clf.predict(test_X_subset)
train_accuracies = accuracy_score(train_Y_subset, train_predictions)
test_accuracies = accuracy_score(test_Y_subset, test_predictions)

print(test_predictions)
print(test_accuracies)

# Decision tree 
P_CV = 0.2
RUNS = 10
MAX_LEAF = 200
print("Start finding max_leaf_nodes hyperparameter from 2-200 in 10 runs")

max_leaf_nodes_range = range(2, MAX_LEAF)
train_accuracies_run = np.zeros((RUNS, len(max_leaf_nodes_range)))
val_accuracies_run = np.zeros((RUNS, len(max_leaf_nodes_range)))

for run in range(RUNS):
    data_train_CV, data_val_CV, labels_train_CV, labels_val_CV = train_test_split(train_X_subset, train_Y_subset, test_size=P_CV, random_state=RANDOM_SEED)
    for i, max_leaf_nodes in enumerate(max_leaf_nodes_range):
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        clf.fit(data_train_CV, labels_train_CV)
        train_predictions = clf.predict(data_train_CV)
        val_predictions = clf.predict(data_val_CV)
        train_accuracies_run[run, i] = accuracy_score(labels_train_CV, train_predictions)
        val_accuracies_run[run, i] = accuracy_score(labels_val_CV, val_predictions)
        
train_accuracies_mean = np.mean(train_accuracies_run, axis=0)
val_accuracies_mean = np.mean(val_accuracies_run, axis=0)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(max_leaf_nodes_range, train_accuracies_mean, label='Train Accuracy')
plt.plot(max_leaf_nodes_range, val_accuracies_mean, label='Validation Accuracy')
plt.xlabel('max_leaf_nodes')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Decision Tree (Averaged over 10 Runs)')
plt.legend()
plt.grid(True)
plt.show()