import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

plt.close('all')
# Define the MSE loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Load dataset
dungeon_data = pd.read_csv('dungeon_sensorstats_partC.csv')

# Select features and target variable
features = ["height", "strength"]
target = "alignment"

# Check for missing values
print("Missing values in dataset:")
print(dungeon_data[features + [target]].isnull().sum())

# Drop rows with NaN values
dungeon_data = dungeon_data.dropna(subset=features + [target])

# Extract features and labels
X = dungeon_data[features]
y = dungeon_data[target]

# Convert alignment labels to numeric (Ordinal Encoding)
alignment_mapping = {'good': 0, 'neutral': 1, 'evil': 2}
y = y.map(alignment_mapping)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model = LogisticRegression()
print("Training model...")
model.fit(X_train, y_train)

# Make predictions on test set
predictions = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, predictions) # for real test
print("Mean Squared Error:", mse)

# Evaluate model performance
print("Test Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Good', 'Neutral', 'Evil']))

plt.rcParams.update({'font.size': 18})
# Visualizing decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X_test['height'], X_test['strength'], c=predictions, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.colorbar(label='Predicted Alignment')
plt.xlabel('Height')
plt.ylabel('Strength')
# plt.title('Predicted Alignment Based on Height & Strength')
plt.show()
