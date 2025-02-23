import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate synthetic data for simplicity
from sklearn.datasets import make_classification

# Step 1: Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

# Step 2: Convert to a DataFrame for better visualization
data = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, 6)])
data['Target'] = y

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='Target'), data['Target'], test_size=0.3, random_state=42)

# Step 4: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 6: Plot feature importances
importances = model.feature_importances_
plt.bar(data.columns[:-1], importances)
plt.title("Result_Accuracy")
plt.xlabel("Semester")
plt.ylabel("Importance Score")
plt.show()
