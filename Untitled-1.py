# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Sample data (replace this with your actual dataset)
data = {
    'Age': [25, 45, 35, 50, 23, 32, 40, 60, 22, 35],
    'Income': [50000, 100000, 75000, 120000, 45000, 70000, 80000, 110000, 30000, 90000],
    'Previous_Purchase': [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],  # 0 = No, 1 = Yes
    'Will_Buy': [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  # 0 = No, 1 = Yes
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (Age, Income, Previous Purchase)
X = df[['Age', 'Income', 'Previous_Purchase']]

# Target (Will Buy)
y = df['Will_Buy']

# Split the dataset into training (70%) and testing sets (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model using the training data
clf.fit(X_train, y_train)

# Predict the responses for the test dataset
y_pred = clf.predict(X_test)

# Model accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=['Age', 'Income', 'Previous_Purchase'], class_names=['No', 'Yes'], filled=True)
plt.show()
