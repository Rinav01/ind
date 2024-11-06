# Import necessary libraries
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn import metrics # type: ignore
from sklearn.tree import plot_tree # type: ignore
import matplotlib.pyplot as plt # type: ignore
# 
# Sample data
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model using the training data
clf.fit(X_train, y_train)

# Predict the responses for the test dataset
y_pred = clf.predict(X_test)

# Model accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Enhance and visualize the decision tree
plt.figure(figsize=(14,10))  # Increase the figure size for better visibility
plot_tree(clf, 
          feature_names=['Age', 'Income', 'Previous_Purchase'], 
          class_names=['No', 'Yes'], 
          filled=True,  # Color the nodes for better separation between classes
          rounded=True,  # Round the edges for a cleaner look
          precision=2,  # Limit decimal precision for easier reading
          fontsize=12)  # Increase font size for better legibility
plt.title('Decision Tree for Predicting Customer Purchase', fontsize=16)  # Add a title
plt.show()
