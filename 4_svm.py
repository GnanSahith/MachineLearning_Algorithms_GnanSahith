
# Support Vector Machine on Breast Cancer Dataset

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
model = SVC()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("SVM Accuracy on Breast Cancer dataset:", accuracy_score(y_test, predictions))
