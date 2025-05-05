import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Your dataset (you can later load this from CSV)
data = pd.DataFrame({
    'age': [22, 28, 24, 35, 45, 23, 30, 32, 26, 34],
    'salary': [50000, 70000, 60000, 100000, 120000, 55000, 80000, 85000, 65000, 95000],
    'experience': [1, 5, 2, 10, 20, 1, 6, 8, 3, 12],
    'hired': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
})

# Encode target
data['hired'] = LabelEncoder().fit_transform(data['hired'])

# Split features and labels
X = data[['age', 'salary', 'experience']]
y = data['hired']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Classifier models
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Training and evaluating
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{model_name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
