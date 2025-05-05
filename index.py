# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset (SMS Spam Collection)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
sms_data = pd.read_csv(url, compression='zip', sep='\t', header=None, names=['label', 'message'])

# Data Preprocessing
sms_data['label'] = LabelEncoder().fit_transform(sms_data['label'])  # Label Encoding
X = sms_data['message']
y = sms_data['label']

# Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Apply PCA for dimensionality reduction (keeping 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train multiple models
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{model_name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

