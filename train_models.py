import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset directly from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# These are the standard 14 column names for this dataset
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load data, handling the '?' characters often found in this specific dataset
df = pd.read_csv(url, names=columns, na_values="?")

# 2. Basic Cleaning (The Cleveland dataset has a few missing values marked as NaN)
df = df.fillna(df.median())

# The target in Cleveland is 0-4 (0=healthy, 1-4=heart disease). 
# We convert it to Binary: 0 (No Disease) and 1 (Disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 2. Preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling is CRITICAL for SVM and Neural Networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Build the Models

# Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # RF doesn't strictly need scaling but works fine with it

# Model 2: SVM
svm_model = SVC(probability=True, kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Model 3: Neural Network (MLP)
nn_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)

# 4. Compare Accuracy
models = {'Random Forest': rf_model, 'SVM': svm_model, 'Neural Network': nn_model}
for name, model in models.items():
    xtest = X_test_scaled if name != 'Random Forest' else X_test
    pred = model.predict(xtest)
    print(f"{name} Accuracy: {accuracy_score(y_test, pred):.2%}")

# 5. Save the models and the scaler for Flask
joblib.dump(rf_model, 'heart_rf_model.pkl')
joblib.dump(svm_model, 'heart_svm_model.pkl')
joblib.dump(nn_model, 'heart_nn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')