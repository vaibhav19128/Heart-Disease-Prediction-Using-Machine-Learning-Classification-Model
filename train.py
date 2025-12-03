
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/heart.csv")

# Basic cleaning (if needed)
df = df.dropna()

# Split features + target
X = df.drop("target", axis=1)
y = df["target"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
)

# -----------------------------
# 1. Logistic Regression
# -----------------------------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
joblib.dump(log_reg, "models/logistic_model.pkl")

# -----------------------------
# 2. Decision Tree
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
joblib.dump(dt, "models/decision_tree.pkl")

# -----------------------------
# 3. Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# -----------------------------
# 4. Neural Network (Keras)
# -----------------------------
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# If dataset is tiny this will be quick; with full data set increase epochs
model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)

model.save("models/neural_net.h5")

# -----------------------------
# Evaluation (Random Forest shown)
# -----------------------------
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
try:
    proba = rf.predict_proba(X_test)[:,1]
    print("ROC AUC:", roc_auc_score(y_test, proba))
except Exception:
    pass
