import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("data/students.csv")

X = data.drop("placed", axis=1)
y = data["placed"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
lr = LogisticRegression(max_iter=1000)

# Train
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)

# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print("Random Forest Accuracy:", rf_acc)
print("Logistic Regression Accuracy:", lr_acc)

# Choose best model
best_model = rf if rf_acc > lr_acc else lr
joblib.dump(best_model, "models/placement_model.pkl")

print("Best model saved.")

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance (Random Forest only)
importances = rf.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(X.columns, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))
