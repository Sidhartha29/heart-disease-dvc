import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned Heart Disease dataset
df = pd.read_csv("data/heart_disease_uci_cleaned.csv")

# Remove ID because it is not a useful prediction feature
X = df.drop(columns=["id", "num"])
y = df["num"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Random Forest Model V2
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("HEART DISEASE RANDOM FOREST - MODEL V2")
print("=" * 50)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/HeartDisease_RF.pkl")

print("\nModel saved successfully:")
print("models/HeartDisease_RF.pkl")