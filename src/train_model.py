import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("models", exist_ok=True)

df    = pd.read_csv("data/landmarks.csv")
X     = df.drop("label", axis=1).values
y     = df["label"].values

le    = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                    max_depth=5, random_state=42)
model.fit(X_train, y_train)

preds    = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, preds, target_names=le.classes_))

# Save confusion matrix image for your report
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, preds, display_labels=le.classes_)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("report/confusion_matrix.png")
plt.show()
print("Confusion matrix saved to report/")

joblib.dump(model, "models/gesture_model.pkl")
joblib.dump(le,    "models/label_encoder.pkl")
print("Model saved to models/")