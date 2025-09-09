# train_model.py
# Usage:
#  python -m venv .venv
#  source .venv/bin/activate          (or .venv\Scripts\activate on Windows)
#  pip install -r requirements.txt
#  python train_model.py

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = Path("data/heart.csv")
MODEL_PATH = Path("model.joblib")
SCALER_PATH = Path("scaler.joblib")

# features typical in UCI heart dataset
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def load_data():
    if DATA_PATH.exists():
        print(f"Found dataset at {DATA_PATH} — loading.")
        df = pd.read_csv(DATA_PATH)
        # Expect column "target" for label (0/1)
        if "target" not in df.columns:
            raise ValueError("data/heart.csv must have a 'target' column (0/1).")
        # attempt to pick required columns if present, else use available numeric columns
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            print("Warning: missing expected columns:", missing)
            # fallback: take numeric columns except target
            X = df.select_dtypes(include=[np.number]).drop(columns=["target"], errors="ignore")
            y = df["target"]
            return X, y
        X = df[FEATURE_NAMES]
        y = df["target"]
        return X, y
    else:
        print("No data/heart.csv found — generating a small synthetic dataset for testing.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=len(FEATURE_NAMES),
                                   n_informative=8, n_redundant=2, random_state=42)
        X = pd.DataFrame(X, columns=FEATURE_NAMES)
        return X, pd.Series(y)

def train_save():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None

    joblib.dump({"model": clf, "feature_names": list(X.columns)}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")
    print(f"Test accuracy: {acc:.4f}")
    if auc is not None:
        print(f"Test ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    train_save()
