# train_model.py (works even without labels)
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_CSV = Path("data/heart.csv")
MODEL_PATH = Path("model.joblib")
SCALER_PATH = Path("scaler.joblib")

def load_data():
    if DATA_CSV.exists():
        print(f"Loading {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)

        if "target" in df.columns:
            print("Found 'target' column — using real labels.")
            y = df["target"]
            X = df.drop(columns=["target"])
        else:
            print("⚠️ No 'target' column found — generating synthetic labels (for demo/testing).")
            X = df.select_dtypes(include=[np.number])
            rng = np.random.RandomState(42)
            y = pd.Series(rng.randint(0, 2, size=len(X)), name="target")

        return X, y

    else:
        print("No CSV found — generating synthetic dataset.")
        from sklearn.datasets import make_classification
        X_arr, y_arr = make_classification(n_samples=200, n_features=13,
                                           n_informative=8, n_redundant=2,
                                           random_state=42)
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
        y = pd.Series(y_arr, name="target")
        return X, y

def train_and_save():
    X, y = load_data()
    print(f"Training with {X.shape[0]} rows and {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    joblib.dump({"model": clf, "feature_names": list(X.columns)}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"✅ Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")
    print(f"Test accuracy (with synthetic labels if no target given): {acc:.4f}")

if __name__ == "__main__":
    train_and_save()
